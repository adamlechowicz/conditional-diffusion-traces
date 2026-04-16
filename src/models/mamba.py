import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaBlock(nn.Module):
    """
    A pure PyTorch implementation of a Mamba (Selective State Space Model) block.
    This avoids custom CUDA kernels, ensuring compatibility with MPS on Apple Silicon.

    Reference: Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
    """
    def __init__(self, d_model, d_state=16, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=4,
            groups=self.d_inner,
            padding=3, # (kernel_size - 1)
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A and D parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D) B: Batch size, L: Sequence length, D: Model dimension
        """
        B, L, D = x.shape

        # 1. Project input
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 2. 1D Convolution
        x = x.transpose(1, 2) # (B, D_inner, L)
        x = self.conv1d(x)[:, :, :L] # Adjust length for causal padding
        x = x.transpose(1, 2) # (B, L, D_inner)
        x = F.silu(x)

        # 3. State Space Model (SSM) computations
        # We need to compute A, B, C, dt
        x_dbl = self.x_proj(x) # (B, L, dt_rank + 2*d_state)
        dt, B_val, C_val = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt) # (B, L, D_inner)
        dt = F.softplus(dt) # softplus to ensure dt is positive

        # Discretize A
        A = -torch.exp(self.A_log.float()) # (D_inner, d_state)

        # Perform SSM recurrence
        # In a pure PyTorch naive implementation, we can use a loop for sequence dimension
        # Note: This loop is slow for long sequences. For a more performant pure-PT approach,
        # one could use parallel associative scan, but a loop works as a fallback baseline.

        y = self._selective_scan(x, dt, A, B_val, C_val) # (B, L, D_inner)

        # 4. Multiply with D
        y = y + x * self.D

        # 5. Multiply with z (gating)
        y = y * F.silu(z)

        # 6. Output projection
        out = self.out_proj(y)

        return out

    def _selective_scan(self, u, delta, A, B, C):
        """
        Naive sequential selective scan for pure PyTorch fallback.
        u: (B, L, D_inner)
        delta: (B, L, D_inner)
        A: (D_inner, d_state)
        B: (B, L, d_state)
        C: (B, L, d_state)
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # Initialize hidden state h
        h = torch.zeros((batch_size, d_inner, d_state), device=u.device, dtype=u.dtype)
        ys = []

        # Pre-compute discretized B values
        # delta is (B, L, D_inner), B is (B, L, d_state)
        # We need delta * B: delta.unsqueeze(-1) * B.unsqueeze(-2) -> (B, L, D_inner, d_state)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)

        # delta * A: delta.unsqueeze(-1) * A -> (B, L, D_inner, d_state)
        delta_A = delta.unsqueeze(-1) * A
        delta_A_exp = torch.exp(delta_A)

        for i in range(seq_len):
            u_i = u[:, i, :] # (B, D_inner)
            delta_A_exp_i = delta_A_exp[:, i, :, :] # (B, D_inner, d_state)
            delta_B_i = delta_B[:, i, :, :] # (B, D_inner, d_state)
            C_i = C[:, i, :] # (B, d_state)

            # h_t = exp(delta * A) * h_{t-1} + (delta * B) * u_t
            h = delta_A_exp_i * h + delta_B_i * u_i.unsqueeze(-1)

            # y_t = C * h_t
            # h is (B, D_inner, d_state), C_i is (B, d_state)
            # We want to multiply h by C and sum over d_state to get (B, D_inner)
            y_i = torch.einsum('bns,bs->bn', h, C_i)
            ys.append(y_i)

        y = torch.stack(ys, dim=1) # (B, L, D_inner)
        return y