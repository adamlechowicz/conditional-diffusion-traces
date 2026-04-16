import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add time embedding
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb.unsqueeze(-1)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.shortcut(x)

class MambaUNet(nn.Module):
    """
    A 1D U-Net using standard 1D Convolutions for down/up sampling
    and Mamba blocks in the deep bottleneck for long sequence modeling.
    """
    def __init__(self, in_channels=1, out_channels=1, model_channels=64, num_res_blocks=2, time_emb_dim=256, mamba_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch_mults = [1, 2, 4]

        channels = model_channels
        self.down_block_chans = [model_channels]

        for level, mult in enumerate(ch_mults):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock1D(channels, out_ch, time_emb_dim))
                channels = out_ch
                self.down_block_chans.append(channels)

            if level != len(ch_mults) - 1:
                self.down_blocks.append(nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1))
                self.down_block_chans.append(channels)

        # Bottleneck (Mamba)
        from .mamba import MambaBlock

        self.mid_mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=channels) for _ in range(mamba_layers)
        ])

        self.mid_norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(mamba_layers)
        ])

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = model_channels * mult

            if level != len(ch_mults) - 1:
                self.up_blocks.append(nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1))

            num_blocks = num_res_blocks + 1 if level != len(ch_mults) - 1 else num_res_blocks
            for _ in range(num_blocks):
                skip_ch = self.down_block_chans.pop()
                self.up_blocks.append(ResidualBlock1D(channels + skip_ch, out_ch, time_emb_dim))
                channels = out_ch

        # Last skip
        skip_ch = self.down_block_chans.pop()
        self.up_blocks.append(ResidualBlock1D(channels + skip_ch, model_channels, time_emb_dim))
        channels = model_channels

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)
        )

    def get_time_embedding(self, timesteps, dim):
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -embeddings)
        embeddings = timesteps.unsqueeze(1).float() * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings

    def forward(self, x, timesteps, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        t_emb = self.get_time_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)

        h = self.init_conv(x)

        skips = [h]

        # Down
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock1D):
                h = block(h, t_emb)
                skips.append(h)
            else: # Conv1d downsample
                h = block(h)
                skips.append(h)

        # Middle
        h = h.transpose(1, 2)
        for mamba, norm in zip(self.mid_mamba_blocks, self.mid_norms):
            residual = h
            h = norm(h)
            h = mamba(h)
            h = h + residual
        h = h.transpose(1, 2)

        # Up
        for block in self.up_blocks:
            if isinstance(block, nn.ConvTranspose1d):
                h = block(h)
            elif isinstance(block, ResidualBlock1D):
                skip = skips.pop()
                if h.shape[-1] != skip.shape[-1]:
                    h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)

        out = self.out_conv(h)
        return out
