import torch
import numpy as np

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class SDEditDiffusion:
    def __init__(self, config):
        self.config = config
        self.timesteps = config.timesteps
        self.beta = linear_beta_schedule(self.timesteps, config.beta_start, config.beta_end)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.t0_ratio = config.t0_ratio

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        return self

    def q_sample(self, x_start, t, noise=None):
        """
        Forward process: Add noise to the data.
        x_t = sqrt(alpha_hat_t) * x_start + sqrt(1 - alpha_hat_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alpha_hat[t])

        # Reshape for broadcasting
        # t is (B,), we want (B, 1, 1) to broadcast with (B, C, L)
        sqrt_alpha_hat_t = sqrt_alpha_hat_t.view(-1, 1, 1)
        sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t.view(-1, 1, 1)

        return sqrt_alpha_hat_t * x_start + sqrt_one_minus_alpha_hat_t * noise

    def sdedit_inject_noise(self, x_ref, noise=None):
        """
        Injects noise into the retrieved reference trace up to timestep t0.
        Returns the noisy trace and the timestep t0.
        """
        B = x_ref.shape[0]
        # t0 is calculated as a fraction of total timesteps
        t0 = int(self.timesteps * self.t0_ratio)
        t = torch.full((B,), t0 - 1, device=x_ref.device, dtype=torch.long)

        x_noisy = self.q_sample(x_ref, t, noise)
        return x_noisy, t

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, cond=None):
        """
        Sample x_{t-1} given x_t and the model predicting noise.
        """
        betas_t = self.beta[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alpha_hat[t]).view(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alpha[t]).view(-1, 1, 1)

        # Model predicts noise
        predicted_noise = model(x, t, cond=cond)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t # using beta as variance
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sdedit_reverse_sample(self, model, x_ref, cond=None):
        """
        SDEdit full process:
        1. Inject noise up to t0
        2. Denoise from t0 down to 0
        """
        device = x_ref.device
        b = x_ref.shape[0]

        # 1. Forward process to intermediate state
        x, t_start_tensor = self.sdedit_inject_noise(x_ref)
        t_start = t_start_tensor[0].item()

        # 2. Reverse process
        for i in reversed(range(0, t_start + 1)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, cond=cond)

        return x
