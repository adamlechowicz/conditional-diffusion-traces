import torch
import numpy as np
from scipy import signal

class Evaluator:
    def __init__(self, config):
        self.config = config

    def compute_rmse(self, pred, target):
        """
        Pointwise RMSE metric. Heavily penalizes stochastic shifts.
        pred, target: (B, C, L)
        """
        mse = torch.mean((pred - target) ** 2)
        return torch.sqrt(mse).item()

    def compute_mae(self, pred, target):
        return torch.mean(torch.abs(pred - target)).item()

    def compute_ramp_rate_wasserstein(self, pred, target):
        """
        Compare the distribution of 1-minute ramp rates (x_t - x_{t-1})
        using an approximation of Wasserstein distance or simply summary stats.
        Since we want this fast, we will compute the difference in standard deviations
        of ramp rates as a proxy for distributional similarity in the tails.
        """
        # Ramp rates
        pred_rr = pred[:, :, 1:] - pred[:, :, :-1]
        target_rr = target[:, :, 1:] - target[:, :, :-1]

        pred_std = torch.std(pred_rr).item()
        target_std = torch.std(target_rr).item()

        # We return the absolute difference in standard deviation of ramp rates
        # Lower is better (distributions match better)
        return abs(pred_std - target_std)

    def compute_psd_error(self, pred, target):
        """
        Frequency Domain evaluation: Compare Power Spectral Density (PSD)
        To verify the correct spectrum of cloud-edge transients.
        """
        pred_np = pred.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()

        if pred_np.ndim == 1:
            pred_np = pred_np[np.newaxis, :]
            target_np = target_np[np.newaxis, :]

        b, l = pred_np.shape
        total_error = 0

        for i in range(b):
            f_p, psd_p = signal.welch(pred_np[i], nperseg=256)
            f_t, psd_t = signal.welch(target_np[i], nperseg=256)

            # Log spectral distance
            err = np.mean(np.abs(np.log10(psd_p + 1e-8) - np.log10(psd_t + 1e-8)))
            total_error += err

        return total_error / b

    def evaluate(self, generated, target):
        """
        Evaluates a batch of generated traces against target traces.
        """
        metrics = {
            "RMSE": self.compute_rmse(generated, target),
            "MAE": self.compute_mae(generated, target),
            "Ramp_Rate_STD_Diff": self.compute_ramp_rate_wasserstein(generated, target),
            "Log_Spectral_Distance": self.compute_psd_error(generated, target)
        }
        return metrics
