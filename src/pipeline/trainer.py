import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, config, model, diffusion, train_loader, val_loader, device):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs} [Train]")
        for batch in pbar:
            self.optimizer.zero_grad()

            x_start = batch["hr_target"].to(self.device)
            cond = batch["cond"].to(self.device)

            # Sample random timesteps
            b = x_start.shape[0]
            t = torch.randint(0, self.config.timesteps, (b,), device=self.device).long()

            # Generate noise
            noise = torch.randn_like(x_start)

            # Forward process: get noisy data at timestep t
            x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

            # Predict noise
            predicted_noise = self.model(x=x_noisy, timesteps=t, cond=cond)

            # Compute MSE loss between true noise and predicted noise
            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0

        for batch in self.val_loader:
            x_start = batch["hr_target"].to(self.device)
            cond = batch["cond"].to(self.device)

            b = x_start.shape[0]
            t = torch.randint(0, self.config.timesteps, (b,), device=self.device).long()

            noise = torch.randn_like(x_start)
            x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

            predicted_noise = self.model(x=x_noisy, timesteps=t, cond=cond)
            loss = F.mse_loss(predicted_noise, noise)

            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        print(f"Starting training on {self.device} for {self.config.epochs} epochs...")

        best_val_loss = float('inf')

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print("  [*] Saved new best model.")

        print("Training complete.")
