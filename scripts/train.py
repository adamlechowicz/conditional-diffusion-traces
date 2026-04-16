import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import config
from src.data.dataset import get_dataloader
from src.models.unet import MambaUNet
from src.models.diffusion import SDEditDiffusion
from src.pipeline.trainer import Trainer

def main():
    # 1. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Update config to ensure short epochs for mock demonstration
    config.epochs = 2
    config.batch_size = 8

    # 2. Data
    print("Loading data...")
    train_loader = get_dataloader(config, mode="train", num_samples=160)
    val_loader = get_dataloader(config, mode="val", num_samples=32)

    # 3. Model Architecture
    # in_channels = 1 (target) + 2 (conditioning) = 3
    print("Initializing Mamba-UNet...")
    model = MambaUNet(
        in_channels=3,
        out_channels=1,
        model_channels=config.d_model,
        num_res_blocks=2,
        mamba_layers=2
    ).to(device)

    # 4. Diffusion Process
    print("Initializing SDEdit Diffusion...")
    diffusion = SDEditDiffusion(config).to(device)

    # 5. Trainer
    trainer = Trainer(
        config=config,
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # 6. Run Training
    trainer.train()

if __name__ == "__main__":
    main()
