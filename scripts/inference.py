import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.config import config
from src.retrieval.vector_db import VectorDB
from src.models.unet import MambaUNet
from src.models.diffusion import SDEditDiffusion
from src.data.dataset import get_dataloader
from src.pipeline.evaluator import Evaluator

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 1. Setup Models
    print("Loading architecture components...")
    model = MambaUNet(
        in_channels=3,
        out_channels=1,
        model_channels=config.d_model,
        num_res_blocks=2,
        mamba_layers=2
    ).to(device)

    # In a real run, we would load the trained weights
    # if os.path.exists("best_model.pth"):
    #     model.load_state_dict(torch.load("best_model.pth", map_location=device))

    model.eval()

    diffusion = SDEditDiffusion(config).to(device)

    # 2. Setup Vector DB
    db = VectorDB(config)
    if os.path.exists("canonical_shapes.index"):
        print("Loading FAISS Index...")
        db.load("canonical_shapes.index")
    else:
        print("Warning: canonical_shapes.index not found. Skipping retrieval step.")
        # In this mock demo we will use the template from the dataloader if DB is missing

    # 3. Get test data
    test_loader = get_dataloader(config, mode="test", num_samples=8)
    batch = next(iter(test_loader))

    cond = batch["cond"].to(device) # (B, 2, L)
    target = batch["hr_target"].to(device) # (B, 1, L)

    b = cond.shape[0]

    # 4. Retrieval
    print("Phase 2: Retrieving Canonical Shapes...")
    if db.is_trained:
        # We query using the low-res conditioning (downsampled, or raw)
        # For simplicity in this demo, we extract the first channel of cond as query
        queries = cond[:, 0, :].cpu().numpy()
        retrieved_templates = db.retrieve(queries, k=1) # (B, 1, L)
        x_ref = torch.tensor(retrieved_templates, dtype=torch.float32).to(device)
    else:
        x_ref = batch["template"].to(device)

    # 5. SDEdit Reverse Generation
    print("Phase 3 & 4: Applying SDEdit & Mamba-UNet Reverse Diffusion...")
    with torch.no_grad():
        generated_traces = diffusion.sdedit_reverse_sample(model, x_ref, cond=cond)

    # 6. Evaluation
    print("Phase 5: Evaluation...")
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(generated_traces, target)

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nSuccess: End-to-end inference pipeline completed.")

if __name__ == "__main__":
    main()
