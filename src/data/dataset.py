import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SolarDataset(Dataset):
    """
    A PyTorch Dataset for Solar Super-Resolution.
    Handles low-resolution weather covariates, site metadata, and high-resolution targets.
    """
    def __init__(self, config, num_samples=1000, mode="train"):
        self.config = config
        self.seq_length = config.seq_length
        self.num_samples = num_samples

        # We will simulate mock data for now.
        # In reality, this would load from disk/xarray/pandas
        print(f"Initializing {mode} dataset with {num_samples} mock samples...")
        self.data = self._generate_mock_dataset()

    def _generate_mock_dataset(self):
        dataset = []
        t = np.linspace(0, np.pi, self.seq_length)
        base_curve = np.sin(t)

        for _ in range(self.num_samples):
            # 1. High-Res Target (1-minute, L=1440)
            noise = np.random.normal(0, 0.1, self.seq_length)
            hr_target = base_curve + noise
            hr_target = np.clip(hr_target, 0, None)

            # 2. Low-Res Conditioning
            # Simulate hourly data (e.g., ERA5 cloud cover), shape (24,)
            # We interpolate it up to (1440,) for simpler network ingestion
            lr_hourly = hr_target.reshape(24, 60).mean(axis=1) + np.random.normal(0, 0.05, 24)
            lr_cond = np.repeat(lr_hourly, 60)

            # 3. Site Metadata Conditioning
            # Simulate continuous metadata (e.g., capacity, azimuth, shading)
            # We treat this as a static vector that we tile to sequence length,
            # or pass through a separate conditioning mechanism.
            # Here we create a (1, L) vector of constant site features.
            site_meta = np.ones(self.seq_length) * np.random.uniform(0.5, 1.5)

            # Stack conditioning: (2, L) -> [lr_weather, site_meta]
            cond = np.stack([lr_cond, site_meta], axis=0)

            # Also create a "mock retrieved template" to simulate the RAG output
            # In a real pipeline, this happens dynamically or is pre-computed
            template_noise = np.random.normal(0, 0.05, self.seq_length)
            retrieved_template = base_curve + template_noise
            retrieved_template = np.clip(retrieved_template, 0, None)

            dataset.append({
                "hr_target": torch.tensor(hr_target, dtype=torch.float32).unsqueeze(0), # (1, L)
                "cond": torch.tensor(cond, dtype=torch.float32), # (2, L)
                "template": torch.tensor(retrieved_template, dtype=torch.float32).unsqueeze(0) # (1, L)
            })

        return dataset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(config, mode="train", num_samples=1000):
    dataset = SolarDataset(config, num_samples=num_samples, mode=mode)
    shuffle = True if mode == "train" else False

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0, # Keeping 0 for simpler local execution
        drop_last=True if mode == "train" else False
    )
    return loader
