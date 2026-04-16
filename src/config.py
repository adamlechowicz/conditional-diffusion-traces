from pydantic import BaseModel

class Config(BaseModel):
    # Data settings
    seq_length: int = 1440  # 1-minute data for a day (24 * 60)
    num_clusters: int = 10000

    # Model architecture
    d_model: int = 64
    n_layers: int = 4
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 4

    # Diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # SDEdit
    t0_ratio: float = 0.5  # Add noise up to this fraction of max timesteps

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10

    # Hardware
    device: str = "mps" # Will be evaluated dynamically, fallback to cpu

config = Config()
