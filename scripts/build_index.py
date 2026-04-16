import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from src.config import config
from src.retrieval.vector_db import VectorDB

def generate_mock_data(num_samples, seq_length):
    """
    Generates mock high-res solar data for testing.
    Shape: a smooth daily curve (like a sine wave) + high frequency noise
    """
    t = np.linspace(0, np.pi, seq_length)
    base_curve = np.sin(t)

    data = []
    for _ in range(num_samples):
        noise = np.random.normal(0, 0.1, seq_length)
        trace = base_curve + noise
        # Ensure non-negative
        trace = np.clip(trace, 0, None)
        data.append(trace)

    return np.array(data, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/nsrdb_1min_daily_traces.npy", help="Path to processed numpy data")
    parser.add_argument("--out_path", type=str, default="canonical_shapes.index", help="Path to save the Faiss index")
    args = parser.parse_args()

    if os.path.exists(args.data_path):
        print(f"Loading real data from {args.data_path}...")
        data = np.load(args.data_path)
    else:
        print(f"Warning: {args.data_path} not found. Falling back to mock data.")
        data = generate_mock_data(15000, config.seq_length)

    print("Initializing Vector DB...")
    # For a small dataset of ~2190 traces, we don't need 10,000 clusters.
    # We will adjust num_clusters based on the data size if it's smaller than the configured amount.
    if data.shape[0] < config.num_clusters:
        config.num_clusters = data.shape[0] // 2 # Just an arbitrary clustering factor for small datasets

    db = VectorDB(config)

    db.build_index(data)

    db.save(args.out_path)
    print(f"Index successfully saved to {args.out_path}")

if __name__ == "__main__":
    main()
