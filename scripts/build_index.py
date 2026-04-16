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
    parser.add_argument("--num_samples", type=int, default=15000, help="Number of mock traces to generate")
    parser.add_argument("--out_path", type=str, default="canonical_shapes.index", help="Path to save the Faiss index")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} mock 1-minute solar traces...")
    mock_data = generate_mock_data(args.num_samples, config.seq_length)

    print("Initializing Vector DB...")
    db = VectorDB(config)

    db.build_index(mock_data)

    db.save(args.out_path)
    print(f"Index successfully saved to {args.out_path}")

if __name__ == "__main__":
    main()
