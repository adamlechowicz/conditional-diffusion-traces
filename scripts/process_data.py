import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob

def process_site_data(filepath, seq_length=1440):
    """
    Reads the raw NSRDB 5-minute CSV, interpolates GHI to 1-minute,
    and chunks it into daily sequences of length L=1440.
    """
    print(f"Processing {filepath}...")
    # The first 2 rows contain metadata and headers
    df = pd.read_csv(filepath, skiprows=2)

    # Create datetime index
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df = df.set_index('datetime')

    # Extract GHI
    ghi = df['GHI'].values

    # We have 5-minute data. For a full year of 365 days, we expect 365 * 24 * 12 = 105120 points
    # Let's ensure there are no missing indices by resampling to exactly 5min intervals first
    df = df.resample('5min').ffill()
    ghi = df['GHI'].values

    # To interpolate to 1-minute, we can use scipy.interpolate
    x_5min = np.arange(len(ghi)) * 5
    f_interp = interp1d(x_5min, ghi, kind='linear', fill_value="extrapolate")

    # Desired 1-minute points
    # Total minutes in the dataset
    total_minutes = len(ghi) * 5
    x_1min = np.arange(total_minutes)

    ghi_1min = f_interp(x_1min)

    # Ensure no negative irradiance
    ghi_1min = np.clip(ghi_1min, 0, None)

    # Now chunk into daily sequences
    # 1 day = 1440 minutes
    num_days = len(ghi_1min) // seq_length

    # Truncate any remainder
    ghi_1min = ghi_1min[:num_days * seq_length]

    daily_chunks = ghi_1min.reshape(num_days, seq_length)

    return daily_chunks

def main():
    raw_files = glob.glob("data/raw/nsrdb_*.csv")

    if not raw_files:
        print("No raw data files found in data/raw/")
        return

    all_chunks = []
    for fp in raw_files:
        try:
            chunks = process_site_data(fp)
            all_chunks.append(chunks)
        except Exception as e:
            print(f"Failed to process {fp}: {e}")

    if all_chunks:
        # Stack all days from all sites
        final_dataset = np.vstack(all_chunks) # (Total_Days, 1440)

        # Save as a numpy array
        out_path = "data/processed/nsrdb_1min_daily_traces.npy"
        np.save(out_path, final_dataset)

        print(f"Processing complete. Saved {final_dataset.shape[0]} daily traces to {out_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
