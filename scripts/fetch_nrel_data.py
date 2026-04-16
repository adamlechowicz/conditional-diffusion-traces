import os
import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

# Get credentials from environment
API_KEY = os.environ.get("NREL_API_KEY")
EMAIL = os.environ.get("NREL_EMAIL")

if not API_KEY or not EMAIL:
    print("Error: NREL_API_KEY and NREL_EMAIL environment variables must be set.")
    exit(1)

# Base URL for the NSRDB PSM v3 1-minute API (NOTE: nlr.gov transition)
BASE_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/psm3-5min-download.csv"
# The API documentation mentions psm3-download for 30/60m and psm3-5min-download for 5m.
# However, 1-minute data from NSRDB is typically found in the Oahu or SRRL datasets,
# or we use the lowest resolution available from PSM (which is 5-min for CONUS).
# Let's target the SRRL (Solar Radiation Research Laboratory) 1-minute data endpoint if possible,
# or default to the standard PSM 5-min and interpolate.
# Wait, the prompt says "1 year of 1-minute solar irradiance data".
# For true 1-minute data across diverse coordinates, we might not get it from standard PSM3 (which is 5min or 30min).
# Let's use the PSM3 5-min data and we will temporally interpolate it to 1-minute in the processing step,
# OR we query the Oahu grid if available. For "diversity of site coordinates", PSM3 5-min is the most robust.

def fetch_data(lat, lon, year="2019", name="user"):
    # According to the NLR documentation, standard PSM 5-min is available via:
    # https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download
    # Wait, the search snippet showed:
    # GOES Conus: PSM v4 Download (/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download)
    # Temporal intervals of 5 minutes.

    url = f"https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"

    params = {
        "api_key": API_KEY,
        "wkt": f"POINT({lon} {lat})",
        "names": year,
        "leap_day": "false",
        "interval": "5",
        "utc": "false",
        "full_name": name,
        "email": EMAIL,
        "affiliation": "Research",
        "reason": "research",
        "attributes": "ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle"
    }

    # Send request
    print(f"Fetching data for ({lat}, {lon}) Year {year}...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Save to raw
        out_path = f"data/raw/nsrdb_{lat}_{lon}_{year}.csv"
        with open(out_path, "wb") as f:
            f.write(response.content)

        print(f"Successfully saved {out_path}")
        return True
    except Exception as e:
        print(f"Failed to fetch ({lat}, {lon}): {e}")
        return False

def main():
    # Diverse set of coordinates (lat, lon) within CONUS (Continental US) to ensure PSM v4 availability
    # 1. Oahu, HI (might not be in CONUS, but worth trying, or use desert)
    # 2. Golden, CO (NREL SRRL)
    # 3. Death Valley, CA (Clear sky, high irradiance)
    # 4. Seattle, WA (Cloudy, dynamic)
    # 5. Miami, FL (Tropical, intermittent clouds)

    sites = [
        (39.74, -105.18), # Golden, CO
        (36.45, -116.86), # Death Valley, CA
        (47.60, -122.33), # Seattle, WA
        (25.76, -80.19),  # Miami, FL
        (33.44, -112.07), # Phoenix, AZ
        (42.36, -71.05),  # Boston, MA
    ]

    year = "2019" # Pre-pandemic year, usually fully complete in datasets

    # We will fetch them sequentially to avoid rate limits
    for lat, lon in sites:
        fetch_data(lat, lon, year)
        time.sleep(2) # brief pause

if __name__ == "__main__":
    main()
