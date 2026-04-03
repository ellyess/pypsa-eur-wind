#!/usr/bin/env python3
"""
Fetch hourly wind generation (onshore + offshore) from ENTSO-E Transparency
Platform for all tier-2 countries and save as a multi-header CSV matching
the format of data/entsoe_generation_2023.csv.

Usage:
    python scripts/fetch_entsoe_wind_generation.py --api-key YOUR_KEY
    # or set the environment variable:
    ENTSOE_API_KEY=YOUR_KEY python scripts/fetch_entsoe_wind_generation.py

Output:
    data/entsoe_generation_2023.csv  (overwrites existing file)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# Tier-2 countries (PyPSA-Eur ISO codes)
TIER2_COUNTRIES = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "ES",
    "FR", "GB", "GR", "HR", "IE", "IT", "LU", "ME", "MK", "NL",
    "NO", "PT", "RS", "SE", "SI",
]

# ENTSO-E uses different codes for some countries
ENTSOE_COUNTRY_MAP = {
    "GR": "GR",  # entsoe-py accepts GR
    "GB": "GB",  # entsoe-py accepts GB
}

OUTPATH = "data/entsoe_generation_2023.csv"
START = "20230101"
END = "20240101"


def fetch_wind_generation(
    client: EntsoePandasClient,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch wind generation for a single country.

    Returns DataFrame with columns 'Wind Onshore' and/or 'Wind Offshore',
    hourly, in MW. Returns None if no data available.
    """
    entsoe_code = ENTSOE_COUNTRY_MAP.get(country, country)

    try:
        gen = client.query_generation(entsoe_code, start=start, end=end, nett=True)
    except NoMatchingDataError:
        return None
    except Exception as e:
        print(f"    Error fetching {country}: {e}")
        return None

    # Resample to hourly and strip timezone
    gen = gen.tz_localize(None) if gen.index.tz is not None else gen
    gen = gen.resample("1h").mean()
    gen = gen.loc[start.tz_localize(None):end.tz_localize(None)]

    # Keep only wind columns
    wind_cols = [c for c in gen.columns if "Wind" in str(c)]
    if not wind_cols:
        return None

    result = pd.DataFrame(index=gen.index)
    for carrier in ["Wind Onshore", "Wind Offshore"]:
        matching = [c for c in wind_cols if carrier in str(c)]
        if matching:
            result[carrier] = gen[matching].sum(axis=1)

    if result.empty or result.columns.empty:
        return None

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ENTSO-E wind generation for tier-2 countries"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ENTSOE_API_KEY"),
        help="ENTSO-E API key (or set ENTSOE_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        default=OUTPATH,
        help=f"Output CSV path (default: {OUTPATH})",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key provided.")
        print("  Use --api-key YOUR_KEY or set ENTSOE_API_KEY environment variable.")
        print("  Register at https://transparency.entsoe.eu/ to get a free API key.")
        sys.exit(1)

    client = EntsoePandasClient(api_key=args.api_key)
    start = pd.Timestamp(START, tz="Europe/Brussels")
    end = pd.Timestamp(END, tz="Europe/Brussels")

    country_dfs = {}
    unavailable = []

    for country in TIER2_COUNTRIES:
        print(f"  Fetching {country}...", end=" ", flush=True)
        df = fetch_wind_generation(client, country, start, end)
        if df is not None and not df.empty:
            country_dfs[country] = df
            cols = list(df.columns)
            print(f"OK ({', '.join(cols)}, {len(df)} rows)")
        else:
            unavailable.append(country)
            print("no wind data")
        # Small delay to avoid rate limiting
        time.sleep(0.5)

    if not country_dfs:
        print("Error: No wind data retrieved for any country.")
        sys.exit(1)

    if unavailable:
        print(f"\n  Countries without wind data: {', '.join(unavailable)}")

    # Build multi-header DataFrame: (country, carrier)
    pieces = []
    for cc, df in sorted(country_dfs.items()):
        for col in df.columns:
            s = df[col].copy()
            s.name = (cc, col)
            pieces.append(s)

    result = pd.concat(pieces, axis=1)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["country", "carrier"])
    result.index.name = "timestamp"

    # Trim to exact 2023 year
    result = result.loc["2023-01-01":"2023-12-31 23:00:00"]

    result.to_csv(args.output)
    n_countries = len(country_dfs)
    print(f"\n  Saved {args.output}: {n_countries} countries, {len(result)} timesteps, {len(result.columns)} columns")


if __name__ == "__main__":
    main()
