#!/usr/bin/env python3
"""
==============================================================================
ATM 531 - HW #3a: Mesonet vs TMY Solar Radiation Comparison
==============================================================================

Compares NYS Mesonet flux-tower incoming shortwave (SW_IN) against NSRDB TMY
Global Horizontal Irradiance (GHI) for Albany and NYC.

Data mapping:
  - FLUX_BKLN (Brooklyn)   <-> 5037116_40.78_-73.97 (NYC TMY)
  - FLUX_VOOR (Voorheesville) <-> 5073330_42.66_-73.75 (Albany TMY)

Both datasets are half-hourly and in UTC. The script:
  1. Discovers matching year coverage (only includes years present in BOTH
     the Mesonet and TMY datasets).
  2. Aggregates half-hourly -> hourly means.
  3. Filters to daytime (value > 0 in BOTH sources for a given hour).
  4. Computes multi-year monthly mean of daytime hourly radiation.
  5. Produces per-site:
       - A 12-month line chart (Mesonet vs TMY)
       - A summary table with difference column

Output : PNG charts + printed tables (+ optional CSV export).

USAGE:
  1. Set MESO_DIR and TMY_DIR below to your local data folders.
  2. Run:  python compare_mesonet_tmy.py
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, glob, re, warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION -- edit these paths to match your local setup
# =====================================================================

MESO_DIR = r"D:\TMY_Data\NYSM"     # folder with Flux_NYSMesonet_YYYYMM.csv
TMY_DIR  = r"D:\TMY_Data"         # folder with NSRDB CSVs
OUT_DIR  = r"D:\TMY_Data"         # where to save charts / CSVs

# Site mapping:  mesonet_stid -> (tmy_file_pattern, display_name)
SITE_MAP = {
    'FLUX_BKLN': {
        'tmy_pattern': '5037116_40.78_-73.97',
        'name': 'NYC (Brooklyn)',
    },
    'FLUX_VOOR': {
        'tmy_pattern': '5073330_42.66_-73.75',
        'name': 'Albany (Voorheesville)',
    },
}

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']


# =====================================================================
# 1. DATA LOADING
# =====================================================================

def load_mesonet_files(meso_dir):
    """
    Load all Flux_NYSMesonet_YYYYMM.csv files.
    Returns a dict:  { stid: DataFrame with [datetime_utc, SW_IN, year, month] }
    """
    pattern = os.path.join(meso_dir, "Flux_NYSMesonet_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        # try underscores (cloud upload renames)
        pattern = os.path.join(meso_dir, "Flux_NYSMesonet_*.csv")
        print(pattern)
        files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} Mesonet file(s)")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        # Standardize column names (strip whitespace)
        df.columns = df.columns.str.strip()
        frames.append(df)

    if not frames:
        return {}

    all_meso = pd.concat(frames, ignore_index=True)
    all_meso['datetime_utc'] = pd.to_datetime(all_meso['datetime'], errors='coerce')
    all_meso = all_meso.dropna(subset=['datetime_utc'])
    all_meso['year']  = all_meso['datetime_utc'].dt.year
    all_meso['month'] = all_meso['datetime_utc'].dt.month

    # Split by station
    result = {}
    for stid in all_meso['stid'].unique():
        sub = all_meso[all_meso['stid'] == stid][
            ['datetime_utc', 'SW_IN', 'year', 'month']].copy()
        sub['SW_IN'] = pd.to_numeric(sub['SW_IN'], errors='coerce')
        result[stid.strip()] = sub
    return result


def load_tmy_files(tmy_dir, file_pattern):
    """
    Load all NSRDB CSVs matching a site pattern.
    Returns DataFrame with [datetime_utc, GHI, year, month].
    """
    all_files = os.listdir(tmy_dir)
    # Match with both dot and underscore variants
    pat_dot   = file_pattern
    pat_under = file_pattern.replace('.', '_')

    matched = sorted([
        os.path.join(tmy_dir, f) for f in all_files
        if f.endswith('.csv') and (pat_dot in f or pat_under in f)
    ])
    print(f"  TMY files for pattern '{file_pattern}': {len(matched)}")

    frames = []
    for fp in matched:
        df = pd.read_csv(fp, skiprows=2)
        df['datetime_utc'] = pd.to_datetime(
            df[['Year','Month','Day','Hour','Minute']].rename(
                columns={'Year':'year','Month':'month','Day':'day',
                         'Hour':'hour','Minute':'minute'}))
        df = df[['datetime_utc', 'GHI', 'Year', 'Month']].copy()
        df.columns = ['datetime_utc', 'GHI', 'year', 'month']
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =====================================================================
# 2. AGGREGATE & COMPARE
# =====================================================================

def aggregate_hourly(df, value_col):
    """
    Aggregate half-hourly data to hourly means.
    Groups by the floored hour, returns [datetime_hr, value, year, month].
    """
    df = df.copy()
    df['datetime_hr'] = df['datetime_utc'].dt.floor('h')
    hourly = df.groupby('datetime_hr').agg({
        value_col: 'mean',
        'year':    'first',
        'month':   'first',
    }).reset_index()
    return hourly


def compute_monthly_comparison(meso_hourly, tmy_hourly, overlapping_years):
    """
    For each month (1-12), compute the mean daytime hourly radiation
    across all overlapping years, filtering to hours where BOTH sources > 0.

    Returns DataFrame with columns:
        Month, Month_Name, Meso_Mean, TMY_Mean, Difference, Pct_Diff
    """
    # Filter to overlapping years
    meso = meso_hourly[meso_hourly['year'].isin(overlapping_years)].copy()
    tmy  = tmy_hourly[tmy_hourly['year'].isin(overlapping_years)].copy()

    # Merge on datetime_hr
    merged = pd.merge(
        meso[['datetime_hr', 'SW_IN', 'month']],
        tmy[['datetime_hr', 'GHI']],
        on='datetime_hr', how='inner'
    )

    # Keep only daytime hours (both > 0)
    daytime = merged[(merged['SW_IN'] > 0) & (merged['GHI'] > 0)]

    if daytime.empty:
        print("    WARNING: No overlapping daytime hours found.")
        return pd.DataFrame()

    # Monthly means
    monthly = daytime.groupby('month').agg(
        Meso_Mean=('SW_IN', 'mean'),
        TMY_Mean=('GHI', 'mean'),
        N_hours=('SW_IN', 'count'),
    ).reset_index()

    monthly['Difference']  = monthly['TMY_Mean'] - monthly['Meso_Mean']
    monthly['Pct_Diff']    = (monthly['Difference'] / monthly['Meso_Mean']) * 100
    monthly['Month_Name']  = monthly['month'].apply(lambda m: MONTH_NAMES[m-1])

    # Reorder
    monthly = monthly[['month', 'Month_Name', 'Meso_Mean', 'TMY_Mean',
                        'Difference', 'Pct_Diff', 'N_hours']]
    return monthly


# =====================================================================
# 3. PLOTTING & REPORTING
# =====================================================================

def plot_comparison(monthly, site_name, out_dir):
    """
    Line chart: monthly average daytime hourly solar radiation,
    one line for Mesonet and one for TMY.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(monthly['month'], monthly['Meso_Mean'],
            'o-', color='#E24A33', linewidth=2, markersize=6,
            label='Mesonet (SW_IN)')
    ax.plot(monthly['month'], monthly['TMY_Mean'],
            's-', color='#348ABD', linewidth=2, markersize=6,
            label='TMY (GHI)')

    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Mean Daytime Hourly Irradiance (W/m²)', fontsize=11)
    ax.set_title(f'{site_name}: Mesonet vs TMY Solar Radiation\n'
                 f'(Multi-Year Average, Daytime Hours Only)', fontsize=12)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir,
                         f"{site_name.replace(' ','_').replace('(','').replace(')','')}"
                         f"_mesonet_vs_tmy.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  Chart saved -> {fname}")


def print_summary_table(monthly, site_name):
    """Print a formatted summary table."""
    print(f"\n{'='*72}")
    print(f"  {site_name}: Monthly Mean Daytime Hourly Solar Radiation")
    print(f"  (Multi-year average, only hours where both Mesonet & TMY > 0)")
    print(f"{'='*72}")
    print(f"  {'Month':>5}  {'Mesonet':>10}  {'TMY':>10}  "
          f"{'Diff':>10}  {'Diff %':>8}  {'N_hrs':>6}")
    print(f"  {'':>5}  {'(W/m²)':>10}  {'(W/m²)':>10}  "
          f"{'(W/m²)':>10}  {'':>8}  {'':>6}")
    print(f"  {'-'*62}")
    for _, r in monthly.iterrows():
        print(f"  {r['Month_Name']:>5}  {r['Meso_Mean']:10.1f}  "
              f"{r['TMY_Mean']:10.1f}  {r['Difference']:+10.1f}  "
              f"{r['Pct_Diff']:+7.1f}%  {r['N_hours']:6.0f}")

    # Annual summary
    annual_meso = monthly['Meso_Mean'].mean()
    annual_tmy  = monthly['TMY_Mean'].mean()
    annual_diff = annual_tmy - annual_meso
    annual_pct  = (annual_diff / annual_meso) * 100 if annual_meso > 0 else 0
    print(f"  {'-'*62}")
    print(f"  {'Ann':>5}  {annual_meso:10.1f}  {annual_tmy:10.1f}  "
          f"{annual_diff:+10.1f}  {annual_pct:+7.1f}%")


# =====================================================================
# 4. MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("ATM 531 - Mesonet vs TMY Solar Radiation Comparison")
    print("=" * 72)

    # --- Load all Mesonet data ---
    meso_data = load_mesonet_files(MESO_DIR)
    if not meso_data:
        print("ERROR: No Mesonet files found. Check MESO_DIR path.")
        return

    print(f"Mesonet stations found: {list(meso_data.keys())}")

    # --- Process each site ---
    for stid, cfg in SITE_MAP.items():
        site_name = cfg['name']
        print(f"\n--- {site_name} ({stid}) ---")

        if stid not in meso_data:
            print(f"  Mesonet station '{stid}' not found in data. Skipping.")
            continue

        meso_df = meso_data[stid]
        meso_years = set(meso_df['year'].unique())
        print(f"  Mesonet years: {sorted(meso_years)}")

        # Load TMY
        tmy_df = load_tmy_files(TMY_DIR, cfg['tmy_pattern'])
        if tmy_df.empty:
            print(f"  No TMY files found for {cfg['tmy_pattern']}. Skipping.")
            continue

        tmy_years = set(tmy_df['year'].unique())
        print(f"  TMY years:     {sorted(tmy_years)}")

        # Overlapping years only
        overlap = sorted(meso_years & tmy_years)
        if not overlap:
            print(f"  No overlapping years between Mesonet and TMY. Skipping.")
            continue
        print(f"  Overlapping:   {overlap}")

        # Aggregate to hourly
        meso_hourly = aggregate_hourly(meso_df, 'SW_IN')
        tmy_hourly  = aggregate_hourly(tmy_df,  'GHI')

        # Compute monthly comparison
        monthly = compute_monthly_comparison(meso_hourly, tmy_hourly, overlap)
        if monthly.empty:
            continue

        # Print table
        print_summary_table(monthly, site_name)

        # Plot
        plot_comparison(monthly, site_name, OUT_DIR)

        # Save CSV
        csv_out = os.path.join(OUT_DIR,
            f"{site_name.replace(' ','_').replace('(','').replace(')','')}"
            f"_mesonet_vs_tmy.csv")
        monthly.to_csv(csv_out, index=False, float_format='%.1f')
        print(f"  Table saved  -> {csv_out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
