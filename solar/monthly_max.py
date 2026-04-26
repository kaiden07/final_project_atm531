"""
==============================================================================
ATM 531 - HW #3a: Compute Monthly Mean-Maximum Tilted Solar Radiation
==============================================================================
Loads multi-year NSRDB data for Albany & NYC, computes hourly tilted-surface
radiation (isotropic model), then for each month finds the mean of the daily
maximum hourly tilted irradiance across all years.

Outputs a small CSV per site that the JS script reads to build the Word table.

Workflow:
  1. For each year-file, compute hourly I_T_iso for every hour.
  2. For each day, find the maximum hourly I_T_iso.
  3. Group those daily maxima by month, average across all years/days.
  4. Result: 12-row table (month, mean_max_IT_iso).
==============================================================================
"""

import numpy as np
import pandas as pd
import os, sys, warnings
warnings.filterwarnings('ignore')

# ---- CONFIGURATION (edit these to match your local paths) ----
DATA_DIR = r"D:\TMY_Data"   # <-- set to your local TMY data folder

SITES = {
    'Albany': {
        'file_pattern': '5073330_42.66_-73.75',
        'latitude':  42.66,
        'longitude': -73.75,
    },
    'NYC': {
        'file_pattern': '5037116_40.78_-73.97',
        'latitude':  40.78,
        'longitude': -73.97,
    },
}

AZIMUTH = 0.0
RHO_G   = 0.2
G_SC    = 1367.0

# ---- Math helpers ----
d2r = np.deg2rad
r2d = np.rad2deg

def solar_declination(n):
    return 23.45 * np.sin(d2r(360.0 * (284.0 + n) / 365.0))

def extraterrestrial_normal(n):
    return G_SC * (1.0 + 0.033 * np.cos(d2r(360.0 * n / 365.0)))

def hour_angle(t_sol):
    return (t_sol - 12.0) * 15.0

def cos_zenith(phi, delta, omega):
    p, d, w = d2r(phi), d2r(delta), d2r(omega)
    return np.cos(p)*np.cos(d)*np.cos(w) + np.sin(p)*np.sin(d)

def cos_incidence(phi, delta, omega, beta, gamma=0.0):
    p, d_, w = d2r(phi), d2r(delta), d2r(omega)
    b, g = d2r(beta), d2r(gamma)
    return (np.sin(d_)*np.sin(p)*np.cos(b)
          - np.sin(d_)*np.cos(p)*np.sin(b)*np.cos(g)
          + np.cos(d_)*np.cos(p)*np.cos(b)*np.cos(w)
          + np.cos(d_)*np.sin(p)*np.sin(b)*np.cos(g)*np.cos(w)
          + np.cos(d_)*np.sin(b)*np.sin(g)*np.sin(w))

def beam_tilt_ratio(ct, cz):
    cz_safe = np.where(cz > 0.01, cz, 1.0)
    return np.where((ct > 0) & (cz > 0.01), ct / cz_safe, 0.0)

def isotropic_model(I, I_d, Rb, beta_deg, rho_g):
    b = d2r(beta_deg)
    I_b = I - I_d
    return np.maximum(
        I_b * Rb
        + I_d * (1.0 + np.cos(b)) / 2.0
        + I * rho_g * (1.0 - np.cos(b)) / 2.0, 0.0)

def equation_of_time(n):
    B = d2r(360.0 * (n - 81.0) / 365.0)
    return 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B)

def utc_to_solar_time(hour_utc, minute_utc, doy, longitude):
    EoT = equation_of_time(doy)
    LC  = 4.0 * longitude
    clock_utc = hour_utc + minute_utc / 60.0
    return clock_utc + (LC + EoT) / 60.0

# ---- File I/O ----
def read_nsrdb(filepath):
    df = pd.read_csv(filepath, skiprows=2)
    df['datetime_utc'] = pd.to_datetime(
        df[['Year','Month','Day','Hour','Minute']].rename(
            columns={'Year':'year','Month':'month','Day':'day',
                     'Hour':'hour','Minute':'minute'}))
    df['doy'] = df['datetime_utc'].dt.dayofyear
    return df

def find_files(site_cfg, data_dir):
    pattern = site_cfg['file_pattern']
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if f.endswith('.csv') and pattern in f])
    if not files:
        files = sorted([os.path.join(data_dir, f)
                        for f in os.listdir(data_dir)
                        if f.endswith('.csv') and pattern.replace('.','_') in f])
    return files

# ---- Main logic ----
def compute_monthly_mean_max(site_name, site_cfg, data_dir):
    """
    For each year-file:
      1. Aggregate half-hourly -> hourly.
      2. Compute I_T_iso for every hour.
      3. Find each day's max I_T_iso.
    Then average daily maxima by month across all years.
    """
    files = find_files(site_cfg, data_dir)
    if not files:
        print(f"  No files found for {site_name}")
        return None

    lat  = site_cfg['latitude']
    lon  = site_cfg['longitude']
    tilt = abs(lat)

    print(f"\n--- {site_name} (lat={lat}, tilt={tilt:.1f}) ---")
    print(f"  Files: {len(files)}")

    all_daily_max = []

    for fp in files:
        df = read_nsrdb(fp)
        year = df['Year'].iloc[0]

        # Aggregate half-hourly -> hourly
        df['datetime_hr'] = df['datetime_utc'].dt.floor('h')
        hourly = df.groupby('datetime_hr').agg({
            'GHI': 'mean', 'DHI': 'mean', 'DNI': 'mean',
            'Solar Zenith Angle': 'mean', 'Surface Albedo': 'mean',
            'Year': 'first', 'Month': 'first', 'Day': 'first',
            'Hour': 'first', 'doy': 'first',
        }).reset_index()

        # Compute tilted radiation
        n     = hourly['doy'].values.astype(float)
        hr    = hourly['Hour'].values.astype(float)
        t_sol = utc_to_solar_time(hr, 0.0, n, lon)
        delta = solar_declination(n)
        omega = hour_angle(t_sol)
        cz    = np.clip(cos_zenith(lat, delta, omega), 0.0, 1.0)
        ct    = cos_incidence(lat, delta, omega, tilt, AZIMUTH)
        Rb    = beam_tilt_ratio(ct, cz)
        alb   = hourly['Surface Albedo'].values
        rho   = np.where(alb > 0, alb, RHO_G)
        I_T   = isotropic_model(hourly['GHI'].values, hourly['DHI'].values,
                                Rb, tilt, rho)

        hourly['I_T_iso'] = I_T

        # Daily max
        daily_max = hourly.groupby(['Year','Month','Day'])['I_T_iso'].max().reset_index()
        daily_max = daily_max[daily_max['I_T_iso'] > 0]
        all_daily_max.append(daily_max)

        print(f"    {year}: {len(hourly)} hourly records")

    combined = pd.concat(all_daily_max, ignore_index=True)

    # Monthly mean of daily maxima
    monthly = combined.groupby('Month')['I_T_iso'].mean().reset_index()
    monthly.columns = ['Month', 'Mean_Max_IT_iso']
    monthly['Mean_Max_IT_iso'] = monthly['Mean_Max_IT_iso'].round(1)

    # Add month names
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly['Month_Name'] = monthly['Month'].map(month_names)

    print(f"\n  Monthly Mean-Maximum Tilted Radiation ({site_name}, Isotropic):")
    for _, r in monthly.iterrows():
        print(f"    {r['Month_Name']:>3}:  {r['Mean_Max_IT_iso']:7.1f} W/m²")

    return monthly


def main():
    print("=" * 60)
    print("Monthly Mean-Maximum Tilted Solar Radiation")
    print("=" * 60)

    results = {}
    for site_name, cfg in SITES.items():
        monthly = compute_monthly_mean_max(site_name, cfg, DATA_DIR)
        if monthly is not None:
            results[site_name] = monthly
            # Save CSV for the JS script to read
            out = os.path.join(DATA_DIR, f"{site_name}_monthly_meanmax.csv")
            monthly.to_csv(out, index=False)
            print(f"  Saved -> {out}")

    print("\nDone. Now run the JS script to append tables to the Word doc.")


if __name__ == '__main__':
    main()