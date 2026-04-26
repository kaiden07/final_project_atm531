"""
==============================================================================
ATM 531 – Principles of Sustainable Energy  |  HW #3a
Estimating Total Hourly Solar Radiation on Inclined Surfaces
==============================================================================

This script reads NSRDB PSM (Physical Solar Model) half-hourly CSV files,
aggregates them to hourly values, computes multi-year averages (2018-2024),
and estimates total hourly solar radiation on an inclined surface using:

  1. Liu & Jordan (1963) Isotropic Diffuse Model   (slide 55)
  2. Perez et al. (1990) Anisotropic Diffuse Model  (slides 64-68)

All equations follow ATM 531 Lecture 6-7 (Spring 2026, Prof. Gonzalez-Cruz).

Input  : NSRDB CSV files (half-hourly, UTC timestamps).
Output : Hourly tilted-surface irradiance for January & July,
         for Albany, NY and New York City.

Usage
-----
  1. Place all NSRDB CSVs in a single directory (DATA_DIR below).
  2. Set the site parameters (latitude, longitude, tilt, azimuth).
  3. Run:  python solar_tilted_nsrdb.py

Author : Kaiden Sookdar
Date   : Spring 2026
==============================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# USER CONFIGURATION  –  edit these paths / parameters as needed
# ===========================================================================

# Directory containing the NSRDB CSV files
DATA_DIR = "D:\TMY_Data"

# Site definitions (derived from filenames in the screenshot)
SITES = {
    'Albany': {
        'file_pattern': '5073330_42_66_-73_75',  # partial filename match
        'latitude':  42.66,
        'longitude': -73.75,
        'tz_offset': -5,          # EST (hours from UTC)
    },
    'NYC': {
        'file_pattern': '5037116_40_78_-73_97',
        'latitude':  40.78,
        'longitude': -73.97,
        'tz_offset': -5,
    },
}

# Panel configuration
TILT       = None    # If None, defaults to site latitude (common rule)
AZIMUTH    = 0.0     # 0 = due south (Northern Hemisphere convention)
RHO_G      = 0.2     # Default ground albedo; code also reads per-row albedo

# ===========================================================================
# Physical Constants
# ===========================================================================
G_SC = 1367.0  # Solar constant [W/m^2]  (slide 29)

# ===========================================================================
# Perez Brightness Coefficients – Table 2.16.1 (slide 66)
# ===========================================================================
PEREZ_COEFFS = np.array([
    [1.000, 1.065, -0.008,  0.588, -0.062, -0.060,  0.072, -0.022],
    [1.065, 1.230,  0.130,  0.683, -0.151, -0.019,  0.066, -0.029],
    [1.230, 1.500,  0.330,  0.487, -0.221,  0.055, -0.064, -0.026],
    [1.500, 1.950,  0.568,  0.187, -0.295,  0.109, -0.152,  0.014],
    [1.950, 2.800,  0.873, -0.392, -0.362,  0.226, -0.462,  0.001],
    [2.800, 4.500,  1.132, -1.237, -0.412,  0.288, -0.823,  0.056],
    [4.500, 6.200,  1.060, -1.600, -0.359,  0.264, -1.127,  0.131],
    [6.200, 99.99,  0.678, -0.327, -0.250,  0.156, -1.377,  0.251],
])

# ===========================================================================
# Helper
# ===========================================================================
d2r = np.deg2rad
r2d = np.rad2deg


# ===========================================================================
# 1. SOLAR GEOMETRY  (slides 31-39)
# ===========================================================================

def solar_declination(n):
    """δ = 23.45 sin[360(284+n)/365]  –  slide 32"""
    return 23.45 * np.sin(d2r(360.0 * (284.0 + n) / 365.0))


def extraterrestrial_normal(n):
    """G_on = G_sc (1 + 0.033 cos(360n/365))  –  slide 31"""
    return G_SC * (1.0 + 0.033 * np.cos(d2r(360.0 * n / 365.0)))


def hour_angle(t_sol):
    """ω = (t_sol − 12) × 15°  –  slide 33"""
    return (t_sol - 12.0) * 15.0


def cos_zenith(phi, delta, omega):
    """cos θ_z = cos φ cos δ cos ω + sin φ sin δ  –  slide 34"""
    p, d, w = d2r(phi), d2r(delta), d2r(omega)
    return np.cos(p)*np.cos(d)*np.cos(w) + np.sin(p)*np.sin(d)


def cos_incidence(phi, delta, omega, beta, gamma=0.0):
    """
    Full incidence-angle equation for a tilted surface  –  slide 38.
    gamma = surface azimuth (0=south, negative=east, positive=west).
    """
    p = d2r(phi);   d_ = d2r(delta); w = d2r(omega)
    b = d2r(beta);  g  = d2r(gamma)
    return (np.sin(d_)*np.sin(p)*np.cos(b)
          - np.sin(d_)*np.cos(p)*np.sin(b)*np.cos(g)
          + np.cos(d_)*np.cos(p)*np.cos(b)*np.cos(w)
          + np.cos(d_)*np.sin(p)*np.sin(b)*np.cos(g)*np.cos(w)
          + np.cos(d_)*np.sin(b)*np.sin(g)*np.sin(w))


def beam_tilt_ratio(ct, cz):
    """R_b = cos θ / cos θ_z  –  slide 43"""
    cz_safe = np.where(cz > 0.01, cz, 1.0)
    return np.where((ct > 0) & (cz > 0.01), ct / cz_safe, 0.0)


def sunset_hour_angle(phi, delta):
    """ω_s = arccos(−tan φ tan δ)  –  slide 39"""
    val = -np.tan(d2r(phi)) * np.tan(d2r(delta))
    return r2d(np.arccos(np.clip(val, -1, 1)))


# ===========================================================================
# 2. DIFFUSE FRACTION – Erbs et al.  (slide 53)
# ===========================================================================

def erbs_diffuse_fraction(kt):
    """Hourly diffuse fraction from clearness index k_T."""
    kt = np.asarray(kt, dtype=float)
    return np.where(kt <= 0.22,
                    1.0 - 0.09 * kt,
                    np.where(kt <= 0.80,
                             0.9511 - 0.1604*kt + 4.388*kt**2
                             - 16.638*kt**3 + 12.336*kt**4,
                             0.165))


# ===========================================================================
# 3. TILTED-SURFACE MODELS
# ===========================================================================

def isotropic_model(I, I_d, Rb, beta_deg, rho_g):
    """
    Liu & Jordan (1963) isotropic model  –  slide 55.
    I_T = I_b R_b + I_d (1+cosβ)/2 + I ρ_g (1−cosβ)/2
    """
    b   = d2r(beta_deg)
    I_b = I - I_d
    return np.maximum(
        I_b * Rb
        + I_d * (1.0 + np.cos(b)) / 2.0
        + I * rho_g * (1.0 - np.cos(b)) / 2.0,
        0.0)


def perez_model(I, I_d, Rb, ct, cz, theta_z_deg, n, beta_deg, rho_g):
    """
    Perez et al. (1990) anisotropic model  –  slides 64-68.
    I_T = I_b R_b
        + I_d (1−F1)(1+cosβ)/2
        + I_d F1 (a/b)
        + I_d F2 sinβ
        + I ρ_g (1−cosβ)/2
    """
    beta = d2r(beta_deg)
    I_b  = I - I_d

    # Beam-normal irradiance
    cz_safe = np.where(cz > 0.01, cz, 1.0)
    I_bn = np.where(cz > 0.01, I_b / cz_safe, 0.0)

    # Extraterrestrial normal
    I_on = extraterrestrial_normal(n)

    # Air mass (capped at cos 85°)
    m = 1.0 / np.maximum(cz, np.cos(d2r(85.0)))

    # Brightness index Δ  (slide 65)
    Delta = np.where(I_on > 0, m * I_d / I_on, 0.0)

    # Clearness index ε  (slide 65, eq 2.16.10)
    tz3   = theta_z_deg ** 3
    denom = np.maximum(I_d + 5.535e-6 * tz3, 1e-6)
    eps   = ((I_d + I_bn) / denom) * (1.0 + 5.535e-6 * tz3)
    eps   = np.clip(eps, 1.0, 99.0)

    # Look up Perez coefficients (Table 2.16.1)
    f11 = np.zeros_like(eps)
    f12 = np.zeros_like(eps)
    f13 = np.zeros_like(eps)
    f21 = np.zeros_like(eps)
    f22 = np.zeros_like(eps)
    f23 = np.zeros_like(eps)
    for row in PEREZ_COEFFS:
        m_ = (eps >= row[0]) & (eps < row[1])
        f11[m_], f12[m_], f13[m_] = row[2], row[3], row[4]
        f21[m_], f22[m_], f23[m_] = row[5], row[6], row[7]

    tz_rad = d2r(theta_z_deg)
    F1 = np.maximum(0.0, f11 + f12 * Delta + tz_rad * f13)
    F2 = f21 + f22 * Delta + tz_rad * f23

    a = np.maximum(0.0, ct)
    b = np.maximum(np.cos(d2r(85.0)), cz)

    I_T = (I_b * Rb
           + I_d * (1.0 - F1) * (1.0 + np.cos(beta)) / 2.0
           + I_d * F1 * (a / b)
           + I_d * F2 * np.sin(beta)
           + I * rho_g * (1.0 - np.cos(beta)) / 2.0)
    return np.maximum(I_T, 0.0)


# ===========================================================================
# 4. SOLAR TIME CONVERSION
# ===========================================================================

def equation_of_time(n):
    """Spencer (1971) Equation of Time [minutes]."""
    B = d2r(360.0 * (n - 81.0) / 365.0)
    return 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B)


def utc_to_solar_time(hour_utc, minute_utc, doy, longitude):
    """
    Convert UTC clock time to local solar time [decimal hours].
    
    Parameters
    ----------
    hour_utc, minute_utc : array – UTC hour and minute.
    doy                  : array – Day of year.
    longitude            : float – Site longitude [degrees, +E/-W].
    """
    EoT = equation_of_time(doy)                       # [min]
    LC  = 4.0 * longitude                             # [min] (from prime meridian)
    clock_utc = hour_utc + minute_utc / 60.0          # decimal UTC hours
    # Solar time = UTC + (longitude correction + EoT) / 60
    # (longitude correction from Greenwich = 4 min per degree of longitude)
    solar_time = clock_utc + (LC + EoT) / 60.0
    return solar_time


# ===========================================================================
# 5. READ & PROCESS NSRDB FILES
# ===========================================================================

def read_nsrdb(filepath):
    """
    Read a single NSRDB PSM CSV file (half-hourly).
    Returns a DataFrame with columns:
        datetime_utc, Year, Month, Day, Hour, Minute, GHI, DHI, DNI,
        Solar Zenith Angle, Surface Albedo, doy
    """
    df = pd.read_csv(filepath, skiprows=2)
    # Build UTC datetime
    df['datetime_utc'] = pd.to_datetime(
        df[['Year','Month','Day','Hour','Minute']].rename(
            columns={'Year':'year','Month':'month','Day':'day',
                     'Hour':'hour','Minute':'minute'}))
    df['doy'] = df['datetime_utc'].dt.dayofyear
    return df


def load_site_data(site_name, site_cfg, data_dir):
    """
    Load all yearly NSRDB CSVs for a site, aggregate half-hourly → hourly,
    then compute multi-year average hourly profiles.
    """
    pattern = site_cfg['file_pattern']
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if f.endswith('.csv') and pattern in f.replace('.', '_')])

    # The filenames have dots replaced with underscores in the upload,
    # so let's match more flexibly
    if not files:
        files = sorted([os.path.join(data_dir, f)
                        for f in os.listdir(data_dir)
                        if f.endswith('.csv') and pattern.replace('.','_') in f])

    if not files:
        print(f"  WARNING: No files found for {site_name} "
              f"(pattern: {pattern}) in {data_dir}")
        return None

    print(f"  Found {len(files)} file(s) for {site_name}:")
    for f in files:
        print(f"    {os.path.basename(f)}")

    frames = []
    for fp in files:
        df = read_nsrdb(fp)
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)

    # ---- Aggregate half-hourly → hourly by averaging each pair ----
    # Create an hourly timestamp (floor to hour)
    all_data['datetime_hr'] = all_data['datetime_utc'].dt.floor('h')
    hourly = all_data.groupby('datetime_hr').agg({
        'GHI':                 'mean',
        'DHI':                 'mean',
        'DNI':                 'mean',
        'Solar Zenith Angle':  'mean',
        'Surface Albedo':      'mean',
        'Year':                'first',
        'Month':               'first',
        'Day':                 'first',
        'Hour':                'first',
        'doy':                 'first',
    }).reset_index()

    # ---- Multi-year average: group by (Month, Day, Hour) ----
    avg = hourly.groupby(['Month', 'Day', 'Hour']).agg({
        'GHI':                'mean',
        'DHI':                'mean',
        'DNI':                'mean',
        'Solar Zenith Angle': 'mean',
        'Surface Albedo':     'mean',
        'doy':                'first',   # representative doy
    }).reset_index()

    avg.rename(columns={'Solar Zenith Angle': 'SZA',
                        'Surface Albedo': 'albedo'}, inplace=True)

    print(f"  Multi-year average: {len(avg)} hourly records "
          f"({len(files)} years)\n")
    return avg


# ===========================================================================
# 6. MAIN COMPUTATION
# ===========================================================================

def compute_tilted(df, latitude, longitude, tilt, azimuth, rho_g_default):
    """
    Given a multi-year-averaged hourly DataFrame, compute tilted-surface
    radiation for both models.
    """
    n   = df['doy'].values.astype(float)
    GHI = df['GHI'].values.astype(float)
    DHI = df['DHI'].values.astype(float)
    DNI = df['DNI'].values.astype(float)
    sza = df['SZA'].values.astype(float)
    alb = df['albedo'].values.astype(float)
    hr  = df['Hour'].values.astype(float)
    mn  = np.zeros_like(hr)   # hourly data, minute = 0

    # Solar time from UTC
    t_sol = utc_to_solar_time(hr, mn, n, longitude)

    # Solar geometry (computed from solar time for consistency)
    delta = solar_declination(n)
    omega = hour_angle(t_sol)
    cz    = cos_zenith(latitude, delta, omega)
    cz    = np.clip(cz, 0.0, 1.0)
    tz_d  = r2d(np.arccos(cz))

    ct = cos_incidence(latitude, delta, omega, tilt, azimuth)
    Rb = beam_tilt_ratio(ct, cz)

    # Use per-row albedo where available; fall back to default
    rho = np.where(alb > 0, alb, rho_g_default)

    # Models
    I_T_iso   = isotropic_model(GHI, DHI, Rb, tilt, rho)
    I_T_perez = perez_model(GHI, DHI, Rb, ct, cz, tz_d, n, tilt, rho)

    df = df.copy()
    df['solar_time']   = t_sol
    df['declination']  = delta
    df['hour_angle']   = omega
    df['zenith_calc']  = tz_d
    df['cos_theta']    = ct
    df['Rb']           = Rb
    df['I_T_iso']      = I_T_iso
    df['I_T_perez']    = I_T_perez

    return df


def summarize_month(df, month, site_name, tilt):
    """Print a summary table for a given month."""
    m = df[df['Month'] == month].copy()
    m = m[m['GHI'] > 0]  # daytime only
    if m.empty:
        print(f"  No daytime data for month {month}")
        return

    month_name = {1: 'January', 7: 'July'}[month]

    # Representative day (mid-month)
    mid_day = m.groupby('Day').first().index[len(m.groupby('Day'))//2]
    day_df  = m[m['Day'] == mid_day].sort_values('Hour')

    print(f"\n{'='*72}")
    print(f"  {site_name}  |  {month_name} (multi-year avg)  |  "
          f"Tilt = {tilt:.1f}°  |  Mid-month day = {mid_day}")
    print(f"{'='*72}")
    print(f"  {'Hr(UTC)':>7} {'SolTime':>7} {'GHI':>7} {'DHI':>7} "
          f"{'DNI':>7} {'SZA':>6} {'Rb':>6} {'IT_iso':>8} {'IT_perez':>9}")
    print(f"  {' ':>7} {'[hr]':>7} {'[W/m²]':>7} {'[W/m²]':>7} "
          f"{'[W/m²]':>7} {'[°]':>6} {'[-]':>6} {'[W/m²]':>8} {'[W/m²]':>9}")
    print(f"  {'-'*68}")
    for _, r in day_df.iterrows():
        print(f"  {r['Hour']:7.0f} {r['solar_time']:7.1f} "
              f"{r['GHI']:7.1f} {r['DHI']:7.1f} {r['DNI']:7.1f} "
              f"{r['zenith_calc']:6.1f} {r['Rb']:6.2f} "
              f"{r['I_T_iso']:8.1f} {r['I_T_perez']:9.1f}")

    # Monthly daily totals (kWh/m²/day)
    daily = m.groupby('Day').agg({'GHI':'sum','I_T_iso':'sum','I_T_perez':'sum'})
    daily_avg_ghi   = daily['GHI'].mean()   / 1000.0
    daily_avg_iso   = daily['I_T_iso'].mean()   / 1000.0
    daily_avg_perez = daily['I_T_perez'].mean() / 1000.0

    print(f"\n  Monthly avg daily totals (kWh/m²/day):")
    print(f"    GHI (horizontal)  : {daily_avg_ghi:.2f}")
    print(f"    Tilted (Isotropic): {daily_avg_iso:.2f}")
    print(f"    Tilted (Perez)    : {daily_avg_perez:.2f}")
    print(f"    Gain Iso vs Horiz : {(daily_avg_iso/daily_avg_ghi - 1)*100:+.1f}%")
    print(f"    Gain Perez vs Iso : {(daily_avg_perez/daily_avg_iso - 1)*100:+.1f}%")


# ===========================================================================
# 7. ENTRY POINT
# ===========================================================================

def main():
    print("=" * 72)
    print("ATM 531 – HW #3a: Solar Radiation on Inclined Surfaces")
    print("Models: Liu & Jordan (Isotropic) + Perez (Anisotropic)")
    print("Data  : NSRDB PSM half-hourly, multi-year average (2018-2024)")
    print("=" * 72)
    print()

    for site_name, cfg in SITES.items():
        print(f"--- Loading data for {site_name} ---")
        avg_df = load_site_data(site_name, cfg, DATA_DIR)

        if avg_df is None:
            continue

        lat  = cfg['latitude']
        lon  = cfg['longitude']
        tilt = TILT if TILT is not None else abs(lat)  # default: tilt = latitude

        print(f"  Computing tilted radiation  (lat={lat}, lon={lon}, "
              f"tilt={tilt:.1f}°, azimuth={AZIMUTH}°)")

        result = compute_tilted(avg_df, lat, lon, tilt, AZIMUTH, RHO_G)

        # Summarize January & July
        for month in [1, 7]:
            summarize_month(result, month, site_name, tilt)

        # Save full result to CSV
        outfile = f"D:\TMY_Data\\{site_name}_tilted_radiation.csv"
        result.to_csv(outfile, index=False, float_format='%.2f')
        print(f"\n  Full results saved → {outfile}\n")

    print("Done.")


if __name__ == '__main__':
    main()
