#!/usr/bin/env python3
"""
==============================================================================
ATM 531 - HW #3, Problem 2
PV System Design & Analysis for Residential Facility near Bear Mountain, NY
==============================================================================

Site     : Just north of Bear Mountain State Park, near West Point, NY
Coords   : 41.32 N, -74.03 W  (NSRDB pixel)
Load     : 8000 kWh/yr (baseline)
Panel    : SunPower E19/318 (spec sheet from HW #3 PDF)
Models   : Liu-Jordan isotropic tilted irradiance
           Duffie-Beckman cell temperature / PV power (lecture slides)

Tasks    :
  1. Load multi-year TMY CSVs, compute hourly tilted irradiance
  2. Model PV array output with temperature and inverter losses
  3. Size the system to meet 8000 kWh/yr
  4. Generate monthly and annual performance
  5. Economic analysis with state/federal rebates
  6. Add a cold-climate heat pump, recompute load and resize array

Author   : [Your Name]
Date     : Spring 2026
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

DATA_DIR = r"D:\TMY_Data"   # folder containing TMY CSV files

# Site: north of Bear Mountain State Park, near West Point, NY
SITE = {
    'name'        : 'Bear_Mountain',
    'file_pattern': '41.32_-74.03',
    'latitude'    : 41.32,
    'longitude'   : -74.03,
}

# --- Load / building ---
ANNUAL_LOAD_KWH   = 8000.0    # baseline electrical consumption [kWh/yr]
HEATED_AREA_M2    = 200.0     # ~2150 sqft typical NY home
HDD_BASE_C        = 18.3      # 65°F base for heating degree days

# --- System sizing defaults ---
TILT              = None      # None -> use site latitude
AZIMUTH           = 0.0       # due south
RHO_G             = 0.2       # default ground albedo (overridden per TMY row)

# --- Inverter / system losses ---
ETA_INVERTER      = 0.96      # inverter efficiency
ETA_WIRING        = 0.98      # DC/AC wiring losses
ETA_SOILING       = 0.97      # dust / soiling
ETA_MISMATCH      = 0.98      # module mismatch
ETA_SYSTEM        = ETA_INVERTER * ETA_WIRING * ETA_SOILING * ETA_MISMATCH  # ~ 0.895

# --- Economics ---
ELEC_PRICE_USD_KWH = 0.20
COST_PER_WATT_LOW  = 3.0
COST_PER_WATT_HIGH = 4.0
FED_REBATE         = 0.30     # 30% federal tax credit
NY_REBATE_CAP      = 5000.0   # NY State rebate capped at $5000

# ===========================================================================
# SunPower E19/318 Panel Specifications  (from HW #3 PDF spec sheet)
# ===========================================================================

PANEL = {
    'model'       : 'SunPower E19/318',
    'P_max'       : 318.0,    # W  peak power at STC
    'eta_ref'     : 0.195,    # 19.5% efficiency at STC
    'V_mpp'       : 54.7,     # V  rated voltage
    'I_mpp'       : 5.82,     # A  rated current
    'V_oc'        : 64.7,     # V  open-circuit voltage
    'I_sc'        : 6.20,     # A  short-circuit current
    'mu_P'        : -0.0038,  # -0.38 %/K  power temp coefficient
    'mu_Voc_V'    : -0.17662, # -176.6 mV/K  => V/K
    'NOCT'        : 45.0,     # C  nominal operating cell temperature
    'length_mm'   : 1559.0,
    'width_mm'    : 1046.0,
    'area_m2'     : 1.559 * 1.046,   # ~1.631 m^2
    'T_ref'       : 25.0,     # STC reference cell temperature [C]
    'G_ref'       : 1000.0,   # STC reference irradiance [W/m^2]
    'tau_alpha'   : 0.90,     # transmittance-absorptance product (typical)
}

# NOCT conditions (standard)
NOCT_COND = {
    'I_T'         : 800.0,   # W/m^2
    'T_a'         : 20.0,    # C
    'wind'        : 1.0,     # m/s
}

# ===========================================================================
# Solar geometry and irradiance (from HW #3a code, trimmed to what's needed)
# ===========================================================================

G_SC = 1367.0
d2r  = np.deg2rad
r2d  = np.rad2deg

def solar_declination(n):
    return 23.45 * np.sin(d2r(360.0 * (284.0 + n) / 365.0))

def hour_angle(t_sol):
    return (t_sol - 12.0) * 15.0

def cos_zenith(phi, delta, omega):
    p, d, w = d2r(phi), d2r(delta), d2r(omega)
    return np.cos(p)*np.cos(d)*np.cos(w) + np.sin(p)*np.sin(d)

def cos_incidence(phi, delta, omega, beta, gamma=0.0):
    p, d_, w = d2r(phi), d2r(delta), d2r(omega)
    b, g     = d2r(beta), d2r(gamma)
    return (np.sin(d_)*np.sin(p)*np.cos(b)
          - np.sin(d_)*np.cos(p)*np.sin(b)*np.cos(g)
          + np.cos(d_)*np.cos(p)*np.cos(b)*np.cos(w)
          + np.cos(d_)*np.sin(p)*np.sin(b)*np.cos(g)*np.cos(w)
          + np.cos(d_)*np.sin(b)*np.sin(g)*np.sin(w))

def beam_tilt_ratio(ct, cz):
    cz_safe = np.where(cz > 0.01, cz, 1.0)
    return np.where((ct > 0) & (cz > 0.01), ct / cz_safe, 0.0)

def isotropic_model(I, I_d, Rb, beta_deg, rho_g):
    """Liu & Jordan isotropic tilted-surface model (slide 55)."""
    b   = d2r(beta_deg)
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

# ===========================================================================
# PV MODEL — equations from lecture slides (Duffie & Beckman)
# ===========================================================================

def cell_temperature(I_T, T_a, panel=PANEL, noct_cond=NOCT_COND):
    """
    Cell temperature from NOCT method.
        T_c = T_a + (I_T / I_T,NOCT) * (T_NOCT - T_a,NOCT) * (1 - eta_c / (tau*alpha))
    Using eta_c ~ eta_ref as a first-order approximation (per slides).
    """
    tau_alpha = panel['tau_alpha']
    eta_c     = panel['eta_ref']
    dT_noct   = panel['NOCT'] - noct_cond['T_a']
    return T_a + (I_T / noct_cond['I_T']) * dT_noct * (1.0 - eta_c / tau_alpha)


def module_efficiency(T_c, panel=PANEL):
    """
    Efficiency vs cell temperature (slide form):
        eta_mp = eta_ref * [1 + (mu_P) * (T_c - T_ref)]
    where mu_P is the fractional power temperature coefficient per deg C.
    """
    return panel['eta_ref'] * (1.0 + panel['mu_P'] * (T_c - panel['T_ref']))


def panel_dc_power(I_T, T_a, panel=PANEL):
    """
    Hourly DC power output of ONE panel [W].
        P_dc = A_c * I_T * eta_mp(T_c)
    """
    T_c     = cell_temperature(I_T, T_a, panel)
    eta_mp  = module_efficiency(T_c, panel)
    P_dc    = panel['area_m2'] * I_T * eta_mp
    return np.maximum(P_dc, 0.0), T_c, eta_mp


# ===========================================================================
# TMY I/O
# ===========================================================================

def read_nsrdb(filepath):
    df = pd.read_csv(filepath, skiprows=2)
    df['datetime_utc'] = pd.to_datetime(
        df[['Year','Month','Day','Hour','Minute']].rename(
            columns={'Year':'year','Month':'month','Day':'day',
                     'Hour':'hour','Minute':'minute'}))
    df['doy'] = df['datetime_utc'].dt.dayofyear
    return df


def load_site(site, data_dir):
    """Load all year files, aggregate half-hourly -> hourly, return concatenated."""
    pat = site['file_pattern']
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if f.endswith('.csv') and pat in f])
    if not files:
        print(f"  No files found for pattern '{pat}' in {data_dir}")
        return None

    print(f"  Found {len(files)} file(s) for {site['name']}")
    frames = []
    for fp in files:
        df = read_nsrdb(fp)
        df['datetime_hr'] = df['datetime_utc'].dt.floor('h')
        hr = df.groupby('datetime_hr').agg({
            'GHI':'mean','DHI':'mean','DNI':'mean',
            'Temperature':'mean','Surface Albedo':'mean',
            'Year':'first','Month':'first','Day':'first','Hour':'first',
            'doy':'first',
        }).reset_index()
        frames.append(hr)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined)} hourly records over {combined['Year'].nunique()} years")
    return combined


# ===========================================================================
# Main computation
# ===========================================================================

def compute_hourly_pv(df, site, tilt, azimuth, panel=PANEL):
    """
    Given an hourly DataFrame, compute:
      - tilted irradiance (isotropic model, diffuse from DHI column)
      - cell temperature
      - DC power per panel
      - AC power per panel (after system losses)
    """
    lat = site['latitude']
    lon = site['longitude']

    n     = df['doy'].values.astype(float)
    hr    = df['Hour'].values.astype(float)
    GHI   = df['GHI'].values.astype(float)
    DHI   = df['DHI'].values.astype(float)
    T_amb = df['Temperature'].values.astype(float)
    alb   = df['Surface Albedo'].values.astype(float)
    rho   = np.where(alb > 0, alb, RHO_G)

    t_sol = utc_to_solar_time(hr, 0.0, n, lon)
    delta = solar_declination(n)
    omega = hour_angle(t_sol)
    cz    = np.clip(cos_zenith(lat, delta, omega), 0.0, 1.0)
    ct    = cos_incidence(lat, delta, omega, tilt, azimuth)
    Rb    = beam_tilt_ratio(ct, cz)

    I_T   = isotropic_model(GHI, DHI, Rb, tilt, rho)

    P_dc, T_c, eta_mp = panel_dc_power(I_T, T_amb, panel)
    P_ac              = P_dc * ETA_SYSTEM

    df = df.copy()
    df['I_T']      = I_T
    df['T_cell']   = T_c
    df['eta_mp']   = eta_mp
    df['P_dc_panel_W'] = P_dc    # per-panel DC
    df['P_ac_panel_W'] = P_ac    # per-panel AC
    return df


def size_system(hourly, annual_target_kwh, panel=PANEL):
    """
    Size the system: number of panels needed so that multi-year mean annual
    AC generation >= annual_target_kwh.
    """
    # Mean annual energy per panel [kWh/yr] = sum(P_ac) / n_years / 1000
    n_years = hourly['Year'].nunique()
    E_per_panel_kwh = hourly['P_ac_panel_W'].sum() / n_years / 1000.0
    n_panels = int(np.ceil(annual_target_kwh / E_per_panel_kwh))
    array_dc_kw = n_panels * panel['P_max'] / 1000.0
    return {
        'n_panels'          : n_panels,
        'array_dc_kw'       : array_dc_kw,
        'E_per_panel_kwh'   : E_per_panel_kwh,
        'annual_gen_kwh'    : n_panels * E_per_panel_kwh,
        'n_years'           : n_years,
    }


def monthly_performance(hourly, n_panels):
    """Monthly average AC generation [kWh/month] across all years."""
    n_years = hourly['Year'].nunique()
    monthly = hourly.groupby(['Year','Month'])['P_ac_panel_W'].sum().reset_index()
    monthly['kwh'] = monthly['P_ac_panel_W'] * n_panels / 1000.0
    mean_monthly = monthly.groupby('Month')['kwh'].mean().reset_index()
    mean_monthly.columns = ['Month','AC_kWh']
    return mean_monthly


def inverter_sizing(n_panels, panel=PANEL):
    """
    Size inverters. DC/AC ratio typically 1.1-1.3.
    We pick a string inverter size close to 0.85 * DC rating.
    """
    dc_kw     = n_panels * panel['P_max'] / 1000.0
    ac_target = dc_kw / 1.15   # DC/AC ratio = 1.15
    # Round to common residential inverter sizes
    common_sizes = [3.0, 3.8, 5.0, 6.0, 7.6, 8.2, 10.0, 11.4]
    chosen = next((s for s in common_sizes if s >= ac_target), common_sizes[-1])
    n_inverters = 1
    if chosen < ac_target:
        n_inverters = int(np.ceil(ac_target / chosen))
    return {
        'dc_kw'         : dc_kw,
        'ac_target_kw'  : ac_target,
        'inverter_kw'   : chosen,
        'n_inverters'   : n_inverters,
        'dc_ac_ratio'   : dc_kw / (chosen * n_inverters),
    }


def economics(array_dc_kw, annual_gen_kwh):
    """Simple payback economics with rebates."""
    watts = array_dc_kw * 1000.0
    results = {}
    for price, label in [(COST_PER_WATT_LOW,'low'), (COST_PER_WATT_HIGH,'high')]:
        gross_cost = watts * price
        fed_rebate = FED_REBATE * gross_cost
        ny_rebate  = min(NY_REBATE_CAP, gross_cost - fed_rebate)
        net_cost   = gross_cost - fed_rebate - ny_rebate
        annual_savings = annual_gen_kwh * ELEC_PRICE_USD_KWH
        payback_years  = net_cost / annual_savings if annual_savings > 0 else float('inf')
        results[label] = {
            'price_per_w'   : price,
            'gross_cost'    : gross_cost,
            'fed_rebate'    : fed_rebate,
            'ny_rebate'     : ny_rebate,
            'net_cost'      : net_cost,
            'annual_savings': annual_savings,
            'payback_years' : payback_years,
        }
    return results


# ===========================================================================
# Heat pump addition
# ===========================================================================

def estimate_heating_load(hourly, heated_area_m2=HEATED_AREA_M2):
    """
    Estimate annual heating energy requirement using heating degree-days.
        Q_heat [kWh/yr] = UA * HDD (kelvin-hours) / 1000
    UA chosen to give ~60 kWh/m2/yr heating load for a typical NY home.
    """
    # HDD calculation (hourly, base 18.3 C)
    T = hourly['Temperature'].values
    dT = np.maximum(HDD_BASE_C - T, 0.0)   # [K] below base per hour
    n_years = hourly['Year'].nunique()
    total_kelvin_hours_per_year = dT.sum() / n_years   # [K*h/yr]

    # Calibrate UA so annual heating demand ~ 60 kWh/m2/yr (Climate Zone 5)
    target_heat_kwh = 60.0 * heated_area_m2
    UA = target_heat_kwh * 1000.0 / total_kelvin_hours_per_year   # [W/K]

    heating_load_hourly = UA * dT   # [W] instantaneous
    heating_kwh_per_year = heating_load_hourly.sum() / n_years / 1000.0
    return heating_kwh_per_year, UA, total_kelvin_hours_per_year, heating_load_hourly


def heat_pump_electrical_load(hourly, UA):
    """
    Convert hourly heating demand [W] -> electrical demand [W] using a
    temperature-dependent COP curve for a cold-climate heat pump.
    Approximate COP as linear between T=-15°C (COP=1.7) and T=8.3°C (COP=3.5).
    """
    T  = hourly['Temperature'].values
    dT = np.maximum(HDD_BASE_C - T, 0.0)
    Q_heat_W = UA * dT

    # Cold climate heat pump COP curve (Mitsubishi H2i / similar)
    COP = np.interp(T, [-15.0, -8.3, 0.0, 8.3, 17.0],
                       [ 1.7,   2.0, 2.5, 3.5, 4.0])
    COP = np.clip(COP, 1.3, 4.5)

    P_elec_W = np.where(Q_heat_W > 0, Q_heat_W / COP, 0.0)
    n_years  = hourly['Year'].nunique()
    heat_kwh_per_year = P_elec_W.sum() / n_years / 1000.0

    # Size the HP to meet the 99th percentile hourly demand
    P_design_kW = np.percentile(Q_heat_W, 99) / 1000.0
    return heat_kwh_per_year, P_design_kW, P_elec_W


# ===========================================================================
# Plotting
# ===========================================================================

def plot_monthly_bar(monthly, title, filepath, annual_load=None, color='#2E5090'):
    """Create a bar chart of monthly AC generation."""
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    # Ensure months are in order 1-12
    m = monthly.set_index('Month').reindex(range(1, 13)).reset_index()
    values = m['AC_kWh'].values
    total  = values.sum()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(month_names, values, color=color, edgecolor='black',
                  linewidth=0.6, zorder=3)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    # Average monthly load reference line
    if annual_load is not None:
        avg_load = annual_load / 12.0
        ax.axhline(avg_load, color='#C0392B', linestyle='--', linewidth=1.5,
                   label=f'Avg monthly load ({avg_load:.0f} kWh)', zorder=2)
        ax.legend(loc='upper right', framealpha=0.9)

    ax.set_ylabel('AC Energy Generated (kWh)', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_title(f'{title}\nAnnual total: {total:,.0f} kWh',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved -> {filepath}")


def plot_comparison_bar(monthly_base, monthly_hp, filepath,
                        baseline_load=None, total_load=None):
    """Side-by-side bar chart comparing baseline vs expanded system."""
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    mb = monthly_base.set_index('Month').reindex(range(1, 13))['AC_kWh'].values
    mh = monthly_hp.set_index('Month').reindex(range(1, 13))['AC_kWh'].values

    x = np.arange(len(month_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width/2, mb, width, label=f'Baseline ({mb.sum():,.0f} kWh/yr)',
                color='#2E5090', edgecolor='black', linewidth=0.6, zorder=3)
    b2 = ax.bar(x + width/2, mh, width, label=f'Expanded w/ HP ({mh.sum():,.0f} kWh/yr)',
                color='#E67E22', edgecolor='black', linewidth=0.6, zorder=3)

    if baseline_load is not None:
        ax.axhline(baseline_load/12, color='#2E5090', linestyle='--',
                   linewidth=1.2, alpha=0.7,
                   label=f'Baseline avg load ({baseline_load/12:.0f} kWh/mo)')
    if total_load is not None:
        ax.axhline(total_load/12, color='#E67E22', linestyle='--',
                   linewidth=1.2, alpha=0.7,
                   label=f'Expanded avg load ({total_load/12:.0f} kWh/mo)')

    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.set_ylabel('AC Energy Generated (kWh)', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_title('Monthly AC Generation: Baseline vs Expanded (with Heat Pump)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved -> {filepath}")


# ===========================================================================
# Pretty printing
# ===========================================================================

def print_header(title):
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


def print_sizing(sizing, inv, label=""):
    print(f"\n  {label}System Sizing:")
    print(f"    Energy per panel/yr : {sizing['E_per_panel_kwh']:8.1f} kWh")
    print(f"    Number of panels    : {sizing['n_panels']:8d}")
    print(f"    Array DC rating     : {sizing['array_dc_kw']:8.2f} kW")
    print(f"    Annual generation   : {sizing['annual_gen_kwh']:8.0f} kWh/yr")
    print(f"    Inverter size       : {inv['inverter_kw']:8.1f} kW  (x {inv['n_inverters']})")
    print(f"    DC/AC ratio         : {inv['dc_ac_ratio']:8.2f}")


def print_monthly(monthly, n_panels, label=""):
    print(f"\n  {label}Monthly AC Generation:")
    print(f"    {'Month':>5} {'kWh':>10}")
    print(f"    {'-'*5} {'-'*10}")
    total = 0.0
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    for _, r in monthly.iterrows():
        print(f"    {month_names[int(r['Month'])-1]:>5} {r['AC_kWh']:10.1f}")
        total += r['AC_kWh']
    print(f"    {'-'*5} {'-'*10}")
    print(f"    {'Ann':>5} {total:10.1f}")


def print_economics(econ, label=""):
    print(f"\n  {label}Economics:")
    print(f"    {'Case':>10} {'Gross $':>12} {'Fed $':>10} {'NY $':>10} "
          f"{'Net $':>12} {'Save/yr':>10} {'Payback':>10}")
    for case, v in econ.items():
        print(f"    {case + ' ($'+str(v['price_per_w'])+'/W)':>10} "
              f"{v['gross_cost']:12,.0f} "
              f"{v['fed_rebate']:10,.0f} "
              f"{v['ny_rebate']:10,.0f} "
              f"{v['net_cost']:12,.0f} "
              f"{v['annual_savings']:10,.0f} "
              f"{v['payback_years']:9.1f}y")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print_header("ATM 531 HW #3-P2: Residential PV System Design")
    print(f"  Site     : {SITE['name']}  ({SITE['latitude']}, {SITE['longitude']})")
    print(f"  Panel    : {PANEL['model']}  ({PANEL['P_max']} W)")
    print(f"  Base load: {ANNUAL_LOAD_KWH} kWh/yr")

    # 1. Load TMY data
    print_header("1. Loading TMY Data")
    hourly = load_site(SITE, DATA_DIR)
    if hourly is None:
        print("  No data. Aborting.")
        return

    # 2. Compute tilted irradiance + PV output
    tilt = TILT if TILT is not None else abs(SITE['latitude'])
    print_header("2. Computing Hourly PV Output")
    print(f"  Tilt = {tilt:.1f}°, Azimuth = {AZIMUTH:.0f}° (south)")
    hourly = compute_hourly_pv(hourly, SITE, tilt, AZIMUTH)
    print(f"  Mean annual tilted irradiance: "
          f"{hourly['I_T'].sum()/hourly['Year'].nunique()/1000:.0f} kWh/m²/yr")

    # 3. BASELINE: Size for 8000 kWh/yr
    print_header("3. Baseline System (8000 kWh/yr Load)")
    sizing_base = size_system(hourly, ANNUAL_LOAD_KWH)
    inv_base    = inverter_sizing(sizing_base['n_panels'])
    print_sizing(sizing_base, inv_base, "Baseline ")

    monthly_base = monthly_performance(hourly, sizing_base['n_panels'])
    print_monthly(monthly_base, sizing_base['n_panels'], "Baseline ")

    econ_base = economics(sizing_base['array_dc_kw'],
                          sizing_base['annual_gen_kwh'])
    print_economics(econ_base, "Baseline ")

    # 4. HEAT PUMP ADDITION
    print_header("4. Heat Pump Addition")
    heat_kwh, UA, KH, Q_heat = estimate_heating_load(hourly)
    print(f"  Estimated annual heating demand  : {heat_kwh:8.0f} kWh(thermal)/yr")
    print(f"  Building UA (calibrated)         : {UA:8.1f} W/K")

    hp_elec_kwh, hp_design_kw, hp_P_elec = heat_pump_electrical_load(hourly, UA)
    print(f"  HP annual electric consumption   : {hp_elec_kwh:8.0f} kWh/yr")
    print(f"  HP design heating capacity (99%) : {hp_design_kw:8.1f} kW")
    print(f"  Seasonal COP (annual avg)        : "
          f"{heat_kwh/hp_elec_kwh if hp_elec_kwh>0 else 0:8.2f}")

    # New total annual load
    total_load_kwh = ANNUAL_LOAD_KWH + hp_elec_kwh
    print(f"\n  Baseline load      : {ANNUAL_LOAD_KWH:8.0f} kWh/yr")
    print(f"  + Heat pump load   : {hp_elec_kwh:8.0f} kWh/yr")
    print(f"  = Total new load   : {total_load_kwh:8.0f} kWh/yr")

    # 5. RESIZE SYSTEM for electrified heating
    print_header("5. Expanded PV System (Baseline + Heat Pump)")
    sizing_hp = size_system(hourly, total_load_kwh)
    inv_hp    = inverter_sizing(sizing_hp['n_panels'])
    print_sizing(sizing_hp, inv_hp, "Expanded ")

    monthly_hp = monthly_performance(hourly, sizing_hp['n_panels'])
    print_monthly(monthly_hp, sizing_hp['n_panels'], "Expanded ")

    econ_hp = economics(sizing_hp['array_dc_kw'], sizing_hp['annual_gen_kwh'])
    print_economics(econ_hp, "Expanded ")

    # 6. SUMMARY / DELTA
    print_header("6. Summary: Baseline vs Expanded")
    delta_panels = sizing_hp['n_panels'] - sizing_base['n_panels']
    delta_kw     = sizing_hp['array_dc_kw'] - sizing_base['array_dc_kw']
    print(f"  Panels : {sizing_base['n_panels']:>4d} -> {sizing_hp['n_panels']:>4d}  "
          f"(+{delta_panels})")
    print(f"  DC kW  : {sizing_base['array_dc_kw']:>5.2f} -> {sizing_hp['array_dc_kw']:>5.2f}  "
          f"(+{delta_kw:.2f})")
    print(f"  Load   : {ANNUAL_LOAD_KWH:>5.0f} -> {total_load_kwh:>5.0f} kWh/yr "
          f"(+{hp_elec_kwh:.0f}, +{100*hp_elec_kwh/ANNUAL_LOAD_KWH:.0f}%)")

    # 7. Save outputs
    print_header("7. Saving Outputs")
    out_dir = DATA_DIR
    hourly_out = os.path.join(out_dir, f"{SITE['name']}_pv_hourly.csv")
    hourly[['datetime_hr','GHI','DHI','Temperature','I_T',
            'T_cell','eta_mp','P_dc_panel_W','P_ac_panel_W']].to_csv(
        hourly_out, index=False, float_format='%.3f')
    print(f"  Hourly PV output  -> {hourly_out}")

    mb_out = os.path.join(out_dir, f"{SITE['name']}_monthly_baseline.csv")
    monthly_base.to_csv(mb_out, index=False, float_format='%.2f')
    print(f"  Monthly baseline  -> {mb_out}")

    mh_out = os.path.join(out_dir, f"{SITE['name']}_monthly_with_hp.csv")
    monthly_hp.to_csv(mh_out, index=False, float_format='%.2f')
    print(f"  Monthly w/ HP     -> {mh_out}")

    # Bar charts
    plot_monthly_bar(
        monthly_base,
        f"Monthly AC Generation - Baseline System ({sizing_base['n_panels']} panels, "
        f"{sizing_base['array_dc_kw']:.1f} kW DC)",
        os.path.join(out_dir, f"{SITE['name']}_monthly_baseline.png"),
        annual_load=ANNUAL_LOAD_KWH,
        color='#2E5090')

    plot_monthly_bar(
        monthly_hp,
        f"Monthly AC Generation - Expanded System w/ Heat Pump "
        f"({sizing_hp['n_panels']} panels, {sizing_hp['array_dc_kw']:.1f} kW DC)",
        os.path.join(out_dir, f"{SITE['name']}_monthly_with_hp.png"),
        annual_load=total_load_kwh,
        color='#E67E22')

    plot_comparison_bar(
        monthly_base, monthly_hp,
        os.path.join(out_dir, f"{SITE['name']}_monthly_comparison.png"),
        baseline_load=ANNUAL_LOAD_KWH,
        total_load=total_load_kwh)

    print("\nDone.")


if __name__ == '__main__':
    main()