"""
5 MW on-shore wind feasibility study — winery industry, Riverhead, LI, NY.

Data: multi-year TMY hourly averages (2014, 2015, 2018, 2019, 2020) at 10 m and 60 m.
Site: 40.92 N, -72.66 W (Riverhead, Long Island).
Turbine: Energie PGE 20/35 — 36.4 kW, 19.2 m rotor, 80 m hub, shear alpha=0.14.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "data" / "tmy_hourly_averages.csv"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# --- Site / turbine constants -------------------------------------------------
RHO = 1.225                  # kg/m^3, standard air density at sea level
ALPHA = 0.14                 # shear coefficient (open terrain, per turbine spec)
HUB_H = 80.0                 # m, turbine hub height
ROTOR_D = 19.2               # m
ROTOR_R = ROTOR_D / 2
ROTOR_A = np.pi * ROTOR_R**2 # m^2
RATED_KW = 36.4              # kW per turbine
CUT_IN = 3.0                 # m/s (from power curve)
CUT_OUT = 25.0               # m/s (from power curve)

PROJECT_MW = 5.0             # total project capacity
N_TURBINES = int(np.ceil(PROJECT_MW * 1000 / RATED_KW))

# Reference heights
H_STD = 9.1                  # m, standard reference height
H_50 = 50.0                  # m
H_MEAS_LOW = 10.0            # m
H_MEAS_HIGH = 60.0           # m

# --- Load data ----------------------------------------------------------------
df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
df.rename(columns={
    "wind speed at 10m (m/s)": "v10",
    "wind speed at 60m (m/s)": "v60",
}, inplace=True)

# --- Extrapolation via power law ---------------------------------------------
# Hourly shear from measured pair (10m, 60m): alpha_hr = ln(v60/v10)/ln(60/10)
# If any hour is calm, fall back to the nominal ALPHA=0.14.
with np.errstate(divide="ignore", invalid="ignore"):
    alpha_hr = np.log(df["v60"] / df["v10"]) / np.log(H_MEAS_HIGH / H_MEAS_LOW)
alpha_hr = alpha_hr.where(np.isfinite(alpha_hr) & (df["v10"] > 0.1), ALPHA)

def extrapolate(v_ref, h_ref, h_target, alpha):
    return v_ref * (h_target / h_ref) ** alpha

# 9.1 m and 50 m sit inside the 10-60 m measurement range, so the hourly fit
# shear gives a clean interpolation. For 80 m (above the top sensor), using
# the same near-surface shear tends to overshoot — extrapolate from the 60 m
# anchor with the turbine-spec shear (0.14, open terrain) instead.
df["v9_1"] = extrapolate(df["v10"], H_MEAS_LOW, H_STD, alpha_hr)
df["v50"]  = extrapolate(df["v10"], H_MEAS_LOW, H_50, alpha_hr)
df["v80"]  = extrapolate(df["v60"], H_MEAS_HIGH, HUB_H, ALPHA)

# --- Wind power density (kWh/m^2) --------------------------------------------
# P_wind = 0.5 * rho * V^3   (W/m^2)
# Energy per hour = P_wind * 1 h  →  divide by 1000 for kWh/m^2
def wpd_kwh_per_m2(v):
    return 0.5 * RHO * v**3 / 1000.0  # kWh/m^2 per hour

df["wpd_9_1"] = wpd_kwh_per_m2(df["v9_1"])
df["wpd_50"]  = wpd_kwh_per_m2(df["v50"])

# Monthly / annual wind resource (summing hourly kWh/m^2)
monthly_res = df.groupby("Month").agg(
    v9_1_mean=("v9_1", "mean"),
    v50_mean=("v50", "mean"),
    v80_mean=("v80", "mean"),
    wpd_9_1_kWh_per_m2=("wpd_9_1", "sum"),
    wpd_50_kWh_per_m2=("wpd_50", "sum"),
).round(3)

annual_res = pd.Series({
    "v9_1_mean_m_s": df["v9_1"].mean(),
    "v50_mean_m_s": df["v50"].mean(),
    "v80_mean_m_s": df["v80"].mean(),
    "wpd_9_1_kWh_per_m2_yr": df["wpd_9_1"].sum(),
    "wpd_50_kWh_per_m2_yr": df["wpd_50"].sum(),
}).round(3)

# --- Turbine power curve (Energie PGE 20/35) ---------------------------------
# Digitized from the manufacturer curve in the project brief (kW vs m/s).
# Peaks at ~36.4 kW near 14 m/s, drops to a ~30 kW plateau from 18-25 m/s,
# hard cut-out at 25 m/s.
curve_v = np.array([0, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.01, 30])
curve_p = np.array([0, 0,   0.2, 1.5, 4, 8, 13, 19, 25, 30, 33.5, 35.5, 36.3, 36.4, 35.5, 33, 31, 30, 30, 30, 30, 30, 30, 30, 30, 0, 0])

def turbine_power(v):
    """kW output given hub-height wind speed. Enforces cut-in and cut-out."""
    v = np.asarray(v, dtype=float)
    p = np.interp(v, curve_v, curve_p, left=0.0, right=0.0)
    p = np.where(v < CUT_IN, 0.0, p)
    p = np.where(v >= CUT_OUT, 0.0, p)
    return p

df["p_turbine_kW"] = turbine_power(df["v80"])

# --- Annual and monthly energy per turbine and for the array -----------------
# Hourly output in kWh = kW * 1 h
df["e_turbine_kWh"] = df["p_turbine_kW"] * 1.0

monthly_energy = df.groupby("Month").agg(
    e_per_turbine_kWh=("e_turbine_kWh", "sum"),
).round(1)
monthly_energy["e_array_kWh"] = (monthly_energy["e_per_turbine_kWh"] * N_TURBINES).round(1)

annual_per_turbine_kWh = df["e_turbine_kWh"].sum()
annual_array_kWh = annual_per_turbine_kWh * N_TURBINES
capacity_factor = annual_per_turbine_kWh / (RATED_KW * 8760)

# --- Land use ----------------------------------------------------------------
# Standard wind-farm spacing: 5D downwind x 3D crosswind (D = rotor diameter).
# Area per turbine = 5D * 3D = 15 * D^2.
area_per_turbine_m2 = 15 * ROTOR_D**2
total_area_m2 = area_per_turbine_m2 * N_TURBINES
total_area_acres = total_area_m2 / 4046.86

# --- Economic model ----------------------------------------------------------
# Assumptions (distributed small-wind, NY 2024-2025 baseline):
CAPEX_PER_KW = 6000          # $/kW installed (small-wind distributed, tall tower)
OM_PER_KW_YR = 50            # $/kW-yr
PROJECT_LIFE_YR = 25
DISCOUNT_RATE = 0.06
ELECTRICITY_PRICE = 0.18     # $/kWh avoided retail cost (LI commercial)
ITC_FRACTION = 0.30          # Federal Investment Tax Credit (Sec 48)
NY_STATE_CREDIT = 0.00       # commercial small wind: NYSERDA rebates vary; assume 0 baseline
# Net metering: excess generation credited at retail (NY net-metering / VDER floor ~retail)
NET_METER_FRACTION = 1.0

project_kW = N_TURBINES * RATED_KW
capex_gross = project_kW * CAPEX_PER_KW
itc_credit = capex_gross * ITC_FRACTION
capex_net = capex_gross - itc_credit - (capex_gross * NY_STATE_CREDIT)
om_year = project_kW * OM_PER_KW_YR
annual_revenue = annual_array_kWh * ELECTRICITY_PRICE * NET_METER_FRACTION
net_cash_flow_year = annual_revenue - om_year
simple_payback_yr = capex_net / net_cash_flow_year

# NPV
years = np.arange(1, PROJECT_LIFE_YR + 1)
discounted = net_cash_flow_year / (1 + DISCOUNT_RATE) ** years
npv = discounted.sum() - capex_net

# LCOE: total discounted cost / total discounted energy
discounted_om = (om_year / (1 + DISCOUNT_RATE) ** years).sum()
discounted_energy = (annual_array_kWh / (1 + DISCOUNT_RATE) ** years).sum()
lcoe = (capex_net + discounted_om) / discounted_energy

# --- Carbon savings ----------------------------------------------------------
# EPA eGRID 2022: NYUP subregion (Upstate NY) ~ 110 kg CO2/MWh.
# NYCW (NYC/LI) ~ 280 kg CO2/MWh — Riverhead is on the LI grid (NYCW).
# Use NYCW: 0.28 kg CO2/kWh as the displaced emission factor.
CO2_KG_PER_KWH = 0.28
annual_co2_avoided_tonnes = annual_array_kWh * CO2_KG_PER_KWH / 1000
lifetime_co2_avoided_tonnes = annual_co2_avoided_tonnes * PROJECT_LIFE_YR

# --- Write outputs -----------------------------------------------------------
monthly_res.to_csv(OUT / "monthly_wind_resource.csv")
monthly_energy.to_csv(OUT / "monthly_energy.csv")
annual_res.to_csv(OUT / "annual_wind_resource.csv", header=["value"])

summary = {
    "Site": "Riverhead, LI, NY (40.92N, -72.66W)",
    "Turbine": "Energie PGE 20/35 (36.4 kW, 19.2 m rotor, 80 m hub)",
    "Number of turbines": N_TURBINES,
    "Installed capacity (kW)": project_kW,
    "Land area (acres, 5Dx3D spacing)": round(total_area_acres, 1),
    "Annual energy per turbine (kWh)": round(annual_per_turbine_kWh, 0),
    "Annual array energy (kWh)": round(annual_array_kWh, 0),
    "Capacity factor": round(capacity_factor, 3),
    "Mean wind speed @ 9.1 m (m/s)": round(float(annual_res["v9_1_mean_m_s"]), 2),
    "Mean wind speed @ 50 m (m/s)": round(float(annual_res["v50_mean_m_s"]), 2),
    "Mean wind speed @ 80 m (m/s)": round(float(annual_res["v80_mean_m_s"]), 2),
    "Wind power density @ 9.1 m (kWh/m^2/yr)": round(float(annual_res["wpd_9_1_kWh_per_m2_yr"]), 1),
    "Wind power density @ 50 m (kWh/m^2/yr)": round(float(annual_res["wpd_50_kWh_per_m2_yr"]), 1),
    "CAPEX gross ($)": f"{capex_gross:,.0f}",
    "ITC credit ($)": f"{itc_credit:,.0f}",
    "CAPEX net of incentives ($)": f"{capex_net:,.0f}",
    "Annual O&M ($/yr)": f"{om_year:,.0f}",
    "Annual revenue ($/yr, net metered)": f"{annual_revenue:,.0f}",
    "Simple payback (yr)": round(simple_payback_yr, 1),
    f"NPV @ {DISCOUNT_RATE*100:.0f}% over {PROJECT_LIFE_YR} yr ($)": f"{npv:,.0f}",
    "LCOE ($/kWh)": round(lcoe, 3),
    "Annual CO2 avoided (tonnes/yr)": round(annual_co2_avoided_tonnes, 1),
    f"Lifetime CO2 avoided ({PROJECT_LIFE_YR} yr, tonnes)": round(lifetime_co2_avoided_tonnes, 0),
}

with open(OUT / "summary.txt", "w") as f:
    for k, v in summary.items():
        f.write(f"{k:50s} : {v}\n")

# --- Plots -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(curve_v, curve_p, "-o", ms=3)
ax.axvline(CUT_IN, color="g", ls="--", alpha=0.5, label=f"cut-in {CUT_IN} m/s")
ax.axvline(CUT_OUT, color="r", ls="--", alpha=0.5, label=f"cut-out {CUT_OUT} m/s")
ax.set_xlabel("Hub-height wind speed (m/s)")
ax.set_ylabel("Turbine power (kW)")
ax.set_title("Energie PGE 20/35 power curve (digitized)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "power_curve.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
months = monthly_energy.index
ax.bar(months, monthly_energy["e_array_kWh"] / 1e6)
ax.set_xlabel("Month")
ax.set_ylabel("Array energy (GWh)")
ax.set_title(f"Monthly net energy — {N_TURBINES} x Energie PGE 20/35")
ax.set_xticks(range(1, 13))
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "monthly_energy.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(months, monthly_res["v9_1_mean"], "-o", label="9.1 m")
ax.plot(months, monthly_res["v50_mean"], "-s", label="50 m")
ax.plot(months, monthly_res["v80_mean"], "-^", label="80 m (hub)")
ax.set_xlabel("Month")
ax.set_ylabel("Mean wind speed (m/s)")
ax.set_title("Monthly mean wind speed — Riverhead TMY")
ax.set_xticks(range(1, 13))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "monthly_wind_speed.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(months - 0.2, monthly_res["wpd_9_1_kWh_per_m2"], width=0.4, label="9.1 m")
ax.bar(months + 0.2, monthly_res["wpd_50_kWh_per_m2"], width=0.4, label="50 m")
ax.set_xlabel("Month")
ax.set_ylabel("Wind power density (kWh/m$^2$)")
ax.set_title("Monthly wind resource — Riverhead TMY")
ax.set_xticks(range(1, 13))
ax.grid(True, alpha=0.3, axis="y")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "monthly_wpd.png", dpi=130)
plt.close(fig)

print("Summary:")
for k, v in summary.items():
    print(f"  {k:50s} : {v}")
print(f"\nOutputs written to {OUT}")
