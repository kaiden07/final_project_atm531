"""
Long Island utility-scale wind feasibility — GE 2.8-127, 100 m hub, 2050 scenario.

Data: 9 years of hourly wind speeds at 100 m (site 2325452, 40.24N -73.51W),
averaged across years to a TMY-style 8760-hour profile.

Outputs (all under wind/):
  data/wind_li_cf_hourly.csv               — per-MW-AC normalized hourly CF
  outputs/2050/annual_wind_resource.csv
  outputs/2050/monthly_wind_resource.csv
  outputs/2050/monthly_energy.csv
  outputs/2050/summary.txt
  outputs/2050/power_curve.png
  outputs/2050/monthly_wind_speed.png
  outputs/2050/monthly_wpd.png
  outputs/2050/monthly_energy.png
  outputs/2050/diurnal_cf.png
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA_SRC = HERE / "data"
DATA_OUT = HERE / "data"
OUT = HERE / "outputs" / "2050"
OUT.mkdir(parents=True, exist_ok=True)

# --- Constants / turbine: GE 2.8-127 ------------------------------------------
RHO = 1.225
RATED_KW = 2800.0
ROTOR_D = 127.0
ROTOR_R = ROTOR_D / 2
ROTOR_A = np.pi * ROTOR_R**2
HUB_H = 100.0
CUT_IN = 3.0
CUT_OUT = 25.0

# Power curve digitized from GE/IEC class III public spec (kW vs m/s, 1.225 kg/m^3).
curve_v = np.array([
    0.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
    8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
    25.0, 25.01, 30.0,
])
curve_p = np.array([
    0, 0, 0, 22, 71, 137, 226, 339, 477, 644, 841, 1071, 1334,
    1632, 1942, 2244, 2484, 2664, 2773, 2799, 2800, 2800, 2800,
    2800, 0, 0,
], dtype=float)

def turbine_power(v):
    v = np.asarray(v, dtype=float)
    p = np.interp(v, curve_v, curve_p, left=0.0, right=0.0)
    p = np.where(v < CUT_IN, 0.0, p)
    p = np.where(v >= CUT_OUT, 0.0, p)
    return p

def wpd_kwh_per_m2(v):
    """Wind power density per hour (kWh/m^2)."""
    return 0.5 * RHO * v**3 / 1000.0

# --- Load multi-year data and average to 8760 ---------------------------------
files = sorted(glob.glob(str(DATA_SRC / "2325452_*.csv")))
if not files:
    raise SystemExit(f"No 100 m wind data files found in {DATA_SRC}")

frames = []
for f in files:
    df = pd.read_csv(f, header=1)
    df.columns = df.columns.str.strip()
    rename_map = {c: "v100" for c in df.columns if c.lower() == "wind speed at 100m (m/s)"}
    df = df.rename(columns=rename_map)
    frames.append(df[["Year", "Month", "Day", "Hour", "Minute", "v100"]])

combined = pd.concat(frames, ignore_index=True)
n_years = combined["Year"].nunique()
yr_min, yr_max = int(combined["Year"].min()), int(combined["Year"].max())

# Drop Feb 29 so the TMY is a clean 8760-hour year
combined = combined[~((combined["Month"] == 2) & (combined["Day"] == 29))]

tmy = (
    combined.groupby(["Month", "Day", "Hour", "Minute"], as_index=False)["v100"]
    .mean()
    .sort_values(["Month", "Day", "Hour", "Minute"])
    .reset_index(drop=True)
)
assert len(tmy) == 8760, f"Expected 8760 rows after averaging, got {len(tmy)}"

tmy["datetime"] = pd.to_datetime({
    "year": 2023, "month": tmy["Month"], "day": tmy["Day"],
    "hour": tmy["Hour"], "minute": tmy["Minute"],
})

# --- Resource and turbine output ---------------------------------------------
tmy["wpd_100"] = wpd_kwh_per_m2(tmy["v100"])
tmy["p_kW"] = turbine_power(tmy["v100"])
tmy["e_turbine_kWh"] = tmy["p_kW"] * 1.0
tmy["cf"] = tmy["p_kW"] / RATED_KW

annual_cf = tmy["cf"].mean()
monthly_cf = tmy.groupby("Month")["cf"].mean()
diurnal_cf = tmy.groupby("Hour")["cf"].mean()

annual_per_turbine_kWh = tmy["e_turbine_kWh"].sum()

# --- Sizing for 2050 target demand --------------------------------------------
TARGET_GWH = 38466.7 * 0.65  # 65 % of 2050 annual load projection (≈ 25,003 GWh)
target_mwh = TARGET_GWH * 1000.0
mw_required = target_mwh / (8760.0 * annual_cf)
n_turbines = int(np.ceil(mw_required * 1000.0 / RATED_KW))
installed_mw = n_turbines * RATED_KW / 1000.0

area_per_turbine_m2 = 5 * ROTOR_D * 3 * ROTOR_D
total_area_km2 = area_per_turbine_m2 * n_turbines / 1e6

annual_array_kWh = annual_per_turbine_kWh * n_turbines

# --- Resource and energy tables ----------------------------------------------
monthly_res = tmy.groupby("Month").agg(
    v100_mean=("v100", "mean"),
    wpd_100_kWh_per_m2=("wpd_100", "sum"),
).round(3)

monthly_energy = tmy.groupby("Month").agg(
    e_per_turbine_kWh=("e_turbine_kWh", "sum"),
).round(1)
monthly_energy["e_array_GWh"] = (monthly_energy["e_per_turbine_kWh"] * n_turbines / 1e6).round(2)
monthly_energy["cf"] = monthly_cf.round(4)

annual_res = pd.Series({
    "v100_mean_m_s": tmy["v100"].mean(),
    "wpd_100_kWh_per_m2_yr": tmy["wpd_100"].sum(),
    "annual_cf": annual_cf,
    "annual_per_turbine_MWh": annual_per_turbine_kWh / 1000.0,
}).round(3)

# --- Hourly CF series ---------------------------------------------------------
out_df = pd.DataFrame({"datetime": tmy["datetime"], "cf": tmy["cf"].round(6)})
out_csv = DATA_OUT / "wind_li_cf_hourly.csv"
out_df.to_csv(out_csv, index=False)

# --- CSV outputs --------------------------------------------------------------
monthly_res.to_csv(OUT / "monthly_wind_resource.csv")
monthly_energy.to_csv(OUT / "monthly_energy.csv")
annual_res.to_csv(OUT / "annual_wind_resource.csv", header=["value"])

summary = {
    "Site": "Long Island, NY (40.24N, -73.51W; site 2325452)",
    "Data window": f"{yr_min}-{yr_max} ({n_years}-year hourly average, 100 m)",
    "Turbine": "GE 2.8-127 (2.8 MW, 127 m rotor, 100 m hub)",
    "Target annual demand (GWh)": f"{TARGET_GWH:,.0f}",
    "Number of turbines": n_turbines,
    "Installed capacity (MW)": f"{installed_mw:,.1f}",
    "Farm area, 5D x 3D spacing (km^2)": round(total_area_km2, 2),
    "Annual energy per turbine (MWh)": round(annual_per_turbine_kWh / 1000, 1),
    "Annual array energy (GWh)": round(annual_array_kWh / 1e6, 1),
    "Annual capacity factor": round(annual_cf, 4),
    "Mean wind speed @ 100 m (m/s)": round(float(annual_res["v100_mean_m_s"]), 2),
    "Wind power density @ 100 m (kWh/m^2/yr)": round(float(annual_res["wpd_100_kWh_per_m2_yr"]), 1),
}
with open(OUT / "summary.txt", "w") as f:
    for k, v in summary.items():
        f.write(f"{k:50s} : {v}\n")

# --- Plots --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(curve_v, curve_p, "-o", ms=3)
ax.axvline(CUT_IN, color="g", ls="--", alpha=0.5, label=f"cut-in {CUT_IN} m/s")
ax.axvline(CUT_OUT, color="r", ls="--", alpha=0.5, label=f"cut-out {CUT_OUT} m/s")
ax.set_xlabel("Hub-height wind speed (m/s)")
ax.set_ylabel("Turbine power (kW)")
ax.set_title("GE 2.8-127 power curve (digitized)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "power_curve.png", dpi=130)
plt.close(fig)

months = monthly_energy.index
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(months, monthly_energy["e_array_GWh"])
ax.set_xlabel("Month")
ax.set_ylabel("Array energy (GWh)")
ax.set_title(f"Monthly net energy — {n_turbines} x GE 2.8-127 ({installed_mw:,.0f} MW)")
ax.set_xticks(range(1, 13))
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "monthly_energy.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(months, monthly_res["v100_mean"], "-^", label="100 m (hub)")
ax.set_xlabel("Month")
ax.set_ylabel("Mean wind speed (m/s)")
ax.set_title("Monthly mean wind speed @ 100 m — Long Island")
ax.set_xticks(range(1, 13))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "monthly_wind_speed.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(months, monthly_res["wpd_100_kWh_per_m2"])
ax.set_xlabel("Month")
ax.set_ylabel("Wind power density (kWh/m$^2$)")
ax.set_title("Monthly wind resource @ 100 m — Long Island")
ax.set_xticks(range(1, 13))
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "monthly_wpd.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(diurnal_cf.index, diurnal_cf.values, "-o")
ax.set_xlabel("Hour of day (local, UTC-5)")
ax.set_ylabel("Capacity factor")
ax.set_title(f"Diurnal mean CF — GE 2.8-127 @ 100 m, Long Island ({n_years}-yr avg)")
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(0.6, diurnal_cf.max() * 1.1))
fig.tight_layout()
fig.savefig(OUT / "diurnal_cf.png", dpi=130)
plt.close(fig)

# --- Console summary ----------------------------------------------------------
print(f"Site: 2325452 (40.24N, -73.51W), {n_years}-year average ({yr_min}-{yr_max})")
print(f"Turbine: GE 2.8-127 @ {HUB_H:.0f} m hub")
print(f"Wrote {len(out_df)} rows to {out_csv}")
print()
print(f"Annual CF: {annual_cf:.4f} ({annual_cf*100:.2f}%)")
print("Monthly CF:")
for m, v in monthly_cf.round(4).items():
    print(f"  {m:2d}: {v:.4f} ({v*100:.2f}%)")
print()
print(f"Target demand:           {TARGET_GWH:,.0f} GWh/yr")
print(f"Required AC capacity:    {mw_required:,.1f} MW")
print(f"Number of turbines:      {n_turbines} x 2.8 MW = {installed_mw:,.1f} MW installed")
print(f"Farm area (5D x 3D):     {total_area_km2:,.2f} km^2")
print(f"Outputs written to:      {OUT}")
