"""
2050 wind-only supply/demand balance.

Inputs:
  load_projections/2050/total/total_load_2050_K.csv      — hourly demand (MW)
  wind/data/wind_li_cf_hourly.csv                        — per-MW-AC CF (8760)

Approach:
  Both files are 8760 hours (no leap day). The CF series is a 9-year TMY-style
  average (datetime stamped to 2023). We align by month-day-hour with the 2050
  load timestamps and scale CF by installed capacity to get wind generation.

Outputs:
  balance/2050/wind_only/hourly_balance_2050.csv        — datetime, load, wind, deficit
  balance/2050/wind_only/monthly_hour_deficit_2050.csv  — month x hour average deficit
  balance/2050/wind_only/monthly_hour_deficit_2050.png  — heatmap
  balance/2050/wind_only/summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "balance" / "2050" / "wind_only"
OUT.mkdir(parents=True, exist_ok=True)

# Sizing (matches wind_li_analysis.py 2050 scenario)
TARGET_GWH = 38466.7 * 0.65  # 65 % of 2050 annual load projection (≈ 25,003 GWh)
RATED_KW = 2800.0

load = pd.read_csv(ROOT / "load_projections" / "2050" / "total" / "total_load_2050_K.csv")
load["hour"] = pd.to_datetime(load["hour"])
load = load.rename(columns={"hour": "datetime"})

wind = pd.read_csv(ROOT / "wind" / "data" / "wind_li_cf_hourly.csv")
wind["datetime"] = pd.to_datetime(wind["datetime"])

# Drop Feb 29 if present (neither 2023 nor 2050 has one — defensive).
load = load[~((load["datetime"].dt.month == 2) & (load["datetime"].dt.day == 29))].reset_index(drop=True)
wind = wind[~((wind["datetime"].dt.month == 2) & (wind["datetime"].dt.day == 29))].reset_index(drop=True)
assert len(load) == 8760 and len(wind) == 8760

# Align by (month, day, hour)
key = ["month", "day", "hour"]
for df in (load, wind):
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour

merged = load.merge(wind[key + ["cf"]], on=key, how="left", validate="1:1")
assert merged["cf"].notna().all()

# Recompute installed capacity for the 2050 target (2475-turbine scenario at 60.81% CF
# now sizes to ~1341 turbines; reuse the same sizing logic).
annual_cf = wind["cf"].mean()
mw_required = (TARGET_GWH * 1000.0) / (8760.0 * annual_cf)
n_turbines = int(np.ceil(mw_required * 1000.0 / RATED_KW))
installed_mw = n_turbines * RATED_KW / 1000.0

merged["wind_MW"] = merged["cf"] * installed_mw
merged["deficit_MW"] = merged["total_MW"] - merged["wind_MW"]  # +ve = deficit, -ve = surplus

hourly = merged[["datetime", "total_MW", "wind_MW", "deficit_MW"]].rename(
    columns={"total_MW": "load_MW"}
)
hourly[["load_MW", "wind_MW", "deficit_MW"]] = hourly[["load_MW", "wind_MW", "deficit_MW"]].round(2)
hourly.to_csv(OUT / "hourly_balance_2050.csv", index=False)

# Monthly x hour-of-day average deficit
merged["mo"] = merged["datetime"].dt.month
merged["hr"] = merged["datetime"].dt.hour
mh = (
    merged.groupby(["mo", "hr"])["deficit_MW"]
    .mean()
    .unstack("hr")
    .round(2)
)
mh.index.name = "month"
mh.columns.name = "hour"
mh.to_csv(OUT / "monthly_hour_deficit_2050.csv")

# Heatmap
fig, ax = plt.subplots(figsize=(11, 5))
vmax = max(abs(mh.values.min()), abs(mh.values.max()))
im = ax.imshow(mh.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(24))
ax.set_xticklabels(range(24))
ax.set_yticks(range(12))
ax.set_yticklabels(range(1, 13))
ax.set_xlabel("Hour of day (local)")
ax.set_ylabel("Month")
ax.set_title(f"2050 wind-only deficit (MW) — {n_turbines} x GE 2.8-127, {installed_mw:,.0f} MW")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Deficit MW (+ = shortfall, - = surplus)")
fig.tight_layout()
fig.savefig(OUT / "monthly_hour_deficit_2050.png", dpi=130)
plt.close(fig)

# Daily average deficit/surplus (365 days)
daily = (
    hourly.assign(date=hourly["datetime"].dt.date)
    .groupby("date", as_index=False)["deficit_MW"]
    .mean()
)
daily["date"] = pd.to_datetime(daily["date"])
daily.to_csv(OUT / "daily_deficit_2050.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4.5))
colors = np.where(daily["deficit_MW"] >= 0, "#c0392b", "#2980b9")
ax.bar(daily["date"], daily["deficit_MW"], color=colors, width=1.0)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Date (2050)")
ax.set_ylabel("Daily mean deficit (MW)")
ax.set_title(f"2050 wind-only daily mean deficit (+ shortfall / - surplus) — {installed_mw:,.0f} MW fleet")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "daily_deficit_2050.png", dpi=130)
plt.close(fig)

# Summary
load_TWh = hourly["load_MW"].sum() / 1e6
wind_TWh = hourly["wind_MW"].sum() / 1e6
deficit_only = hourly["deficit_MW"].clip(lower=0).sum() / 1e6
surplus_only = (-hourly["deficit_MW"].clip(upper=0)).sum() / 1e6
hours_deficit = (hourly["deficit_MW"] > 0).sum()
hours_surplus = (hourly["deficit_MW"] < 0).sum()

with open(OUT / "summary.txt", "w") as f:
    lines = [
        f"Scenario              : 2050 wind-only (target {TARGET_GWH:,.0f} GWh/yr from wind)",
        f"Wind fleet            : {n_turbines} x GE 2.8-127 = {installed_mw:,.1f} MW AC",
        f"Annual wind CF        : {annual_cf:.4f} ({annual_cf*100:.2f}%)",
        f"",
        f"Annual load           : {load_TWh:,.2f} TWh",
        f"Annual wind energy    : {wind_TWh:,.2f} TWh ({wind_TWh/load_TWh*100:.1f}% of load)",
        f"Energy deficit (sum>0): {deficit_only:,.2f} TWh",
        f"Energy surplus (sum<0): {surplus_only:,.2f} TWh",
        f"Hours with deficit    : {hours_deficit:,} / 8760 ({hours_deficit/8760*100:.1f}%)",
        f"Hours with surplus    : {hours_surplus:,} / 8760 ({hours_surplus/8760*100:.1f}%)",
        f"Peak hourly deficit   : {hourly['deficit_MW'].max():,.0f} MW",
        f"Peak hourly surplus   : {(-hourly['deficit_MW']).max():,.0f} MW",
    ]
    f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

print(f"\nOutputs written to: {OUT}")
