"""
Diurnal generation vs demand profile for 2050.

Usage:
    python diurnal_profile.py wind
    python diurnal_profile.py solar

Reads generation from hourly_balance_2050.csv (same directory — contains wind_MW and solar_MW).
Computes mean load and mean generation by hour of day (0-23) across all 8760 hours,
then writes a CSV and two plots (annual average + month-faceted).

Outputs -> diurnal/<source>/
    diurnal_profile.csv
    diurnal_annual.png   — single panel, full-year average
    diurnal_monthly.png  — 12-panel faceted by month
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if len(sys.argv) != 2 or sys.argv[1].lower() not in ("wind", "solar"):
    sys.exit("Usage: python diurnal_profile.py [wind|solar]")

SOURCE = sys.argv[1].lower()
HERE = Path(__file__).parent
OUT = HERE / "diurnal" / SOURCE
OUT.mkdir(parents=True, exist_ok=True)

gen_col = "wind_MW" if SOURCE == "wind" else "solar_MW"
label = "Wind generation" if SOURCE == "wind" else "Solar generation"
color_gen = "#2980b9" if SOURCE == "wind" else "#e67e22"

bal = pd.read_csv(HERE / "hourly_balance_2050.csv", parse_dates=["datetime"])
assert len(bal) == 8760

bal["hour"] = bal["datetime"].dt.hour
bal["month"] = bal["datetime"].dt.month

# --- Annual diurnal average ---------------------------------------------------
diurnal = (
    bal.groupby("hour")[["load_MW", gen_col]]
    .mean()
    .round(2)
    .rename(columns={gen_col: "gen_MW"})
)
diurnal.index.name = "hour"
diurnal["surplus_MW"] = (diurnal["gen_MW"] - diurnal["load_MW"]).round(2)
diurnal.to_csv(OUT / "diurnal_profile.csv")

# Annual plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(diurnal.index, diurnal["load_MW"], color="#c0392b", lw=2, label="Load")
ax.plot(diurnal.index, diurnal["gen_MW"], color=color_gen, lw=2, label=label)
ax.fill_between(
    diurnal.index, diurnal["gen_MW"], diurnal["load_MW"],
    where=diurnal["gen_MW"] >= diurnal["load_MW"],
    interpolate=True, alpha=0.25, color=color_gen, label="Surplus",
)
ax.fill_between(
    diurnal.index, diurnal["gen_MW"], diurnal["load_MW"],
    where=diurnal["gen_MW"] < diurnal["load_MW"],
    interpolate=True, alpha=0.25, color="#c0392b", label="Shortfall",
)
ax.set_xlabel("Hour of day")
ax.set_ylabel("MW")
ax.set_title(f"2050 diurnal profile — {label} vs demand (annual average)")
ax.set_xticks(range(0, 24, 2))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "diurnal_annual.png", dpi=130)
plt.close(fig)

# --- Monthly facet (3 rows × 4 cols) -----------------------------------------
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey=True, sharex=True)
for m, ax in zip(range(1, 13), axes.flat):
    sub = (
        bal[bal["month"] == m]
        .groupby("hour")[["load_MW", gen_col]]
        .mean()
        .rename(columns={gen_col: "gen_MW"})
    )
    ax.plot(sub.index, sub["load_MW"], color="#c0392b", lw=1.4, label="Load")
    ax.plot(sub.index, sub["gen_MW"], color=color_gen, lw=1.4, label=label)
    ax.fill_between(
        sub.index, sub["gen_MW"], sub["load_MW"],
        where=sub["gen_MW"] >= sub["load_MW"],
        interpolate=True, alpha=0.25, color=color_gen,
    )
    ax.fill_between(
        sub.index, sub["gen_MW"], sub["load_MW"],
        where=sub["gen_MW"] < sub["load_MW"],
        interpolate=True, alpha=0.25, color="#c0392b",
    )
    ax.set_title(month_names[m - 1], fontsize=10)
    ax.set_xticks(range(0, 24, 6))
    ax.grid(True, alpha=0.25)

for ax in axes[-1]:
    ax.set_xlabel("Hour")
for ax in axes[:, 0]:
    ax.set_ylabel("MW")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle(f"2050 monthly diurnal profile — {label} vs demand", fontsize=12, y=1.01)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / "diurnal_monthly.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# Console summary
print(f"Source  : {SOURCE}")
print(f"Outputs : {OUT}")
print()
print(f"{'Hour':>4}  {'Load (MW)':>10}  {'Gen (MW)':>10}  {'Surplus (MW)':>13}")
print("-" * 45)
for h, row in diurnal.iterrows():
    print(f"{h:4d}  {row['load_MW']:10,.0f}  {row['gen_MW']:10,.0f}  {row['surplus_MW']:+13,.0f}")
