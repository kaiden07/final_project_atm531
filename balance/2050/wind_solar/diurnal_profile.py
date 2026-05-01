"""
Diurnal generation vs demand profile for 2050.

Usage:
    python diurnal_profile.py wind
    python diurnal_profile.py solar
    python diurnal_profile.py wind_solar    # stacked wind + solar

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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

VALID = ("wind", "solar", "wind_solar")
if len(sys.argv) != 2 or sys.argv[1].lower() not in VALID:
    sys.exit(f"Usage: python diurnal_profile.py [{ '|'.join(VALID) }]")

SOURCE = sys.argv[1].lower()
HERE = Path(__file__).parent
OUT = HERE / "diurnal" / SOURCE
OUT.mkdir(parents=True, exist_ok=True)

WIND_COLOR = "#2980b9"
SOLAR_COLOR = "#e67e22"
LOAD_COLOR = "#c0392b"

bal = pd.read_csv(HERE / "hourly_balance_2050.csv", parse_dates=["datetime"])
assert len(bal) == 8760
bal["hour"] = bal["datetime"].dt.hour
bal["month"] = bal["datetime"].dt.month

if SOURCE == "wind_solar":
    gen_cols = ["wind_MW", "solar_MW"]
    title_label = "Wind + Solar generation"
else:
    gen_cols = ["wind_MW" if SOURCE == "wind" else "solar_MW"]
    title_label = "Wind generation" if SOURCE == "wind" else "Solar generation"

# --- Annual diurnal average ---------------------------------------------------
diurnal = bal.groupby("hour")[["load_MW"] + gen_cols].mean().round(2)
diurnal["gen_total_MW"] = diurnal[gen_cols].sum(axis=1).round(2)
diurnal["surplus_MW"] = (diurnal["gen_total_MW"] - diurnal["load_MW"]).round(2)
diurnal.index.name = "hour"
diurnal.to_csv(OUT / "diurnal_profile.csv")

# --- Annual plot --------------------------------------------------------------
SURPLUS_COLOR = "#27ae60"

def annual_plot(ax, df, with_legend_labels=True):
    total = df[gen_cols].sum(axis=1)
    if SOURCE == "wind_solar":
        ax.plot(df.index, df["wind_MW"], color=WIND_COLOR, lw=1.6,
                label="Wind" if with_legend_labels else None)
        ax.plot(df.index, df["solar_MW"], color=SOLAR_COLOR, lw=1.6,
                label="Solar" if with_legend_labels else None)
        ax.plot(df.index, total, color="#34495e", lw=1.8, ls="--",
                label="Wind + Solar" if with_legend_labels else None)
    else:
        col = gen_cols[0]
        color = WIND_COLOR if SOURCE == "wind" else SOLAR_COLOR
        ax.plot(df.index, df[col], color=color, lw=2.0,
                label=title_label if with_legend_labels else None)
    ax.plot(df.index, df["load_MW"], color=LOAD_COLOR, lw=2.2,
            label="Load" if with_legend_labels else None)
    # Shaded deficit / surplus between total generation and load
    ax.fill_between(df.index, total, df["load_MW"],
                    where=total < df["load_MW"], interpolate=True,
                    alpha=0.25, color=LOAD_COLOR,
                    label="Shortfall" if with_legend_labels else None)
    ax.fill_between(df.index, total, df["load_MW"],
                    where=total >= df["load_MW"], interpolate=True,
                    alpha=0.25, color=SURPLUS_COLOR,
                    label="Surplus" if with_legend_labels else None)

fig, ax = plt.subplots(figsize=(9, 5))
annual_plot(ax, diurnal)
ax.set_xlabel("Hour of day")
ax.set_ylabel("MW")
ax.set_title(f"2050 diurnal profile — {title_label} vs demand (annual average)")
ax.set_xticks(range(0, 24, 2))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "diurnal_annual.png", dpi=130)
plt.close(fig)

# --- Monthly facet ------------------------------------------------------------
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey=True, sharex=True)
for m, ax in zip(range(1, 13), axes.flat):
    sub = bal[bal["month"] == m].groupby("hour")[["load_MW"] + gen_cols].mean()
    annual_plot(ax, sub, with_legend_labels=(m == 1))
    ax.set_title(month_names[m - 1], fontsize=10)
    ax.set_xticks(range(0, 24, 6))
    ax.grid(True, alpha=0.25)

for ax in axes[-1]:
    ax.set_xlabel("Hour")
for ax in axes[:, 0]:
    ax.set_ylabel("MW")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle(f"2050 monthly diurnal profile — {title_label} vs demand", fontsize=12, y=1.01)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / "diurnal_monthly.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# --- Console summary ----------------------------------------------------------
print(f"Source  : {SOURCE}")
print(f"Outputs : {OUT}\n")
header = f"{'Hour':>4}  {'Load (MW)':>10}  " + "  ".join(f"{c:>10}" for c in gen_cols) + f"  {'Total Gen':>10}  {'Surplus':>10}"
print(header)
print("-" * len(header))
for h, row in diurnal.iterrows():
    parts = [f"{h:4d}", f"{row['load_MW']:10,.0f}"]
    parts += [f"{row[c]:10,.0f}" for c in gen_cols]
    parts += [f"{row['gen_total_MW']:10,.0f}", f"{row['surplus_MW']:+10,.0f}"]
    print("  ".join(parts))
