"""
Wind-max + solar 2050 analysis at increased wind fleet sizes.

Reproduces the wind_solar workflow (hourly balance, monthly/daily/hourly deficit
plots, summary, and diurnal profiles for wind/solar/wind_solar) for a chosen
number of SG 14-222 DD turbines. Solar is held constant (12.5 M panels) using
the same source CSV as the base wind_solar scenario.

Usage:
    python wind_max_solar_2050.py 778
    python wind_max_solar_2050.py 1015
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if len(sys.argv) != 2:
    sys.exit("Usage: python wind_max_solar_2050.py <n_turbines>")

N_TURB = int(sys.argv[1])
RATED_MW = 15.0
SOLAR_PANELS = 12_500_000
INSTALLED_MW = N_TURB * RATED_MW

ROOT = Path(__file__).parent.parent
OUT = ROOT / "balance" / "2050" / "wind_max_solar" / str(N_TURB)
OUT.mkdir(parents=True, exist_ok=True)

WIND_COLOR = "#2980b9"
SOLAR_COLOR = "#e67e22"
LOAD_COLOR = "#c0392b"
SURPLUS_COLOR = "#27ae60"

# --- Inputs -------------------------------------------------------------------
load = pd.read_csv(ROOT / "load_projections" / "2050" / "total" / "total_load_2050_K.csv")
load["datetime"] = pd.to_datetime(load["hour"])
load = load[~((load["datetime"].dt.month == 2) & (load["datetime"].dt.day == 29))].reset_index(drop=True)

cf = pd.read_csv(ROOT / "wind" / "data" / "wind_li_cf_hourly.csv")
cf["datetime"] = pd.to_datetime(cf["datetime"])
cf = cf[~((cf["datetime"].dt.month == 2) & (cf["datetime"].dt.day == 29))].reset_index(drop=True)

for df in (load, cf):
    df["mo"] = df["datetime"].dt.month
    df["d"] = df["datetime"].dt.day
    df["h"] = df["datetime"].dt.hour

merged = load.merge(cf[["mo", "d", "h", "cf"]], on=["mo", "d", "h"], validate="1:1")
assert len(merged) == 8760

solar = pd.read_csv(ROOT / "balance" / "solar_hourly_balance.csv",
                    skiprows=1, header=None, names=["solar_MW"])
n_neg = (solar["solar_MW"] < 0).sum()
if n_neg:
    print(f"  Clipping {n_neg} negative solar value(s) to 0")
solar["solar_MW"] = solar["solar_MW"].clip(lower=0)

# --- Build hourly balance ----------------------------------------------------
bal = pd.DataFrame({
    "datetime": merged["datetime"],
    "load_MW": merged["total_MW"],
    "wind_MW": merged["cf"] * INSTALLED_MW,
    "solar_MW": solar["solar_MW"].values,
})
bal["deficit_MW"] = bal["load_MW"] - bal["wind_MW"] - bal["solar_MW"]
for c in ["load_MW", "wind_MW", "solar_MW", "deficit_MW"]:
    bal[c] = bal[c].round(2)

bal[["datetime", "load_MW", "wind_MW", "solar_MW", "deficit_MW"]].to_csv(
    OUT / "hourly_balance_2050.csv", index=False
)

# --- Monthly x hour-of-day deficit -------------------------------------------
bal["month"] = bal["datetime"].dt.month
bal["hr"] = bal["datetime"].dt.hour
mh = bal.groupby(["month", "hr"])["deficit_MW"].mean().unstack("hr").round(2)
mh.index.name = "month"
mh.columns.name = "hour"
mh.to_csv(OUT / "monthly_hour_deficit_2050.csv")

vmax = max(abs(mh.values.min()), abs(mh.values.max()))
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(mh.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
ax.set_yticks(range(12)); ax.set_yticklabels(range(1, 13))
ax.set_xlabel("Hour of day (local)"); ax.set_ylabel("Month")
ax.set_title(f"2050 wind+solar deficit (MW) — {N_TURB} x SG 14-222 DD ({INSTALLED_MW:,.0f} MW) + {SOLAR_PANELS/1e6:.1f}M solar panels")
fig.colorbar(im, ax=ax, label="Deficit MW (+ shortfall, − surplus)")
fig.tight_layout(); fig.savefig(OUT / "monthly_hour_deficit_2050.png", dpi=130); plt.close(fig)

# --- Daily mean deficit -------------------------------------------------------
daily = bal.assign(date=bal["datetime"].dt.date).groupby("date", as_index=False)["deficit_MW"].mean()
daily["date"] = pd.to_datetime(daily["date"])
daily.to_csv(OUT / "daily_deficit_2050.csv", index=False)

colors = np.where(daily["deficit_MW"] >= 0, LOAD_COLOR, "#2980b9")
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(daily["date"], daily["deficit_MW"], color=colors, width=1.0)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Date (2050)"); ax.set_ylabel("Daily mean deficit (MW)")
ax.set_title(f"2050 wind+solar daily mean deficit — {N_TURB} turbines ({INSTALLED_MW:,.0f} MW) + 12.5M solar panels")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "daily_deficit_2050.png", dpi=130); plt.close(fig)

# --- Hourly deficit -----------------------------------------------------------
bal[["datetime", "deficit_MW"]].to_csv(OUT / "hourly_deficit_2050.csv", index=False)

fig, ax = plt.subplots(figsize=(14, 4.5))
pos = bal["deficit_MW"].clip(lower=0); neg = bal["deficit_MW"].clip(upper=0)
ax.fill_between(bal["datetime"], pos, color=LOAD_COLOR, alpha=0.7, label="Shortfall")
ax.fill_between(bal["datetime"], neg, color="#2980b9", alpha=0.7, label="Surplus")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Date (2050)"); ax.set_ylabel("Deficit (MW)")
ax.set_title(f"2050 wind+solar hourly deficit — {N_TURB} turbines ({INSTALLED_MW:,.0f} MW) + 12.5M solar panels")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper right"); ax.grid(True, alpha=0.2, axis="y")
fig.tight_layout(); fig.savefig(OUT / "hourly_deficit_2050.png", dpi=130); plt.close(fig)

# --- Diurnal profiles (wind, solar, wind_solar) -------------------------------
def annual_plot(ax, df, mode, with_legend=True):
    if mode == "wind_solar":
        gens = ["wind_MW", "solar_MW"]
        ax.plot(df.index, df["wind_MW"], color=WIND_COLOR, lw=1.6, label="Wind" if with_legend else None)
        ax.plot(df.index, df["solar_MW"], color=SOLAR_COLOR, lw=1.6, label="Solar" if with_legend else None)
        total = df[gens].sum(axis=1)
        ax.plot(df.index, total, color="#34495e", lw=1.8, ls="--", label="Wind + Solar" if with_legend else None)
    else:
        col = "wind_MW" if mode == "wind" else "solar_MW"
        color = WIND_COLOR if mode == "wind" else SOLAR_COLOR
        lab = "Wind generation" if mode == "wind" else "Solar generation"
        ax.plot(df.index, df[col], color=color, lw=2.0, label=lab if with_legend else None)
        total = df[col]
    ax.plot(df.index, df["load_MW"], color=LOAD_COLOR, lw=2.2, label="Load" if with_legend else None)
    ax.fill_between(df.index, total, df["load_MW"],
                    where=total < df["load_MW"], interpolate=True,
                    alpha=0.25, color=LOAD_COLOR, label="Shortfall" if with_legend else None)
    ax.fill_between(df.index, total, df["load_MW"],
                    where=total >= df["load_MW"], interpolate=True,
                    alpha=0.25, color=SURPLUS_COLOR, label="Surplus" if with_legend else None)

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for mode in ("wind", "solar", "wind_solar"):
    sub_dir = OUT / "diurnal" / mode
    sub_dir.mkdir(parents=True, exist_ok=True)
    cols = ["wind_MW", "solar_MW"] if mode == "wind_solar" else (
           ["wind_MW"] if mode == "wind" else ["solar_MW"])

    diurnal = bal.groupby("hr")[["load_MW"] + cols].mean().round(2)
    diurnal.index.name = "hour"
    diurnal["gen_total_MW"] = diurnal[cols].sum(axis=1).round(2)
    diurnal["surplus_MW"] = (diurnal["gen_total_MW"] - diurnal["load_MW"]).round(2)
    diurnal.to_csv(sub_dir / "diurnal_profile.csv")

    title_label = {"wind": "Wind generation", "solar": "Solar generation",
                   "wind_solar": "Wind + Solar generation"}[mode]

    fig, ax = plt.subplots(figsize=(9, 5))
    annual_plot(ax, diurnal, mode)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("MW")
    ax.set_title(f"2050 diurnal — {title_label} vs demand ({N_TURB} turbines, annual avg)")
    ax.set_xticks(range(0, 24, 2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(sub_dir / "diurnal_annual.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey=True, sharex=True)
    for m, ax in zip(range(1, 13), axes.flat):
        sub = bal[bal["month"] == m].groupby("hr")[["load_MW"] + cols].mean()
        annual_plot(ax, sub, mode, with_legend=(m == 1))
        ax.set_title(month_names[m - 1], fontsize=10)
        ax.set_xticks(range(0, 24, 6)); ax.grid(True, alpha=0.25)
    for ax in axes[-1]: ax.set_xlabel("Hour")
    for ax in axes[:, 0]:
        ax.set_ylabel("MW")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"2050 monthly diurnal — {title_label} vs demand ({N_TURB} turbines)", fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(sub_dir / "diurnal_monthly.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# --- Summary ------------------------------------------------------------------
load_TWh = bal["load_MW"].sum() / 1e6
wind_TWh = bal["wind_MW"].sum() / 1e6
solar_TWh = bal["solar_MW"].sum() / 1e6
renew_TWh = wind_TWh + solar_TWh
deficit_pos = bal["deficit_MW"].clip(lower=0).sum() / 1e6
surplus_neg = (-bal["deficit_MW"].clip(upper=0)).sum() / 1e6
hours_def = (bal["deficit_MW"] > 0).sum()
hours_sur = (bal["deficit_MW"] < 0).sum()

lines = [
    f"Scenario              : 2050 wind-max + solar (wind = {N_TURB} turbines)",
    f"Wind fleet            : {N_TURB} x SG 14-222 DD = {INSTALLED_MW:,.0f} MW AC",
    f"Solar farm            : {SOLAR_PANELS:,} panels (held constant)",
    "",
    f"Annual load           : {load_TWh:,.2f} TWh",
    f"Annual wind energy    : {wind_TWh:,.2f} TWh ({wind_TWh/load_TWh*100:.1f}% of load)",
    f"Annual solar energy   : {solar_TWh:,.2f} TWh ({solar_TWh/load_TWh*100:.1f}% of load)",
    f"Combined renewable    : {renew_TWh:,.2f} TWh ({renew_TWh/load_TWh*100:.1f}% of load)",
    "",
    f"Energy deficit (sum>0): {deficit_pos:,.2f} TWh",
    f"Energy surplus (sum<0): {surplus_neg:,.2f} TWh",
    f"Hours with deficit    : {hours_def:,} / 8760 ({hours_def/8760*100:.1f}%)",
    f"Hours with surplus    : {hours_sur:,} / 8760 ({hours_sur/8760*100:.1f}%)",
    f"Peak hourly deficit   : {bal['deficit_MW'].max():,.0f} MW",
    f"Peak hourly surplus   : {(-bal['deficit_MW']).max():,.0f} MW",
]
print("\n".join(lines))
with open(OUT / "summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nOutputs: {OUT}")
