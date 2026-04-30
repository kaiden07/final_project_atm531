"""
2050 wind + solar combined supply/demand balance.

Inputs:
  balance/2050/wind_only/hourly_balance_2050.csv  — datetime, load_MW, wind_MW, (old deficit)
  balance/solar_hourly_balance.csv                — 8761 lines; skip line 1; rows = solar MW (8760 h)

Solar alignment: positional — row 0 of solar (after skip) = Jan 1 00:00, matching row 0 of wind balance.

Outputs (balance/2050/wind_solar/):
  hourly_balance_2050.csv           — datetime, load_MW, wind_MW, solar_MW, deficit_MW
  monthly_hour_deficit_2050.csv     — month x hour average deficit (MW)
  monthly_hour_deficit_2050.png
  daily_deficit_2050.csv
  daily_deficit_2050.png
  summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).parent.parent
OUT = ROOT / "balance" / "2050" / "wind_solar"
OUT.mkdir(parents=True, exist_ok=True)

SOLAR_PANELS = 12_500_000

# --- Load inputs ---------------------------------------------------------------
wind_bal = pd.read_csv(
    ROOT / "balance" / "2050" / "wind_only" / "hourly_balance_2050.csv",
    parse_dates=["datetime"],
)
assert len(wind_bal) == 8760

solar = pd.read_csv(
    ROOT / "balance" / "solar_hourly_balance.csv",
    skiprows=1, header=None, names=["solar_MW"],
)
assert len(solar) == 8760, f"Expected 8760 solar rows, got {len(solar)}"

# Clip out two known spurious negative values in the source data (physical minimum = 0)
n_neg = (solar["solar_MW"] < 0).sum()
if n_neg:
    print(f"  Clipping {n_neg} negative solar value(s) to 0 (data artifact)")
solar["solar_MW"] = solar["solar_MW"].clip(lower=0)

# Positional alignment: both sorted Jan-1 00:00 → Dec-31 23:00
wind_bal["solar_MW"] = solar["solar_MW"].values
wind_bal["deficit_MW"] = wind_bal["load_MW"] - wind_bal["wind_MW"] - wind_bal["solar_MW"]
wind_bal[["load_MW", "wind_MW", "solar_MW", "deficit_MW"]] = (
    wind_bal[["load_MW", "wind_MW", "solar_MW", "deficit_MW"]].round(2)
)

# --- Hourly CSV ----------------------------------------------------------------
out_cols = ["datetime", "load_MW", "wind_MW", "solar_MW", "deficit_MW"]
wind_bal[out_cols].to_csv(OUT / "hourly_balance_2050.csv", index=False)

# --- Monthly x hour-of-day average deficit ------------------------------------
wind_bal["month"] = wind_bal["datetime"].dt.month
wind_bal["hr"] = wind_bal["datetime"].dt.hour

mh = (
    wind_bal.groupby(["month", "hr"])["deficit_MW"]
    .mean()
    .unstack("hr")
    .round(2)
)
mh.index.name = "month"
mh.columns.name = "hour"
mh.to_csv(OUT / "monthly_hour_deficit_2050.csv")

vmax = max(abs(mh.values.min()), abs(mh.values.max()))
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(mh.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(24))
ax.set_xticklabels(range(24))
ax.set_yticks(range(12))
ax.set_yticklabels(range(1, 13))
ax.set_xlabel("Hour of day (local)")
ax.set_ylabel("Month")
ax.set_title("2050 wind + solar deficit (MW) — monthly × diurnal mean")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Deficit MW (+ = shortfall, − = surplus)")
fig.tight_layout()
fig.savefig(OUT / "monthly_hour_deficit_2050.png", dpi=130)
plt.close(fig)

# --- Daily average deficit/surplus --------------------------------------------
daily = (
    wind_bal.assign(date=wind_bal["datetime"].dt.date)
    .groupby("date", as_index=False)["deficit_MW"]
    .mean()
)
daily["date"] = pd.to_datetime(daily["date"])
daily.to_csv(OUT / "daily_deficit_2050.csv", index=False)

colors = np.where(daily["deficit_MW"] >= 0, "#c0392b", "#2980b9")
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(daily["date"], daily["deficit_MW"], color=colors, width=1.0)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Date (2050)")
ax.set_ylabel("Daily mean deficit (MW)")
ax.set_title("2050 wind + solar — daily mean deficit (red = shortfall, blue = surplus)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "daily_deficit_2050.png", dpi=130)
plt.close(fig)

# --- Summary ------------------------------------------------------------------
load_TWh = wind_bal["load_MW"].sum() / 1e6
wind_TWh = wind_bal["wind_MW"].sum() / 1e6
solar_TWh = wind_bal["solar_MW"].sum() / 1e6
renew_TWh = wind_TWh + solar_TWh
deficit_pos = wind_bal["deficit_MW"].clip(lower=0).sum() / 1e6
surplus_neg = (-wind_bal["deficit_MW"].clip(upper=0)).sum() / 1e6
hours_def = (wind_bal["deficit_MW"] > 0).sum()
hours_sur = (wind_bal["deficit_MW"] < 0).sum()
peak_def = wind_bal["deficit_MW"].max()
peak_sur = (-wind_bal["deficit_MW"]).max()

lines = [
    f"Scenario              : 2050 wind + solar",
    f"Solar farm            : {SOLAR_PANELS:,} panels",
    f"",
    f"Annual load           : {load_TWh:,.2f} TWh",
    f"Annual wind energy    : {wind_TWh:,.2f} TWh ({wind_TWh/load_TWh*100:.1f}% of load)",
    f"Annual solar energy   : {solar_TWh:,.2f} TWh ({solar_TWh/load_TWh*100:.1f}% of load)",
    f"Combined renewable    : {renew_TWh:,.2f} TWh ({renew_TWh/load_TWh*100:.1f}% of load)",
    f"",
    f"Energy deficit (sum>0): {deficit_pos:,.2f} TWh",
    f"Energy surplus (sum<0): {surplus_neg:,.2f} TWh",
    f"Hours with deficit    : {hours_def:,} / 8760 ({hours_def/8760*100:.1f}%)",
    f"Hours with surplus    : {hours_sur:,} / 8760 ({hours_sur/8760*100:.1f}%)",
    f"Peak hourly deficit   : {peak_def:,.0f} MW",
    f"Peak hourly surplus   : {peak_sur:,.0f} MW",
]
print("\n".join(lines))
with open(OUT / "summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nOutputs written to: {OUT}")
