"""
Wind-only deficit reduction sensitivity: minimum turbines to hit hour-coverage thresholds.

For each hour, the turbines required to zero the deficit is load_MW / (cf * 2.8 MW).
We sweep turbine counts 1–5000 and report coverage at key thresholds.

Outputs (balance/2050/wind_max/):
  summary.txt
  deficit_vs_turbines.png      — % deficit-free hours vs turbine count
  hourly_deficit_<N>.csv/.png  — hourly deficit at each threshold fleet size
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).parent.parent
OUT = ROOT / "balance" / "2050" / "wind_max"
OUT.mkdir(parents=True, exist_ok=True)

RATED_MW = 2.8
MAX_TURBINES = 5000
CURRENT_TURBINES = 1677  # 65% demand target fleet

bal = pd.read_csv(
    ROOT / "balance" / "2050" / "wind_only" / "hourly_balance_2050.csv",
    parse_dates=["datetime"],
)
assert len(bal) == 8760

# Recover normalized CF from current fleet
bal["cf"] = bal["wind_MW"] / (CURRENT_TURBINES * RATED_MW)

# Hours requiring N turbines to zero deficit = ceil(load / (cf * RATED_MW))
bal["turbines_needed"] = np.where(
    bal["cf"] > 0,
    np.ceil(bal["load_MW"] / (bal["cf"] * RATED_MW)),
    np.inf,
)

n_hours = len(bal)
zero_cf_hours = int((bal["cf"] == 0).sum())

# --- Coverage curve -----------------------------------------------------------
turbine_range = np.arange(0, MAX_TURBINES + 1, 10)
coverage_pct = np.array([(bal["turbines_needed"] <= t).sum() / n_hours * 100
                          for t in turbine_range])

# --- Threshold analysis -------------------------------------------------------
results = {}
sorted_needed = np.sort(bal["turbines_needed"].values)

for pct in [100, 90, 80, 70]:
    target_uncovered = int(np.ceil(n_hours * (1 - pct / 100)))
    idx = n_hours - target_uncovered - 1
    T = sorted_needed[idx]
    if np.isinf(T) or T > MAX_TURBINES:
        at_cap = (bal["turbines_needed"] <= MAX_TURBINES).sum() / n_hours * 100
        results[pct] = {"achievable": False, "turbines": None, "at_cap_pct": round(at_cap, 2)}
    else:
        T_int = int(T)
        actual_pct = (bal["turbines_needed"] <= T_int).sum() / n_hours * 100
        results[pct] = {"achievable": True, "turbines": T_int, "actual_pct": round(actual_pct, 2)}

# Coverage at exactly 5000
at_cap_pct = (bal["turbines_needed"] <= MAX_TURBINES).sum() / n_hours * 100

# --- Coverage curve plot ------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(turbine_range, coverage_pct, color="#2c7bb6", lw=1.8)
ax.axvline(CURRENT_TURBINES, color="gray", ls="--", lw=1, label=f"Current fleet ({CURRENT_TURBINES})")
for pct, res in results.items():
    if res["achievable"]:
        ax.axhline(pct, color="orange", ls=":", lw=0.8)
        ax.axvline(res["turbines"], color="orange", ls=":", lw=0.8)
        ax.scatter([res["turbines"]], [pct], color="orange", zorder=5, s=40,
                   label=f"{pct}% → {res['turbines']:,} turbines")
ax.axhline(at_cap_pct, color="#d7191c", ls="--", lw=1,
           label=f"Max cap ({MAX_TURBINES}) → {at_cap_pct:.1f}%")
ax.set_xlabel("Number of GE 2.8-127 turbines")
ax.set_ylabel("Hours with zero deficit (%)")
ax.set_title("Wind-only deficit-free hour coverage vs fleet size (2050)")
ax.set_xlim(0, MAX_TURBINES)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "deficit_vs_turbines.png", dpi=130)
plt.close(fig)

# --- Hourly deficit plots at key fleet sizes ----------------------------------
def hourly_deficit_plot(n_turb, label):
    gen = bal["cf"] * n_turb * RATED_MW
    deficit = (bal["load_MW"] - gen).round(2)
    pct_free = (deficit <= 0).mean() * 100

    out_csv = OUT / f"hourly_deficit_{n_turb}.csv"
    pd.DataFrame({"datetime": bal["datetime"], "deficit_MW": deficit}).to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    pos = deficit.clip(lower=0)
    neg = deficit.clip(upper=0)
    ax.fill_between(bal["datetime"], pos, color="#c0392b", alpha=0.75, label="Shortfall")
    ax.fill_between(bal["datetime"], neg, color="#2980b9", alpha=0.75, label="Surplus")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Date (2050)")
    ax.set_ylabel("Deficit (MW)")
    ax.set_title(f"2050 wind-only hourly deficit — {n_turb:,} turbines ({n_turb * RATED_MW:,.0f} MW) | {pct_free:.1f}% hours deficit-free  [{label}]")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / f"hourly_deficit_{n_turb}.png", dpi=130)
    plt.close(fig)
    return pct_free

# 70% threshold fleet
n_70 = results[70]["turbines"]
pct_70 = hourly_deficit_plot(n_70, "70% threshold")

# 5000 cap
pct_cap = hourly_deficit_plot(MAX_TURBINES, "5000-turbine cap")

# --- Summary ------------------------------------------------------------------
lines = [
    "Minimum turbines to zero deficit — threshold analysis",
    "======================================================",
    f"Turbine model      : GE 2.8-127, {RATED_MW} MW AC",
    f"Maximum cap        : {MAX_TURBINES} turbines = {MAX_TURBINES * RATED_MW:,.0f} MW",
    f"Current fleet      : {CURRENT_TURBINES} turbines = {CURRENT_TURBINES * RATED_MW:,.0f} MW (65% demand target)",
    f"Hours with cf = 0  : {zero_cf_hours}  (wind cannot cover load these hours regardless of fleet size)",
    "",
    "Threshold results:",
]
for pct, res in results.items():
    if res["achievable"]:
        lines.append(f"  {pct}% hours deficit-free : {res['turbines']:,} turbines "
                     f"= {res['turbines'] * RATED_MW:,.0f} MW  (actual {res['actual_pct']:.2f}%)")
    else:
        lines.append(f"  {pct}% hours deficit-free : NOT achievable within {MAX_TURBINES}-turbine cap "
                     f"(at cap: {res['at_cap_pct']:.2f}%)")
lines += [
    "",
    f"At maximum cap ({MAX_TURBINES} turbines, {MAX_TURBINES * RATED_MW:,.0f} MW):",
    f"  {at_cap_pct:.2f}% of hours are deficit-free",
    f"  {100 - at_cap_pct:.2f}% of hours still have a shortfall",
]

print("\n".join(lines))
with open(OUT / "summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nOutputs: {OUT}")
