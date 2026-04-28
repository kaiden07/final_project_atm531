#!/usr/bin/env python3
"""
build_total_2050_K.py

Aggregate three Zone K (Long Island) load components for 2050 onto a single
hourly timeline:

  baseline_MW : non-EV baseline load
      /Profiles/baselines_df_no_ev_final/hourly/k_hourly_2020_2050.csv
      (full 2020-2050 hourly series, single Load column)

  ev_MW       : slow-adoption EV load
      /Profiles/baselines_df_ev_final/ev_baselines_hourly_profiles/
      slow_adoption/zones/K/hourly_ev_load_profiles_<month>.csv
      (one row per year, 24 hour-of-day columns; we use the 2050 row for
       each month and broadcast across every day of that month)

  hvac_MW     : WRF-derived HVAC load (UCP 1.3 km BAF, "nypa" column)
      /531_final_project/load_projections/2050/HVAC_only/
      compare_hvac_baf_ZoneK_2050.csv

Output:
  /531_final_project/load_projections/2050/total/total_load_2050_K.csv
      hour,baseline_MW,ev_MW,hvac_MW,total_MW

Also prints the 2050 annual energy demand in GWh.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd

ZONE = "K"
YEAR = 2050

BASELINE_CSV = Path(
    "/network/rit/lab/CUERG/kaiden/Profiles/"
    "baselines_df_no_ev_final/hourly/k_hourly_2020_2050.csv"
)
EV_DIR = Path(
    "/network/rit/lab/CUERG/kaiden/Profiles/"
    "baselines_df_ev_final/ev_baselines_hourly_profiles/"
    "normal_adoption/zones/K"
)
HVAC_CSV = Path(
    "/network/rit/lab/CUERG/kaiden/531_final_project/load_projections/2050/"
    "HVAC_only/compare_hvac_baf_ZoneK_2050.csv"
)
OUT_DIR = Path(
    "/network/rit/lab/CUERG/kaiden/531_final_project/load_projections/2050/total"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / f"total_load_{YEAR}_{ZONE}.csv"


def load_baseline() -> pd.Series:
    df = pd.read_csv(BASELINE_CSV)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[df["Datetime"].dt.year == YEAR].set_index("Datetime")
    s = pd.to_numeric(df["Load"], errors="coerce")
    s.name = "baseline_MW"
    return s


def load_hvac() -> pd.Series:
    df = pd.read_csv(HVAC_CSV)
    df["Time_Local"] = pd.to_datetime(df["Time_Local"])
    s = pd.to_numeric(df["ZoneK_nypa_MW"], errors="coerce")
    s.index = df["Time_Local"]
    # Fall-back DST transition produces a duplicate local hour; average the
    # two values so neither is dropped on reindex.
    s = s.groupby(level=0).mean()
    s.name = "hvac_MW"
    return s


def load_ev_for_year(year: int) -> pd.Series:
    """Build an 8760-row (or 8784 leap) EV series for the given year by
    repeating each month's 24-hour pattern across every day in that month."""
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    rows = []
    for mi, mname in enumerate(months, start=1):
        f = EV_DIR / f"hourly_ev_load_profiles_{mname}.csv"
        df = pd.read_csv(f)
        if "Year" not in df.columns:
            raise ValueError(f"{f}: missing Year column")
        row = df[df["Year"] == year]
        if row.empty:
            raise ValueError(f"{f}: no row for year {year}")
        diurnal = [float(row.iloc[0][f"Hour_{h}"]) for h in range(24)]
        ndays = calendar.monthrange(year, mi)[1]
        for d in range(1, ndays + 1):
            for h in range(24):
                rows.append((pd.Timestamp(year, mi, d, h), diurnal[h]))
    s = pd.Series(
        data=[v for _, v in rows],
        index=pd.DatetimeIndex([t for t, _ in rows]),
        name="ev_MW",
    )
    return s


def main() -> None:
    baseline = load_baseline()
    hvac = load_hvac()
    ev = load_ev_for_year(YEAR)

    # Canonical hourly timeline for 2050 (clock-time, no DST gaps).
    full_idx = pd.date_range(f"{YEAR}-01-01 00:00:00",
                             f"{YEAR}-12-31 23:00:00", freq="h")
    df = pd.DataFrame(index=full_idx)
    df.index.name = "hour"
    df["baseline_MW"] = baseline.reindex(full_idx)
    df["ev_MW"]       = ev.reindex(full_idx)
    df["hvac_MW"]     = hvac.reindex(full_idx)

    # Fill the few missing hours (DST gap + last 5 hours of Dec 31 in
    # both baseline and HVAC) with the previous day's same-hour value.
    miss_before = df[["baseline_MW", "hvac_MW"]].isna().sum().to_dict()
    for col in ("baseline_MW", "hvac_MW"):
        # previous day same hour, then forward fill any leftover
        df[col] = df[col].fillna(df[col].shift(24)).ffill().bfill()
    miss_after = df[["baseline_MW", "hvac_MW"]].isna().sum().to_dict()

    df["total_MW"] = df["baseline_MW"] + df["ev_MW"] + df["hvac_MW"]
    df = df.round(3)
    df.to_csv(OUT_CSV)

    annual_mwh = float(df["total_MW"].sum())  # 1-hour steps -> MW = MWh
    annual_gwh = annual_mwh / 1000.0

    print(f"Wrote {OUT_CSV}  (rows={len(df)})")
    print(f"Missing-before fill: {miss_before}")
    print(f"Missing-after  fill: {miss_after}")
    print()
    print(f"Annual energy demand, Zone K, {YEAR}:")
    print(f"  baseline = {df['baseline_MW'].sum()/1000:>10.1f} GWh")
    print(f"  EV       = {df['ev_MW'].sum()/1000:>10.1f} GWh")
    print(f"  HVAC     = {df['hvac_MW'].sum()/1000:>10.1f} GWh")
    print(f"  TOTAL    = {annual_gwh:>10.1f} GWh")


if __name__ == "__main__":
    main()
