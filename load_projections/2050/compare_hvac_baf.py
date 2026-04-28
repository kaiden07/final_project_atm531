#!/usr/bin/env python3
"""
compare_hvac_baf.py

Reads WRF predictions directly from Azure Blob Storage (the same source used
by validation/energy/diag/NYPA/Open_blob.ipynb) and computes hourly HVAC_MW
per NYISO zone (I / J / K) for each of five BUILD_AREA_FRACTION sources:

  no_baf    — raw sum( CM_AC_URB3D × cell_area ) over urban LU cells.
               Effective BAF = 1; upper bound / overestimate.
  cuerg     — BAF from cuerg_data/.../wrfinput_d02   (BAF≈0 outside d03)
  swain     — BAF from mswain/.../wrfinput_d02       (canonical WUDAPT)
  forecast  — BAF from kaiden/.../wrfinput_d02       (current uWRF-4.2.2)
  nypa      — BAF from validation/.../NYPA/wrfndi_1km_UCP
               (exact blob d02 grid, covers all three zones)

BAF and LU_INDEX are pulled from local wrfinput files and nearest-neighbour
remapped onto whichever blob domain (d01 or d02) is requested. The blob d02
grid (141×162, DX=1333 m) covers all three zones; that is the default.

Outputs (per year, per zone):
  Zone{I,J,K}/compare_hvac_baf_{zone}_{YEAR}.csv  — Time_Local, Time_UTC,
       run_date, plus one column per valid BAF source (round to 2 decimals).
       Sources whose mean BAF over the zone's urban cells is below
       BAF_MIN_THRESHOLD (0.01) are dropped per zone.
  Zone{I,J,K}/compare_hvac_baf_{zone}_{YEAR}.png  — timeseries plot.

Usage:
    python compare_hvac_baf.py                            # default 2021-2025, d02
    python compare_hvac_baf.py --years 2030 2031          # any year
    python compare_hvac_baf.py --domain d01               # any blob domain
    python compare_hvac_baf.py --years 2024 --max-days 5  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import traceback
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import zarr
from azure.storage.blob import ContainerClient

# ── Blob configuration ─────────────────────────────────────────────────────
SAS_URL = (
    "https://agileloadforecasting.blob.core.windows.net/ualbany"
    "?sp=rl&st=2026-04-28T18:10:38Z&se=2027-01-01T03:25:38Z"
    "&spr=https&sv=2025-11-05&sr=c"
    "&sig=%2Ffdz2j445tKrJ184l%2BJaOdYFmi8%2BBs4ao8wQG18N0HI%3D"
)
BLOB_DATA_PATH = "final_uWRF_simulations/ssp45_cooler/2d_vars"
BLOB_NAME_FMT  = "zarr_2d_vars_{dom}_{year:04d}_{month:02d}.zarr"

# ── Paths & zone configuration ─────────────────────────────────────────────
SHP_DIR = Path(
    "/network/rit/lab/CUERG/NYISO_zones_shapefiles/"
    "all_shapefiles_zones/NYCA_Shapefile_Transfer"
)
ZONE_CFG = {
    "I": {"shp": SHP_DIR / "NYCA_I.shp"},
    "J": {"shp": SHP_DIR / "NYCA_J.shp"},
    "K": {"shp": SHP_DIR / "NYCA_K.shp"},
}

# Five BAF sources — single wrfinput per source (NN-remapped onto blob grid).
# We pick the source-domain whose grid covers the entire NYISO region;
# for cuerg/swain/forecast that is the d02 wrfinput, for nypa it is the 4 km
# wrfndi (the 1 km wrfndi only covers the NYC region).
BAF_SOURCES: dict[str, str | None] = {
    "no_baf":   None,
    "cuerg":    "/network/rit/lab/CUERG/cuerg_data/New_York_Forecast/wrf_outputs/wrfinput_d02",
    "swain":    "/network/rit/lab/CUERG/mswain/NYC_Forecast/uWRFV4.2/WRF_wudapt/run/wrfinput_d02",
    "forecast": "/network/rit/lab/CUERG/kaiden/wrf/4.2.2_Forecast_NYC/uWRF/run/wrfinput_d02",
    "nypa":     "/network/rit/lab/CUERG/kaiden/wrf/4.2.2_Forecast_NYC/validation/energy/diag/NYPA/wrfndi_1km_UCP",
}

# LU_INDEX source per blob domain. For d02 we use the NYPA 1 km wrfndi which
# is the *exact same grid* as the blob d02 (141×162, identical lat/lon),
# so no NN remap is needed. For d01 we fall back to the NYPA 4 km wrfndi.
LU_SOURCES: dict[str, str] = {
    "d01": "/network/rit/lab/CUERG/kaiden/wrf/4.2.2_Forecast_NYC/validation/energy/diag/NYPA/wrfndi_4km_UCP",
    "d02": "/network/rit/lab/CUERG/kaiden/wrf/4.2.2_Forecast_NYC/validation/energy/diag/NYPA/wrfndi_1km_UCP",
}

HVAC_ROOT = Path(
    "/network/rit/lab/CUERG/kaiden/wrf/4.2.2_Forecast_NYC/"
    "validation/energy/diag/NYPA/hist_hvac"
)


def _zone_output_dir(zone: str, output_dir: Path | None) -> Path:
    """Where CSV/PNG for `zone` should be written. Flat under `output_dir`
    when given, else legacy HVAC_ROOT/Zone{X}/ layout."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    d = HVAC_ROOT / f"Zone{zone}"
    d.mkdir(parents=True, exist_ok=True)
    return d

BAF_MIN_THRESHOLD = 0.01

LOCAL_TZ = ZoneInfo("America/New_York")
UTC      = ZoneInfo("UTC")

SOURCE_NAMES  = ["no_baf", "cuerg", "swain", "forecast", "nypa"]
SOURCE_COLORS = {"no_baf": "k", "cuerg": "tab:red",
                 "swain": "tab:blue", "forecast": "tab:green",
                 "nypa": "tab:orange"}
SOURCE_LABELS = {
    "no_baf":   "No BAF  (BAF=1, upper bound)",
    "cuerg":    "CUERG wrfinput  (BAF near-zero for I/K)",
    "swain":    "Swain WUDAPT  (canonical)",
    "forecast": "Kaiden forecast wrfinput",
    "nypa":     "NYPA UCP  (1.3 km, native blob d02 grid)",
}

BASE_FIELDS = ["Time_Local", "Time_UTC", "run_date"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("compare_baf")


# ── Zone masking (same logic as diag_hvac_spatial) ─────────────────────────

def _cell_corner_coords(lons, lats):
    def _pad(a):
        M, N = a.shape
        p = np.empty((M + 2, N + 2), dtype=a.dtype)
        p[1:-1, 1:-1] = a
        p[0,    1:-1] = 2 * a[0,  :] - a[1,  :]
        p[-1,   1:-1] = 2 * a[-1, :] - a[-2, :]
        p[1:-1, 0]    = 2 * a[:, 0]  - a[:, 1]
        p[1:-1, -1]   = 2 * a[:, -1] - a[:, -2]
        p[0,  0] = 2*p[0,  1] - p[0,  2]
        p[0, -1] = 2*p[0, -2] - p[0, -3]
        p[-1, 0] = 2*p[-1, 1] - p[-1, 2]
        p[-1,-1] = 2*p[-1,-2] - p[-1,-3]
        return p
    lp, la = _pad(lons), _pad(lats)
    cL = 0.25*(lp[:-1,:-1]+lp[1:,:-1]+lp[:-1,1:]+lp[1:,1:])
    cA = 0.25*(la[:-1,:-1]+la[1:,:-1]+la[:-1,1:]+la[1:,1:])
    return cL, cA


def build_zone_mask(shp_path, lons, lats):
    import shapely
    zone_poly = gpd.read_file(shp_path).geometry.union_all()
    cL, cA = _cell_corner_coords(lons, lats)
    M, N = lons.shape
    rings = np.empty((M*N, 5, 2))
    rings[:,0,0]=cL[:-1,:-1].ravel(); rings[:,0,1]=cA[:-1,:-1].ravel()
    rings[:,1,0]=cL[:-1, 1:].ravel(); rings[:,1,1]=cA[:-1, 1:].ravel()
    rings[:,2,0]=cL[1:, 1:].ravel();  rings[:,2,1]=cA[1:, 1:].ravel()
    rings[:,3,0]=cL[1:,:-1].ravel();  rings[:,3,1]=cA[1:,:-1].ravel()
    rings[:,4]=rings[:,0]
    return shapely.contains(zone_poly, shapely.polygons(rings)).reshape(M, N)


# ── NN remap from a wrfinput-like file onto an arbitrary target grid ───────

def _read_wrfinput_field(path: str, field: str):
    ds = xr.open_dataset(path)
    arr  = ds[field].isel(Time=0).values
    lons = ds["XLONG"].isel(Time=0).values
    lats = ds["XLAT"].isel(Time=0).values
    ds.close()
    return arr, lons, lats


def _nn_remap(src_arr: np.ndarray,
              src_lons: np.ndarray, src_lats: np.ndarray,
              dst_lons: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    if (src_arr.shape == dst_lons.shape
        and np.allclose(src_lons, dst_lons, atol=1e-4)
        and np.allclose(src_lats, dst_lats, atol=1e-4)):
        return src_arr
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([src_lons.ravel(), src_lats.ravel()]))
    _, idx = tree.query(
        np.column_stack([dst_lons.ravel(), dst_lats.ravel()]), k=1)
    return src_arr.ravel()[idx].reshape(dst_lons.shape)


# ── Blob helpers ───────────────────────────────────────────────────────────

_container_client: ContainerClient | None = None

def _cc() -> ContainerClient:
    global _container_client
    if _container_client is None:
        _container_client = ContainerClient.from_container_url(SAS_URL)
    return _container_client


def open_blob_month(year: int, month: int, dom: str) -> xr.Dataset | None:
    """Open one monthly zarr store from the blob, or None if absent."""
    name = BLOB_NAME_FMT.format(dom=dom, year=year, month=month)
    prefix = f"{BLOB_DATA_PATH}/{name}"
    try:
        store = zarr.ABSStore(client=_cc(), prefix=prefix)
        return xr.open_dataset(store, engine="zarr", consolidated=True)
    except Exception as e:
        log.warning("Cannot open blob %s: %s", prefix, e)
        return None


def _decode_times_utc(times_arr) -> list[datetime]:
    out = []
    for t in times_arr:
        s = t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
        out.append(datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")
                           .replace(tzinfo=UTC))
    return out


# ── Grid construction (single global grid per (domain) per process) ────────

_grid: dict[str, dict] = {}


def build_grid(ds: xr.Dataset, dom: str) -> dict:
    if dom in _grid:
        return _grid[dom]

    xlat = ds["XLAT"].values
    xlon = ds["XLONG"].values
    if xlat.ndim == 3:
        xlat = xlat[0]
        xlon = xlon[0]
    dx = float(ds.attrs.get("DX", ds.attrs.get("dx", 1333.0)))
    log.info("Building grid for %s shape=%s dx=%.0f m", dom, xlat.shape, dx)

    # LU_INDEX from canonical wrfinput, NN-remap if necessary.
    lu_path = LU_SOURCES.get(dom)
    if not lu_path:
        raise ValueError(f"No LU source configured for domain {dom!r}")
    lu_src, ll, la = _read_wrfinput_field(lu_path, "LU_INDEX")
    lu = _nn_remap(lu_src, ll, la, xlon, xlat).astype(int)
    urban = np.isin(lu, (31, 32, 33))
    log.info("LU source %s: %d urban cells on target grid", lu_path, int(urban.sum()))

    g = {
        "lons": xlon, "lats": xlat, "lu": lu, "urban": urban,
        "cell_area": dx * dx, "shape": xlat.shape,
        "zone_masks": {}, "baf": {}, "valid_sources": {},
    }

    for z, cfg in ZONE_CFG.items():
        m = build_zone_mask(cfg["shp"], xlon, xlat)
        g["zone_masks"][z] = m
        log.info("  Zone %s: %d polygon cells, %d urban-overlap", z,
                 int(m.sum()), int((m & urban).sum()))

    for src in SOURCE_NAMES:
        path = BAF_SOURCES[src]
        if path is None:
            g["baf"][src] = None
            continue
        baf_src, sl, sa = _read_wrfinput_field(path, "BUILD_AREA_FRACTION")
        g["baf"][src] = _nn_remap(baf_src, sl, sa, xlon, xlat)
        log.info("  BAF %s: src=%s, target-max=%.3f",
                 src, baf_src.shape, float(g["baf"][src].max()))

    for z in "IJK":
        m_urban = g["zone_masks"][z] & urban
        valid = []
        for src in SOURCE_NAMES:
            baf = g["baf"][src]
            if baf is None:
                valid.append(src)
            else:
                mean_baf = float(baf[m_urban].mean()) if m_urban.any() else 0.0
                if mean_baf >= BAF_MIN_THRESHOLD:
                    valid.append(src)
                else:
                    log.info("  Excluding source '%s' for Zone %s (BAF mean=%.4f)",
                             src, z, mean_baf)
        g["valid_sources"][z] = valid
        log.info("  Zone %s valid sources: %s", z, valid)

    _grid[dom] = g
    return g


# ── HVAC computation ────────────────────────────────────────────────────────

def hvac_mw_all_sources(cm_ac: np.ndarray, g: dict, zone: str) -> dict[str, float]:
    mask = g["zone_masks"].get(zone)
    valid = g["valid_sources"].get(zone, SOURCE_NAMES)
    if mask is None:
        return {s: float("nan") for s in SOURCE_NAMES}
    m_urban = mask & g["urban"]
    if not m_urban.any():
        return {s: float("nan") for s in SOURCE_NAMES}
    ca = g["cell_area"]
    out = {}
    for src in SOURCE_NAMES:
        if src not in valid:
            out[src] = float("nan")
            continue
        baf = g["baf"][src]
        if baf is None:
            w = float(cm_ac[m_urban].sum()) * ca
        else:
            w = float((cm_ac[m_urban] * baf[m_urban]).sum()) * ca
        out[src] = round(w / 1e6, 2)
    return out


# ── Year driver ─────────────────────────────────────────────────────────────

def process_year(year: int, dom: str, max_days: int | None,
                 zones: str = "IJK", output_dir: Path | None = None):
    """Iterate Jan(year) .. Jan(year+1) zarr stores; emit per-hour rows
    whose local-time year == ``year``."""
    months_to_load = [(year, m) for m in range(1, 13)] + [(year + 1, 1)]

    # Open one CSV writer per zone.
    zone_files: dict[str, tuple] = {}
    for zone in zones:
        p = _zone_output_dir(zone, output_dir) / f"compare_hvac_baf_Zone{zone}_{year}.csv"
        fh = open(p, "w", newline="")
        cols = BASE_FIELDS + [f"Zone{zone}_{s}_MW" for s in SOURCE_NAMES]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        fh.flush()
        zone_files[zone] = (fh, w, cols, p)

    grid: dict | None = None
    days_seen: set[date] = set()
    all_rows: list[dict] = []
    stop_early = False

    try:
        for (yy, mm) in months_to_load:
            if stop_early:
                break
            ds = open_blob_month(yy, mm, dom)
            if ds is None:
                continue

            t0 = datetime.now()
            try:
                if grid is None:
                    grid = build_grid(ds, dom)

                times_utc = _decode_times_utc(ds["Times"].values)
                cm_full = ds["CM_AC_URB3D"].values  # (Time, sn, we) for the month

                rows_this_month = 0
                for i, utc_dt in enumerate(times_utc):
                    local_dt = utc_dt.astimezone(LOCAL_TZ)
                    if local_dt.year != year:
                        continue
                    rd = local_dt.date()

                    if max_days and rd not in days_seen and len(days_seen) >= max_days:
                        log.info("max-days cap reached (%d days), stopping early.", max_days)
                        stop_early = True
                        break
                    days_seen.add(rd)

                    row = {
                        "Time_Local": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "Time_UTC":   utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "run_date":   rd.isoformat(),
                    }
                    cm_slice = cm_full[i]
                    for z in zones:
                        vals = hvac_mw_all_sources(cm_slice, grid, z)
                        for src, v in vals.items():
                            row[f"Zone{z}_{src}_MW"] = v
                    for zone, (fh, w, cols, _) in zone_files.items():
                        w.writerow({k: row.get(k, "") for k in cols})
                    all_rows.append(row)
                    rows_this_month += 1

                for fh, _, _, _ in zone_files.values():
                    fh.flush()
                dt = (datetime.now() - t0).total_seconds()
                log.info("  %d-%02d -> %d rows (%.1fs, days seen=%d)",
                         yy, mm, rows_this_month, dt, len(days_seen))
            except Exception as e:
                log.error("Month %d-%02d failed: %s\n%s",
                          yy, mm, e, traceback.format_exc())
            finally:
                ds.close()
    finally:
        _finish(year, all_rows, zone_files, zones, output_dir)


def _finish(year: int, all_rows: list[dict], zone_files: dict,
            zones: str, output_dir: Path | None):
    for zone, (fh, _, _, p) in zone_files.items():
        try:
            fh.close()
        except Exception:
            pass
        log.info("Year %d Zone %s: %s (rows=%d)", year, zone, p, len(all_rows))
    plot_year(year, all_rows, zones, output_dir)


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_year(year: int, rows: list[dict],
              zones: str = "IJK", output_dir: Path | None = None):
    if not rows:
        log.warning("Year %d: no data to plot.", year)
        return
    import pandas as pd
    df = pd.DataFrame.from_records(rows)
    df["Time_Local"] = pd.to_datetime(df["Time_Local"])
    df = df.sort_values("Time_Local").reset_index(drop=True)

    for zone in zones:
        valid_srcs = [
            s for s in SOURCE_NAMES
            if df.get(f"Zone{zone}_{s}_MW", pd.Series(dtype=float)).notna().any()
        ]
        if not valid_srcs:
            log.warning("Zone %s year %d: no valid sources to plot.", zone, year)
            continue

        fig, ax = plt.subplots(figsize=(18, 5))
        for src in valid_srcs:
            col = f"Zone{zone}_{src}_MW"
            s = pd.to_numeric(df[col], errors="coerce")
            ax.plot(df["Time_Local"], s,
                    color=SOURCE_COLORS[src], linewidth=0.7,
                    alpha=0.85, label=SOURCE_LABELS[src])
        ax.set_ylabel("HVAC Load (MW)", fontsize=11)
        ax.set_xlabel("Local Time (America/New_York)", fontsize=10)
        ax.set_title(
            f"Zone {zone}  —  {year}  |  BAF Source Comparison  (blob ssp45_cooler)\n"
            "HVAC_MW = Σ( CM_AC_URB3D × BAF × cell_area ) / 1e6  [urban LU 31/32/33]",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        png_path = _zone_output_dir(zone, output_dir) / f"compare_hvac_baf_Zone{zone}_{year}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        log.info("Saved %s", png_path)


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025],
                    help="Year(s) of local-time HVAC to emit. Default 2021-2025.")
    ap.add_argument("--domain", default="d02", choices=("d01", "d02"),
                    help="Blob domain to read (default d02; covers all NYISO zones I/J/K).")
    ap.add_argument("--max-days", type=int, default=None,
                    help="Cap days per year for smoke tests.")
    ap.add_argument("--zones", default="IJK",
                    help="Concatenation of zone letters to process (subset of IJK). "
                         "Default 'IJK'. Use 'K' for Long Island only.")
    ap.add_argument("--output-dir", default=None,
                    help="Override output directory. CSVs/PNGs are written flat "
                         "into this dir instead of HVAC_ROOT/Zone{X}/.")
    args = ap.parse_args()

    if args.domain not in LU_SOURCES:
        sys.exit(f"No LU_SOURCES entry for domain {args.domain!r}")

    zones = "".join(z for z in args.zones if z in "IJK")
    if not zones:
        sys.exit(f"--zones {args.zones!r} contains no valid zone letters (I/J/K).")

    output_dir = Path(args.output_dir) if args.output_dir else None

    for yr in args.years:
        log.info("=" * 60)
        log.info("Year %d  (domain=%s, zones=%s)", yr, args.domain, zones)
        log.info("=" * 60)
        process_year(yr, args.domain, args.max_days,
                     zones=zones, output_dir=output_dir)

    log.info("Done. Output in %s", output_dir if output_dir else HVAC_ROOT)


if __name__ == "__main__":
    main()
