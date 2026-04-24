import pandas as pd
import glob
import os

# ── Configuration ────────────────────────────────────────────────────────────
CSV_DIR = "."                        # Folder containing your CSV files (change if needed)
OUTPUT_FILE = "tmy_hourly_averages.csv"

# Column names from your data
TIME_COLS = ["Year", "Month", "Day", "Hour", "Minute"]
COL_10M = "wind speed at 10m (m/s)"
COL_60M = "wind speed at 60m (m/s)"
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath):
    """Load a wind data CSV, skipping the metadata header row."""
    # Row 0 has site metadata (SiteID, Timezone, etc.), so real headers are row 1
    df = pd.read_csv(filepath, header=1)
    df.columns = df.columns.str.strip()
    return df

def main():
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in '{CSV_DIR}'. Check your CSV_DIR setting.")
        return

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  {f}")

    # Load all files into a list of DataFrames
    dfs = [load_csv(f) for f in csv_files]

    # Stack all DataFrames on top of each other
    combined = pd.concat(dfs, ignore_index=True)

    # Group by Month/Day/Hour/Minute and average wind speeds across all years
    averaged = (
        combined
        .groupby(["Month", "Day", "Hour", "Minute"], as_index=False)[[COL_10M, COL_60M]]
        .mean()
        .round(4)
    )

    # Use the time columns from the first file as the row-order template
    template = dfs[0][TIME_COLS].reset_index(drop=True)
    output_df = pd.concat([template, averaged[[COL_10M, COL_60M]]], axis=1)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved hourly averages to '{OUTPUT_FILE}'")
    print(f"Output shape: {output_df.shape[0]} rows x {output_df.shape[1]} columns")

if __name__ == "__main__":
    main()
