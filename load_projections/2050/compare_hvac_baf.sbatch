#!/bin/bash
#SBATCH --job-name=hvac_2050_LI
#SBATCH --output=/network/rit/lab/CUERG/kaiden/531_final_project/load_projections/2050/HVAC_only/logs/hvac_2050_%j.log
#SBATCH --error=/network/rit/lab/CUERG/kaiden/531_final_project/load_projections/2050/HVAC_only/logs/hvac_2050_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# ─────────────────────────────────────────────────────────────────────────────
# Long Island Demand Project, Task 1a.
# Compute hourly HVAC load (CM_AC_URB3D) for Long Island (NYISO Zone K) in
# 2050, using the UCP 1.3 km dataset for BUILD_AREA_FRACTION (the "nypa"
# source = wrfndi_1km_UCP), reading WRF predictions directly from Azure
# Blob Storage (final_uWRF_simulations/ssp45_cooler/2d_vars, blob d02 grid).
#
# Output: HVAC_only/compare_hvac_baf_ZoneK_2050.csv (hourly).
#
# Usage:
#   sbatch compare_hvac_baf.sbatch
#
# Overrides via environment:
#   sbatch --export=ALL,BAF_YEARS=2050,BAF_ZONES=K compare_hvac_baf.sbatch
# ─────────────────────────────────────────────────────────────────────────────

BASE="/network/rit/lab/CUERG/kaiden/531_final_project/load_projections/2050"
OUTDIR="${BASE}/HVAC_only"
PYTHON="/network/rit/lab/CUERG/kaiden/miniconda3/envs/WRF_New/bin/python"
SCRIPT="${BASE}/compare_hvac_baf.py"

mkdir -p "${OUTDIR}/logs"

YEARS="${BAF_YEARS:-2050}"
DOMAIN="${BAF_DOMAIN:-d02}"
ZONES="${BAF_ZONES:-K}"

CMD="${PYTHON} ${SCRIPT} --years ${YEARS} --domain ${DOMAIN} --zones ${ZONES} --output-dir ${OUTDIR}"
[[ -n "${BAF_MAX_DAYS}" ]] && CMD="${CMD} --max-days ${BAF_MAX_DAYS}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Host: $(hostname)"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Running: ${CMD}"
echo
${CMD}
rc=$?
echo
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Done (exit ${rc})."
exit ${rc}
