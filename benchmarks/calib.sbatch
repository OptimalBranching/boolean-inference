#!/bin/bash
#SBATCH -J p0calib
#SBATCH -p i64m512u
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH --array=0-8%9
# Phase 0 calibration: direct kissat solve of the WHOLE instance at sizes
# {24,26,28} x instances {0,1,2} = 9 tasks, to map the hardness ramp (wall time)
# and set the meaningful measurement size + per-cube timeout. Reuses the existing
# gate_b_run.py kissat arm (export_dimacs VE0 -> kissat -q, with a per-instance
# timeout; a timeout is a censored data point, not a failure).
set -euo pipefail

REPO="${PHASE0_REPO:?set PHASE0_REPO}"
TIMEOUT="${PHASE0_TIMEOUT:-1200}"      # per-instance kissat seconds
cd "$REPO"
export PATH="$HOME/kissat/build:$PATH"  # gate_b_run.py's kissat arm calls bare `kissat`

SIZES=(24 26 28)
i=$SLURM_ARRAY_TASK_ID
size=${SIZES[$(( i / 3 ))]}
inst=$(( i % 3 ))

WORK="$HOME/jhspoolers/p0calib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK" logs

echo "task=$i size=f$size inst=$inst timeout=${TIMEOUT}s host=$(hostname)"
python3 benchmarks/gate_b_run.py --size "$size" --arm kissat --timeout "$TIMEOUT" \
  --instances "$inst" --bin target/release/examples --data benchmarks/data --work "$WORK"

rm -rf "$WORK"
