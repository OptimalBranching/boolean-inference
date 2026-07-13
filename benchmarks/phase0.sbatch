#!/bin/bash
#SBATCH -J phase0
#SBATCH -p i64m512u
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --array=0-9%10
# One array task = ONE 22x22 (44-bit) instance: exhaustively sample its
# search-tree nodes, conquer each with kissat, correlate the candidate cutoff
# signals against realized difficulty. 22 is the largest size where EXHAUSTIVE
# conquer stays tractable (whole instance solves in seconds). 10 instances =
# tasks 0..9. Submit the sizing probe first with `sbatch --array=0 ...`, read
# results/f22_i0.json, then run the full array. Aggregate with
# phase0_aggregate.py.
set -euo pipefail

REPO="${PHASE0_REPO:?set PHASE0_REPO}"
KISSAT="${PHASE0_KISSAT:?set PHASE0_KISSAT}"
CUBE_TIMEOUT="${PHASE0_CUBE_TIMEOUT:-60}"      # per-cube kissat seconds (safety)
cd "$REPO"

SIZES=(22)
i=$SLURM_ARRAY_TASK_ID
size=${SIZES[0]}
index=$i

WORK="$HOME/jhspoolers/phase0-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK" results

echo "task=$i size=f$size index=$index cube_timeout=${CUBE_TIMEOUT}s host=$(hostname)"
python3 benchmarks/phase0_run.py \
  --size "$size" --index "$index" \
  --bin target/release/examples --data benchmarks/data \
  --work "$WORK" --out results \
  --kissat "$KISSAT" --cube-timeout "$CUBE_TIMEOUT"

rm -rf "$WORK"
