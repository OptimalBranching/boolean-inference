#!/bin/bash
#SBATCH -J calib-tseitin
#SBATCH -p i64m512u
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 02:00:00
# B-calibration on Tseitin (3-regular expander, UNSAT): one array task = one
# (v, seed). Generate the instance, sample a subtree-complete depth-D prefix
# with narrow region rules (max_rows MR), conquer every node with kissat, and
# run the oracle-frontier DP + B-regret sweep (hyp E3). Array task i maps to
# v=V_LIST[i / #seeds], seed=1 + i % #seeds.
set -euo pipefail

REPO="${CALIB_REPO:-$HOME/boolean-inference}"
V_LIST=(${CALIB_V:-80 90})
NSEEDS="${CALIB_SEEDS:-3}"
DEPTH="${CALIB_DEPTH:-6}"
MR="${CALIB_MAXROWS:-4}"
CUBE_TIMEOUT="${CALIB_CUBE_TIMEOUT:-300}"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

i=$SLURM_ARRAY_TASK_ID
v=${V_LIST[$((i / NSEEDS))]}
seed=$((1 + i % NSEEDS))
WORK="$HOME/jhspoolers/calib-${SLURM_ARRAY_JOB_ID}-${i}"
OUT="results/calib_tseitin"
TAG="t${v}_s${seed}_d${DEPTH}_mr${MR}"
mkdir -p "$WORK" "$OUT" logs

echo "task=$i v=$v seed=$seed depth=$DEPTH max_rows=$MR host=$(hostname)"

python3 benchmarks/csp/gen_tseitin.py --family regular --v "$v" --d 3 \
  --seed "$seed" --csp "$WORK/inst.csp" --cnf "$WORK/inst.cnf"

# Narrow rules (max_rows MR) keep the tree deep rather than wide; the .csp and
# .cnf share variable numbering by construction.
target/release/examples/gen_cubes "$WORK/inst.csp" "$WORK/inst.cubes" \
  most "$MR" --max-depth "$DEPTH" 2> "$WORK/gen.log"
tail -1 "$WORK/gen.log"
SAMPLING_TIME=$(sed -En 's/.*sampling_time=([0-9.]+)s.*/\1/p' "$WORK/gen.log")

runscribe run --hyp E3 --tag "$TAG" -- \
  python3 benchmarks/frontier_dp.py "$WORK/inst.cnf" "$WORK/inst.cubes" \
    --sampling-time "$SAMPLING_TIME" \
    --kissat cnc-tools/bin/kissat \
    --jobs "$SLURM_CPUS_PER_TASK" \
    --cube-timeout "$CUBE_TIMEOUT" \
    --cache "$OUT/${TAG}.conquer.csv" \
    --out-prefix "$OUT/${TAG}"

cp "$WORK/inst.cubes.meta" "$OUT/${TAG}.cubes.meta"
rm -rf "$WORK"
echo "task=$i done"
