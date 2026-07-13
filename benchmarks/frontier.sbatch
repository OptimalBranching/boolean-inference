#!/bin/bash
#SBATCH -J frontier18
#SBATCH -p i64m512u
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --array=0-9
# One array task = ONE 18x18 factoring instance: exhaustively sample its
# branching tree (gen_cubes, no cutoff), conquer every node with kissat, then
# run the oracle-frontier DP + rule-regret analysis (frontier_dp.py, hyp E3).
# Smoke-test first on the debug partition with `sbatch -p debug -t 00:25:00
# --array=0 benchmarks/frontier.sbatch`, then the full array on i64m512u.
# Results land in $REPO/results/frontier/; runscribe records detached (pull
# from the host with `runscribe pull`).
set -euo pipefail

REPO="${FRONTIER_REPO:-$HOME/boolean-inference}"
SIZE="${FRONTIER_SIZE:-18}"
DEPTH="${FRONTIER_DEPTH:-}"          # empty => exhaustive; N => --max-depth N
CUBE_TIMEOUT="${FRONTIER_CUBE_TIMEOUT:-60}"
export PATH="$HOME/.local/bin:$PATH"   # runscribe CLI
cd "$REPO"

i=$SLURM_ARRAY_TASK_ID
WORK="$HOME/jhspoolers/frontier-${SLURM_ARRAY_JOB_ID}-${i}"
OUT="results/frontier"
# Depth is part of every artifact name: a different depth is a different DFS
# node order, and a stale conquer cache would silently attach times to the
# wrong nodes.
TAG="f${SIZE}_i${i}_d${DEPTH:-full}"
mkdir -p "$WORK" "$OUT" logs

echo "task=$i size=${SIZE}x${SIZE} host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"

# Instance i = line i+1 of the jsonl (10 instances per size).
sed -n "$((i + 1))p" "benchmarks/data/factoring_${SIZE}x${SIZE}.jsonl" > "$WORK/inst.json"
[ -s "$WORK/inst.json" ] || { echo "no instance $i"; exit 1; }

# VE budget MUST be 0: keeps CNF numbering aligned with gen_cubes literals.
target/release/examples/export_dimacs "$WORK/inst.json" "$SIZE" 0 > "$WORK/inst.cnf"

# Subtree-complete sampling: exhaustive when DEPTH is empty, else a depth-D
# prefix with boundary-flagged leaves — both are valid frontier_dp input
# (--max-nodes would truncate mid-subtree and is not used here).
target/release/examples/gen_cubes "$WORK/inst.json" "$WORK/inst.cubes" \
  ${DEPTH:+--max-depth "$DEPTH"} 2> "$WORK/gen.log"
tail -1 "$WORK/gen.log"
SAMPLING_TIME=$(sed -En 's/.*sampling_time=([0-9.]+)s.*/\1/p' "$WORK/gen.log")

runscribe run --hyp E3 --tag "f${SIZE}-i${i}-d${DEPTH:-full}" -- \
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
