#!/bin/bash
#SBATCH -J f28prod
#SBATCH -p i64m512u
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=64G
#SBATCH -t 06:00:00
# Production offline CnC on ONE 28x28 factoring instance, 64-core conquer:
#   1. baseline: single-core kissat on the full CNF (capped at BASELINE_CAP)
#   2. cubing: march-style REFUTATION-ADAPTIVE cutoff (--cutoff adaptive) —
#      threshold starts 0, is raised by the cuber's own refuted branches,
#      decays per emitted cube. No hand-set budget; --max-depth is only a
#      walltime guard (depth-capped boundary cubes keep the frontier complete
#      and the solve sound).
#   3. conquer: all cubes, 64 workers, stop at first SAT (instance is SAT)
set -euo pipefail

REPO="${PROD_REPO:-$HOME/boolean-inference}"
SIZE=28
INDEX="${PROD_INDEX:-0}"
BASELINE_CAP="${PROD_BASELINE_CAP:-7200}"
CUBE_TIMEOUT="${PROD_CUBE_TIMEOUT:-600}"
DEPTH_GUARD="${PROD_DEPTH_GUARD:-60}"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

WORK="$HOME/jhspoolers/f28prod-${SLURM_JOB_ID}"
OUT="results/f28_prod"
mkdir -p "$WORK" "$OUT" logs
KISSAT="$REPO/cnc-tools/bin/kissat"

sed -n "$((INDEX + 1))p" "benchmarks/data/factoring_${SIZE}x${SIZE}.jsonl" > "$WORK/inst.json"
target/release/examples/export_dimacs "$WORK/inst.json" "$SIZE" 0 > "$WORK/inst.cnf"
echo "instance: $(head -c 200 "$WORK/inst.json" | grep -o '"target":[0-9]*' || true)"

# --- 1. single-core baseline (skippable: run it as its own 1-core job so the
#        many-core conquer job stays small enough to backfill) ---
BWALL=skipped; BRES=skipped
if [ -z "${PROD_SKIP_BASELINE:-}" ]; then
  echo "=== baseline kissat (1 core, cap ${BASELINE_CAP}s) ==="
  B0=$(date +%s.%N)
  set +e
  timeout "$BASELINE_CAP" "$KISSAT" --relaxed "$WORK/inst.cnf" > "$WORK/base.out" 2>&1
  BSTAT=$?
  set -e
  B1=$(date +%s.%N)
  BWALL=$(echo "$B1 - $B0" | bc)
  BRES=$(grep -m1 '^s ' "$WORK/base.out" || echo "s CENSORED(cap)")
  echo "baseline: $BRES wall=${BWALL}s exit=$BSTAT"
fi

# --- 2. cubing: adaptive feedback cutoff (no hand-set budget) ---
echo "=== cubing (adaptive, depth guard $DEPTH_GUARD) ==="
target/release/examples/gen_cubes "$WORK/inst.json" "$WORK/frontier.icnf" \
  --cutoff adaptive --frontier-only --max-depth "$DEPTH_GUARD" \
  --max-seconds "${PROD_CUBING_CAP:-900}" 2> "$WORK/gen.log"
tail -1 "$WORK/gen.log"
N=$(grep -c '^a' "$WORK/frontier.icnf" || true)
echo "frontier cubes: $N"
TAG="f${SIZE}_i${INDEX}_adaptive"
cp "$WORK/frontier.icnf" "$OUT/${TAG}.icnf"
cp "$WORK/frontier.icnf.meta" "$OUT/${TAG}.icnf.meta"
cp "$WORK/gen.log" "$OUT/${TAG}.gen.log"

# --- 3. 64-way parallel conquer, stop at first SAT ---
echo "=== conquer (64 workers, stop-on-sat) ==="
runscribe run --hyp E3 --tag "f${SIZE}-prod-adaptive-64c" -- \
  python3 benchmarks/conquer_parallel.py "$WORK/inst.cnf" "$WORK/frontier.icnf" \
    --kissat "$KISSAT" --jobs "$SLURM_CPUS_PER_TASK" \
    --cube-timeout "$CUBE_TIMEOUT" --stop-on-sat \
    --out "$OUT/${TAG}.conquer.csv"

echo "baseline_wall=${BWALL}s baseline_res=$BRES ncubes=$N"
rm -rf "$WORK"
