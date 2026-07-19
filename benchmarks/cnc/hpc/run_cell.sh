#!/bin/bash
set -euo pipefail

: "${BI51_REPO:?set BI51_REPO to the repository checkout}"
: "${BI51_ARTIFACT_ROOT:?set BI51_ARTIFACT_ROOT to the hard-regime artifact root}"
: "${BI51_CELL_LIST:?set BI51_CELL_LIST to an immutable cell-list text file}"
: "${BI51_SCRATCH_ROOT:?set BI51_SCRATCH_ROOT to an HPC2 SSD scratch directory}"
: "${SLURM_ARRAY_TASK_ID:?cell execution must run as a Slurm array}"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
CELL_ID=$(sed -n "${LINE_NUMBER}p" "$BI51_CELL_LIST")
if [[ -z "$CELL_ID" ]]; then
  echo "no cell at array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

cd "$BI51_REPO"
source .venv/bin/activate

python -m benchmarks.cnc.run_hard_regime_cell \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  "$BI51_ARTIFACT_ROOT/run-matrix.json" \
  "$BI51_ARTIFACT_ROOT/toolchain.json" \
  --cell-id "$CELL_ID" \
  --instance-root "$BI51_ARTIFACT_ROOT/instances" \
  --output-root "$BI51_ARTIFACT_ROOT/runs" \
  --temp-root "$BI51_SCRATCH_ROOT"
