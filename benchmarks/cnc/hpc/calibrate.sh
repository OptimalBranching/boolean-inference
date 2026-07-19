#!/bin/bash
set -euo pipefail

: "${BI51_REPO:?set BI51_REPO to the repository checkout}"
: "${BI51_ARTIFACT_ROOT:?set BI51_ARTIFACT_ROOT to the hard-regime artifact root}"
: "${SLURM_ARRAY_TASK_ID:?calibration must run as a six-element Slurm array}"

case "$SLURM_ARRAY_TASK_ID" in
  0) PRODUCT_WIDTH=64; SELECTOR=region ;;
  1) PRODUCT_WIDTH=64; SELECTOR=structure-blind ;;
  2) PRODUCT_WIDTH=72; SELECTOR=region ;;
  3) PRODUCT_WIDTH=72; SELECTOR=structure-blind ;;
  4) PRODUCT_WIDTH=80; SELECTOR=region ;;
  5) PRODUCT_WIDTH=80; SELECTOR=structure-blind ;;
  *) echo "invalid calibration array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

cd "$BI51_REPO"
source .venv/bin/activate

python -m benchmarks.cnc.calibrate_hard_regime \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  "$BI51_ARTIFACT_ROOT/instances/manifest.jsonl" \
  --product-width "$PRODUCT_WIDTH" \
  --selector "$SELECTOR" \
  --cuber target/release/cnc_cuber \
  --out-dir "$BI51_ARTIFACT_ROOT/calibration/p${PRODUCT_WIDTH}/${SELECTOR}"
