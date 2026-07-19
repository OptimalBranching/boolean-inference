#!/bin/bash
set -euo pipefail

: "${BI51_REPO:?set BI51_REPO to the repository checkout}"
: "${BI51_ARTIFACT_ROOT:?set BI51_ARTIFACT_ROOT to the hard-regime artifact root}"

cd "$BI51_REPO"
source .venv/bin/activate

python -m benchmarks.cnc.hard_regime_matrix build-matrix \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  "$BI51_ARTIFACT_ROOT/instances/manifest.jsonl" \
  --calibration-root "$BI51_ARTIFACT_ROOT/calibration" \
  --toolchain "$BI51_ARTIFACT_ROOT/toolchain.json" \
  --out "$BI51_ARTIFACT_ROOT/run-matrix.json"

python -m benchmarks.cnc.hard_regime_matrix list-cells \
  "$BI51_ARTIFACT_ROOT/run-matrix.json" --set monolithic \
  --out "$BI51_ARTIFACT_ROOT/monolithic-cells.txt"
python -m benchmarks.cnc.hard_regime_matrix list-cells \
  "$BI51_ARTIFACT_ROOT/run-matrix.json" --set pilot-cnc \
  --out "$BI51_ARTIFACT_ROOT/pilot-cnc-cells.txt"
python -m benchmarks.cnc.hard_regime_matrix list-cells \
  "$BI51_ARTIFACT_ROOT/run-matrix.json" --set remaining-cnc \
  --out "$BI51_ARTIFACT_ROOT/remaining-cnc-cells.txt"
