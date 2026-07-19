#!/bin/bash
set -euo pipefail

: "${BI51_REPO:?set BI51_REPO to the repository checkout}"
: "${BI51_ARTIFACT_ROOT:?set BI51_ARTIFACT_ROOT to the hard-regime artifact root}"
: "${BI51_SCOPE:?set BI51_SCOPE to pilot or full}"

cd "$BI51_REPO"
source .venv/bin/activate

REPORT_ROOT="$BI51_ARTIFACT_ROOT/${BI51_SCOPE}-report"

python -m benchmarks.cnc.verify_hard_regime \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  "$BI51_ARTIFACT_ROOT/instances/manifest.jsonl" \
  --calibration-root "$BI51_ARTIFACT_ROOT/calibration" \
  --toolchain "$BI51_ARTIFACT_ROOT/toolchain.json" \
  --matrix "$BI51_ARTIFACT_ROOT/run-matrix.json" \
  --runs-root "$BI51_ARTIFACT_ROOT/runs" \
  --scope "$BI51_SCOPE"

python -m benchmarks.cnc.aggregate_hard_regime \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  "$BI51_ARTIFACT_ROOT/run-matrix.json" \
  --runs-root "$BI51_ARTIFACT_ROOT/runs" \
  --scope "$BI51_SCOPE" --out-dir "$REPORT_ROOT"

python -m benchmarks.cnc.verify_hard_regime \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  "$BI51_ARTIFACT_ROOT/instances/manifest.jsonl" \
  --calibration-root "$BI51_ARTIFACT_ROOT/calibration" \
  --toolchain "$BI51_ARTIFACT_ROOT/toolchain.json" \
  --matrix "$BI51_ARTIFACT_ROOT/run-matrix.json" \
  --runs-root "$BI51_ARTIFACT_ROOT/runs" \
  --scope "$BI51_SCOPE" --aggregate "$REPORT_ROOT/aggregate.json"
