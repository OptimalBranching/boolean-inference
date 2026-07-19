#!/bin/bash
set -euo pipefail

: "${BI51_REPO:?set BI51_REPO to the repository checkout}"
: "${BI51_ARTIFACT_ROOT:?set BI51_ARTIFACT_ROOT to the hard-regime artifact root}"

cd "$BI51_REPO"
source .venv/bin/activate

python -m benchmarks.cnc.hard_regime validate-contract \
  benchmarks/cnc/contracts/hard-regime-v1.yaml
python -m pytest tests/test_hard_regime.py tests/test_calibrate_hard_regime.py \
  tests/test_hard_regime_matrix.py tests/test_run_hard_regime_cell.py \
  tests/test_verify_hard_regime.py tests/test_aggregate_hard_regime.py -q
cargo test --test cnc_cuber_trace
cargo build --release --bin cnc_cuber
python -m benchmarks.cnc.hard_regime generate-targets \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  --out "$BI51_ARTIFACT_ROOT/smoke-targets.jsonl"
