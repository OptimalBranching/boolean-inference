#!/bin/bash
set -euo pipefail

: "${BI51_REPO:?set BI51_REPO to the repository checkout}"
: "${BI51_ARTIFACT_ROOT:?set BI51_ARTIFACT_ROOT to the hard-regime artifact root}"

cd "$BI51_REPO"
source .venv/bin/activate

python -m benchmarks.cnc.hard_regime materialize \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  --out-dir "$BI51_ARTIFACT_ROOT/instances"

python -m benchmarks.cnc.hard_regime_matrix lock-toolchain \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  --cuber target/release/cnc_cuber \
  --kissat cnc-tools/bin/kissat-8af8e56f174b778aef3aa45af9f739b2a5f492c2 \
  --march-cu cnc-tools/bin/march_cu-705b60c6491ef2b61988b3ce6ac674be1b90571d \
  --repository-revision "$(git rev-parse HEAD)" \
  --out "$BI51_ARTIFACT_ROOT/toolchain.json"
