# Auditable Cube-and-Conquer measurements

## Issue #51 hard-UNSAT factoring study

The frozen product-width study contract is
`contracts/hard-regime-v1.yaml`. Product width and factor-input width are
separate required fields: the declared pairs are 64/32, 72/36, and 80/40.
Targets are deterministic full-product-width primes inside the reachable
unsigned multiplier range and above the factor-input range, so each pinned
instance is UNSAT by construction.

Generate and independently verify the 39 declared targets before materializing
the array-ripple CircuitSAT/CNF pairs:

```bash
python3 -m benchmarks.cnc.hard_regime generate-targets \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  --out artifacts/cnc-hard-regime/targets.jsonl
python3 -m benchmarks.cnc.hard_regime verify-targets \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  artifacts/cnc-hard-regime/targets.jsonl
python3 -m benchmarks.cnc.hard_regime materialize \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  --out-dir artifacts/cnc-hard-regime/instances
```

Raw run artifacts remain outside Git. The contract, checksummed artifact
manifest, aggregate statistics, primary table, and report are the reviewable
payload.

Freeze a selector's three thresholds from exactly the three calibration
instances at one product width. This produces one shared threshold per band;
it never searches on a held-out instance:

```bash
python3 -m benchmarks.cnc.calibrate_hard_regime \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  artifacts/cnc-hard-regime/instances/manifest.jsonl \
  --product-width 64 --selector region \
  --cuber target/release/cnc_cuber \
  --out-dir artifacts/cnc-hard-regime/calibration/p64/region
```

After all six width/selector calibration locks exist, capture the exact tool
binaries and build the immutable 249-cell matrix (39 hardness rows plus 210
held-out CnC rows):

```bash
python3 -m benchmarks.cnc.hard_regime_matrix lock-toolchain \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  --cuber target/release/cnc_cuber \
  --kissat cnc-tools/bin/kissat-8af8e56f174b778aef3aa45af9f739b2a5f492c2 \
  --march-cu cnc-tools/bin/march_cu-705b60c6491ef2b61988b3ce6ac674be1b90571d \
  --repository-revision "$(git rev-parse HEAD)" \
  --out artifacts/cnc-hard-regime/toolchain.json
python3 -m benchmarks.cnc.hard_regime_matrix build-matrix \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  artifacts/cnc-hard-regime/instances/manifest.jsonl \
  --calibration-root artifacts/cnc-hard-regime/calibration \
  --toolchain artifacts/cnc-hard-regime/toolchain.json \
  --out artifacts/cnc-hard-regime/run-matrix.json
```

Every Slurm task executes exactly one immutable cell and writes
`cells/<cell-id>/terminal.json` last. Re-running a completed cell is idempotent;
timeouts and errors are terminal outcomes rather than missing rows:

```bash
python3 -m benchmarks.cnc.run_hard_regime_cell \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  artifacts/cnc-hard-regime/run-matrix.json \
  artifacts/cnc-hard-regime/toolchain.json \
  --cell-id CELL_ID \
  --instance-root artifacts/cnc-hard-regime/instances \
  --output-root artifacts/cnc-hard-regime/runs
```

Verify either the pilot gate (all 39 monolithic rows plus the 72 pilot cells)
or the complete 249-cell table. The verifier reconstructs raw scheduling and
rejects missing cells, width confusion, split leakage, per-test thresholds,
mixed CNFs, malformed branch assignments/refutation reasons, and censored cubes
omitted from work/span accounting. Harness-error terminals remain visible and
are explicitly excluded from any claim that per-cube work/span reconstructed:

```bash
python3 -m benchmarks.cnc.verify_hard_regime \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  artifacts/cnc-hard-regime/instances/manifest.jsonl \
  --calibration-root artifacts/cnc-hard-regime/calibration \
  --toolchain artifacts/cnc-hard-regime/toolchain.json \
  --matrix artifacts/cnc-hard-regime/run-matrix.json \
  --runs-root artifacts/cnc-hard-regime/runs --scope pilot
```

Generate raw paired rows, log-log frontier-budget interpolation on common
support, deterministic instance bootstrap intervals, the primary table, and a
short report only after verification succeeds:

```bash
python3 -m benchmarks.cnc.aggregate_hard_regime \
  benchmarks/cnc/contracts/hard-regime-v1.yaml \
  artifacts/cnc-hard-regime/run-matrix.json \
  --runs-root artifacts/cnc-hard-regime/runs --scope pilot \
  --out-dir artifacts/cnc-hard-regime/pilot-report
```

Pass `--aggregate artifacts/cnc-hard-regime/pilot-report/aggregate.json` to the
verifier for a final regeneration check that also rejects duplicate instance
IDs in any bootstrap group.

## HPC2 execution

The scripts in `hpc/` contain no hidden Slurm resource defaults. Supply every
partition, walltime, task/CPU count, memory limit, array range, working
directory, environment export, and stdout/stderr path explicitly to `sbatch`.
Use `BI51_SCRATCH_ROOT` on the HPC2 `jhspoolers` SSD for temporary per-cube CNF
files; persistent instances, raw results, and reports remain under
`BI51_ARTIFACT_ROOT` on HDD. Validate `smoke.sh` in the free `debug` partition
before materialization, calibration, or a run-cell array.

## Current CnC comparison

The primary comparison uses online stopping rules on both sides:

- Region cubing reads structured CircuitSAT and stops at classical CC
  difficulty `D^2(D+I)/N > threshold`. Calibrate the threshold by frontier
  count only with `calibrate_cc_difficulty.py`.
- `march_cu` reads a globally encoded CNF and uses its upstream default dynamic
  cutoff unchanged. Do not pass `-d`, `-n`, `-e`, or `-f` in the primary arm.
- Both frontiers are conquered by the same solver and resource policy. Report
  preprocessing, cubing, and conquer work/span separately and end to end.

Machine-specific Slurm wrappers are intentionally not versioned. Build the
frontiers with `cnc_cuber`, `calibrate_cc_difficulty.py`, and upstream
`march_cu`, then run both arms through `conquer_parallel.py`. Rejected
static/product cutoff workflows are not maintained.

Each measurement bundle is a directory containing a hash-linked `bundle.json`,
the input DIMACS file, a frontier JSONL, a monotonic event JSONL, per-cube raw
result JSONL, and a SAT witness when applicable. The verifier treats those raw
records as authoritative and independently reconstructs:

- complete, non-overlapping frontier coverage;
- every cube's solved, cancelled, timed-out, or never-started lifecycle;
- cubing wall/CPU time, conquer CPU work and scheduled makespan;
- orchestration and end-to-end wall time;
- maximum worker concurrency and the aggregate verdict;
- input, tool, and executable provenance plus SAT witness validity.

Run the positive fixture and the generated negative controls from the
repository root:

```bash
python3 benchmarks/cnc/verify_measurements.py \
  --bundle tests/fixtures/cnc/measurement-valid

python3 -m unittest tests/test_cnc_measurements.py
```

The exhaustive frontier representation is intended for small audit fixtures.
Large production runs should add a branching-tree certificate before using
this format beyond a tractable number of frontier variables.
