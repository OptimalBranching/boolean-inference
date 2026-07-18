# Auditable Cube-and-Conquer measurements

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
