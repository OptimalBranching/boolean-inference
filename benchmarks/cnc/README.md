# Factoring Cube-and-Conquer

This directory contains only the operations needed for factoring runs:

1. generate paired SAT/UNSAT `n×n` factoring instances in CircuitSAT and CNF;
2. solve a CNF directly with Kissat;
3. cube a CNF with `march_cu`, then solve the cubes in parallel with Kissat;
4. use the Rust solver either to export a complete frontier or to stream cubes
   directly into a bounded parallel Kissat pool;
5. solve a frozen frontier in Python for reproducible paper measurements.

## Generate instances

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.factoring \
  --width 26 --width 28 --count 10 \
  --out-dir artifacts/factoring
```

Each manifest row points to `instance.circuitsat.json` and `instance.cnf` with
matching SAT/UNSAT metadata and hashes.

## Direct Kissat

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.solve INSTANCE.cnf \
  --kissat cnc-tools/bin/kissat \
  --timeout-s 600 --out-dir artifacts/direct
```

## march_cu then parallel Kissat

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.cubing march INSTANCE.cnf \
  --march-cu cnc-tools/bin/march_cu \
  --kissat cnc-tools/bin/kissat --workers 32 \
  --out-dir artifacts/march
```

Pass `--remaining-vars N` to override `march_cu`'s dynamic cutoff.

## Rust solver: export all cubes

```sh
cargo build --release --bin cnc_cuber
target/release/cnc_cuber INSTANCE.circuitsat.json \
  --cc-threshold 65536 -o artifacts/project/frontier.icnf
```

This mode finishes the whole cubing traversal and never starts Kissat.

## Rust solver: streaming Cube-and-Conquer

```sh
target/release/cnc_cuber INSTANCE.circuitsat.json \
  --cc-threshold 65536 \
  --solve-cnf INSTANCE.cnf \
  --kissat cnc-tools/bin/kissat --workers 32
```

The Rust cuber submits each open leaf immediately to a bounded worker pool.
Exit codes follow SAT conventions: `10` for SAT and `20` for UNSAT.

## Analyze a frozen frontier

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.cubing frontier \
  INSTANCE.cnf artifacts/project/frontier.icnf \
  --kissat cnc-tools/bin/kissat --workers 32 \
  --out-dir artifacts/project-analysis
```

The Python path never invokes the project cuber. It consumes a complete,
frozen frontier and records per-cube timing, decisions, conflicts, and aggregate
statistics. March and project frontiers can therefore use the same analysis
backend.
