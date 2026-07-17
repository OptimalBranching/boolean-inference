# Benchmark data pipeline

This directory turns the scope in `benchmarks/scope/benchmark-scope.yaml` into
reproducible instances. Generated data and run manifests live outside Git;
generator revisions and artifact hashes should travel with the archived
dataset.

The project method consumes structure-preserving CircuitSAT JSON. A DIMACS
baseline is always encoded from that same CircuitSAT document, so a comparison
does not accidentally use two different circuits.

## 1. Generate mathematical targets

Generate public target records separately from the private factor oracle:

```bash
python3 -m benchmarks.pipeline.cli targets \
  --width 24 --width 32 --count 20 --seed-base 20260709 \
  --out benchmarks/data/targets.jsonl \
  --oracle-out benchmarks/data/private/factor-oracle.jsonl
```

This 24/32-bit command is a local smoke example only. Formal benchmark factor
widths are 64/96/128 bits and should be materialized through the bounded HPC2
workflow rather than on a workstation.

Do not pass or publish the oracle file as solver input. Every target in the
public JSONL is later reused across every selected multiplier architecture.

## 2. Produce raw structural multipliers

Array and Karatsuba have native generators:

```bash
python3 -m benchmarks.pipeline.cli multiplier \
  --bits 32 --architecture array-ripple \
  --out benchmarks/data/raw/array-ripple-32.json

python3 -m benchmarks.pipeline.cli multiplier \
  --bits 32 --architecture karatsuba --base-case 4 \
  --out benchmarks/data/raw/karatsuba-32.json
```

Use Multgen for Wallace/Dadda and Booth combinations, and GenMul as an
independent Verilog source. Their exact CLI flags are release-specific: pin a
Git revision, record `./multgen -help` or the GenMul choices, and generate one
Verilog module per architecture and width. Then normalize each module without
ABC optimization:

```bash
python3 -m benchmarks.pipeline.cli import-verilog generated.v \
  --top multiplier --source-id multgen --source-revision COMMIT \
  --architecture wallace-ripple \
  --out benchmarks/data/raw/wallace-ripple-32.json
```

The importer reads Verilog or SystemVerilog, runs
`proc; flatten; techmap; simplemap; opt_clean`, records the source hash, and
converts the resulting single-bit cells to CircuitSAT. It supports
AND/OR/XOR/NOT, their inverted variants, buffers, and muxes, and fails closed on
unknown cells.

## 3. Pin the same target across architectures

Raw files can use `{bits}` in their path template:

```bash
python3 -m benchmarks.pipeline.cli factor \
  --targets benchmarks/data/targets.jsonl \
  --netlist 'array-ripple=benchmarks/data/raw/array-ripple-{bits}.json' \
  --netlist 'wallace-ripple=benchmarks/data/raw/wallace-ripple-{bits}.json' \
  --netlist 'dadda-ripple=benchmarks/data/raw/dadda-ripple-{bits}.json' \
  --netlist 'booth-r4-wallace=benchmarks/data/raw/booth-r4-wallace-{bits}.json' \
  --netlist 'booth-r4-dadda=benchmarks/data/raw/booth-r4-dadda-{bits}.json' \
  --netlist 'karatsuba=benchmarks/data/raw/karatsuba-{bits}.json' \
  --default-product-port product --require-all \
  --out-dir benchmarks/data/factoring
```

If an upstream module names its product port differently, add
`--product-port ARCH=PORT`. Each output directory contains CircuitSAT, its
same-circuit Tseitin CNF, a metadata sidecar, and a sorted `manifest.jsonl`.
Run the project method directly on the structural artifact:

```bash
cargo run --release -- benchmarks/data/factoring/INSTANCE.circuitsat.json
```

The `.cnf` sibling is only for external SAT baselines; the
`boolean-inference` CLI intentionally rejects it.

## 4. Generate EPFL preimages

EPFL provides fixed Verilog circuits. Import the original `divisor`, `square`,
or `square-root` module, then sample deterministic inputs, simulate their
outputs, hide the inputs, and pin the outputs:

```bash
python3 -m benchmarks.pipeline.cli import-verilog divisor.v \
  --top divisor --out benchmarks/data/raw/epfl-divisor.json

python3 -m benchmarks.pipeline.cli preimages \
  benchmarks/data/raw/epfl-divisor.json \
  --family epfl-divisor --count 20 --seed 20260709 \
  --out-dir benchmarks/data/epfl/divisor
```

All-zero inputs, constant output vectors, and duplicate pinned outputs are
rejected. The sampled witness constructs the pinned instance but is not written
to the public metadata.

## 5. Generate equivalence miters

Build a global miter:

```bash
python3 -m benchmarks.pipeline.cli miter left.json right.json \
  --left-output product --right-output product \
  --out benchmarks/data/equivalence/left-vs-right.json
```

Add `--bit 17` for a bit-level miter. The mismatch reduction is a balanced
binary OR tree, keeping CircuitSAT gate arity and Tseitin clause width small.
If upstream modules use different input names, repeat
`--input-map LEFT_PORT=RIGHT_PORT` for every input pair.
Convert it with:

```bash
python3 -m benchmarks.pipeline.cli cnf \
  benchmarks/data/equivalence/left-vs-right.json \
  --out benchmarks/data/equivalence/left-vs-right.cnf
```

Equivalent circuits produce UNSAT instances.

## 6. Acquire public fixed suites

Public SAT Competition artifacts are downloaded rather than regenerated:

```bash
python3 -m benchmarks.pipeline.cli fetch URL \
  --sha256 EXPECTED_SHA256 --out benchmarks/data/public/archive.tar.zst
```

The first acquisition prints the digest; subsequent runs must provide it. The
script writes a provenance sidecar containing the URL and digest.

## 7. Validate and collect

```bash
python3 -m benchmarks.pipeline.cli validate \
  --circuitsat instance.json --cnf instance.cnf \
  --solver /path/to/kissat --expect sat

python3 -m benchmarks.pipeline.cli manifest \
  benchmarks/data/factoring/instances-24 \
  benchmarks/data/factoring/instances-32 \
  --base benchmarks/data/factoring \
  --out /path/to/dataset/manifest.jsonl
```

Validation checks CircuitSAT references, DIMACS counts and clause widths, and
optionally an independent SAT/UNSAT verdict. Manifest collection accepts one or
more dataset roots, re-hashes every referenced artifact, rebases paths relative
to the combined manifest, and rejects duplicate instance IDs.
