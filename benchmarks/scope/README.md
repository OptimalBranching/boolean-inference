# Benchmark scope

`benchmark-scope.yaml` freezes which problem structures belong in the study.
It intentionally does not freeze widths, instance counts, tuning data,
execution resources, metrics, or success criteria. Those choices belong to a
later evaluation protocol, after every required family is importable and pilot
runs identify non-trivial scales.

The primary benchmark has three parts:

1. matched factoring instances across six multiplier architectures;
2. public multiplier-equivalence miters from SAT Competition artifacts; and
3. divisor, square, and square-root circuits from the EPFL combinational suite.

The same semiprime must be reused across all controlled multiplier
architectures. This isolates circuit topology from variation in the target
integer. General-Boolean CSP families remain conditional diagnostics and are
not pooled with the arithmetic results.

The local structural generator supplies Array and native Karatsuba circuits;
Multgen and GenMul cover the simple/Booth and tree matrix. The CNF-only
Purdom-Sabry generator is retained as an independent verdict cross-check rather
than being passed off as a structure-preserving input.

Audit the human-readable scope, its structural coverage, and its lock:

```bash
python3 -m pip install -r benchmarks/scope/requirements.txt
python3 benchmarks/scope/audit.py audit \
  benchmarks/scope/benchmark-scope.yaml \
  benchmarks/scope/benchmark-scope.lock
```

Expected output:

```text
PASS completeness: scope matches the schema and has no unresolved fields
PASS multiplier-coverage: required multiplier structures and matched targets are explicit
PASS breadth: external miters and non-multiplication arithmetic are included
PASS boundaries: conditional families and justified exclusions are explicit
PASS freeze: canonical digest matches the scope lock
```

Run the negative controls and lock tests with:

```bash
python3 -m unittest tests/test_benchmark_scope.py
```

The generation, conversion, pairing, miter, acquisition, and validation
commands are implemented and documented in
[`benchmarks/pipeline/README.md`](../pipeline/README.md). Exact benchmark sizes
remain deliberately deferred until every external architecture is imported and
pilot difficulty is measured.
