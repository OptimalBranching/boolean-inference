# Benchmark scope

`benchmark-scope.yaml` freezes which problem structures belong in the study.
It intentionally does not freeze widths, instance counts, tuning data,
execution resources, metrics, or success criteria. Those choices belong to a
later evaluation protocol, after every required family is importable and pilot
runs identify non-trivial scales.

The primary benchmark has three parts:

1. matched factoring instances across six multiplier architectures;
2. public multiplier-equivalence miters from SAT Competition artifacts; and
3. divider, square, and square-root circuits from the EPFL combinational suite.

The same semiprime must be reused across all controlled multiplier
architectures. This isolates circuit topology from variation in the target
integer. General-Boolean CSP families remain conditional diagnostics and are
not pooled with the arithmetic results.

Multgen covers the simple/Booth and array/tree matrix; the Purdom-Sabry
generator supplies the recursive Karatsuba member. GenMul is retained as an
independent implementation cross-check rather than treating one generator as
ground truth.

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

The current local factoring reduction is the `array-ripple` member. The next
implementation stage is an importer/generator pipeline for the other required
architectures and public suites; it is not appropriate to choose benchmark
sizes before that pipeline exists.
