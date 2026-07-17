# Benchmark scope

`benchmark-scope.yaml` records which problem structures belong in the study.
For controlled factoring it also freezes the validated 64/96/128-bit factor
width ladder. The 24/32-bit corpora are generation smoke tests and cannot enter
reported benchmark results. Instance counts, tuning data, execution resources,
metrics, and success criteria belong to a later evaluation protocol.

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

Audit the human-readable scope and its structural coverage:

```bash
python3 -m pip install -r benchmarks/scope/requirements.txt
python3 benchmarks/scope/audit.py benchmarks/scope/benchmark-scope.yaml
```

Expected output:

```text
PASS completeness: scope is complete and versioned
PASS multiplier-coverage: required structures and widths are explicit
PASS breadth: external miters and non-multiplier arithmetic are included
PASS boundaries: evaluation choices remain out of scope
```

Run the negative controls and lock tests with:

```bash
python3 -m unittest tests/test_benchmark_scope.py
```

The generation, conversion, pairing, miter, acquisition, and validation
commands are implemented and documented in
[`benchmarks/pipeline/README.md`](../pipeline/README.md). Exact instance counts
remain deliberately deferred; widths and structural populations are part of
the frozen scope.
