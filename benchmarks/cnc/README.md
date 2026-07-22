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
matching SAT/UNSAT metadata and hashes. CircuitSAT is the canonical semantic
source; the CNF is generated from that exact circuit, preserves its variables
as the leading DIMACS ids, and records a versioned `cnf_encoding`. Historical
`array-tseitin-v1` files remain reproducibility artifacts, not a second source
for new factoring instances.

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
  --cc-threshold 65536 -o artifacts/project/frontier.icnf \
  --trace artifacts/project/nodes.jsonl --trace-replay
```

This mode finishes the whole cubing traversal and never starts Kissat.
Inputs may be native CircuitSAT JSON, native extensional Boolean CSP, or
DIMACS. The `.csp` format retains each `<scope> : <allowed configurations>`
line as one relation tensor; it is intended for transfer tests where the
structure-aware cuber must see semantics that a flattened CNF does not expose.

`--branch-solver tail-greedy` starts from the full-row branches and rejects any
GreedyMerge whose measured reduction is worse than the weakest initial child.

Trace schema v2 records `rule_diagnostics` for every structure-aware branch:
the focus variable, region tensor/variable/boundary counts, joined and
probe-surviving row counts, closed-region status, branching vector, and gamma.
It declares `search_semantics: "sat-decision"`. Ordinary open-region rules are
configuration covers; closed regions may select one representative witness, so
the full frontier is satisfiability-preserving but is neither a model-space
cover nor a partition. The structure-blind control records
`rule_diagnostics: null` and does not run region probes.

`--trace-replay` is optional and requires a region trace. It evaluates binary
focus-variable branching and NaiveBranch against the identical residual state
used by the selected Greedy rule; it never changes the emitted frontier. It is
intentionally off in production because the counterfactual probes add work.
In this mode the cuber also checks every selected cover against the complete
probe-surviving table and records `cover_verified: true`; a failure aborts the
run rather than emitting evidence from an invalid rule.

For full-tree attribution, `--branch-solver naive` keeps the native region,
focus-variable selection, feasibility probes, and cutoff logic fixed while
replacing GreedyMerge with one full-assignment branch per surviving
configuration. `--selector structure-blind` is the separate binary control that
removes region construction entirely. Because these arms evolve different
residual states after the first split, compare complete response curves at
matched frontier scale rather than pairing later nodes.

Validate and aggregate a replay trace with:

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.trace_mechanism \
  artifacts/project/nodes.jsonl --require-replay --pretty
```

The summary reports local rule geometry and stage cost only. It must be joined
to per-cube conquer and residual records at the instance level before making a
runtime-mechanism claim. It also reports syntactic sibling disjointness: two
clauses that contradict on a shared assigned variable are certainly disjoint;
two compatible clauses are only *potentially* overlapping because their native
intersection may still be infeasible. This audits how partition-like a cover is
without upgrading the SAT-decision contract to a model partition.

For a frozen multi-instance cohort, verify that each instrumented frontier is
byte-identical to its registered cubing result and aggregate with instances as
the statistical units:

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.trace_mechanism_cohort \
  --trace-dir artifacts/traces \
  --frontier-dir artifacts/frontiers \
  --cubing-result-dir artifacts/cubing-results \
  --region-conquer-dir artifacts/region-conquer \
  --baseline-conquer-dir artifacts/baseline-conquer --pretty
```

The cohort report separates same-state rule evidence from end-to-end outcomes.
Its feature/outcome correlations are explicitly exploratory; a same-state
replay does not substitute for a full-tree counterfactual whose later residual
states necessarily diverge.

Once full-tree counterfactual frontiers have complete per-cube conquer records,
compare them with a common verifier and percentile definition:

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.full_tree_attribution \
  --arm greedy greedy.icnf greedy.jsonl \
  --arm naive naive.icnf naive.jsonl \
  --arm binary binary.icnf binary.jsonl \
  --reference greedy --pretty
```

The verifier requires a one-to-one `cube_index` match, checks decision-literal
counts, and rejects non-UNSAT uncensored outcomes before computing ratios.
Elapsed-time ratios are omitted by default; enable `--compare-elapsed` only when
every arm used the same hardware, solver binary, worker count, and load. Add
`--cubing-seconds NAME SECONDS` for every arm to compare measured
conquer-makespan plus cubing time end to end.

Audit what each exact cube leaves after propagation, without applying the
cuber's unexported domination or failed-literal choices:

```sh
cargo build --release --bin residual_audit
target/release/residual_audit INSTANCE.circuitsat.json frontier.icnf \
  -o native-audit.jsonl --arm greedy
target/release/residual_audit INSTANCE.cnf frontier.icnf \
  -o cnf-audit.jsonl --arm greedy
```

Each schema-v2 record contains the ordered cube literals, aggregate residual
features, and exact fixed-variable/value bit signatures in original-variable
order. On a DIMACS network, table GAC is ordinary clause BCP. Compare several
complete arms and join them to conquer records with:

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.residual_mediation \
  --arm greedy greedy.native.jsonl greedy.cnf.jsonl greedy-conquer.jsonl \
  --arm naive naive.native.jsonl naive.cnf.jsonl naive-conquer.jsonl \
  --arm binary binary.native.jsonl binary.cnf.jsonl binary-conquer.jsonl \
  --reference binary --pretty
```

The joiner requires a contiguous one-to-one cube-index match and verifies each
ordered-literal SHA-256 before reporting residual distributions, exact
native/CNF propagation agreement, within-arm exploratory associations, and the
hardest 5% conflict share. Cubes are nested observations, so these associations
are diagnostics rather than independent-sample significance tests.

Aggregate treatment/reference ratios with instances, rather than cubes, as the
units using `residual_mediation_cohort`. Repeat one seven-path tuple per frozen
instance:

```sh
PYTHONPATH=. python3 -m benchmarks.cnc.residual_mediation_cohort \
  --treatment region --reference march \
  --instance i0 region.native.jsonl region.cnf.jsonl region-conquer.jsonl \
                march.native.jsonl march.cnf.jsonl march-conquer.jsonl \
  --pretty
```

The cohort output reports geometric-mean ratios and feature/outcome Spearman
associations. Its `inverse_count_scaled_p95_ratio` is deliberately labeled a
granularity sensitivity: it multiplies the observed p95 ratio by the cube-count
ratio under a unit inverse-scaling assumption and is not a fitted or causal
adjustment.

When `trace_mechanism` is given `--conquer`, it also follows each cutoff path's
selected measure reductions and reports whether the weakest root child
concentrates frontier cubes or conflicts. This distinguishes a favorable
aggregate gamma from a finite-worker straggler bottleneck.

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
