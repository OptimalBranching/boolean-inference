# Post-Prefix-Sharing Profile — Where the Time Went Next

**Date:** 2026-07-02
**Build:** prefix-sharing lookahead (commit 0e34edd), `--profile profiling`,
samply, `factoring_22x22` VE10 (816 vars / 332 tensors, 19761/45322 nodes,
~7.5s). 7506 weighted samples.

## Context

Prefix-sharing (region-feasibility trie DFS) cut VE10 wall-clock ~19% (9.18s →
7.45s), node counts identical. This profile finds the NEXT lever.

## Self-time (top)

| % self | function |
|---|---|
| **34.7%** | `propagate::propagate_core_rescan` ← the OLD linear propagator |
| 26.2% | `ct::ct_propagate` (fast; used by feasible_configs + difflookahead probe) |
| 10.8% | `propagate::probe` |
| 6.9% | `ct::RSparseBitSet::intersect_index` |
| 4.7% | `ct::RSparseBitSet::intersect_with_mask` |
| 2.25% | `trail::restore_to` |
| 1.9% | `measure::measure_core` |
| 1.76% | `_platform_memset` (allocation) |
| 1.1% | `ct::enqueue_var_change` |
| 11.24% (incl) | `Vec::from_iter` (allocation) |

## Inclusive (callers)

| % incl | function |
|---|---|
| 95.3% | `selector::findbest` |
| 61.6% | `table::compute_branching_result` |
| **40.7%** | `GreedyMerge` (the branching-rule measurement / set-cover) |
| 39.0% | `ct_propagate` |
| **37.1%** | `RuleProblem::…Branch…` = `apply_branch` (measurement path) |
| 34.7% | `propagate_core_rescan` |
| 29.4% | `probe` (the **difflookahead** candidate probes — still per-candidate) |
| 20.5% | `feasible_configs` |
| 19.8% | `descend` |

## The two levers (both are "propagate per candidate from base")

### Lever 1 — the branching-rule MEASUREMENT path (~40% inclusive, 34.7% self)

`GreedyMerge` computes each candidate row's `size_reduction = measure(before) −
measure(after)` via `RuleProblem::apply_branch`. Today `apply_branch` (adapter.rs):

1. `self.doms.clone()` — per candidate,
2. `SolverBuffer::new(&self.cn)` — **allocates a fresh buffer per candidate**
   (queue + `in_queue[332]` + `mask_scratch` + `dirty[332]`) → the memset /
   Vec::from_iter allocation cost,
3. `propagate_core_rescan` — the **slow linear rescan** (34.7% self).

The entire rest of the solver runs fast CT; this path alone is on the old
propagator. It was left on rescan historically because cloning CT tables per
apply_branch caused an allocation storm — but that is avoidable: apply the branch
on the **live** `(doms, tables)` via the trail (`mark → apply → ct_propagate →
snapshot doms → restore`), cloning only the resulting `doms` (already cloned
today), never the tables. Behavior-preserving: CT and rescan reach the same GAC
fixpoint → identical measures → identical rule → identical nodes (golden test
guards). **Feasibility precondition being verified:** ob-core must call
`apply_branch` only single-level from the root (never on a returned sub-problem)
so the shared-live-tables + restore discipline is sound.

Expected: replaces the 34.7% slow rescan with ~2–3× faster CT and removes the
per-candidate buffer allocation. Plausibly the largest remaining single win.

### Lever 2 — difflookahead candidate probing (~29% inclusive)

`selector::findbest`'s DiffLookahead probes both polarities of ~16 candidate vars,
each via `probe` (CT from base). This is the difflookahead prefix-sharing the
prefix-sharing spec explicitly deferred. Sharing is weaker here (independent
single-var fixes), but the IS_FIXED short-circuit (skip a candidate already fixed
by an earlier probe) is cheap and the pattern is the same. Its own spec.

## Recommended order

1. **Measurement-path CT conversion** (Lever 1) — biggest, isolated to
   `adapter.rs` + a shared CT scratch; guarded by the golden 19761/45322 test.
2. **difflookahead prefix-sharing** (Lever 2) — separate spec after Lever 1.

Both must keep node counts bit-identical (behavior-preserving); measure each with
runscribe A/B on `factoring_22x22` VE10.

## Reproduce

```
cargo build --profile profiling --example solve_circuit
samply record --unstable-presymbolicate --save-only -o psl_prof.json -- \
  ./target/profiling/examples/solve_circuit tests/fixtures/factoring_22x22.circuitsat.json 22 difflook 10
python3 parse_prof2.py psl_prof.json psl_prof.syms.json
```
