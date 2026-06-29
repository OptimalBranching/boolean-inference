# Bounded-Width VE Canonicalizer (Rust port) — Design

**Status:** approved (brainstorming), ready for implementation plan
**Date:** 2026-06-29
**Branch:** `framework-apply-branch` (Rust crate `boolean-inference`)

## Goal

Port the Julia `bounded_ve_canonicalize` (static, width-aware variable-elimination preprocessing) to the Rust solver. It reshapes the raw gate-level `ConstraintNetwork` into a coarser one by bucket-eliminating variables whose elimination width stays within a budget, in weighted-min-fill order. This (a) shrinks the branch set several-fold (the factoring speedup) and (b) produces larger tensors — the regime where the search/propagation cost actually lives.

## Background & motivation

The Rust solver currently runs the **raw** network (`network_from_* → from_network → bbsat`, no preprocessing). For factoring this is all arity-3 gates (≤4 support rows). The Julia VE canonicalizer (`src/preprocessing/canonicalize.jl`, commit eb47a15) cuts the treewidth proxy 50-70% for a several-fold speedup. Measured in Julia on `Factoring(10,10,1040399)`: raw 620 vars/620 tensors/max-arity 3 → at `budget_B`=8/12/16/20 the branch set shrinks to 180/126/98/76 vars with max arity 8/12/15/19. This port brings that to Rust.

## Key decisions (locked in brainstorming)

1. **`sc = out.len() + 1`, not OMEinsum.** Julia estimates the elimination space-complexity via OMEinsum's contraction optimizer. Porting that is impractical. Our **relational** bucket contraction joins all of `v`'s incident tensors into one relation over `neighbors(v) ∪ {v}` and projects `v` out, so the largest intermediate is exactly `out.len()+1` variables — `sc = out.len()+1` is *exact for our engine* and is the quantity bounding the produced tensor's dense size. Accepted consequence: the eliminated-variable set may differ slightly from Julia's at a given `budget_B` (NOT bit-exact parity); validated by measuring branch-set/arity reduction against the Julia ballpark and by correct factoring.
2. **Reuse `setup_problem` for final compression + compose `orig_to_new`.** The VE loop produces surviving `(var_axes, dense)` tensors in cn's compressed-var space; `setup_problem` then dedups, drops eliminated vars (now in no tensor), rebuilds `v2t`, and yields a cn→new `orig_to_new`. Compose with the input's `orig_to_new` to index original var ids.
3. **Lazy `BinaryHeap`** min-ordered by `(fill, sc)` replaces Julia's keyed `PriorityQueue`; stale entries are re-scored and skipped on pop.
4. **Scope:** core canonicalizer + `solve_circuit` example wiring (factor bits protected, end-to-end factoring validation). `solve_dimacs` default path unchanged.
5. **Correctness gate:** solution-preservation *projected onto protected variables* (brute force on small instances) + correct factoring end-to-end.

## Non-goals

- OMEinsum / contraction-order optimizer port (decision 1).
- Baking VE into the default `solve_dimacs` path or a public `solve_factoring` API (deferred; example proves it).
- Trail / Compact-Table / incremental propagation (a *separate*, later step — now justified because VE produces large tensors, but out of scope here).
- Back-substitution to recover eliminated (non-protected) variables (Julia doesn't either; only protected vars are read).

## Architecture

A new self-contained module `src/canonicalize.rs` holding the VE loop, reusing `contract.rs` (relational join) for bucket contraction and `network.rs::setup_problem` for the final build/compression. The `solve_circuit` example gains an optional canonicalization step before search.

## Components

### `bounded_ve_canonicalize` (`src/canonicalize.rs`)

```rust
/// Statically reshape `cn` by bucket-eliminating variables whose elimination width
/// (`out.len()+1`) is `<= budget_b`, in weighted-min-fill order. `protected` lists
/// variable ids in `cn`'s OWN compressed space that must never be eliminated (read-out
/// vars, e.g. factor bits); they survive into the result and are read via the result's
/// `orig_to_new`. Produced tensors are capped at 32 variables (TensorData limit).
/// Returns a new compressed `ConstraintNetwork` whose `orig_to_new` indexes the same
/// original var ids as `cn`.
pub fn bounded_ve_canonicalize(
    cn: &ConstraintNetwork,
    budget_b: usize,
    protected: &[usize],
) -> ConstraintNetwork;
```

Internal working state (cn compressed-var space, `nv = cn.vars.len()`):
- `live: Vec<LiveTensor>` where `LiveTensor { var_axes: Vec<usize>, dense: Vec<bool> }`, seeded from `cn.tensors` (each `var_axes = t.var_axes.clone()`, `dense = cn.data(t).dense.clone()`).
- `active: Vec<bool>` (one per `live` slot).
- `v2t: Vec<Vec<usize>>` incidence (`v2t[v]` = slots whose `var_axes` contain `v`), seeded from `cn.v2t`.
- `protected_set: ` a `Vec<bool>` of length `nv`.

Helpers:
- `active_incident(v) -> Vec<usize>`: `v2t[v]` filtered by `active`.
- `out_vars(tids, v) -> Vec<usize>`: sorted-unique union of `live[t].var_axes` for `t in tids`, minus `v`.
- `fill_count(out) -> usize`: number of pairs `(out[i], out[j])` (`i<j`) that do NOT already share an active tensor (scan `v2t[out[i]]` for an active slot containing `out[j]`). Exact port of Julia `fill_count`.
- `score(v) -> (bool, usize, usize)` = `(eligible, fill, sc)`:
  - `protected_set[v]` ⇒ `(false, 0, usize::MAX)`.
  - `tids = active_incident(v)`; empty ⇒ `(false, 0, usize::MAX)`.
  - `out = out_vars(tids, v)`; `sc = out.len() + 1`.
  - eligible iff `out.len() <= 32 && sc <= budget_b`.

Heap: `BinaryHeap<HeapItem>` where `HeapItem { fill: usize, sc: usize, var: usize }` with `Ord` giving a **min-heap on `(fill, sc)`** (i.e. `Reverse`-style: smaller `(fill, sc)` pops first; tie-break is irrelevant to correctness). Seed with every eligible var.

Main loop — pop `item`, then **re-score `item.var`** (lazy staleness): if not currently eligible, skip; if `(fill, sc)` no longer matches the popped item, skip (a newer entry exists). Otherwise eliminate `v`:
1. `tids = active_incident(v)`; `out = out_vars(tids, v)`.
2. **Bucket contraction (reuse `contract.rs`):** for each `t in tids`, build a `Relation` over `live[t].var_axes` from `live[t].dense` (all rows where dense is true, re-encoded to sorted-var order — see `full_relation` below). `join_all` them → relation over `out ∪ {v}`. Project onto `out` (drop `v`), dedup → rows over `out`.
3. **Densify:** `dense = vec![false; 1 << out.len()]`; for each projected row (already a bitmask over `out` order) set `dense[row] = true`.
4. **Merge in place:** `keep = tids[0]`; remove all `tids` from the incidence of every var they touch; set `active[t]=false` for `t in tids[1..]`; set `live[keep] = LiveTensor { var_axes: out.clone(), dense }`; push `keep` into `v2t[x]` for each `x in out`. (`v` now has no active incident tensor → eliminated.)
5. **Re-score neighbors:** for each `u in out`, if `score(u).eligible` push a fresh `HeapItem`; stale old entries are skipped on later pops.

Finalize:
- Collect surviving tensors: `for t where active[t]`, push `(live[t].var_axes.clone(), live[t].dense.clone())`.
- `new_cn = setup_problem(nv, surviving_var_axes, surviving_dense)` — handles dedup, var compression, `v2t`, and a cn→new `orig_to_new`.
- Compose: `let mut orig_to_new = vec![None; cn.orig_to_new.len()]; for (orig, &cnid) in cn.orig_to_new.iter().enumerate() { if let Some(c) = cnid { orig_to_new[orig] = new_cn.orig_to_new[c]; } }` then return `ConstraintNetwork { orig_to_new, ..new_cn }`.

### `full_relation` (`src/contract.rs`)

`tensor_relation` slices by `doms`; with an all-`BOTH` doms it returns the full relation over all the tensor's vars. To avoid threading a dummy doms vector, add:

```rust
/// The full boolean relation of a dense truth table over `var_axes` (no domain
/// slicing): rows = configs where `dense` is true, re-encoded over `var_axes` SORTED
/// ascending (canonical bit order), deduplicated.
pub fn dense_relation(var_axes: &[usize], dense: &[bool]) -> Relation;
```

Bit order: `var_axes` may be unsorted; `dense[config]` has bit `i` = `var_axes[i]`. Sort `(var, axis_pos)` by var; row bit `j` = original bit at `axis_pos` of entry `j`. Returns `Relation { vars: sorted_vars, rows }`. (This mirrors the sorting in `tensor_relation`.)

### `solve_circuit` example wiring (`examples/solve_circuit.rs`)

New optional 4th arg `budget_B` (usize, default 0 = off). When `> 0`:
1. `protected`: for each factor-bit wire name (`p{i}`, `q{i}` for `i in 1..=bits`), look up `name_to_orig[name]`, map through `cp.network.orig_to_new[orig]` to a compressed id; collect the `Some` ones.
2. `let cn2 = bounded_ve_canonicalize(&cp.network, budget_B, &protected);`
3. Rebuild a `CircuitProblem { network: cn2, name_to_orig: cp.name_to_orig }` (name→orig unchanged; `wire_value` reads `cn2.orig_to_new`).
4. Solve `cn2`, decode factors via `wire_value` as today, print branch-set sizes (`cp.network.vars.len()` → `cn2.vars.len()`).

## Data flow

```
load CircuitSAT JSON ─► CircuitProblem{ network(raw, tiny tensors), name_to_orig }
   budget_B>0 ─► protected = factor-bit compressed ids
              ─► bounded_ve_canonicalize(network, budget_B, protected)
                    seed live/active/v2t ─► heap by (fill,sc)
                    loop: pop v ─► relations(join_all) ─► project v ─► densify ─► merge
                    finalize ─► setup_problem ─► compose orig_to_new
              ─► network2 (fewer vars, larger tensors)
   from_network(network2) ─► bbsat ─► solution ─► wire_value(factor bits) ─► p,q
```

## Error handling / edge cases

- **Produced empty support** (join yields no rows ⇒ all-false dense): kept as a tensor; `from_network`'s root propagation detects the contradiction → UNSAT. No special-casing.
- **Single incident tensor**: `out` = that tensor's other vars; elimination = projecting `v` out of one tensor. Handled by the same path (`join_all` of one relation = itself).
- **`budget_b < 2`**: every elimination has `out.len() >= 1` ⇒ `sc >= 2`; nothing eligible ⇒ returns an equivalent (recompressed) network.
- **Protected var**: never scored eligible; survives into `new_cn` (appears in ≥1 surviving tensor as long as it had one; an isolated protected var with no tensor is compressed out by `setup_problem` exactly as in the raw path — its value is then free, same as today).
- **Arity cap**: `out.len() > 32` ⇒ ineligible (never produce a tensor beyond `TensorData`'s 32-var limit); also bounded well below by `budget_b`.
- **`u64` row cap**: `out ∪ {v}` join uses `Relation` rows as `u64`; `out.len()+1 <= budget_b <= ` (practically ≤ ~24) ≤ 64, safe. Plan asserts `budget_b <= 32`.

## Testing

Unit (`src/canonicalize.rs` tests):
1. **Eliminate a chain var:** `(x0∨x1)∧(x1∨x2)`, eliminate `x1` (budget ≥ 3, protected = []) ⇒ one tensor over `{x0,x2}` whose support = the projection (here all 4 configs, since x1 can absorb). Assert var count drops and the surviving relation over `{x0,x2}` matches brute force.
2. **Protected var is never eliminated:** same network, `protected=[x1]` ⇒ `x1` survives.
3. **Solution-preservation (property):** for a few small hand-built networks + a `protected` subset + budget, enumerate all satisfying assignments of the original over ALL vars, project to `protected`; do the same for the canonicalized network (over its vars, mapped back via `orig_to_new`); assert the projected solution SETS are equal.
4. **Budget gating:** `budget_b=1` ⇒ no elimination (network equivalent); larger budget ⇒ fewer vars.
5. **Arity cap:** every produced tensor has `var_axes.len() <= budget_b - 1`.
6. **`dense_relation`** (in `contract.rs` tests): unsorted `var_axes` re-encodes to sorted order correctly (compare to a brute-force relation).

End-to-end (extend `tests/factoring.rs` or the example, run manually):
7. Canonicalize the committed `factoring_15` fixture (protect `p1..p4`,`q1..q4`) at `budget_B ∈ {6, 10}`, solve, assert `p*q == 15` and `vars` strictly fewer than raw.

No-regression: `cargo test` (incl. `acceptance`, `factoring`) stays green (default path unchanged).

## Files

| File | Change |
|------|--------|
| `src/canonicalize.rs` | **new** — `bounded_ve_canonicalize`, `LiveTensor`, scoring, heap, tests |
| `src/lib.rs` | add `pub mod canonicalize;` |
| `src/contract.rs` | add `pub fn dense_relation(var_axes, dense) -> Relation` + test |
| `examples/solve_circuit.rs` | optional `budget_B` arg: protect factor bits, canonicalize, solve, decode, print reduction |
```
