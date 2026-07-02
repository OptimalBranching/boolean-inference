# Contract Primitive + End-to-End-Sparse Canonicalize — Design

**Date:** 2026-07-02
**Status:** Approved design, ready for implementation plan.
**Research basis:** `docs/research/2026-07-02-sparse-contraction-wcoj.md`

## Goal

Extract the single relational-contraction operation that `canonicalize`'s
variable-elimination step and `contract_region` both perform — "join relations on
shared variables, then project onto a kept variable set" — into one shared
primitive, and make `canonicalize` operate on sparse relations end to end (no dense
`2^arity` truth table is ever materialized along its path).

## Non-Goals (explicitly deferred)

- **No `feasible_configs` fusion.** It is a third caller of the same conceptual
  descent, but folding it in is out of scope here.
- **No worst-case-optimal / generic join.** The `contract` kernel keeps its current
  binary-join internals. Per the research gate, generic join is a wash on factoring
  (arity 2–3, small regions) and only wins on cyclic/tight structured-CSP regions,
  which do not exist in the codebase yet. The `contract` signature is designed so
  the kernel can be swapped to a generic join later without touching callers.

## Invariant

**Behavior-preserving.** Node counts, the factoring solution, and every existing
test must be unchanged. This is a refactor: same results, cleaner structure, and no
`2^arity` dense allocation on the canonicalize path.

---

## Components

### 1. `Relation::project` — `src/contract.rs`

```rust
impl Relation {
    /// Project each row onto `keep` (a subset of `self.vars`, ascending). Rows are
    /// re-encoded over `keep` bit order, then sorted and deduplicated. Every entry
    /// of `keep` must be present in `self.vars`.
    pub fn project(&self, keep: &[usize]) -> Relation
}
```

This captures the projection loop currently duplicated in `contract_region`
(`contract.rs`, the `output_vars` re-encode) and in `canonicalize`'s VE step (the
`out` re-encode). Implementation: for each `row`, for each `v` in `keep`, locate its
bit position via `self.vars.binary_search(&v)`, copy that bit into the new row at
its `keep`-index; collect, `sort_unstable`, `dedup`.

### 2. `contract` — `src/contract.rs`

```rust
/// Join all `rels` on shared variables, then project onto `keep` (a subset of the
/// union of all rels' vars, ascending). The single contraction primitive shared by
/// `contract_region` and `canonicalize`'s VE step.
///
/// Binary-join internals for now (`join_all`); the signature admits a generic-join
/// kernel later with no change to callers.
pub fn contract(rels: Vec<Relation>, keep: &[usize]) -> Relation {
    join_all(rels).project(keep)
}
```

Precondition (inherited from `join_all`): `rels` is non-empty. Both call sites
already guarantee this.

### 3. Support-based network assembly — `src/network.rs`

Extract a shared assembly core; feed it from two entry points.

```rust
/// Shared assembly: dedup TensorData, compress unused variables, remap axes, build
/// v2t. Dedup key is `(var_axes.len(), support)` — see Edge Cases §1.
fn assemble(
    var_num: usize,
    tensors: Vec<(Vec<usize> /* var_axes */, Vec<u32> /* support (ascending) */)>,
) -> ConstraintNetwork

/// Existing dense entry (dimacs / circuit / tests). Converts each dense truth table
/// to its support, then calls `assemble`. Signature unchanged.
pub fn setup_problem(
    var_num: usize,
    tensors_to_vars: Vec<Vec<usize>>,
    tensor_data: Vec<Vec<bool>>,
) -> ConstraintNetwork

/// New sparse entry. Each `Relation` contributes `(rel.vars, rel.rows as u32)`.
/// Used by `canonicalize` so it never densifies.
pub fn setup_from_relations(var_num: usize, rels: Vec<Relation>) -> ConstraintNetwork
```

`setup_problem` keeps its exact current signature and semantics (existing callers
untouched); the only change is that its body funnels through `assemble` after
deriving support from each dense table. `assemble` builds each `TensorData` from its
support (reusing the OR/AND aggregate derivation already in `TensorData::from_dense`;
factor that aggregate step so both dense and support construction share it).

`rel.rows` are `u64` bitmasks over `rel.vars` (ascending) order; since arity ≤ 32 a
row fits `u32`, so `rel.rows[i] as u32` is the support config. `rel.vars` is the
`var_axes`.

### 4. `canonicalize` rewrite — `src/canonicalize.rs`

- Replace the `LiveTensor { rel: Relation }` newtype with `live: Vec<Relation>`
  alongside the existing `active: Vec<bool>`. All `live[t].rel.vars` become
  `live[t].vars`.
- **Seed:** `support_relation(&t.var_axes, cn.support(t))` (already exists).
- **VE step:** the manual `join_all` + projection + densify block becomes
  `let merged = contract(incident_rels, &out);` and the produced slot stores
  `merged` directly. `incident_rels = tids.iter().map(|&t| live[t].clone()).collect()`.
- **Finalize:** collect surviving relations and call
  `setup_from_relations(nv, surviving)`. The `vec![false; 1 << rel.vars.len()]`
  densify loop is deleted — this is the change that actually removes the `2^arity`
  allocation from the canonicalize path.

The VE scheduling (min-fill heap, `budget_b`, protected vars, staleness) is
unchanged — only the contraction kernel and the finalize boundary change.

### 5. `contract_region` rewrite — `src/contract.rs`

The body becomes: build `rels` via `tensor_relation` (doms-sliced) as today, then
`let r = contract(rels, &output_vars);` and return `(r.rows, output_vars)`. This
replaces the inline `join_all` + projection.

---

## Data Flow

```
tensors (sparse support)
  ├─ canonicalize:
  │     support_relation ─▶ [ VE step: contract(incident, out) ]* ─▶ setup_from_relations
  └─ contract_region:
        tensor_relation (doms-sliced) ─▶ contract(rels, unfixed_region_vars) ─▶ configs
```

Everything is `Relation` (sparse) throughout; `contract` is the sole contraction
point.

---

## Edge Cases (normative)

1. **Dedup key includes arity.** `assemble` must key dedup on
   `(var_axes.len(), support)`, not `support` alone: identical support lists can
   mean different relations at different arities (e.g. `{0, 1}` is "either value of
   one var" at arity 1, but configs `00`/`01` at arity 2). This matches the current
   dense-keyed dedup, where differing dense-table lengths already separate by arity.
2. **Empty relation (infeasible region / eliminated-to-UNSAT).** `contract` may
   return a `Relation` with empty `rows`. `setup_from_relations`/`assemble` must
   accept an empty support (an all-false constraint), exactly as the current dense
   path accepts an all-false table.
3. **Arity-0 relation.** When `out` is empty (a variable eliminated with no
   surviving neighbours), the produced relation has `vars = []` and `rows` either
   `[]` (UNSAT constant) or `[0]` (SAT constant, the single empty config). The
   current code already produces and handles arity-0 tensors (`vec![false; 1 << 0]`);
   `assemble` must handle a zero-length `var_axes` with support `[]` or `[0]`.
4. **Axis-order canonicalization.** `Relation.vars` is ascending, so surviving
   tensors carry sorted axes. This is already the case after the prior dense-field
   removal and is solution-equivalent (a tensor's relation is unchanged by axis
   relabeling); existing brute-force tests confirm it.

---

## Testing

**Behavior-preserving — existing tests unchanged and passing:**
- `canonicalize` solution-equivalence tests (`solutions_projected` brute force over
  protected vars).
- `contract_region` brute-force equivalence (`two_tensor_join_matches_bruteforce`,
  etc.) and `region_cache` memoization tests.
- The factoring integration tests (`factoring_15_*`), which pin node-count-sensitive
  end-to-end behavior.

**New unit tests:**
- `Relation::project`: projecting a small relation onto a subset yields the expected
  rows (including a dedup case where two rows collapse after projection).
- `contract`: on a 2–3 tensor region, `contract(rels, keep)` equals
  `join_all(rels)` followed by a hand-written projection onto `keep` (guards the
  extraction).

No differential old-vs-new harness is needed: the brute-force and integration tests
already pin the observable behavior the refactor must preserve.

---

## Sequencing note for the plan

`Relation::project` and `contract` come first (pure, unit-tested). Then
`contract_region` swaps onto them (small, covered by existing brute tests). Then the
`assemble` core + `setup_from_relations` (covered by existing `setup_problem` tests
plus reuse). Finally `canonicalize` is rewritten onto all of the above (covered by
its solution-equivalence tests). Each step is independently testable and
behavior-preserving.
