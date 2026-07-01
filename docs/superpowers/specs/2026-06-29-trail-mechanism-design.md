# Trail-Based State Restoration — Design

**Status:** approved (brainstorming), ready for implementation plan
**Date:** 2026-06-29
**Branch:** `framework-apply-branch` (Rust crate `boolean-inference`)

## Goal

Replace the current **copying** state-restoration in the branch-and-reduce search
with a **trail** (undo log), as a clean, correct foundation. This is *not* a raw-speed
change — profiling showed copying/allocation is ~0.7% of runtime while `propagate_core`
is ~55%. The trail's value is: (1) it removes the copying-driven design the author
dislikes (`SolverBuffer::scratch_doms`), and (2) it is the necessary substrate for a
later, separately-scoped step: incremental / reversible-bitset propagation (Compact-Table),
which is what will actually attack the 55%.

This step must be **behavior-preserving**, with exactly one intentional, sound
control-flow change (the contradiction guard, §6), explicitly approved.

## Non-goals (deferred)

- Incremental propagation / reversible sparse bit-sets (the 55% win) — separate spec.
- Converting recursion to an explicit search stack — recursion is kept.
- Touching ob-core's `BranchAndReduceProblem` copying contract — left as-is (§7).
- Parallel search, GPU region contraction — orthogonal axes, out of scope.

## Background: current restoration is copying

State is a `doms: Vec<DomainMask>` (2-bit per var: `NONE`=00, `D0`=01, `D1`=10, `BOTH`=11).
Restoration today is **copying**:

- `probe_assignment` (`src/propagate.rs:108`) does `scratch_doms.copy_from_slice(base_doms)`,
  applies a partial assignment, propagates, and returns a borrowed slice of the scratch.
- `bbsat_rec` (`src/solver.rs:101-103`) probes, then `scratch.to_vec()` clones a fresh
  owned `doms` and recurses on it.

Every probe restarts from an immutable parent vector, so branches never interfere.

`propagate_core` mutates `doms` in three places, and **all three must be trailed**:
1. `apply_updates` narrowing `BOTH → D0/D1` (`src/propagate.rs:91`).
2. the direct-assign seeding loop in `probe_assignment` (`src/propagate.rs:123`).
3. the **contradiction sentinel** `doms[0] = NONE` on an unsatisfiable tensor
   (`src/propagate.rs:158`).

## Architecture

`doms` becomes a single **shared `&mut`** buffer threaded through the recursion. A
`Trail` records the old value before every write; backtracking pops the trail and
restores. Recursion is kept; the trail's `mark`/`restore_to` map onto recursion
enter/return.

### The one invariant (entire correctness story)

> **Every write to `doms` during apply + propagate is preceded by `trail.record(var, old_value)`** — including the contradiction sentinel.

Given this, `restore_to(mark)` reverts any sequence of narrowings and/or a contradiction
exactly.

## Components

### 1. `Trail` (new file `src/trail.rs`)

```rust
use crate::domain::DomainMask;

/// Undo log of domain writes. `restore_to` reverts to a prior `mark()`.
#[derive(Default)]
pub struct Trail {
    entries: Vec<(u32, DomainMask)>, // (var_id, old_mask), in write order
}

impl Trail {
    pub fn new() -> Trail { Trail { entries: Vec::new() } }

    /// A restore point = current length.
    #[inline]
    pub fn mark(&self) -> usize { self.entries.len() }

    /// Record the OLD value of `var` before it is overwritten.
    #[inline]
    pub fn record(&mut self, var: usize, old: DomainMask) {
        self.entries.push((var as u32, old));
    }

    /// Pop entries back to `mark`, restoring each `var` to its recorded old value (LIFO).
    #[inline]
    pub fn restore_to(&mut self, doms: &mut [DomainMask], mark: usize) {
        while self.entries.len() > mark {
            let (v, old) = self.entries.pop().expect("len > mark");
            doms[v as usize] = old;
        }
    }

    /// Drop all entries without restoring (for the discarded-trail / clone-and-return path).
    #[inline]
    pub fn clear(&mut self) { self.entries.clear(); }
}
```

**No timestamping is needed — but NOT because each var is written once.** That earlier
rationale is **false**: a var *can* appear on the live trail more than once (e.g. the
direct-assign loop writes `doms[0]`, then the same propagation overwrites it with the
`NONE` sentinel). The reason timestamping is unnecessary is simpler and more robust:
**we push every write and pop LIFO with the explicit recorded old value.** Duplicates
restore correctly under LIFO.

**Footgun warnings (do NOT do any of these):**
- Do **not** add a "each var appears at most once" assertion — it is false.
- Do **not** dedup the trail by var.
- Do **not** "restore to `BOTH`" instead of the recorded old value. Wrong because:
  root propagation can leave fixed domains in `problem.doms`; recursion branches from
  already-fixed parent states; and the sentinel overwrites `doms[0]` whose old value may
  be `D0`/`D1`.

### 2. New propagation primitive (`src/propagate.rs`)

`propagate_core` and `apply_updates` gain a `&mut Trail` parameter and call
`trail.record(var, doms[var])` immediately before each write (including the sentinel at
`:158`).

A new in-place primitive replaces the copy semantics for the search:

```rust
/// Apply the partial assignment (`vars[i]` set to bit `i` of `value` where bit `i` of
/// `mask` is 1) to the SHARED `doms` in place, recording every write on `trail`, then
/// propagate to a GAC fixpoint. Returns whether the result is a contradiction
/// (`doms[0] == NONE`). The caller inspects `doms`, then calls
/// `trail.restore_to(doms, mark)` (with the `mark` taken *before* this call) to undo.
pub fn apply_and_propagate(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    trail: &mut Trail,
    buffer: &mut SolverBuffer,
    vars: &[usize],
    mask: u64,
    value: u64,
) -> bool /* contradicted */;
```

Implementation notes:
- **Keep entry cleanup** exactly as `probe_assignment:108-112` does today: clear
  `buffer.queue` and reset all `buffer.in_queue` flags at the start. Although
  `propagate_core` already cleans the worklist on both exit paths, keeping entry cleanup
  preserves the current tolerance to a dirty buffer; cheap insurance.
- Direct-assign loop records old before writing, then seeds the worklist (same as today).
- Returns `doms[0] == NONE` after propagation.

### 3. `SolverBuffer` (`src/problem.rs`)

Delete the `scratch_doms` field. `SolverBuffer` shrinks to `{ queue, in_queue,
connection_scores }`. (`SolverBuffer::new` gets correspondingly cheaper, a minor bonus on
the ob-core `apply_branch` path.)

### 4. `probe_assignment` — kept as a copy-based compatibility wrapper (`src/propagate.rs`)

`probe_assignment` is `pub` and used by `examples/phase2_demo.rs`,
`examples/phase2_perf.rs`, and the `adapter.rs` test `apply_branch_matches_probe_assignment`.
**Do not delete it.** Reimplement it on top of `apply_and_propagate` with a throwaway
trail, returning an **owned** `Vec<DomainMask>`:

```rust
/// Copy-based probe (compatibility wrapper for examples/tests): clones `base_doms`,
/// applies + propagates on the copy, returns the owned result. The search path uses
/// `apply_and_propagate` directly instead.
pub fn probe_assignment(
    cn: &ConstraintNetwork,
    buffer: &mut SolverBuffer,
    base_doms: &[DomainMask],
    vars: &[usize],
    mask: u64,
    value: u64,
) -> Vec<DomainMask> {
    let mut doms = base_doms.to_vec();
    let mut trail = Trail::new(); // discarded — we keep the mutated copy
    apply_and_propagate(cn, &mut doms, &mut trail, buffer, vars, mask, value);
    doms
}
```

Return type changes from `&[DomainMask]` to `Vec<DomainMask>`; callers that use the result
as a slice still compile (`Vec` derefs to `[T]`). The two examples and the adapter test
may need a one-line adjustment if they bound the old borrow; update them as needed.

### 5. `compute_branching_result` feasibility loop (`src/table.rs:43-51`)

Each cached config is probed and discarded; today each restarts from `doms` via copy.
With the shared trail, every config probe must restore so mutations don't accumulate:

```rust
for &config in &cached_configs {
    if (config & check_mask) != check_value { continue; }
    let mark = trail.mark();
    let contradicted = apply_and_propagate(cn, doms, trail, buffer, &region_vars, full_mask, config);
    if !contradicted { feasible.push(config); }
    trail.restore_to(doms, mark);
}
```

`compute_branching_result` gains `doms: &mut [DomainMask]` and `trail: &mut Trail`
parameters.

### 6. `bbsat_rec` branch descent + contradiction guard (`src/solver.rs`)

`bbsat_rec` takes `doms: &mut Vec<DomainMask>` and `trail: &mut Trail` (no longer `doms`
by value). The success leaf clones `doms` into `Solve.solution` (one clone, winning path
only). The 2-SAT leaf reads `doms` (unchanged).

Branch loop — **mark inside each iteration**, guard on contradiction, restore every
iteration:

```rust
stats.record_branch(clauses.len() as u64);
for cl in &clauses {
    stats.record_visit();                       // unchanged: visit counted per branch
    let mark = trail.mark();
    let contradicted = apply_and_propagate(ctx.cn, doms, trail, buffer, &variables, cl.mask, cl.val);
    if !contradicted {
        let result = bbsat_rec(ctx, cache, stats, buffer, doms, trail);
        if result.found { return result; }      // doms already holds the witness; leaf cloned it
    }
    trail.restore_to(doms, mark);               // undo this branch (incl. any sentinel) before next
}
```

**Contradiction guard (the one intentional, approved behavior change):** today the loop
recurses into a contradicted child unconditionally (`solver.rs:101-103`); the child's
`findbest` returns `None` and it backtracks. Skipping the recursion when
`contradicted == true` is **sound** (a GAC contradiction means no satisfying completion
exists in that branch) and preserves both the verdict and the visit/branch stats (the
skipped child records nothing). It also removes the hazard of recursing on a shared
`doms` with `doms[0] = NONE` (which `compute_query_masks` would treat as *unconstrained*).

`bbsat` (the entry, `src/solver.rs:31`) creates `let mut doms = problem.doms.clone();` and
`let mut trail = Trail::new();`, then recurses on `&mut doms, &mut trail` (disjoint field
borrows from `problem.buffer`/`problem.stats`, as today). Operating on a working clone
leaves `problem.doms` intact for the caller.

### 7. DiffLookahead probing (`src/selector.rs:104-120`)

The current code probes both polarities (`c0`, `c1` alias the same scratch) and computes
`f0,d0,f1,d1` *before* the failed-literal check. The trail version must reproduce this
**exact sequencing** — compute each polarity's scalars *before* restoring, and probe BOTH
before deciding:

```rust
for &u in &cands {
    let m0 = trail.mark();
    let f0 = apply_and_propagate(cn, doms, trail, buffer, &[u], 1, 0);
    let d0 = if f0 { 0 } else { sum_active_degree(cn, doms) };   // read BEFORE restore
    trail.restore_to(doms, m0);

    let m1 = trail.mark();
    let f1 = apply_and_propagate(cn, doms, trail, buffer, &[u], 1, 1);
    let d1 = if f1 { 0 } else { sum_active_degree(cn, doms) };   // read BEFORE restore
    trail.restore_to(doms, m1);

    if f0 || f1 { chosen = Some(u); break; }                     // failed literal -> take now
    let s = d0.max(d1);
    if s < best { best = s; chosen = Some(u); }
}
```

This removes the old `c0`/`c1` aliasing footgun while preserving identical behavior (same
`sum_active_degree`, same failed-literal detection). `select_var_difflookahead` gains
`doms: &mut [DomainMask]` and `trail: &mut Trail`.

`findbest` (`src/selector.rs:149`) threads `doms: &mut [DomainMask]` and `trail: &mut Trail`
through to `select_var_difflookahead` / `compute_branching_result`. `MostOccurrence`'s
`select_var_most_occurrence` is read-only over `doms` and needs no probe restore.

### 8. ob-core integration — untouched, with a documented exception (`src/adapter.rs`)

`RuleProblem::apply_branch` keeps its own `doms.clone()` contract. It has its own
direct-assign writes (`adapter.rs:69-78`) *before* calling `propagate_core`. These writes
stay **untrailed**, and `apply_branch` passes a throwaway local `Trail` to the new
`propagate_core` signature. This is correct **only because** `apply_branch` clones `doms`
and returns the mutated copy — it never restores. Document this exception inline. The
existing per-call `SolverBuffer::new` allocation remains a known (negligible) follow-up
cost; out of scope here.

## Data flow

```
bbsat: doms = problem.doms.clone(); trail = Trail::new()
  └─ bbsat_rec(&mut doms, &mut trail)
       count_unfixed == 0 ─────────────▶ return Solve{found, solution: doms.clone()}
       is_two_sat ─────────────────────▶ solve_2sat(doms)  (read-only)
       findbest(&mut doms, &mut trail)
          DiffLookahead: per candidate  mark→apply→read scalar→restore (×2 polarities)
          compute_branching_result: feasibility loop  mark→apply→keep-if-feasible→restore
            ob-core optimal_rule  (RuleProblem clones doms; isolated copying)
       branch loop: per clause  mark→apply→[guard: recurse if not contradicted]→restore
```

## Error handling / edge cases

- **Contradiction during a probe:** the sentinel write `doms[0]=NONE` is trailed; both
  the affected `doms` entries and the worklist are restored/cleaned, so the next probe
  starts clean.
- **Contradicted branch in `bbsat_rec`:** guarded — not recursed; restored before the next
  branch.
- **Var 0 written twice (direct-assign then sentinel):** allowed; LIFO restore reverts
  both in reverse order to the original value.
- **`restore_to` precondition:** `mark` must be `<= trail.len()`; always true because each
  caller takes `mark` before the writes it later restores.

## Testing / correctness gates

Primary gate — **behavior preservation**:
1. The 233-instance brute-force acceptance harness stays green (same SAT/UNSAT verdicts,
   valid assignments).
2. All existing unit tests in `propagate.rs`, `solver.rs`, `table.rs`, `selector.rs`,
   `adapter.rs` pass (updated for the new signatures, asserting the same propagated results).
3. Factoring-15 integration test + 12/20-bit factoring runs produce identical factors.

New targeted tests:
4. **Trail round-trip:** random `doms` + random partial assignment ⇒
   `mark; apply_and_propagate; restore_to(mark)` leaves `doms` byte-identical.
5. **Var-0 duplicate restore:** a probe that direct-assigns `doms[0]` then triggers the
   `NONE` sentinel, restored, leaves `doms[0]` at its original value.
6. **Parent-fixed contradiction restore:** with parent `doms[0]` already `D0`/`D1`, a
   contradicting probe (sentinel overwrites it) then restore reverts to that fixed value
   (proves "restore-to-BOTH" would be wrong).
7. **Repeated-probe purity:** repeated table-feasibility and DiffLookahead probes leave
   `doms`, `buffer.queue`, and `buffer.in_queue` unchanged from before the probes.
8. **Branch-loop immediate contradiction:** a branch that contradicts is skipped (not
   recursed), `stats` match the recurse-then-fail baseline, and the search continues to the
   next branch.

## Files

| File | Change |
|------|--------|
| `src/trail.rs` | **new** — `Trail` struct + unit tests |
| `src/lib.rs` | add `pub mod trail;` |
| `src/propagate.rs` | thread `&mut Trail` through `propagate_core`/`apply_updates`; record before every write incl. sentinel; add `apply_and_propagate`; reimplement `probe_assignment` as copy wrapper; delete the `scratch_doms` swap dance |
| `src/problem.rs` | delete `SolverBuffer::scratch_doms` |
| `src/solver.rs` | `bbsat_rec(&mut doms, &mut trail)`; branch loop w/ mark-per-iteration + contradiction guard; success-leaf clone; `bbsat` sets up `doms`/`trail` |
| `src/table.rs` | `compute_branching_result(&mut doms, &mut trail)`; feasibility loop mark/apply/restore |
| `src/selector.rs` | `findbest`/`select_var_difflookahead(&mut doms, &mut trail)`; exact two-polarity sequencing |
| `src/adapter.rs` | `apply_branch` passes a throwaway `Trail` to `propagate_core`; document the untrailed-direct-write exception; update the test that calls `probe_assignment` |
| `examples/phase2_demo.rs`, `examples/phase2_perf.rs` | adjust to the owned-`Vec` return of `probe_assignment` if needed |
```
