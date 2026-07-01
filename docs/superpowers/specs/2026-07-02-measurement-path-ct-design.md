# Measurement-Path CT Conversion — Design Spec

**Date:** 2026-07-02
**Status:** design, ready for planning
**Backing profile:** [`docs/research/2026-07-02-post-prefix-sharing-profile.md`](../../research/2026-07-02-post-prefix-sharing-profile.md).
**ob-core feasibility:** confirmed `apply_branch` is only ever called single-level
on the root problem, its sub-problem measured immediately and dropped before the
next call (ob-core `branch.rs::size_reduction`), so a mark/apply/propagate/restore
scheme on a shared per-node CT store is sound.

## Goal

Convert the branching-rule **measurement path** (`adapter.rs::apply_branch`, the
biggest remaining hotspot at ~35% self-time) from the slow linear
`propagate_core_rescan` to the fast Compact-Table `ct_propagate`, and eliminate
the per-candidate `SolverBuffer` allocation — while keeping node counts
bit-identical (golden test 19761 / 45322 on `factoring_22x22`).

## Background — the current cost

`RuleProblem::apply_branch(&self, clause, variables)` (adapter.rs) runs per
candidate branch that `GreedyMerge` evaluates (O(rows²) calls per node):

1. `self.doms.clone()` — a per-candidate `Vec<DomainMask>` clone;
2. `SolverBuffer::new(&self.cn)` — a **fresh buffer allocation per candidate**
   (`queue` + `in_queue[n_tensors]` + `mask_scratch` + `dirty[n_tensors]`);
3. `propagate_core_rescan` — the **slow linear support rescan** (34.7% self time,
   vs CT which the rest of the solver uses).

It was deliberately left on rescan because giving each `RuleProblem` its own CT
tables and cloning them per `apply_branch` caused an allocation storm. The fix is
to share ONE CT store per node across all its candidate probes, restoring it
between candidates via the trail — never cloning tables per candidate.

## Correctness foundation

CT and the rescan propagator compute the **same GAC fixpoint** (already proven by
the CT differential oracle and the node-identical golden test). So the resulting
`doms` an `apply_branch` returns is identical whichever propagator runs →
`measure_core` reads identical values → `size_reduction` is identical →
`GreedyMerge` picks the identical rule → identical branching. The golden
19761/45322 test is the behavior-preservation gate.

The mark/apply/propagate/restore discipline is the same one `probe` and
`feasible_configs` already use and validate: one trail epoch per `apply_branch`
(`open` → `mark` → apply literals → `ct_propagate` → snapshot doms → `restore_to`).
By ob-core's single-level usage, the store is always back at the node base before
the next `apply_branch`.

## Design — thread-local scratch, live state swapped in (no bulk copy)

A single thread-local CT store that `apply_branch` (which only has `&self`) can
reach. `compute_branching_result` **swaps the solver's live CT state into it**
around the `optimal_rule` call — O(1) `Vec`-header swaps, NOT an element copy —
then swaps it back:

```rust
struct MeasureScratch {
    doms: Vec<DomainMask>,          // working copy of base doms; at base between probes
    tables: Vec<RSparseBitSet>,     // THE live tables (swapped in); at base between probes
    buffer: SolverBuffer,           // THE live buffer (swapped in)
    trail: Trail,                   // THE live trail (swapped in)
}
thread_local! { static MEASURE_SCRATCH: RefCell<MeasureScratch> = ...; }
```

**Swap-in (once per node, in `compute_branching_result` immediately before
`optimal_rule`)** — no bulk data copy; the live `tables`/`buffer`/`trail` the
solver already holds (at the node base fixpoint after `feasible_configs` restored
them) are moved into the scratch by swapping the container headers:

```rust
MEASURE_SCRATCH.with(|s| {
    let s = &mut *s.borrow_mut();
    std::mem::swap(&mut s.tables, tables);   // tables: &mut Vec<RSparseBitSet> — O(1) header swap
    std::mem::swap(&mut s.buffer, buffer);   // buffer: &mut SolverBuffer
    std::mem::swap(&mut s.trail,  trail);    // trail:  &mut Trail
    s.doms.clear();
    s.doms.extend_from_slice(doms);          // doms is a &mut [DomainMask] slice — small copy (816)
});
let problem = RuleProblem::new(Arc::clone(cn), Arc::clone(masks), doms.to_vec());
let result = solver.optimal_rule(&problem, &table, &unfixed_vars, &MeasureAdapter(measure));
MEASURE_SCRATCH.with(|s| {                   // swap the (restored-to-base) live state back out
    let s = &mut *s.borrow_mut();
    std::mem::swap(&mut s.tables, tables);
    std::mem::swap(&mut s.buffer, buffer);
    std::mem::swap(&mut s.trail,  trail);
});
```

Because every `apply_branch` fully restores the scratch to base (below), the
live `tables`/`buffer`/`trail` swapped back out are byte-identical to what the
solver handed in (the trail's monotonic `epoch` is advanced — harmless, never
reused). The solver's `&mut tables`/`buffer`/`trail` are idle during
`optimal_rule` (it reads only the `RuleProblem`), so lending them to the scratch
is sound.

**`apply_branch(&self, clause, variables)` new body:**

```
MEASURE_SCRATCH.with(|s| {
    let s = &mut *s.borrow_mut();
    s.trail.open();
    let m = s.trail.mark();
    // buffer is clean (drained by previous ct_propagate); assert in debug
    for (i, &var) in variables.iter().enumerate() {
        if (clause.mask >> i) & 1 == 1 {
            let nd = if (clause.val >> i) & 1 == 1 { D1 } else { D0 };
            if s.doms[var] != nd {
                s.trail.record_dom(var, s.doms[var]);
                s.doms[var] = nd;
                enqueue_var_change(&self.cn, &mut s.buffer, var);
            }
        }
    }
    ct_propagate(&self.cn, &mut s.doms, &self.masks, &mut s.tables, &mut s.buffer, &mut s.trail);
    let snapshot = s.doms.clone();                 // owned doms for the returned sub-problem
    s.trail.restore_to(m, &mut s.doms, &mut s.tables);
    (RuleProblem { cn: Arc::clone(&self.cn), doms: snapshot, masks: Arc::clone(&self.masks) }, 0.0)
})
```

`measure(sub)` reads `sub.doms` (the snapshot); `measure(root)` reads the root
`RuleProblem.doms` (untouched base). Both correct.

### `RuleProblem` change

Add the CT masks so `apply_branch` can propagate:

```rust
pub struct RuleProblem {
    pub cn: Arc<ConstraintNetwork>,
    pub masks: Arc<Vec<TableMasks>>,
    pub doms: Vec<DomainMask>,
}
```

`compute_branching_result` already holds `masks: &Arc<Vec<TableMasks>>` and builds
the `RuleProblem` — it passes `Arc::clone(masks)`.

### Scratch sizing is automatic (a benefit of swap-in)

The scratch's `tables`/`buffer` are the solver's own live, correctly-sized
structures (swapped in), so there is nothing to size or rebuild. The thread-local
initializes to an empty `MeasureScratch::default()` (all `Vec::new()`); the first
swap fills it and leaves the caller holding the empty placeholder for the duration
of `optimal_rule`, and the swap-back restores. Between solves — even of different
networks on the same thread — the scratch ends each `compute_branching_result`
holding the empty placeholder, so there is no cross-network staleness. This
requires a cheap empty constructor: `SolverBuffer::default()` and `Trail::default()`
(both all-empty); add them if absent.

## Reentrancy / safety

`apply_branch` never re-enters `compute_branching_result` (it only fixes vars,
propagates, measures), so the thread-local is primed and consumed within one
`compute_branching_result` call, never nested. Single-threaded solve → no
contention. Each `apply_branch` fully restores the scratch before returning, so
ob-core's next root-level call starts from the node base.

## Testing

1. **Golden node-identity (primary gate):** `tests/ct_acceptance.rs` stays green —
   19761 / 45322 on `factoring_22x22`. If it changes, the conversion is not
   behavior-preserving — stop.
2. **`apply_branch` differential:** keep/extend the existing
   `apply_branch_matches_probe` test — the doms an `apply_branch` returns must
   equal the trailed `probe` result on the same base+clause (CT == CT now, but the
   test also guards the scratch prime/restore plumbing). Add a multi-call variant:
   two consecutive `apply_branch` calls on the same primed root return results
   independent of order (restore integrity between candidates), and the scratch is
   left at base after each.
3. **Full suite** `cargo test` green (the adapter unit tests, factoring_15, the CT
   differential oracle, etc.).
4. **Perf (runscribe A/B):** `factoring_22x22` VE10 before/after; expect wall-clock
   down, nodes identical. If the per-node table copy makes it a WASH or REGRESSION,
   that is the risk this gate exists to catch — see Risks.

## Performance expectation (honest)

Replaces the 34.7%-self linear rescan with ~2–3× faster CT and removes the
per-candidate buffer allocation. The only new per-node cost is three O(1)
`Vec`-header swaps plus one small `doms` copy (816 bytes) — no bulk table copy,
so the historical per-candidate-table-clone regression does not recur. Net
expected positive; must be measured.

## Risks

- **Swap-back leaves live state altered.** Every `apply_branch` restores the
  scratch to base before returning, and `optimal_rule` does not otherwise touch
  the live containers, so the swapped-back `tables`/`buffer` are byte-identical
  and the `trail` differs only in its (monotonic, never-reused) `epoch`.
  Mitigation: the golden node-identity test + the `apply_branch` restore-integrity
  test verify end-to-end that the node base is preserved.
- **Buffer/worklist leak between candidates.** Mitigation: `ct_propagate` drains
  clean on both paths; `debug_assert!(s.buffer.queue.is_empty())` before each
  `apply_branch`'s `enqueue_var_change`.
- **Panic-safety of the swap.** If `optimal_rule` could panic between swap-in and
  swap-back, the live state would be stranded in the thread-local. The solver
  treats an ob-core error as unrecoverable (`.expect(...)`), so this is not a
  correctness concern in normal operation; keep the swap-in/out pair adjacent with
  no `?`/early-return between them.

## Out of scope

- difflookahead prefix-sharing (Lever 2 in the profile doc) — its own spec.
- Memoizing `size_reduction` by clause — only if the perf gate shows apply_branch
  count (not per-call cost) still dominates after this change.
- Cheaper surrogate measure (`evalvar`) — changes node counts, not
  behavior-preserving, explicitly excluded.
