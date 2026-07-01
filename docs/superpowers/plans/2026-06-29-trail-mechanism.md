# Trail-Based State Restoration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the search's copy-based state restoration with a trail (undo log), behavior-preserving except one approved sound pruning change (the contradiction guard).

**Architecture:** `doms` becomes a single shared `&mut` buffer threaded through the recursion. A `Trail` records `(var, old_value)` before every write to `doms`; backtracking pops the trail (LIFO) to restore. The copy-based `probe_assignment` is kept as a thin compatibility wrapper for examples/tests; the search switches to an in-place `apply_and_propagate`. ob-core's `BranchAndReduceProblem` copying contract is left untouched.

**Tech Stack:** Rust (crate `boolean-inference`), `cargo test`. Dep: `optimal-branching-core` (git). Branch: `framework-apply-branch`.

**Spec:** `docs/superpowers/specs/2026-06-29-trail-mechanism-design.md`

## Global Constraints

- **The one invariant:** every write to `doms` during apply+propagate is preceded by `trail.record(var, old_value)` — including the contradiction sentinel `doms[0] = NONE`.
- **Trail discipline:** push every write; pop LIFO; store the explicit recorded old value. **Never** add a "var appears once" assertion, **never** dedup by var, **never** "restore to `BOTH`". Duplicate var entries are expected and correct under LIFO.
- **Behavior-preserving:** `tests/acceptance.rs` and `tests/factoring.rs` must stay green; existing unit tests assert the same propagated results. The **only** intentional behavior change is the contradiction guard in `bbsat_rec` (Task 5), which is sound (GAC contradiction ⇒ no solution in that branch) and preserves verdicts and `Stats`.
- **ob-core untouched:** `RuleProblem::apply_branch` keeps `doms.clone()`; its pre-propagate direct writes stay untrailed (safe only because it clones + returns, never restores). It passes a throwaway local `Trail` to `propagate_core`.
- `DomainMask`: `NONE`=00, `D0`=01, `D1`=10, `BOTH`=11; `is_fixed()` is true only for `D0`/`D1`. Contradiction sentinel = `doms[0] == NONE`.
- All `cargo` commands run from `/Users/xiweipan/Codes/boolean-inference`.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/trail.rs` | The `Trail` undo log | **new** (Task 1) |
| `src/lib.rs` | module list | add `pub mod trail;` (Task 1) |
| `src/propagate.rs` | GAC propagation + probe primitives | trail-thread `propagate_core`/`apply_updates`; add `apply_and_propagate`; `probe_assignment` → throwaway-trail (T2) then Vec wrapper (T6) |
| `src/problem.rs` | `SolverBuffer`, `TnProblem`, root propagation | throwaway trail at `from_network` (T2); delete `scratch_doms` (T6) |
| `src/table.rs` | `compute_branching_result` | `&mut doms`+`&mut trail`; feasibility loop → `apply_and_propagate`+restore (T3) |
| `src/solver.rs` | `bbsat`/`bbsat_rec` recursion | `&mut doms`+`&mut trail`; branch loop → `apply_and_propagate`+mark/guard/restore (T5) |
| `src/selector.rs` | `findbest`, var selection | `&mut doms`+`&mut trail`; DiffLookahead → `apply_and_propagate`+restore (T3 plumb, T4 switch) |
| `src/adapter.rs` | ob-core wiring | throwaway trail at `apply_branch`'s `propagate_core` call (T2) |
| `examples/phase2_demo.rs`, `examples/phase2_perf.rs` | demos | adjust to `probe_assignment`'s owned-`Vec` return (T6) |

---

### Task 1: `Trail` undo log

**Files:**
- Create: `src/trail.rs`
- Modify: `src/lib.rs:1` (add module)

**Interfaces:**
- Consumes: `crate::domain::DomainMask`
- Produces: `Trail` with `new() -> Trail`, `mark(&self) -> usize`, `record(&mut self, var: usize, old: DomainMask)`, `restore_to(&mut self, doms: &mut [DomainMask], mark: usize)`, `clear(&mut self)`.

- [ ] **Step 1: Write `src/trail.rs` with the struct and failing tests**

```rust
use crate::domain::DomainMask;

/// Undo log of domain writes. Record the OLD value before every write to `doms`;
/// `restore_to(mark)` reverts (LIFO) to a prior `mark()`. Duplicate entries for the
/// same var are expected (e.g. a var written, then overwritten by the contradiction
/// sentinel) and restore correctly under LIFO. Do NOT dedup, assert uniqueness, or
/// "restore to BOTH" — the recorded old value is authoritative.
#[derive(Default)]
pub struct Trail {
    entries: Vec<(u32, DomainMask)>,
}

impl Trail {
    pub fn new() -> Trail {
        Trail { entries: Vec::new() }
    }

    /// A restore point = the current trail length.
    #[inline]
    pub fn mark(&self) -> usize {
        self.entries.len()
    }

    /// Record the OLD value of `var` before it is overwritten.
    #[inline]
    pub fn record(&mut self, var: usize, old: DomainMask) {
        self.entries.push((var as u32, old));
    }

    /// Pop entries back to `mark`, restoring each var to its recorded old value (LIFO).
    #[inline]
    pub fn restore_to(&mut self, doms: &mut [DomainMask], mark: usize) {
        while self.entries.len() > mark {
            let (v, old) = self.entries.pop().expect("len > mark");
            doms[v as usize] = old;
        }
    }

    /// Drop all entries without restoring (for the discarded-trail / clone-and-return path).
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_then_restore_reverts_writes() {
        let mut doms = vec![DomainMask::BOTH, DomainMask::BOTH];
        let mut t = Trail::new();
        let m = t.mark();
        t.record(0, doms[0]);
        doms[0] = DomainMask::D1;
        t.record(1, doms[1]);
        doms[1] = DomainMask::D0;
        t.restore_to(&mut doms, m);
        assert_eq!(doms, vec![DomainMask::BOTH, DomainMask::BOTH]);
    }

    #[test]
    fn restore_handles_duplicate_var_lifo() {
        // var 0 written twice (BOTH->D1->NONE); LIFO must restore to the ORIGINAL BOTH.
        let mut doms = vec![DomainMask::BOTH];
        let mut t = Trail::new();
        let m = t.mark();
        t.record(0, doms[0]);
        doms[0] = DomainMask::D1;
        t.record(0, doms[0]);
        doms[0] = DomainMask::NONE;
        t.restore_to(&mut doms, m);
        assert_eq!(doms[0], DomainMask::BOTH);
    }

    #[test]
    fn restore_to_an_intermediate_mark() {
        let mut doms = vec![DomainMask::BOTH];
        let mut t = Trail::new();
        t.record(0, doms[0]);
        doms[0] = DomainMask::D0;
        let m = t.mark(); // checkpoint with doms[0] == D0
        t.record(0, doms[0]);
        doms[0] = DomainMask::D1;
        t.restore_to(&mut doms, m);
        assert_eq!(doms[0], DomainMask::D0); // back to the checkpoint, not the start
    }

    #[test]
    fn clear_drops_entries_without_touching_doms() {
        let mut doms = vec![DomainMask::D1];
        let mut t = Trail::new();
        t.record(0, DomainMask::BOTH);
        t.clear();
        assert_eq!(t.mark(), 0);
        assert_eq!(doms[0], DomainMask::D1); // unchanged
    }
}
```

- [ ] **Step 2: Register the module** — add to `src/lib.rs` after `pub mod table;` (keep alphabetical-ish ordering; place near `pub mod twosat;`):

```rust
pub mod trail;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test --lib trail`
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/trail.rs src/lib.rs
git commit -m "feat(trail): add Trail undo log (mark/record/restore_to/clear)"
```

---

### Task 2: Make propagation trail-aware + add `apply_and_propagate`

Trail-thread `propagate_core`/`apply_updates`, recording before every write (incl. the sentinel). Add the in-place `apply_and_propagate`. Keep `probe_assignment` behaviorally identical (copy) by handing `propagate_core` a throwaway trail. Update every other `propagate_core` caller (`problem.rs`, `adapter.rs`, the propagate tests) to pass a throwaway trail. **No search behavior changes** — the search still calls the copy-based `probe_assignment`.

**Files:**
- Modify: `src/propagate.rs` (`propagate_core`, `apply_updates`, `probe_assignment`; add `apply_and_propagate`; tests)
- Modify: `src/problem.rs:72` (`from_network`)
- Modify: `src/adapter.rs:87` (`apply_branch`)

**Interfaces:**
- Consumes: `crate::trail::Trail`
- Produces:
  - `propagate_core(cn: &ConstraintNetwork, doms: &mut [DomainMask], trail: &mut Trail, buffer: &mut SolverBuffer)`
  - `apply_and_propagate(cn: &ConstraintNetwork, doms: &mut [DomainMask], trail: &mut Trail, buffer: &mut SolverBuffer, vars: &[usize], mask: u64, value: u64) -> bool` (returns `contradicted`)
  - `probe_assignment` signature UNCHANGED in this task (still returns `&[DomainMask]`).

- [ ] **Step 1: Add the trail import and update `apply_updates` to record before writing**

In `src/propagate.rs`, add to the top imports:
```rust
use crate::trail::Trail;
```

Replace `apply_updates` (currently `src/propagate.rs:62-96`) with the trail-aware version (note the new `trail` parameter and the `trail.record` before the write):
```rust
#[inline]
fn apply_updates(
    doms: &mut [DomainMask],
    cn: &ConstraintNetwork,
    var_axes: &[usize],
    valid_or: u32,
    valid_and: u32,
    trail: &mut Trail,
    queue: &mut Vec<usize>,
    in_queue: &mut [bool],
) {
    for (i, &var_id) in var_axes.iter().enumerate() {
        let old = doms[var_id];
        if old == DomainMask::D0 || old == DomainMask::D1 {
            continue;
        }
        let bit = 1u32 << i;
        let can_be_1 = valid_or & bit != 0;
        let must_be_1 = valid_and & bit != 0;
        let new_dom = if must_be_1 {
            DomainMask::D1
        } else if can_be_1 {
            DomainMask::BOTH
        } else {
            DomainMask::D0
        };
        debug_assert!(
            new_dom != DomainMask::NONE,
            "apply_updates must never narrow to NONE"
        );
        if new_dom != old {
            trail.record(var_id, old);
            doms[var_id] = new_dom;
            enqueue_neighbors(queue, in_queue, &cn.v2t[var_id]);
        }
    }
}
```

- [ ] **Step 2: Update `propagate_core` to take a trail and record the sentinel**

Replace `propagate_core` (currently `src/propagate.rs:144-178`) with:
```rust
/// Drain the worklist seeded in `buffer.queue` / `buffer.in_queue`, recording every
/// domain write on `trail`. On an unsatisfiable tensor, records and sets `doms[0] = NONE`
/// (contradiction sentinel).
pub fn propagate_core(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    trail: &mut Trail,
    buffer: &mut SolverBuffer,
) {
    let mut head = 0usize;
    while head < buffer.queue.len() {
        let tensor_id = buffer.queue[head];
        head += 1;
        buffer.in_queue[tensor_id] = false;

        let tensor = &cn.tensors[tensor_id];
        let (m0, m1) = compute_query_masks(doms, &tensor.var_axes);
        let td = cn.data(tensor);
        let (valid_or, valid_and, found) =
            scan_supports(&td.support, td.support_or, td.support_and, m0, m1);
        if !found {
            trail.record(0, doms[0]);
            doms[0] = DomainMask::NONE;
            for &t in &buffer.queue[head..] {
                buffer.in_queue[t] = false;
            }
            buffer.queue.clear();
            return;
        }
        apply_updates(
            doms,
            cn,
            &tensor.var_axes,
            valid_or,
            valid_and,
            trail,
            &mut buffer.queue,
            &mut buffer.in_queue,
        );
    }
    buffer.queue.clear();
}
```

- [ ] **Step 3: Add `apply_and_propagate` and make `probe_assignment` hand it a throwaway trail**

Replace `probe_assignment` (currently `src/propagate.rs:100-140`) with the in-place primitive PLUS the unchanged-signature wrapper:
```rust
/// Apply the partial assignment (`vars[i]` set to bit `i` of `value` where bit `i` of
/// `mask` is 1) to the SHARED `doms` in place, recording every write on `trail`, then
/// propagate to a GAC fixpoint. Returns whether the result is a contradiction
/// (`doms[0] == NONE`). The caller takes `trail.mark()` BEFORE this call and uses
/// `trail.restore_to(doms, mark)` to undo.
pub fn apply_and_propagate(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    trail: &mut Trail,
    buffer: &mut SolverBuffer,
    vars: &[usize],
    mask: u64,
    value: u64,
) -> bool {
    // Entry cleanup: ensure a clean worklist (matches the old probe_assignment).
    buffer.queue.clear();
    for b in buffer.in_queue.iter_mut() {
        *b = false;
    }
    for (i, &var_id) in vars.iter().enumerate() {
        if (mask >> i) & 1 == 1 {
            let new_dom = if (value >> i) & 1 == 1 {
                DomainMask::D1
            } else {
                DomainMask::D0
            };
            if doms[var_id] != new_dom {
                trail.record(var_id, doms[var_id]);
                doms[var_id] = new_dom;
                for &t_idx in &cn.v2t[var_id] {
                    if !buffer.in_queue[t_idx] {
                        buffer.in_queue[t_idx] = true;
                        buffer.queue.push(t_idx);
                    }
                }
            }
        }
    }
    propagate_core(cn, doms, trail, buffer);
    doms[0] == DomainMask::NONE
}

/// Copy-based probe (kept for examples/tests). Clones `base_doms`, applies + propagates
/// on the copy via a throwaway trail, returns the borrowed scratch result. The search
/// path uses `apply_and_propagate` directly.
pub fn probe_assignment<'b>(
    cn: &ConstraintNetwork,
    buffer: &'b mut SolverBuffer,
    base_doms: &[DomainMask],
    vars: &[usize],
    mask: u64,
    value: u64,
) -> &'b [DomainMask] {
    buffer.scratch_doms.copy_from_slice(base_doms);
    buffer.queue.clear();
    for b in buffer.in_queue.iter_mut() {
        *b = false;
    }
    for (i, &var_id) in vars.iter().enumerate() {
        if (mask >> i) & 1 == 1 {
            let new_dom = if (value >> i) & 1 == 1 {
                DomainMask::D1
            } else {
                DomainMask::D0
            };
            if buffer.scratch_doms[var_id] != new_dom {
                buffer.scratch_doms[var_id] = new_dom;
                for &t_idx in &cn.v2t[var_id] {
                    if !buffer.in_queue[t_idx] {
                        buffer.in_queue[t_idx] = true;
                        buffer.queue.push(t_idx);
                    }
                }
            }
        }
    }
    let mut scratch = std::mem::take(&mut buffer.scratch_doms);
    let mut trail = Trail::new(); // throwaway — the copy is discarded by the caller
    propagate_core(cn, &mut scratch, &mut trail, buffer);
    buffer.scratch_doms = scratch;
    &buffer.scratch_doms
}
```

- [ ] **Step 4: Update the other `propagate_core` callers to pass a throwaway trail**

`src/problem.rs:72` — in `from_network`, replace:
```rust
        crate::propagate::propagate_core(&static_cn, &mut doms, &mut buffer);
```
with:
```rust
        let mut trail = crate::trail::Trail::new();
        crate::propagate::propagate_core(&static_cn, &mut doms, &mut trail, &mut buffer);
```

`src/adapter.rs:87` — in `apply_branch`, replace:
```rust
        propagate_core(&self.cn, &mut doms, &mut buffer);
```
with (add the import note inline):
```rust
        // apply_branch's direct-assign writes above are intentionally UNTRAILED: this
        // method clones `doms` and returns the mutated copy, never restoring, so the
        // throwaway trail only needs to satisfy propagate_core's signature.
        let mut trail = crate::trail::Trail::new();
        propagate_core(&self.cn, &mut doms, &mut trail, &mut buffer);
```

- [ ] **Step 5: Update the propagate.rs unit tests for the new signature + add trail tests**

In `src/propagate.rs` tests, the three `propagate_core(&cn, &mut doms, &mut buf)` calls (currently lines 225, 238, 250) each need a trail. For each, insert before the call:
```rust
        let mut trail = Trail::new();
```
and change the call to:
```rust
        propagate_core(&cn, &mut doms, &mut trail, &mut buf);
```

Add these new tests to the `tests` module in `src/propagate.rs`:
```rust
    #[test]
    fn apply_and_propagate_then_restore_is_identity() {
        let cn = setup_problem(2, vec![vec![0, 1]], vec![vec![false, true, true, true]]);
        let mut doms = vec![DomainMask::BOTH, DomainMask::BOTH];
        let before = doms.clone();
        let mut trail = Trail::new();
        let mut buf = SolverBuffer::new(&cn);
        let mark = trail.mark();
        let contradicted = apply_and_propagate(&cn, &mut doms, &mut trail, &mut buf, &[0], 1, 0);
        assert!(!contradicted);
        assert_eq!(doms[0], DomainMask::D0);
        assert_eq!(doms[1], DomainMask::D1); // forced by (x0 OR x1)
        trail.restore_to(&mut doms, mark);
        assert_eq!(doms, before);
    }

    #[test]
    fn apply_and_propagate_contradiction_restores_parent_fixed_value() {
        // x0 already fixed to D0; forcing x1=0 violates (x0 OR x1) -> contradiction.
        // The sentinel overwrites doms[0]; restore must return it to D0, NOT BOTH.
        let cn = setup_problem(2, vec![vec![0, 1]], vec![vec![false, true, true, true]]);
        let mut doms = vec![DomainMask::D0, DomainMask::BOTH];
        let before = doms.clone();
        let mut trail = Trail::new();
        let mut buf = SolverBuffer::new(&cn);
        let mark = trail.mark();
        let contradicted = apply_and_propagate(&cn, &mut doms, &mut trail, &mut buf, &[1], 1, 0);
        assert!(contradicted);
        assert_eq!(doms[0], DomainMask::NONE);
        trail.restore_to(&mut doms, mark);
        assert_eq!(doms, before); // doms[0] back to D0
        assert!(buf.queue.is_empty());
        assert!(buf.in_queue.iter().all(|&q| !q));
    }

    #[test]
    fn var0_written_then_sentinel_restores_lifo() {
        // T0=(x0 OR x1), T1=(NOT x0). Direct-assign x0=1 writes doms[0]=D1 (trailed),
        // then propagation hits T1 and writes the sentinel doms[0]=NONE (trailed again).
        let cn = setup_problem(
            2,
            vec![vec![0, 1], vec![0]],
            vec![vec![false, true, true, true], vec![true, false]],
        );
        let mut doms = vec![DomainMask::BOTH, DomainMask::BOTH];
        let before = doms.clone();
        let mut trail = Trail::new();
        let mut buf = SolverBuffer::new(&cn);
        let mark = trail.mark();
        let contradicted = apply_and_propagate(&cn, &mut doms, &mut trail, &mut buf, &[0], 1, 1);
        assert!(contradicted);
        trail.restore_to(&mut doms, mark);
        assert_eq!(doms, before); // both writes to var0 undone in reverse
    }
```

- [ ] **Step 6: Run tests**

Run: `cargo test`
Expected: all existing tests pass (behavior unchanged); the 3 new `propagate` tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/propagate.rs src/problem.rs src/adapter.rs
git commit -m "feat(propagate): trail-aware propagate_core + apply_and_propagate (search unchanged)"
```

---

### Task 3: Feasibility loop on the trail (+ plumb `&mut doms`/`&mut trail`)

Thread `&mut doms` and `&mut trail` from `bbsat` down through `bbsat_rec` → `findbest` → `compute_branching_result`, and switch the feasibility loop in `compute_branching_result` to `apply_and_propagate` + restore. The `bbsat_rec` branch loop and DiffLookahead still use the copy-based `probe_assignment` (they receive `&mut doms` but pass a reborrowed `&` / leave their own probe sites for Tasks 4–5). Behavior identical.

**Files:**
- Modify: `src/solver.rs` (`bbsat`, `bbsat_rec` signature + the `findbest` call)
- Modify: `src/selector.rs` (`findbest` signature; pass `&mut`/`&` onward)
- Modify: `src/table.rs` (`compute_branching_result` signature + feasibility loop; tests)

**Interfaces:**
- Consumes: `apply_and_propagate`, `Trail`.
- Produces:
  - `compute_branching_result(cache: &mut RegionCache, cn: &Arc<ConstraintNetwork>, doms: &mut [DomainMask], trail: &mut Trail, buffer: &mut SolverBuffer, var_id: usize, measure: Measure, solver: &BranchSolver) -> (Option<Vec<Clause>>, Vec<usize>)`
  - `Selector::findbest(&self, cache: &mut RegionCache, cn: &Arc<ConstraintNetwork>, doms: &mut [DomainMask], trail: &mut Trail, buffer: &mut SolverBuffer, measure: Measure, solver: &BranchSolver) -> (Option<Vec<Clause>>, Vec<usize>)`
  - `bbsat_rec(ctx: &SearchCtx, cache: &mut RegionCache, stats: &mut Stats, buffer: &mut SolverBuffer, doms: &mut Vec<DomainMask>, trail: &mut Trail) -> Solve`

- [ ] **Step 1: Convert `compute_branching_result`'s feasibility loop**

In `src/table.rs`: add `use crate::propagate::apply_and_propagate;` and `use crate::trail::Trail;` to the imports (keep the existing `use crate::propagate::probe_assignment;` for now — it becomes unused after this task; remove it in this same edit to avoid a warning).

Change the signature (currently `src/table.rs:16-24`) to add `doms: &mut [DomainMask]` and `trail: &mut Trail`:
```rust
pub fn compute_branching_result(
    cache: &mut RegionCache,
    cn: &Arc<ConstraintNetwork>,
    doms: &mut [DomainMask],
    trail: &mut Trail,
    buffer: &mut SolverBuffer,
    var_id: usize,
    measure: Measure,
    solver: &BranchSolver,
) -> (Option<Vec<Clause>>, Vec<usize>) {
```

Replace the feasibility loop (currently `src/table.rs:42-51`):
```rust
    let mut feasible: Vec<u64> = Vec::new();
    for &config in &cached_configs {
        if (config & check_mask) != check_value {
            continue;
        }
        let scratch = probe_assignment(cn, buffer, doms, &region_vars, full_mask, config);
        if scratch[0] != DomainMask::NONE {
            feasible.push(config);
        }
    }
```
with the trail version (mutate shared `doms`, then restore each probe):
```rust
    let mut feasible: Vec<u64> = Vec::new();
    for &config in &cached_configs {
        if (config & check_mask) != check_value {
            continue;
        }
        let mark = trail.mark();
        let contradicted =
            apply_and_propagate(cn, doms, trail, buffer, &region_vars, full_mask, config);
        if !contradicted {
            feasible.push(config);
        }
        trail.restore_to(doms, mark);
    }
```

Note: `doms.to_vec()` at the `RuleProblem::new` call (currently `src/table.rs:96`) is unchanged and now reads the restored node state — correct, because every feasibility probe restored.

- [ ] **Step 2: Update `compute_branching_result` tests for the new signature**

In `src/table.rs` tests, both call sites (currently lines ~125 and ~151) pass `&mut buf`; add `&mut doms`/`&mut trail`. The two tests build `doms` as `let doms = vec![...]`; change those `let doms` to `let mut doms`, add `let mut trail = Trail::new();`, and update the calls. For `branching_result_covers_the_table`:
```rust
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut cache = RegionCache::new(&cn, &doms, 2, 10);
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let before = doms.clone();
        let (clauses, vars) = compute_branching_result(
            &mut cache, &cn, &mut doms, &mut trail, &mut buf, 1,
            Measure::NumUnfixedVars, &BranchSolver::Ip(IPSolver::default()),
        );
        assert_eq!(doms, before, "feasibility probes must leave doms unchanged");
        assert!(buf.queue.is_empty());
        assert!(buf.in_queue.iter().all(|&q| !q));
```
For `no_feasible_config_returns_none`, similarly switch `let cur` to `let mut cur = ...`, add `let mut trail = Trail::new();`, and call `compute_branching_result(&mut cache, &cn, &mut cur, &mut trail, &mut buf, 1, ...)`.

- [ ] **Step 3: Plumb `&mut doms`/`&mut trail` through `findbest`**

In `src/selector.rs`: add `use crate::trail::Trail;`. Change `findbest` (currently `src/selector.rs:149-175`) to:
```rust
    pub fn findbest(
        &self,
        cache: &mut RegionCache,
        cn: &Arc<ConstraintNetwork>,
        doms: &mut [DomainMask],
        trail: &mut Trail,
        buffer: &mut SolverBuffer,
        measure: Measure,
        solver: &BranchSolver,
    ) -> (Option<Vec<Clause>>, Vec<usize>) {
        match *self {
            Selector::MostOccurrence { .. } => {
                let var_id = match select_var_most_occurrence(cn, doms, buffer) {
                    Some(v) => v,
                    None => return (None, Vec::new()),
                };
                compute_branching_result(cache, cn, doms, trail, buffer, var_id, measure, solver)
            }
            Selector::DiffLookahead { pool, .. } => {
                // select_var_difflookahead still uses copy-based probing in this task;
                // it is converted to the trail in Task 4. It takes `&*doms` (read-only here).
                let var_id = match select_var_difflookahead(cn, doms, buffer, pool) {
                    Some(v) => v,
                    None => return (None, Vec::new()),
                };
                compute_branching_result(cache, cn, doms, trail, buffer, var_id, measure, solver)
            }
        }
    }
```
(`select_var_most_occurrence` and `select_var_difflookahead` keep their current `&[DomainMask]` signatures; `&mut [DomainMask]` coerces to `&[DomainMask]` at the call.)

- [ ] **Step 4: Plumb `&mut doms`/`&mut trail` through `bbsat`/`bbsat_rec`**

In `src/solver.rs`: add `use crate::trail::Trail;`.

Replace `bbsat` (currently `src/solver.rs:31-52`) — create the trail and a working `doms`, recurse with `&mut`:
```rust
pub fn bbsat(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
) -> Solve {
    problem.stats.reset();
    let (k, max_tensors) = selector.k_max();
    let mut cache = RegionCache::new(&problem.static_cn, &problem.doms, k, max_tensors);
    let mut doms = problem.doms.clone();
    let mut trail = Trail::new();

    let ctx = SearchCtx {
        cn: &problem.static_cn,
        selector,
        measure,
        solver,
    };
    let stats = &mut problem.stats;
    let buffer = &mut problem.buffer;
    bbsat_rec(&ctx, &mut cache, stats, buffer, &mut doms, &mut trail)
}
```

Change `bbsat_rec`'s signature and its leaf/`findbest` call (currently `src/solver.rs:54-97`). In THIS task the branch loop is left on `probe_assignment` (converted in Task 5). New signature + body down through `findbest`:
```rust
fn bbsat_rec(
    ctx: &SearchCtx,
    cache: &mut RegionCache,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    trail: &mut Trail,
) -> Solve {
    if count_unfixed(doms) == 0 {
        return Solve {
            found: true,
            solution: doms.clone(),
            stats: stats.clone(),
        };
    }

    if is_two_sat(ctx.cn, doms) {
        return match solve_2sat(ctx.cn, doms) {
            Some(sol) => Solve {
                found: true,
                solution: sol,
                stats: stats.clone(),
            },
            None => Solve {
                found: false,
                solution: Vec::new(),
                stats: stats.clone(),
            },
        };
    }

    let (clauses, variables) =
        ctx.selector
            .findbest(cache, ctx.cn, doms, trail, buffer, ctx.measure, ctx.solver);
    let clauses = match clauses {
        Some(c) => c,
        None => {
            return Solve {
                found: false,
                solution: Vec::new(),
                stats: stats.clone(),
            }
        }
    };

    stats.record_branch(clauses.len() as u64);
    for cl in &clauses {
        stats.record_visit();
        let scratch = probe_assignment(ctx.cn, buffer, doms, &variables, cl.mask, cl.val);
        let mut sub = scratch.to_vec();
        let result = bbsat_rec(ctx, cache, stats, buffer, &mut sub, trail);
        if result.found {
            return result;
        }
    }
    Solve {
        found: false,
        solution: Vec::new(),
        stats: stats.clone(),
    }
}
```
This keeps EXACT copy behavior in the interim: `probe_assignment` returns the copy-based scratch, `scratch.to_vec()` makes an independent child vector `sub`, and the recursion descends into `&mut sub` — the parent `doms` is never mutated by the branch loop (no swap needed). The trail is threaded through but the branch loop itself performs no trail ops in this task; the child's feasibility-loop probes are mark/restore-balanced, so the trail returns to its entry length after each child. The branch loop is converted to `apply_and_propagate` + the contradiction guard in Task 5. `count_unfixed`, `is_two_sat`, `solve_2sat` accept `doms: &mut Vec<DomainMask>` via deref coercion to `&[DomainMask]`.

- [ ] **Step 5: Run tests**

Run: `cargo test`
Expected: all tests pass, including the new `compute_branching_result` purity asserts. Behavior unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/table.rs src/selector.rs src/solver.rs
git commit -m "feat(table): feasibility loop on the trail; plumb &mut doms/trail through search"
```

---

### Task 4: DiffLookahead on the trail

Convert `select_var_difflookahead` to mutate shared `doms` via `apply_and_propagate` + restore, with exact two-polarity sequencing (compute each polarity's scalars before restoring; probe BOTH before the failed-literal check). Removes the old `c0`/`c1` scratch aliasing.

**Files:**
- Modify: `src/selector.rs` (`select_var_difflookahead` signature + body; `findbest` DiffLookahead arm; tests)

**Interfaces:**
- Produces: `select_var_difflookahead(cn: &ConstraintNetwork, doms: &mut [DomainMask], trail: &mut Trail, buffer: &mut SolverBuffer, pool: usize) -> Option<usize>`

- [ ] **Step 1: Rewrite `select_var_difflookahead`**

In `src/selector.rs`, add `use crate::propagate::apply_and_propagate;` (replace the `use crate::propagate::probe_assignment;` import — it becomes unused after this task). Replace the probe loop (currently `src/selector.rs:81-122`):
```rust
pub(crate) fn select_var_difflookahead(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    trail: &mut Trail,
    buffer: &mut SolverBuffer,
    pool: usize,
) -> Option<usize> {
    compute_connection_scores(cn, doms, buffer);
    let mut cands: Vec<usize> = (0..doms.len())
        .filter(|&i| !doms[i].is_fixed() && buffer.connection_scores[i] > 0.0)
        .collect();
    if cands.is_empty() {
        return None;
    }
    cands.sort_by(|&a, &b| {
        buffer.connection_scores[b]
            .partial_cmp(&buffer.connection_scores[a])
            .expect("finite scores")
    });
    cands.truncate(pool);

    let mut best = usize::MAX;
    let mut chosen: Option<usize> = None;
    for &u in &cands {
        let m0 = trail.mark();
        let f0 = apply_and_propagate(cn, doms, trail, buffer, &[u], 1, 0);
        let d0 = if f0 { 0 } else { sum_active_degree(cn, doms) }; // read BEFORE restore
        trail.restore_to(doms, m0);

        let m1 = trail.mark();
        let f1 = apply_and_propagate(cn, doms, trail, buffer, &[u], 1, 1);
        let d1 = if f1 { 0 } else { sum_active_degree(cn, doms) }; // read BEFORE restore
        trail.restore_to(doms, m1);

        if f0 || f1 {
            chosen = Some(u); // failed literal => forced; take it now
            break;
        }
        let s = d0.max(d1);
        if s < best {
            best = s;
            chosen = Some(u);
        }
    }
    Some(chosen.unwrap_or(cands[0]))
}
```

- [ ] **Step 2: Update the `findbest` DiffLookahead arm to pass `&mut trail`**

In `src/selector.rs` `findbest`, change the DiffLookahead arm call to:
```rust
                let var_id = match select_var_difflookahead(cn, doms, trail, buffer, pool) {
                    Some(v) => v,
                    None => return (None, Vec::new()),
                };
```
(remove the interim "still uses copy-based probing" comment from Task 3).

- [ ] **Step 3: Update the DiffLookahead test for the new signature**

In `src/selector.rs` tests, `difflookahead_takes_a_failed_literal_immediately` (currently line ~210) calls `select_var_difflookahead(&cn, &doms, &mut buf, 16)`. Change `let doms` to `let mut doms`, add `let mut trail = Trail::new();`, and call:
```rust
        assert_eq!(
            select_var_difflookahead(&cn, &mut doms, &mut trail, &mut buf, 16),
            Some(0)
        );
        // purity: probes leave doms and the worklist untouched
        assert_eq!(doms, vec![DomainMask::BOTH; 5]);
        assert!(buf.queue.is_empty());
        assert!(buf.in_queue.iter().all(|&q| !q));
```

- [ ] **Step 4: Run tests**

Run: `cargo test`
Expected: all pass; DiffLookahead picks the same var (0) with identical failed-literal behavior, and the purity asserts hold.

- [ ] **Step 5: Commit**

```bash
git add src/selector.rs
git commit -m "feat(selector): DiffLookahead probes on the trail (exact two-polarity sequencing)"
```

---

### Task 5: Branch loop on the trail + contradiction guard

Convert the `bbsat_rec` branch loop from copy-then-recurse to mark / `apply_and_propagate` / **guard** / recurse / restore. This is the one intentional behavior change: a contradicted branch is restored and skipped (not recursed), which is sound and preserves verdicts + `Stats`.

**Files:**
- Modify: `src/solver.rs` (`bbsat_rec` branch loop; imports; tests)

**Interfaces:**
- Consumes: `apply_and_propagate`, `Trail`.

- [ ] **Step 1: Replace the interim copy branch loop with the trail + guard**

In `src/solver.rs`: replace `use crate::propagate::probe_assignment;` with `use crate::propagate::apply_and_propagate;`. Replace the branch loop in `bbsat_rec` (the interim swap-based loop from Task 3) with:
```rust
    stats.record_branch(clauses.len() as u64);
    for cl in &clauses {
        stats.record_visit();
        let mark = trail.mark();
        let contradicted =
            apply_and_propagate(ctx.cn, doms, trail, buffer, &variables, cl.mask, cl.val);
        if !contradicted {
            let result = bbsat_rec(ctx, cache, stats, buffer, doms, trail);
            if result.found {
                return result; // doms holds the witness; the success leaf already cloned it
            }
        }
        trail.restore_to(doms, mark);
    }
    Solve {
        found: false,
        solution: Vec::new(),
        stats: stats.clone(),
    }
```

- [ ] **Step 2: Add a contradiction-guard test**

Add to the `tests` module in `src/solver.rs`:
```rust
    #[test]
    fn solves_with_an_immediately_contradictory_branch() {
        // Forces real branching where one polarity of the chosen var contradicts via GAC,
        // so the guard must skip recursion on that branch and still find the solution.
        // (x1∨x2∨x3) ∧ (¬x1∨x2) ∧ (¬x2∨x3) — satisfiable; degree-3 clause forces branching.
        let (s, cn) = solve_cnf("p cnf 3 3\n1 2 3 0\n-1 2 0\n-2 3 0\n");
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test`
Expected: all pass. The existing `solves_a_satisfiable_3sat`, `proves_an_unsatisfiable_3sat`, `solves_a_pure_2sat_via_the_leaf` still hold (same verdicts, same `branching_nodes` / `total_visited_nodes`).

- [ ] **Step 4: Commit**

```bash
git add src/solver.rs
git commit -m "feat(solver): branch loop on the trail + sound contradiction guard"
```

---

### Task 6: Delete `scratch_doms`, convert `probe_assignment` to a Vec wrapper, validate

The search no longer uses `probe_assignment` (only `examples/` and the `adapter.rs` test do). Convert it to a thin wrapper over `apply_and_propagate` returning an owned `Vec`, delete `SolverBuffer::scratch_doms`, fix the examples, and run the full behavior-preservation gate.

**Files:**
- Modify: `src/propagate.rs` (`probe_assignment` → Vec wrapper)
- Modify: `src/problem.rs` (delete `SolverBuffer::scratch_doms`)
- Modify: `examples/phase2_demo.rs`, `examples/phase2_perf.rs`
- Modify: `src/adapter.rs` (the `apply_branch_matches_probe_assignment` test, if needed)

**Interfaces:**
- Produces: `probe_assignment(cn: &ConstraintNetwork, buffer: &mut SolverBuffer, base_doms: &[DomainMask], vars: &[usize], mask: u64, value: u64) -> Vec<DomainMask>`

- [ ] **Step 1: Delete `scratch_doms` from `SolverBuffer`**

In `src/problem.rs`, remove the `scratch_doms` field (line 31) and its initializer (line 42):
```rust
pub struct SolverBuffer {
    pub queue: Vec<usize>,
    pub in_queue: Vec<bool>,
    pub connection_scores: Vec<f64>,
}

impl SolverBuffer {
    pub fn new(cn: &ConstraintNetwork) -> SolverBuffer {
        let n_tensors = cn.tensors.len();
        let n_vars = cn.vars.len();
        SolverBuffer {
            queue: Vec::with_capacity(n_tensors),
            in_queue: vec![false; n_tensors],
            connection_scores: vec![0.0; n_vars],
        }
    }
}
```

- [ ] **Step 2: Rewrite `probe_assignment` as a Vec wrapper**

In `src/propagate.rs`, replace the copy-based `probe_assignment` from Task 2 with:
```rust
/// Copy-based probe (compatibility wrapper for examples/tests): clones `base_doms`,
/// applies + propagates on the copy via a throwaway trail, and returns the owned result.
/// The search path uses `apply_and_propagate` directly.
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

The existing `probe_assignment_propagates_from_base` test (in `src/propagate.rs` tests) now binds an owned `Vec`; `has_contradiction(result)` takes `&[DomainMask]`, so change it to `has_contradiction(&result)` and `result[0]`/`result[1]` index the `Vec` directly (unchanged).

- [ ] **Step 3: Fix the examples for the owned-`Vec` return**

`examples/phase2_demo.rs` (currently lines 58-62): `res` is now `Vec<DomainMask>`. Change the uses to borrow:
```rust
        let res = probe_assignment(&p.static_cn, &mut buf, &p.doms, &[vid], 1, bit);
        if has_contradiction(&res) {
            println!("assume x{var_1based}{label} : CONTRADICTION — this branch is UNSAT");
        } else {
            println!("assume x{var_1based}{label} : {}", fmt(&res));
        }
```

`examples/phase2_perf.rs` (currently line 58-59): `res` is now `Vec<DomainMask>`; `res[v].0` still works. No change needed (the per-iteration `to_vec()` inside the wrapper is exactly the copy cost this demo measures). Update the stale comment on line 3 if desired (optional).

- [ ] **Step 4: Confirm the adapter test still compiles**

`src/adapter.rs` test `apply_branch_matches_probe_assignment` (line ~178): `expected` is now `Vec<DomainMask>`; `assert_eq!(sub.doms, expected)` compares `Vec == Vec` — fine. No change required. Build will confirm.

- [ ] **Step 5: Run the full gate**

Run: `cargo build --examples`
Expected: examples compile.

Run: `cargo test`
Expected: all unit + integration tests pass, including `tests/acceptance.rs` (the 233-instance brute-force harness) and `tests/factoring.rs` (factoring-15 decodes to valid factors).

Run: `cargo test --test acceptance`
Expected: PASS (behavior preserved across all instances).

Run: `cargo test --test factoring`
Expected: PASS (`p * q == 15`).

- [ ] **Step 6: Commit**

```bash
git add src/propagate.rs src/problem.rs src/adapter.rs examples/phase2_demo.rs examples/phase2_perf.rs
git commit -m "refactor: delete SolverBuffer::scratch_doms; probe_assignment is now a copy wrapper"
```

---

## Self-Review

**1. Spec coverage:**
- Trail struct + footgun rules → Task 1 + Global Constraints. ✓
- The one invariant (3 write sites incl. sentinel) → Task 2 Steps 1–2. ✓
- `apply_and_propagate` returns `contradicted` + entry cleanup → Task 2 Step 3. ✓
- Delete `scratch_doms` → Task 6 Step 1. ✓
- `probe_assignment` kept as copy wrapper → Task 6 Step 2. ✓
- Feasibility loop on trail → Task 3. ✓
- DiffLookahead exact sequencing (probe both, scalars before restore) → Task 4. ✓
- Branch loop + contradiction guard (mark-per-iteration) → Task 5. ✓
- ob-core untouched + documented exception → Task 2 Step 4. ✓
- All 8 spec tests covered: round-trip (T2), var-0 duplicate (T2), parent-fixed restore (T2), repeated-probe purity table (T3) + DiffLookahead (T4), branch-loop contradiction skip (T5), acceptance + factoring (T6). ✓

**2. Placeholder scan:** No `TBD`/`add error handling`/`similar to`/prose-only code steps. Every code step has complete code. ✓

**3. Type consistency:** `Trail` API (`new`/`mark`/`record`/`restore_to`/`clear`) consistent T1→T6. `propagate_core`/`apply_and_propagate`/`compute_branching_result`/`findbest`/`select_var_difflookahead`/`bbsat_rec` signatures match between their producing task and consuming tasks. `apply_and_propagate` returns `bool` (`contradicted`) everywhere. `probe_assignment` returns `&[DomainMask]` in T2 (interim) and `Vec<DomainMask>` from T6 — the only callers across that change are examples + the adapter test, both fixed in T6. ✓
```
