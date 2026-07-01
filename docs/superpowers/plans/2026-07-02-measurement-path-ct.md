# Measurement-Path CT Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `adapter.rs::apply_branch` (the branching-rule measurement path,
~35% self-time) from the slow linear `propagate_core_rescan` to fast CT, by
lending the solver's live CT state to a thread-local scratch around `optimal_rule`
— keeping node counts bit-identical (19761 / 45322 on `factoring_22x22`).

**Architecture:** A thread-local `MeasureScratch` holds the live `doms`/`tables`/
`buffer`/`trail`, swapped in (O(1) `Vec`-header swaps) by `compute_branching_result`
immediately around the `optimal_rule` call and swapped back after. `apply_branch`
(`&self`, called only single-level from root by ob-core) reaches the scratch,
does `open → apply literals → ct_propagate → snapshot doms → restore`, and returns
the snapshot as the sub-problem's `doms`. CT and rescan reach the same GAC
fixpoint, so measures, the chosen rule, and node counts are unchanged.

**Tech Stack:** Rust; CT substrate (`ct.rs`, `trail.rs`); ob-core
`BranchAndReduceProblem`; `cargo test`, `runscribe`.

## Global Constraints

- **Behavior-preserving:** `tests/ct_acceptance.rs` stays green —
  `branching_nodes == 19761`, `total_visited_nodes == 45322`. Node counts must
  not change.
- **apply_branch feasibility (proven):** ob-core calls `apply_branch` only
  single-level on the root problem, measures the sub-problem immediately, and
  drops it before the next call. The scratch is restored to node base after every
  `apply_branch`.
- **Swap safety:** the swap-in and swap-back must bracket `optimal_rule` with no
  early return between them (use the `with_measure_scratch` scoped helper).
- **CT epoch discipline:** one `trail.open()` per `apply_branch`; `restore_to`
  reverts it. Buffer drained per call (`ct_propagate` guarantees it); assert.
- The live `tables`/`buffer` swapped back out must be byte-identical to what was
  handed in; the `trail` may differ only by advanced (monotonic) `epoch`.

---

### Task 1: Atomic measurement-path CT conversion

**Files:**
- Modify: `src/problem.rs` (add `impl Default for SolverBuffer`)
- Modify: `src/trail.rs` (add `impl Default for Trail`)
- Modify: `src/adapter.rs` (`MeasureScratch` + thread-local + `with_measure_scratch`;
  add `masks` to `RuleProblem`; rewrite `apply_branch`; update adapter unit tests)
- Modify: `src/table.rs` (pass `masks` to `RuleProblem::new`; wrap `optimal_rule`
  in `with_measure_scratch`)
- Test: `tests/ct_acceptance.rs` (unchanged — the golden gate)

**Interfaces:**
- Consumes: `ct::{ct_propagate, enqueue_var_change, RSparseBitSet, TableMasks}`,
  `trail::Trail`, `problem::SolverBuffer`, `domain::DomainMask`. `ct_propagate`
  sets `doms[0]=NONE` (trailed) on contradiction and drains the buffer clean.
- Produces: `RuleProblem { cn: Arc<ConstraintNetwork>, masks: Arc<Vec<TableMasks>>,
  doms: Vec<DomainMask> }`, `RuleProblem::new(cn, masks, doms)`,
  `pub(crate) fn with_measure_scratch<R>(doms: &[DomainMask], tables: &mut
  Vec<RSparseBitSet>, buffer: &mut SolverBuffer, trail: &mut Trail, f: impl
  FnOnce() -> R) -> R`.

- [ ] **Step 1: Add empty constructors**

In `src/problem.rs`, after the `SolverBuffer` struct + `impl SolverBuffer`, add
(field list must match the struct — `queue: Vec<usize>`, `in_queue: Vec<bool>`,
`mask_scratch: Vec<u64>`, `dirty: Vec<u32>`):

```rust
impl Default for SolverBuffer {
    /// All-empty placeholder used only as the swap partner in the measure-scratch
    /// (never used for propagation until a real, sized buffer is swapped in).
    fn default() -> Self {
        SolverBuffer {
            queue: Vec::new(),
            in_queue: Vec::new(),
            mask_scratch: Vec::new(),
            dirty: Vec::new(),
        }
    }
}
```

In `src/trail.rs`, after `impl Trail`, add:

```rust
impl Default for Trail {
    fn default() -> Self {
        Trail::new()
    }
}
```

- [ ] **Step 2: Build (scaffolding compiles)**

Run: `cargo build 2>&1 | tail -5`
Expected: builds (warnings about unused `Default` impls are acceptable until they
are used in later steps).

- [ ] **Step 3: Add the scratch, thread-local, and scoped helper to `adapter.rs`**

In `src/adapter.rs`, update the imports and add the scratch machinery. Replace the
import block near the top:

```rust
use std::cell::RefCell;
use std::sync::Arc;

use optimal_branching_core::{
    BranchAndReduceProblem, BranchingRuleSolver, BranchingTable, Clause, Error, GreedyMerge,
    IPSolver, LPSolver, Measure as ObMeasure, NaiveBranch, OptimalBranchingResult,
};

use crate::ct::{ct_propagate, enqueue_var_change, RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::{measure_core, Measure};
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::trail::Trail;
```

(Remove the old `use crate::propagate::propagate_core_rescan;` — CT replaces it.)

Add, after the imports:

```rust
/// Per-node CT store for the branching-rule measurement path. The live
/// `tables`/`buffer`/`trail` are swapped in around `optimal_rule` (see
/// `with_measure_scratch`); `apply_branch` reaches them here. `doms` is a working
/// copy of the node base, restored to base after every `apply_branch`.
#[derive(Default)]
struct MeasureScratch {
    doms: Vec<DomainMask>,
    tables: Vec<RSparseBitSet>,
    buffer: SolverBuffer,
    trail: Trail,
}

thread_local! {
    static MEASURE_SCRATCH: RefCell<MeasureScratch> = RefCell::new(MeasureScratch::default());
}

/// Lend the live CT state to the thread-local measure scratch for the duration of
/// `f` (which drives `optimal_rule`, hence `apply_branch`), then take it back.
/// `mem::swap` moves the container headers (O(1), no element copy). Every
/// `apply_branch` restores the scratch to base, so on return `tables`/`buffer`
/// are byte-identical and `trail` differs only by advanced epoch. Keep the two
/// swap blocks bracketing `f` with no early return between them.
pub(crate) fn with_measure_scratch<R>(
    doms: &[DomainMask],
    tables: &mut Vec<RSparseBitSet>,
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    f: impl FnOnce() -> R,
) -> R {
    MEASURE_SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        std::mem::swap(&mut s.tables, tables);
        std::mem::swap(&mut s.buffer, buffer);
        std::mem::swap(&mut s.trail, trail);
        s.doms.clear();
        s.doms.extend_from_slice(doms);
    });
    let r = f();
    MEASURE_SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        std::mem::swap(&mut s.tables, tables);
        std::mem::swap(&mut s.buffer, buffer);
        std::mem::swap(&mut s.trail, trail);
    });
    r
}
```

- [ ] **Step 4: Add `masks` to `RuleProblem` and rewrite `apply_branch`**

Replace the `RuleProblem` struct, its `new`, and the `apply_branch` impl:

```rust
#[derive(Clone)]
pub struct RuleProblem {
    pub cn: Arc<ConstraintNetwork>,
    pub masks: Arc<Vec<TableMasks>>,
    pub doms: Vec<DomainMask>,
}

impl RuleProblem {
    pub fn new(
        cn: Arc<ConstraintNetwork>,
        masks: Arc<Vec<TableMasks>>,
        doms: Vec<DomainMask>,
    ) -> RuleProblem {
        RuleProblem { cn, masks, doms }
    }
}

impl BranchAndReduceProblem for RuleProblem {
    type LocalValue = f64;

    fn is_empty(&self) -> bool {
        self.doms.iter().all(|d| d.is_fixed())
    }

    /// Apply `clause` over `variables` on the thread-local measure scratch (the
    /// node's live CT store, at base), run CT to a fixpoint, snapshot the
    /// resulting domains as the returned sub-problem, and restore the scratch to
    /// base. Behavior-identical to the old clone-doms + rescan path (CT and rescan
    /// reach the same GAC fixpoint) but ~2-3x faster and allocation-free.
    /// Precondition (ob-core guarantee): called only single-level from the root,
    /// with the scratch primed by `with_measure_scratch`.
    fn apply_branch(&self, clause: &Clause, variables: &[usize]) -> (RuleProblem, f64) {
        let snapshot = MEASURE_SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            s.trail.open();
            let m = s.trail.mark();
            debug_assert!(s.buffer.queue.is_empty(), "measure scratch buffer must be drained");
            for (i, &var_id) in variables.iter().enumerate() {
                if (clause.mask >> i) & 1 == 1 {
                    let new_dom = if (clause.val >> i) & 1 == 1 {
                        DomainMask::D1
                    } else {
                        DomainMask::D0
                    };
                    if s.doms[var_id] != new_dom {
                        s.trail.record_dom(var_id, s.doms[var_id]);
                        s.doms[var_id] = new_dom;
                        enqueue_var_change(&self.cn, &mut s.buffer, var_id);
                    }
                }
            }
            ct_propagate(
                &self.cn,
                &mut s.doms,
                &self.masks,
                &mut s.tables,
                &mut s.buffer,
                &mut s.trail,
            );
            let snap = s.doms.clone();
            s.trail.restore_to(m, &mut s.doms, &mut s.tables);
            snap
        });
        (
            RuleProblem {
                cn: Arc::clone(&self.cn),
                masks: Arc::clone(&self.masks),
                doms: snapshot,
            },
            0.0,
        )
    }
}
```

- [ ] **Step 5: Update `table.rs` — pass `masks`, wrap `optimal_rule`**

In `src/table.rs::compute_branching_result`, replace the rule-solving block:

```rust
    let problem = RuleProblem::new(Arc::clone(cn), doms.to_vec());
    let result = solver
        .optimal_rule(&problem, &table, &unfixed_vars, &MeasureAdapter(measure))
        .expect("optimal_branching_rule failed on a non-empty branching table");
    (Some(result.optimal_rule.clauses), unfixed_vars)
```

with (note the `use crate::adapter::with_measure_scratch;` import at the top of the
file, alongside the existing `RuleProblem`/`MeasureAdapter` imports):

```rust
    let problem = RuleProblem::new(Arc::clone(cn), Arc::clone(masks), doms.to_vec());
    // Lend the live CT state to the measure scratch so apply_branch propagates
    // with CT instead of the linear rescan. apply_branch restores it to base
    // after every candidate, so `doms`/`tables`/`buffer`/`trail` are unchanged here.
    let result = with_measure_scratch(doms, tables, buffer, trail, || {
        solver.optimal_rule(&problem, &table, &unfixed_vars, &MeasureAdapter(measure))
    })
    .expect("optimal_branching_rule failed on a non-empty branching table");
    (Some(result.optimal_rule.clauses), unfixed_vars)
```

- [ ] **Step 6: Update the adapter unit tests to prime the scratch**

`apply_branch` now reads the thread-local scratch, so the adapter tests must prime
it. In `src/adapter.rs` `mod tests`, update the `rule_problem` helper and the
`apply_branch_matches_probe` / `apply_branch_shares_the_network` tests:

```rust
    use crate::ct::build_tables;
    use crate::propagate::probe;

    /// Build a `RuleProblem` at `doms` with CT masks (apply_branch uses CT scratch).
    fn rule_problem(cn: &ConstraintNetwork, doms: Vec<DomainMask>) -> RuleProblem {
        let (masks, _tables) = build_tables(cn);
        RuleProblem::new(Arc::new(cn.clone()), Arc::new(masks), doms)
    }

    #[test]
    fn apply_branch_matches_probe() {
        let cn = or_chain();
        let base = vec![DomainMask::BOTH; 3];
        let vars = vec![0usize, 1, 2];
        let clause = Clause::new(0b001, 0b000); // x0 = 0

        let (masks, mut tables) = build_tables(&cn);
        let masks = Arc::new(masks);
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let p = RuleProblem::new(Arc::new(cn.clone()), Arc::clone(&masks), base.clone());

        // Prime the measure scratch with the base state, then apply_branch.
        let (sub, local) =
            with_measure_scratch(&base, &mut tables, &mut buf, &mut trail, || {
                p.apply_branch(&clause, &vars)
            });
        assert_eq!(local, 0.0);

        // Expected via the trailed CT probe over a fresh (doms, tables).
        let (masks2, mut tables2) = build_tables(&cn);
        let mut doms2 = base.clone();
        let mut buf2 = SolverBuffer::new(&cn);
        let mut trail2 = Trail::new();
        let expected = probe(
            &cn, &mut doms2, &masks2, &mut tables2, &mut buf2, &mut trail2,
            &vars, clause.mask, clause.val, |d| d.to_vec(),
        );
        assert_eq!(sub.doms, expected, "apply_branch must equal probe");
        assert_eq!(sub.doms[0], DomainMask::D0);
        assert_eq!(sub.doms[1], DomainMask::D1); // forced by (x0∨x1)

        // The live tables/buffer are swapped back at base: a fresh probe still works.
        assert!(buf.queue.is_empty());
    }

    #[test]
    fn apply_branch_shares_the_network() {
        let cn = or_chain();
        let base = vec![DomainMask::BOTH; 3];
        let (masks, mut tables) = build_tables(&cn);
        let masks = Arc::new(masks);
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let p = RuleProblem::new(Arc::new(cn.clone()), Arc::clone(&masks), base.clone());
        let (sub, _) = with_measure_scratch(&base, &mut tables, &mut buf, &mut trail, || {
            p.apply_branch(&Clause::new(0b010, 0b010), &[0, 1, 2])
        });
        assert!(Arc::ptr_eq(&p.cn, &sub.cn));
    }
```

The `is_empty_tracks_unfixed_vars`, `measure_adapter_matches_measure_core`, and
`optimal_branching_rule_via_ipsolver_covers_table` tests build a `RuleProblem` via
`rule_problem`; the IPSolver end-to-end test additionally calls
`optimal_branching_rule` which invokes `apply_branch`, so wrap ITS solve in
`with_measure_scratch` too:

```rust
    #[test]
    fn optimal_branching_rule_via_ipsolver_covers_table() {
        let cn = or_chain();
        let base = vec![DomainMask::BOTH; 3];
        let (masks, mut tables) = build_tables(&cn);
        let masks = Arc::new(masks);
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let p = RuleProblem::new(Arc::new(cn.clone()), Arc::clone(&masks), base.clone());
        let table = BranchingTable::new(3, vec![vec![2], vec![3], vec![5], vec![6], vec![7]]);
        let vars = vec![0usize, 1, 2];
        let result = with_measure_scratch(&base, &mut tables, &mut buf, &mut trail, || {
            IPSolver::default()
                .optimal_branching_rule(&p, &table, &vars, &MeasureAdapter(Measure::NumUnfixedVars))
                .expect("rule")
        });
        assert!(!result.optimal_rule.clauses.is_empty());
        assert!(table.covered_by(&DNF { clauses: result.optimal_rule.clauses }));
    }
```

- [ ] **Step 7: Build warning-clean**

Run: `cargo build --release 2>&1 | tail -20`
Expected: builds; no `unused import` (`propagate_core_rescan` removed from adapter)
and no unused-variable warnings. `propagate_core_rescan` itself stays defined in
`propagate.rs` (still used by that module's tests as the differential oracle).

- [ ] **Step 8: Run adapter unit tests**

Run: `cargo test --lib adapter`
Expected: PASS (all adapter tests, including `apply_branch_matches_probe`).

- [ ] **Step 9: Golden node-identity gate**

Run: `cargo test --release --test ct_acceptance`
Expected: PASS — `branching_nodes == 19761`, `total_visited_nodes == 45322`. If it
FAILS, the conversion is not behavior-preserving — STOP and report BLOCKED with
the actual numbers; do not adjust the engine to force them.

- [ ] **Step 10: Full suite**

Run: `cargo test`
Expected: PASS (lib + integration, incl. `factoring_15`, the CT differential
oracle, and the prefix-sharing tests).

- [ ] **Step 11: Commit**

```bash
git add src/adapter.rs src/table.rs src/problem.rs src/trail.rs
git commit -m "perf(adapter): CT measurement path via swapped-in thread-local scratch"
```

---

### Task 2: Measure the speedup (runscribe)

**Files:** none (measurement only). Uses the `solve_circuit` example and
`runscribe`, mirroring the prefix-sharing A/B.

**Interfaces:**
- Consumes: `cargo run --release --example solve_circuit -- tests/fixtures/factoring_22x22.circuitsat.json 22 difflook 10`
  prints `... branching_nodes=19761 visited=45322 time=<T>s`.

- [ ] **Step 1: Open a hypothesis under the perf goal**

Run:
```bash
runscribe hyp new A --from A6 -m "measurement-path CT: same nodes, lower wall time (thread-local scratch)"
```
Expected: allocates `A7`.

- [ ] **Step 2: Build both sides and A/B them**

Build and time the current (after) build:
```bash
cargo build --release --example solve_circuit
for i in 1 2 3; do runscribe run --hyp A7 --tag after-ve10 -- \
  ./target/release/examples/solve_circuit tests/fixtures/factoring_22x22.circuitsat.json 22 difflook 10; done
```
Then the before build (the commit prior to Task 1):
```bash
git checkout -q <Task-1-parent-commit>
cargo build --release --example solve_circuit
for i in 1 2 3; do runscribe run --hyp A7 --tag before-ve10 -- \
  ./target/release/examples/solve_circuit tests/fixtures/factoring_22x22.circuitsat.json 22 difflook 10; done
git checkout -q framework-apply-branch
cargo build --release --example solve_circuit
```
Expected: both print `branching_nodes=19761 visited=45322` (identical); `after`
time lower than `before`.

- [ ] **Step 3: Record the observation**

Fill the A7 `## Observation` with before/after wall-clock and the per-run node
counts (must be 19761/45322 on both). If node counts differ, STOP — regression.
If `after` is a wash or slower, record it honestly — that triggers the spec's
fallback discussion (the swap should be near-free, so a regression would indicate
CT-per-candidate is not cheaper than rescan here, an unexpected result worth
investigating before further work).

Run: `runscribe index`
Expected: tables rebuilt.

- [ ] **Step 4: Commit any ledger/notes**

```bash
git add -A && git commit -m "chore(perf): runscribe A/B of measurement-path CT conversion" || echo "nothing to commit"
```

---

## Self-Review

**1. Spec coverage:**
- Thread-local scratch + swap-in/out → Task 1 Step 3 (`with_measure_scratch`). ✅
- `apply_branch` CT rewrite (open/apply/ct_propagate/snapshot/restore) → Step 4. ✅
- `masks` on `RuleProblem` → Step 4; passed from `table.rs` → Step 5. ✅
- `Default` for `SolverBuffer`/`Trail` (scratch init + swap partner) → Step 1. ✅
- Swap safety (adjacent, no early return) → `with_measure_scratch` encapsulates. ✅
- Buffer-drained debug_assert → Step 4. ✅
- Golden gate + differential (`apply_branch_matches_probe`) + full suite → Steps
  8-10. ✅
- Perf A/B, nodes identical → Task 2. ✅
- Out-of-scope (difflookahead, memoization, evalvar) → not planned. ✅

**2. Placeholder scan:** the only literal to fill is `<Task-1-parent-commit>` in
Task 2 Step 2 — the controller records this commit before dispatching Task 1 (it
is `git rev-parse HEAD` at plan start). No code placeholders.

**3. Type consistency:** `RuleProblem::new(cn, masks, doms)` is used identically in
Step 4 (def), Step 5 (`table.rs`), and Step 6 (tests). `with_measure_scratch(doms:
&[DomainMask], tables: &mut Vec<RSparseBitSet>, buffer: &mut SolverBuffer, trail:
&mut Trail, f)` matches at all call sites (table.rs + 3 tests). `masks` is
`Arc<Vec<TableMasks>>` everywhere; `table.rs` holds `masks: &Arc<Vec<TableMasks>>`
so `Arc::clone(masks)` is correct.
