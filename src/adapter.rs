//! Adapter wiring our SAT problem into optimal-branching-core's
//! `BranchAndReduceProblem` / `Measure` contract, so that any
//! `BranchingRuleSolver` — `IPSolver`/`LPSolver` (via the blanket impl) or
//! `GreedyMerge`/`NaiveBranch` — can produce the branching rule through one
//! unified entry point: `BranchingRuleSolver::optimal_branching_rule`.
//!
//! We deliberately do NOT adopt ob-core's full `branch_and_reduce` driver: it
//! returns a `ResultAlgebra` value (a problem size), not a satisfying
//! assignment, so it cannot yield a SAT witness. We keep our own `bbsat_rec`
//! recursion and use the framework only to compute the rule.
//!
//! `RuleProblem` shares the network behind an `Arc`: cloning a problem (which
//! `BranchAndReduceProblem` requires, and which `apply_branch` does per branch)
//! bumps the refcount and copies only `doms`, never the network — this is what
//! makes adopting the framework cheap instead of the per-node network clone the
//! Phase 3 decoupling was introduced to avoid.

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
    /// Per-node memoization of `apply_branch` results keyed by (clause.mask,
    /// clause.val). `variables` is constant across a node's `optimal_rule` call,
    /// so (mask,val) uniquely identifies a branch from the fixed node base;
    /// GreedyMerge's O(rows²) merge loop re-evaluates many clauses, so caching the
    /// propagated domains avoids re-propagating them. Cleared per node in
    /// `with_measure_scratch`. Measured ~8% wall-clock on factoring_22x22 VE10.
    cache: std::collections::HashMap<(u64, u64), Vec<DomainMask>>,
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
        s.cache.clear();
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

/// A clone-cheap view of the SAT problem at one search node, sized to feed
/// `optimal_branching_rule`. Cloning bumps the network `Arc` refcount and
/// deep-copies only `doms`. CT tables are shared via `masks` so `apply_branch`
/// can propagate with CT via the thread-local measure scratch.
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
    /// `optimal_branching_rule` never reads `LocalValue` — it only uses
    /// `size_reduction` = `measure(before) − measure(after)`. `LocalValue`
    /// matters solely to the full `branch_and_reduce` driver, which we do not
    /// use. We report `0.0` honestly rather than fabricate a count.
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
        let key = (clause.mask, clause.val);
        let snapshot = MEASURE_SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            if let Some(cached) = s.cache.get(&key) {
                return cached.clone();
            }
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
            s.cache.insert(key, snap.clone());
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

/// Exposes our `measure_core` as ob-core's `Measure` trait over `RuleProblem`.
/// `Output = f64` matches `measure_core`'s return type and satisfies the
/// framework's `From<u32> + Into<f64>` bounds.
pub struct MeasureAdapter(pub Measure);

impl ObMeasure<RuleProblem> for MeasureAdapter {
    type Output = f64;

    fn measure(&self, problem: &RuleProblem) -> f64 {
        measure_core(&problem.cn, &problem.doms, self.0)
    }

    /// Not consulted by `optimal_branching_rule` nor by our path.
    fn delta(&self, _problem: &RuleProblem, _removed: &[usize]) -> f64 {
        0.0
    }
}

/// A runtime-selectable branching-rule solver.
///
/// ob-core's `BranchingRuleSolver::optimal_branching_rule<P, M>` is a generic
/// method, which makes the trait non-object-safe — there is no
/// `Box<dyn BranchingRuleSolver>`. So we enumerate the closed set of solvers and
/// dispatch with a `match`, the same enum-dispatch pattern as `Selector` and
/// `Measure`. `IPSolver`/`LPSolver` reach the entry point via ob-core's blanket
/// `impl<S: SetCoverSolver> BranchingRuleSolver for S`; `GreedyMerge`/
/// `NaiveBranch` implement it directly.
pub enum BranchSolver {
    Ip(IPSolver),
    Lp(LPSolver),
    Greedy(GreedyMerge),
    Naive(NaiveBranch),
}

impl BranchSolver {
    /// Compute the branching rule for `table` over `variables` through the one
    /// unified entry point, regardless of which concrete solver is selected.
    pub fn optimal_rule(
        &self,
        problem: &RuleProblem,
        table: &BranchingTable,
        variables: &[usize],
        measure: &MeasureAdapter,
    ) -> Result<OptimalBranchingResult, Error> {
        match self {
            BranchSolver::Ip(s) => s.optimal_branching_rule(problem, table, variables, measure),
            BranchSolver::Lp(s) => s.optimal_branching_rule(problem, table, variables, measure),
            BranchSolver::Greedy(s) => s.optimal_branching_rule(problem, table, variables, measure),
            BranchSolver::Naive(s) => s.optimal_branching_rule(problem, table, variables, measure),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ct::build_tables;
    use crate::network::setup_problem;
    use crate::problem::SolverBuffer;
    use crate::propagate::probe;
    use crate::trail::Trail;
    use optimal_branching_core::{BranchingRuleSolver, BranchingTable, IPSolver, DNF};

    fn or_chain() -> ConstraintNetwork {
        // (x0∨x1) ∧ (x1∨x2): forces x1=1 ⇒ both satisfied; x0=0 ⇒ x1=1 forced.
        let or2 = vec![false, true, true, true];
        setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![or2.clone(), or2])
    }

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

    #[test]
    fn is_empty_tracks_unfixed_vars() {
        let cn = or_chain();
        let unfixed = rule_problem(&cn, vec![DomainMask::BOTH; 3]);
        assert!(!unfixed.is_empty());
        let fixed = rule_problem(&cn, vec![DomainMask::D1, DomainMask::D0, DomainMask::D1]);
        assert!(fixed.is_empty());
    }

    #[test]
    fn measure_adapter_matches_measure_core() {
        let cn = or_chain();
        let doms = vec![DomainMask::BOTH, DomainMask::D1, DomainMask::BOTH];
        let p = rule_problem(&cn, doms.clone());
        for m in [
            Measure::NumUnfixedVars,
            Measure::NumUnfixedTensors,
            Measure::NumHardTensors,
        ] {
            let adapter = MeasureAdapter(m);
            assert_eq!(
                ObMeasure::measure(&adapter, &p),
                measure_core(&cn, &doms, m)
            );
        }
    }

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
}
