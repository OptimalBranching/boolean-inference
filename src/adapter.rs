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

use std::sync::Arc;

use optimal_branching_core::{
    BranchAndReduceProblem, BranchingRuleSolver, BranchingTable, Clause, Error, GreedyMerge,
    IPSolver, LPSolver, Measure as ObMeasure, NaiveBranch, OptimalBranchingResult,
};

use crate::ct::{ct_propagate, RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::{measure_core, Measure};
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::trail::Trail;

/// A clone-cheap view of the SAT problem at one search node, sized to feed
/// `optimal_branching_rule`. Cloning bumps the network/masks `Arc` refcounts and
/// deep-copies only `doms` and the live-row sets `tables`.
#[derive(Clone)]
pub struct RuleProblem {
    pub cn: Arc<ConstraintNetwork>,
    pub doms: Vec<DomainMask>,
    pub masks: Arc<Vec<TableMasks>>,
    pub tables: Vec<RSparseBitSet>,
}

impl RuleProblem {
    pub fn new(
        cn: Arc<ConstraintNetwork>,
        doms: Vec<DomainMask>,
        masks: Arc<Vec<TableMasks>>,
        tables: Vec<RSparseBitSet>,
    ) -> RuleProblem {
        RuleProblem {
            cn,
            doms,
            masks,
            tables,
        }
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

    /// Apply `clause` over `variables` (bit `i` ⇒ `variables[i]`) on a fresh
    /// copy of `(doms, tables)`, run Compact-Table GAC to a fixpoint, and return
    /// the resulting sub-problem. The mutations are trailed against a throwaway
    /// `Trail` that is dropped with the call (the clone is kept, never restored).
    /// On an unsatisfiable branch `ct_propagate` sets `doms[0] = NONE` (the
    /// contradiction sentinel), which the caller detects via the measure /
    /// feasibility check.
    fn apply_branch(&self, clause: &Clause, variables: &[usize]) -> (RuleProblem, f64) {
        let mut doms = self.doms.clone();
        let mut tables = self.tables.clone();
        // A private propagation worklist for this call. Allocating a buffer per
        // `apply_branch` is a known follow-up cost (a pooled buffer would avoid
        // it); correctness and the unified API come first.
        let mut buffer = SolverBuffer::new(&self.cn);
        let mut trail = Trail::new(); // throwaway; entries die with this call
        trail.open();
        for (i, &var_id) in variables.iter().enumerate() {
            if (clause.mask >> i) & 1 == 1 {
                let new_dom = if (clause.val >> i) & 1 == 1 {
                    DomainMask::D1
                } else {
                    DomainMask::D0
                };
                if doms[var_id] != new_dom {
                    trail.record_dom(var_id, doms[var_id]);
                    doms[var_id] = new_dom;
                    for &t_idx in &self.cn.v2t[var_id] {
                        if !buffer.in_queue[t_idx] {
                            buffer.in_queue[t_idx] = true;
                            buffer.queue.push(t_idx);
                        }
                    }
                }
            }
        }
        ct_propagate(
            &self.cn,
            &mut doms,
            &self.masks,
            &mut tables,
            &mut buffer,
            &mut trail,
        );
        (
            RuleProblem {
                cn: Arc::clone(&self.cn),
                doms,
                masks: Arc::clone(&self.masks),
                tables,
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
    use crate::problem::{has_contradiction, SolverBuffer};
    use crate::propagate::probe;
    use crate::trail::Trail;
    use optimal_branching_core::{BranchingRuleSolver, BranchingTable, IPSolver, DNF};

    fn or_chain() -> ConstraintNetwork {
        // (x0∨x1) ∧ (x1∨x2): forces x1=1 ⇒ both satisfied; x0=0 ⇒ x1=1 forced.
        let or2 = vec![false, true, true, true];
        setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![or2.clone(), or2])
    }

    /// Build a fresh `RuleProblem` at `doms` with its own masks/tables (no root
    /// propagation applied — the tables start all-live over the network).
    fn rule_problem(cn: &ConstraintNetwork, doms: Vec<DomainMask>) -> RuleProblem {
        let (masks, tables) = build_tables(cn);
        RuleProblem::new(Arc::new(cn.clone()), doms, Arc::new(masks), tables)
    }

    #[test]
    fn apply_branch_matches_probe() {
        let cn = or_chain();
        let base = vec![DomainMask::BOTH; 3];
        let vars = vec![0usize, 1, 2];
        // Branch: set x0 = 0 (mask bit0, val bit0 = 0).
        let clause = Clause::new(0b001, 0b000);

        let p = rule_problem(&cn, base.clone());
        let (sub, local) = p.apply_branch(&clause, &vars);
        assert_eq!(local, 0.0);

        // Expected via the trailed `probe` over a fresh (doms, tables).
        let (masks, mut tables) = build_tables(&cn);
        let mut doms = base.clone();
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let expected = probe(
            &cn,
            &mut doms,
            &masks,
            &mut tables,
            &mut buf,
            &mut trail,
            &vars,
            clause.mask,
            clause.val,
            |d| d.to_vec(),
        );

        assert_eq!(sub.doms, expected, "apply_branch must equal probe");
        assert!(!has_contradiction(&sub.doms));
        assert_eq!(sub.doms[0], DomainMask::D0);
        assert_eq!(sub.doms[1], DomainMask::D1); // forced by (x0∨x1)
    }

    #[test]
    fn apply_branch_shares_the_network() {
        let cn = or_chain();
        let p = rule_problem(&cn, vec![DomainMask::BOTH; 3]);
        let (sub, _) = p.apply_branch(&Clause::new(0b010, 0b010), &[0, 1, 2]);
        // Cloning the problem must not deep-copy the network.
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
        // End-to-end through the UNIFIED entry point: a BranchingRuleSolver
        // (IPSolver via the blanket impl) computes the rule from the framework's
        // own size_reduction (apply_branch + measure). Proves GreedyMerge will
        // slot into the same call shape.
        let cn = or_chain();
        let p = rule_problem(&cn, vec![DomainMask::BOTH; 3]);
        // Feasible configs of (x0∨x1)∧(x1∨x2) over [x0,x1,x2] (bit i = xi).
        let table = BranchingTable::new(3, vec![vec![2], vec![3], vec![5], vec![6], vec![7]]);
        let vars = vec![0usize, 1, 2];

        let result = IPSolver::default()
            .optimal_branching_rule(&p, &table, &vars, &MeasureAdapter(Measure::NumUnfixedVars))
            .expect("rule");
        assert!(!result.optimal_rule.clauses.is_empty());
        assert!(table.covered_by(&DNF {
            clauses: result.optimal_rule.clauses
        }));
    }
}
