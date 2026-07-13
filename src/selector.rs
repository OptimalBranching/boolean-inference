use std::sync::Arc;

use optimal_branching_core::Clause;

use crate::adapter::BranchSolver;
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::table::compute_branching_result;
use crate::trail::Trail;
use crate::util::{active_tensors, is_entailed};

/// Fill `buffer.occurrence_scores`: each active NON-ENTAILED tensor adds 1 to
/// each of its unfixed variables — plain occurrence counting, no structural
/// weighting. Entailed tensors (every combo of their unfixed vars satisfying)
/// constrain nothing and are skipped, so a var scores > 0 iff some constraint
/// still bites it; score-0 unfixed vars are FREE (any value extends) and are
/// batch-fixed by `findbest`'s fallback instead of branched.
pub(crate) fn compute_occurrence_scores(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
    masks: &[TableMasks],
) {
    for s in buffer.occurrence_scores.iter_mut() {
        *s = 0.0;
    }
    for tid in active_tensors(cn, doms) {
        if is_entailed(cn, tid, doms, masks) {
            continue;
        }
        for &v in &cn.tensors[tid].var_axes {
            if !doms[v].is_fixed() {
                buffer.occurrence_scores[v] += 1.0;
            }
        }
    }
}

/// Highest-occurrence unfixed variable in `scope` (first one at the max).
/// `None` means no scope var is constrained by a non-entailed tensor — all of
/// them are free; the caller batch-fixes them without branching.
pub(crate) fn select_var_most_occurrence(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
    scope: &[usize],
    masks: &[TableMasks],
) -> Option<usize> {
    compute_occurrence_scores(cn, doms, buffer, masks);
    let mut max_score = 0.0f64;
    let mut var_id: Option<usize> = None;
    for &i in scope {
        if doms[i].is_fixed() {
            continue;
        }
        if buffer.occurrence_scores[i] > max_score {
            max_score = buffer.occurrence_scores[i];
            var_id = Some(i);
        }
    }
    var_id
}

/// Pool size for the failed-literal reduce step: the top-`FAILED_LITERAL_POOL`
/// occurrence-scored unfixed vars are probed per node (each probe is one CT
/// propagation). A documented trade-off knob, not an instance-fitted constant —
/// bigger pool finds more forced literals at more probing cost; its payoff is
/// the forced-move hit rate, the feature the scaling sweep (goal C) measures.
/// Matches the old DiffLookahead pool so probing cost is comparable.
pub(crate) const FAILED_LITERAL_POOL: usize = 16;

/// The top-`pool` unfixed vars by occurrence score (highest first; ties keep
/// ascending var-id order). Vars scoring 0 are FREE (all incident tensors
/// entailed) and excluded — probing them is pointless (any value extends).
/// This is the candidate set the failed-literal reduce (`propagate.rs`) probes;
/// occurrence focuses the bounded probe budget on the most-constrained vars.
pub(crate) fn occurrence_pool(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
    masks: &[TableMasks],
    pool: usize,
) -> Vec<usize> {
    compute_occurrence_scores(cn, doms, buffer, masks);
    let mut cands: Vec<usize> = (0..doms.len())
        .filter(|&i| !doms[i].is_fixed() && buffer.occurrence_scores[i] > 0.0)
        .collect();
    cands.sort_by(|&a, &b| {
        buffer.occurrence_scores[b]
            .partial_cmp(&buffer.occurrence_scores[a])
            .expect("finite scores")
    });
    cands.truncate(pool);
    cands
}

#[derive(Clone, Copy, Debug)]
pub enum Selector {
    MostOccurrence {
        /// Row budget for `grow_region` — the branching-table size cap.
        max_rows: usize,
    },
    /// CONTROL ARM: plain 2-way variable branching ({v=0, v=1}) with the same
    /// variable choice as `MostOccurrence` but NO region machinery — no growth,
    /// no feasibility probe, no closed-region shortcut, no rule solver. Isolates
    /// what the branching layer earns over the shared reductions (GAC,
    /// domination, failed-literal, components). Runscribe goal C.
    BinaryOccurrence,
}

impl Selector {
    /// The row budget regions are grown with (the binary control arm grows none).
    pub fn max_rows(&self) -> usize {
        match *self {
            Selector::MostOccurrence { max_rows } => max_rows,
            Selector::BinaryOccurrence => 0,
        }
    }

    /// Pick a focus variable inside `scope` (the caller's connected component;
    /// pass all vars when undecomposed) and compute its branching rule from a
    /// region grown fresh at the current `doms`. Returns the rule's clauses
    /// (or `None` for a no-op), the variables they range over, and the rule's
    /// branching factor γ (see `compute_branching_result`; `f64::NAN` for the
    /// `BinaryOccurrence` control arm, which never feeds the cutoff signal
    /// analysis). Port of `selector.jl::findbest`.
    #[allow(clippy::too_many_arguments)]
    pub fn findbest(
        &self,
        cn: &Arc<ConstraintNetwork>,
        doms: &mut [DomainMask],
        buffer: &mut SolverBuffer,
        measure: Measure,
        solver: &BranchSolver,
        masks: &Arc<Vec<TableMasks>>,
        tables: &mut Vec<RSparseBitSet>,
        trail: &mut Trail,
        scope: &[usize],
    ) -> (Option<Vec<Clause>>, Vec<usize>, f64) {
        let var_id = select_var_most_occurrence(cn, doms, buffer, scope, masks);
        let var_id = match var_id {
            Some(v) => v,
            None => {
                // Every unfixed scope var is FREE: all its tensors are
                // entailed, so any value extends any solution of the rest.
                // Fix them in ONE single-branch clause — no alternatives are
                // needed for completeness, and entailment is preserved under
                // slicing so batch-fixing is sound. A clause holds at most 64
                // literals (u64 mask); any surplus is fixed at the next node.
                //
                // In the solver this arm is shadowed by domination (a free
                // var's full tables flip everywhere, so `dominate_fixpoint`
                // fixes it before findbest runs); it stays as the selector's
                // own completeness guarantee — findbest must not depend on
                // which reductions ran before it. Exercised directly by
                // `free_vars_are_batch_fixed_in_one_branch`.
                let vars: Vec<usize> = scope
                    .iter()
                    .copied()
                    .filter(|&v| !doms[v].is_fixed())
                    .take(64)
                    .collect();
                debug_assert!(!vars.is_empty(), "findbest requires an unfixed scope var");
                let mask = if vars.len() == 64 {
                    u64::MAX
                } else {
                    (1u64 << vars.len()) - 1
                };
                // Single-branch (batch-fix) rule: γ = 1, matching the closed
                // region in `compute_branching_result`.
                return (Some(vec![Clause::new(mask, 0)]), vars, 1.0);
            }
        };
        if matches!(*self, Selector::BinaryOccurrence) {
            // Control arm: branch the chosen var both ways. Trivially complete;
            // everything the region layer adds is deliberately absent. γ is not
            // meaningful for this arm (no region rule) — report NaN; it never
            // participates in the cutoff signal analysis.
            return (
                Some(vec![Clause::new(1, 0), Clause::new(1, 1)]),
                vec![var_id],
                f64::NAN,
            );
        }
        compute_branching_result(
            cn,
            doms,
            buffer,
            var_id,
            self.max_rows(),
            measure,
            solver,
            masks,
            tables,
            trail,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    fn or3() -> Vec<bool> {
        vec![false, true, true, true, true, true, true, true]
    }
    #[test]
    fn most_occurrence_picks_highest_occurrence() {
        // T0 over [0,1,2], T1 over [1,2,3]: occurrences v0=1, v1=2, v2=2, v3=1
        // -> argmax is v1 (first to reach the max).
        let cn = setup_problem(4, vec![vec![0, 1, 2], vec![1, 2, 3]], vec![or3(), or3()]);
        let doms = vec![DomainMask::BOTH; 4];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, _t) = crate::ct::build_tables(&cn);
        assert_eq!(
            select_var_most_occurrence(&cn, &doms, &mut buf, &[0, 1, 2, 3], &masks),
            Some(1)
        );
        assert_eq!(buf.occurrence_scores, vec![1.0, 2.0, 2.0, 1.0]);
        // Scope restriction: excluding v1 shifts the argmax to v2.
        assert_eq!(
            select_var_most_occurrence(&cn, &doms, &mut buf, &[0, 2, 3], &masks),
            Some(2)
        );
    }

    #[test]
    fn selector_is_complete_on_binary_only_residuals() {
        // Pure binary network: every var still scores (occurrence counting has
        // no degree threshold), so selection works without any 2-SAT shortcut.
        let or2 = vec![false, true, true, true];
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![or2.clone(), or2]);
        let doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, _t) = crate::ct::build_tables(&cn);
        // occurrences: v0=1, v1=2, v2=1 -> v1.
        assert_eq!(
            select_var_most_occurrence(&cn, &doms, &mut buf, &[0, 1, 2], &masks),
            Some(1)
        );
    }

    #[test]
    fn occurrence_pool_ranks_by_score_and_drops_free_vars() {
        // T0 over [0,1,2], T1 over [1,2,3]: scores v0=1,v1=2,v2=2,v3=1.
        let cn = setup_problem(4, vec![vec![0, 1, 2], vec![1, 2, 3]], vec![or3(), or3()]);
        let doms = vec![DomainMask::BOTH; 4];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, _t) = crate::ct::build_tables(&cn);
        // Top-2 by score: v1, v2 (both score 2, ascending id order).
        assert_eq!(occurrence_pool(&cn, &doms, &mut buf, &masks, 2), vec![1, 2]);
        // Unbounded: all four, highest-first; ties ascending.
        assert_eq!(
            occurrence_pool(&cn, &doms, &mut buf, &masks, 16),
            vec![1, 2, 0, 3]
        );
    }

    #[test]
    fn free_vars_are_batch_fixed_in_one_branch() {
        // A single FULL tensor over [0,1]: entailed at the root, so both vars
        // are free — findbest must return one single-branch clause fixing both
        // to 0 instead of branching over configs.
        let full2 = vec![true, true, true, true];
        let cn = Arc::new(setup_problem(2, vec![vec![0, 1]], vec![full2]));
        let mut doms = vec![DomainMask::BOTH; 2];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        let sel = Selector::MostOccurrence { max_rows: 32 };
        let (clauses, vars, gamma) = sel.findbest(
            &cn,
            &mut doms,
            &mut buf,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(optimal_branching_core::IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
            &[0, 1],
        );
        assert_eq!(vars, vec![0, 1]);
        assert_eq!(
            gamma, 1.0,
            "batch-fix of free vars is a single branch, γ = 1"
        );
        let clauses = clauses.expect("free vars still produce a rule");
        assert_eq!(clauses.len(), 1, "one branch, no alternatives");
        assert_eq!(clauses[0].mask, 0b11);
        assert_eq!(clauses[0].val, 0b00);
    }

    #[test]
    fn most_occurrence_findbest_returns_a_rule() {
        let cn = Arc::new(setup_problem(
            4,
            vec![vec![0, 1, 2], vec![1, 2, 3]],
            vec![or3(), or3()],
        ));
        let mut doms = vec![DomainMask::BOTH; 4];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        let sel = Selector::MostOccurrence { max_rows: 32 };
        let (clauses, vars, gamma) = sel.findbest(
            &cn,
            &mut doms,
            &mut buf,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(optimal_branching_core::IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
            &[0, 1, 2, 3],
        );
        assert!(clauses.is_some());
        assert!(!vars.is_empty());
        assert!(gamma >= 1.0, "a real covering rule has γ ≥ 1, got {gamma}");
    }
}
