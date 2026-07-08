use std::sync::Arc;

use optimal_branching_core::Clause;

use crate::adapter::BranchSolver;
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{has_contradiction, SolverBuffer};
use crate::propagate::probe;
use crate::table::compute_branching_result;
use crate::trail::Trail;
use crate::util::get_active_tensors;

/// Fill `buffer.occurrence_scores`: each ACTIVE tensor (at least one unfixed
/// var) adds 1 to each of its unfixed variables — plain occurrence counting,
/// no structural weighting. Every unfixed var still constrained by an active
/// tensor scores > 0; vars whose tensors are all entailed score 0 and are
/// handled by the completeness fallback in the selectors.
pub(crate) fn compute_occurrence_scores(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
) {
    for s in buffer.occurrence_scores.iter_mut() {
        *s = 0.0;
    }
    for tid in get_active_tensors(cn, doms) {
        for &v in &cn.tensors[tid].var_axes {
            if !doms[v].is_fixed() {
                buffer.occurrence_scores[v] += 1.0;
            }
        }
    }
}

/// Completeness fallback: the first unfixed variable in `scope`. Reached when
/// no unfixed scope var scores > 0 (every remaining constraint is entailed);
/// such vars still must be branched to reach the all-fixed SAT leaf.
fn first_unfixed(doms: &[DomainMask], scope: &[usize]) -> Option<usize> {
    scope.iter().copied().find(|&i| !doms[i].is_fixed())
}

/// Highest-occurrence unfixed variable in `scope` (first one at the max);
/// falls back to the first unfixed scope var when nothing scores (all
/// remaining tensors entailed). `None` only when every scope variable is
/// fixed — the caller's SAT leaf.
pub(crate) fn select_var_most_occurrence(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
    scope: &[usize],
) -> Option<usize> {
    compute_occurrence_scores(cn, doms, buffer);
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
    var_id.or_else(|| first_unfixed(doms, scope))
}

// NOTE: iterates ALL tensors (not just active), matching Julia's `_sum_active_degree`; fixed vars contribute 0, so the result is correct — the full scan is intentional.
/// Sum of unfixed-variable degrees over ALL tensors. Port of
/// `selector.jl::_sum_active_degree`.
fn sum_active_degree(cn: &ConstraintNetwork, doms: &[DomainMask]) -> usize {
    let mut s = 0usize;
    for t in &cn.tensors {
        for &v in &t.var_axes {
            if !doms[v].is_fixed() {
                s += 1;
            }
        }
    }
    s
}

/// Difficulty-guided lookahead: among the top-`pool` scope candidates (by
/// occurrence score), probe both polarities and pick the var whose HARDER
/// child has the lowest active-degree; take a failed literal immediately.
/// Falls back to the first unfixed scope var when nothing scores (all
/// remaining tensors entailed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_var_difflookahead(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    buffer: &mut SolverBuffer,
    pool: usize,
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    trail: &mut Trail,
    scope: &[usize],
) -> Option<usize> {
    compute_occurrence_scores(cn, doms, buffer);
    let mut cands: Vec<usize> = scope
        .iter()
        .copied()
        .filter(|&i| !doms[i].is_fixed() && buffer.occurrence_scores[i] > 0.0)
        .collect();
    if cands.is_empty() {
        // No active tensor constrains any unfixed scope var: probing is
        // pointless, any unfixed var makes progress.
        return first_unfixed(doms, scope);
    }
    // Highest score first (stable: ties keep ascending var-id order).
    cands.sort_by(|&a, &b| {
        buffer.occurrence_scores[b]
            .partial_cmp(&buffer.occurrence_scores[a])
            .expect("finite scores")
    });
    cands.truncate(pool);

    let mut best = usize::MAX;
    let mut chosen: Option<usize> = None;
    for &u in &cands {
        let (f0, d0) = probe(cn, doms, masks, tables, buffer, trail, &[u], 1, 0, |d| {
            (
                has_contradiction(d),
                if has_contradiction(d) {
                    0
                } else {
                    sum_active_degree(cn, d)
                },
            )
        });
        let (f1, d1) = probe(cn, doms, masks, tables, buffer, trail, &[u], 1, 1, |d| {
            (
                has_contradiction(d),
                if has_contradiction(d) {
                    0
                } else {
                    sum_active_degree(cn, d)
                },
            )
        });
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

#[derive(Clone, Copy, Debug)]
pub enum Selector {
    MostOccurrence {
        /// Row budget for `grow_region` — the branching-table size cap.
        max_rows: usize,
    },
    DiffLookahead {
        max_rows: usize,
        pool: usize,
    },
}

impl Selector {
    /// The row budget regions are grown with.
    pub fn max_rows(&self) -> usize {
        match *self {
            Selector::MostOccurrence { max_rows } => max_rows,
            Selector::DiffLookahead { max_rows, .. } => max_rows,
        }
    }

    /// Pick a focus variable inside `scope` (the caller's connected component;
    /// pass all vars when undecomposed) and compute its branching rule from a
    /// region grown fresh at the current `doms`. Returns the rule's clauses
    /// (or `None` for a no-op) and the variables they range over. Port of
    /// `selector.jl::findbest`.
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
    ) -> (Option<Vec<Clause>>, Vec<usize>) {
        let var_id = match *self {
            Selector::MostOccurrence { .. } => select_var_most_occurrence(cn, doms, buffer, scope),
            Selector::DiffLookahead { pool, .. } => {
                select_var_difflookahead(cn, doms, buffer, pool, masks, tables, trail, scope)
            }
        };
        let var_id = match var_id {
            Some(v) => v,
            None => return (None, Vec::new()),
        };
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
        assert_eq!(
            select_var_most_occurrence(&cn, &doms, &mut buf, &[0, 1, 2, 3]),
            Some(1)
        );
        assert_eq!(buf.occurrence_scores, vec![1.0, 2.0, 2.0, 1.0]);
        // Scope restriction: excluding v1 shifts the argmax to v2.
        assert_eq!(
            select_var_most_occurrence(&cn, &doms, &mut buf, &[0, 2, 3]),
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
        // occurrences: v0=1, v1=2, v2=1 -> v1.
        assert_eq!(
            select_var_most_occurrence(&cn, &doms, &mut buf, &[0, 1, 2]),
            Some(1)
        );
    }

    #[test]
    fn difflookahead_takes_a_failed_literal_immediately() {
        // T0 hard OR over [0,1,2] gives v0 a positive score (so it's a candidate).
        // Binary clauses make x0=1 contradict: (¬x0∨x3)(¬x0∨x4)(¬x3∨¬x4).
        let f_imp = vec![true, false, true, true]; // ¬a∨b over [a,b]: forbids (a=1,b=0)
        let f_nand = vec![true, true, true, false]; // ¬a∨¬b over [a,b]: forbids (1,1)
        let cn = setup_problem(
            5,
            vec![vec![0, 1, 2], vec![0, 3], vec![0, 4], vec![3, 4]],
            vec![or3(), f_imp.clone(), f_imp, f_nand],
        );
        let mut doms = vec![DomainMask::BOTH; 5];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let mut trail = Trail::new();
        // x0=1 cascades 3->1,4->1 then (¬x3∨¬x4) fails: failed literal -> pick x0.
        assert_eq!(
            select_var_difflookahead(
                &cn,
                &mut doms,
                &mut buf,
                16,
                &masks,
                &mut tables,
                &mut trail,
                &[0, 1, 2, 3, 4]
            ),
            Some(0)
        );
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
        let (clauses, vars) = sel.findbest(
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
    }
}
