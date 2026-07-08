use std::sync::Arc;

use optimal_branching_core::Clause;

use crate::adapter::BranchSolver;
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{has_contradiction, SolverBuffer};
use crate::propagate::probe;
use crate::region::RegionCache;
use crate::table::compute_branching_result;
use crate::trail::Trail;
use crate::util::get_active_tensors;

/// Fill `buffer.connection_scores`: each hard tensor (unfixed degree > 2) adds
/// `degree - 2` to each of its unfixed variables. Port of
/// `selector.jl::compute_var_cover_scores_weighted`.
pub(crate) fn compute_connection_scores(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
) {
    for s in buffer.connection_scores.iter_mut() {
        *s = 0.0;
    }
    for tid in get_active_tensors(cn, doms) {
        let vars = &cn.tensors[tid].var_axes;
        let degree = vars.iter().filter(|&&v| !doms[v].is_fixed()).count();
        if degree > 2 {
            let weight = (degree - 2) as f64;
            for &v in vars {
                if !doms[v].is_fixed() {
                    buffer.connection_scores[v] += weight;
                }
            }
        }
    }
}

/// Highest-connection-score unfixed variable (first one at the max). `None` if
/// no unfixed var has a positive score (the residual is 2-SAT — handled upstream).
pub(crate) fn select_var_most_occurrence(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
) -> Option<usize> {
    compute_connection_scores(cn, doms, buffer);
    let mut max_score = 0.0f64;
    let mut var_id: Option<usize> = None;
    for i in 0..doms.len() {
        if doms[i].is_fixed() {
            continue;
        }
        if buffer.connection_scores[i] > max_score {
            max_score = buffer.connection_scores[i];
            var_id = Some(i);
        }
    }
    var_id
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

/// Difficulty-guided lookahead: among the top-`pool` candidates (by connection
/// score), probe both polarities and pick the var whose HARDER child has the
/// lowest active-degree; take a failed literal immediately. Port of
/// `selector.jl`'s `DiffLookaheadSelector` `findbest` var-choice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_var_difflookahead(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    buffer: &mut SolverBuffer,
    pool: usize,
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    trail: &mut Trail,
) -> Option<usize> {
    compute_connection_scores(cn, doms, buffer);
    let mut cands: Vec<usize> = (0..doms.len())
        .filter(|&i| !doms[i].is_fixed() && buffer.connection_scores[i] > 0.0)
        .collect();
    if cands.is_empty() {
        return None;
    }
    // Highest score first (stable: ties keep ascending var-id order, like Julia).
    cands.sort_by(|&a, &b| {
        buffer.connection_scores[b]
            .partial_cmp(&buffer.connection_scores[a])
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
    /// The row budget the `RegionCache` should be built with.
    pub fn max_rows(&self) -> usize {
        match *self {
            Selector::MostOccurrence { max_rows } => max_rows,
            Selector::DiffLookahead { max_rows, .. } => max_rows,
        }
    }

    /// Pick a focus variable and compute its branching rule from its cached
    /// root region, conditioned on the current `doms`. Returns the rule's
    /// clauses (or `None` for a no-op) and the variables they range over.
    /// Port of `selector.jl::findbest`.
    #[allow(clippy::too_many_arguments)]
    pub fn findbest(
        &self,
        cache: &mut RegionCache,
        cn: &Arc<ConstraintNetwork>,
        doms: &mut [DomainMask],
        buffer: &mut SolverBuffer,
        measure: Measure,
        solver: &BranchSolver,
        masks: &Arc<Vec<TableMasks>>,
        tables: &mut Vec<RSparseBitSet>,
        trail: &mut Trail,
    ) -> (Option<Vec<Clause>>, Vec<usize>) {
        let var_id = match *self {
            Selector::MostOccurrence { .. } => select_var_most_occurrence(cn, doms, buffer),
            Selector::DiffLookahead { pool, .. } => {
                select_var_difflookahead(cn, doms, buffer, pool, masks, tables, trail)
            }
        };
        let var_id = match var_id {
            Some(v) => v,
            None => return (None, Vec::new()),
        };
        compute_branching_result(
            cache, cn, doms, buffer, var_id, measure, solver, masks, tables, trail,
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
    fn most_occurrence_picks_highest_connection_score() {
        // Two hard (degree-3) tensors: T0 over [0,1,2], T1 over [1,2,3].
        // scores: v0=1, v1=2, v2=2, v3=1 -> argmax is v1 (first to reach the max).
        let cn = setup_problem(4, vec![vec![0, 1, 2], vec![1, 2, 3]], vec![or3(), or3()]);
        let doms = vec![DomainMask::BOTH; 4];
        let mut buf = SolverBuffer::new(&cn);
        assert_eq!(select_var_most_occurrence(&cn, &doms, &mut buf), Some(1));
        assert_eq!(buf.connection_scores, vec![1.0, 2.0, 2.0, 1.0]);
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
                &mut trail
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
        let mut cache = RegionCache::new(&cn, &doms, 32);
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        let sel = Selector::MostOccurrence { max_rows: 32 };
        let (clauses, vars) = sel.findbest(
            &mut cache,
            &cn,
            &mut doms,
            &mut buf,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(optimal_branching_core::IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
        );
        assert!(clauses.is_some());
        assert!(!vars.is_empty());
    }
}
