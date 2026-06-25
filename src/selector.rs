use optimal_branching_core::{Clause, SetCoverSolver};

use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{has_contradiction, SolverBuffer};
use crate::propagate::probe_assignment;
use crate::region::RegionCache;
use crate::table::compute_branching_result;
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
pub(crate) fn select_var_difflookahead(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
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
        let c0 = probe_assignment(cn, buffer, doms, &[u], 1, 0);
        let f0 = has_contradiction(c0);
        let d0 = if f0 { 0 } else { sum_active_degree(cn, c0) }; // last use of c0
        let c1 = probe_assignment(cn, buffer, doms, &[u], 1, 1);
        let f1 = has_contradiction(c1);
        let d1 = if f1 { 0 } else { sum_active_degree(cn, c1) }; // last use of c1
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
        k: usize,
        max_tensors: usize,
    },
    DiffLookahead {
        k: usize,
        max_tensors: usize,
        pool: usize,
    },
}

impl Selector {
    /// The `(k, max_tensors)` the `RegionCache` should be built with.
    pub fn k_max(&self) -> (usize, usize) {
        match *self {
            Selector::MostOccurrence { k, max_tensors } => (k, max_tensors),
            Selector::DiffLookahead { k, max_tensors, .. } => (k, max_tensors),
        }
    }

    /// Pick a focus variable and compute its branching rule. Returns the rule's
    /// clauses (or `None` for a no-op) and the variables they range over.
    /// Port of `selector.jl::findbest`.
    pub fn findbest<SC: SetCoverSolver>(
        &self,
        cache: &mut RegionCache,
        cn: &ConstraintNetwork,
        doms: &[DomainMask],
        buffer: &mut SolverBuffer,
        measure: Measure,
        solver: &SC,
    ) -> (Option<Vec<Clause>>, Vec<usize>) {
        match *self {
            Selector::MostOccurrence { .. } => {
                let var_id = match select_var_most_occurrence(cn, doms, buffer) {
                    Some(v) => v,
                    None => return (None, Vec::new()),
                };
                compute_branching_result(cache, cn, doms, buffer, var_id, measure, solver)
            }
            Selector::DiffLookahead { pool, .. } => {
                let var_id = match select_var_difflookahead(cn, doms, buffer, pool) {
                    Some(v) => v,
                    None => return (None, Vec::new()),
                };
                // The lookahead probing scribbled in the cache; clear it so the
                // chosen var's table is computed clean (matches Julia's empty!).
                buffer.branching_cache.clear();
                compute_branching_result(cache, cn, doms, buffer, var_id, measure, solver)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    fn or3() -> Vec<bool> {
        vec![false, true, true, true, true, true, true, true]
    }
    fn or2() -> Vec<bool> {
        vec![false, true, true, true]
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
        let doms = vec![DomainMask::BOTH; 5];
        let mut buf = SolverBuffer::new(&cn);
        // x0=1 cascades 3->1,4->1 then (¬x3∨¬x4) fails: failed literal -> pick x0.
        assert_eq!(select_var_difflookahead(&cn, &doms, &mut buf, 16), Some(0));
    }

    #[test]
    fn most_occurrence_findbest_returns_a_rule() {
        let cn = setup_problem(4, vec![vec![0, 1, 2], vec![1, 2, 3]], vec![or3(), or3()]);
        let doms = vec![DomainMask::BOTH; 4];
        let mut cache = RegionCache::new(&cn, &doms, 1, 2);
        let mut buf = SolverBuffer::new(&cn);
        let sel = Selector::MostOccurrence {
            k: 1,
            max_tensors: 2,
        };
        let (clauses, vars) = sel.findbest(
            &mut cache,
            &cn,
            &doms,
            &mut buf,
            Measure::NumUnfixedVars,
            &optimal_branching_core::IPSolver::default(),
        );
        assert!(clauses.is_some());
        assert!(!vars.is_empty());
    }
}
