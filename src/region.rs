use rustc_hash::FxHashSet;

use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;

#[derive(Clone, Debug)]
pub struct Region {
    /// Focus variable the region was grown from.
    pub id: usize,
    /// Region tensor ids, ascending.
    pub tensors: Vec<usize>,
    /// Region variable ids, ascending.
    pub vars: Vec<usize>,
}

#[inline]
fn tensor_is_hard(cn: &ConstraintNetwork, doms: &[DomainMask], tid: usize) -> bool {
    // Port of domain.jl `is_hard`: active degree (unfixed-var count) > 2.
    cn.tensors[tid]
        .var_axes
        .iter()
        .filter(|&&v| !doms[v].is_fixed())
        .count()
        > 2
}

/// BFS from `focus`, alternating var -> tensor -> var for `k` hops, collecting at
/// most `max_tensors` tensors and every unfixed variable reached. Port of
/// `knn.jl::_k_neighboring`. `hard_only` restricts collection to degree-3+ tensors.
pub fn k_neighboring_impl(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    focus: usize,
    max_tensors: usize,
    k: usize,
    hard_only: bool,
) -> Region {
    debug_assert!(!doms[focus].is_fixed(), "focus variable must be unfixed");

    let mut visited_vars: FxHashSet<usize> = FxHashSet::default();
    let mut visited_tensors: FxHashSet<usize> = FxHashSet::default();
    let mut collected_vars: Vec<usize> = vec![focus];
    let mut collected_tensors: Vec<usize> = Vec::new();
    visited_vars.insert(focus);

    let mut var_queue: Vec<usize> = vec![focus];

    for _hop in 0..k {
        // Step 1: current variables -> new tensors (respecting the cap).
        let mut tensor_queue: Vec<usize> = Vec::new();
        'collect: for &var_id in &var_queue {
            for &tid in &cn.v2t[var_id] {
                if visited_tensors.contains(&tid) {
                    continue;
                }
                if hard_only && !tensor_is_hard(cn, doms, tid) {
                    continue;
                }
                visited_tensors.insert(tid);
                collected_tensors.push(tid);
                tensor_queue.push(tid);
                if collected_tensors.len() >= max_tensors {
                    break 'collect;
                }
            }
        }

        // Step 2: new tensors -> next-layer unfixed variables (always harvested,
        // even when the tensor cap was just hit — matches Julia).
        var_queue = Vec::new();
        for &tid in &tensor_queue {
            for &nv in &cn.tensors[tid].var_axes {
                if !visited_vars.contains(&nv) && !doms[nv].is_fixed() {
                    visited_vars.insert(nv);
                    collected_vars.push(nv);
                    var_queue.push(nv);
                }
            }
        }

        if collected_tensors.len() >= max_tensors {
            break;
        }
    }

    collected_vars.sort_unstable();
    collected_tensors.sort_unstable();
    Region {
        id: focus,
        tensors: collected_tensors,
        vars: collected_vars,
    }
}

/// Region growth with `hard_only = false`. Port of `knn.jl::k_neighboring`.
pub fn k_neighboring(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    focus: usize,
    max_tensors: usize,
    k: usize,
) -> Region {
    k_neighboring_impl(cn, doms, focus, max_tensors, k, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    // A 5-var OR-chain: T0[0,1] T1[1,2] T2[2,3] T3[3,4], each a binary OR.
    fn chain() -> ConstraintNetwork {
        let or2 = vec![false, true, true, true];
        setup_problem(
            5,
            vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]],
            vec![or2.clone(), or2.clone(), or2.clone(), or2],
        )
    }

    #[test]
    fn k1_collects_incident_tensors_and_their_vars() {
        let cn = chain();
        let doms = vec![DomainMask::BOTH; 5];
        let r = k_neighboring(&cn, &doms, 2, 10, 1);
        assert_eq!(r.id, 2);
        assert_eq!(r.tensors, vec![1, 2]); // tensors incident to var 2
        assert_eq!(r.vars, vec![1, 2, 3]); // their unfixed vars
    }

    #[test]
    fn k2_expands_one_more_hop() {
        let cn = chain();
        let doms = vec![DomainMask::BOTH; 5];
        let r = k_neighboring(&cn, &doms, 2, 10, 2);
        assert_eq!(r.tensors, vec![0, 1, 2, 3]);
        assert_eq!(r.vars, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn max_tensors_cap_truncates() {
        let cn = chain();
        let doms = vec![DomainMask::BOTH; 5];
        let r = k_neighboring(&cn, &doms, 2, 2, 2);
        assert_eq!(r.tensors.len(), 2); // capped in the first hop
        assert_eq!(r.tensors, vec![1, 2]);
        // Step 2 still harvests the capped tensors' vars before stopping.
        assert_eq!(r.vars, vec![1, 2, 3]);
    }

    #[test]
    fn hard_only_skips_binary_tensors() {
        // T0 is a hard (degree-3) tensor over [0,1,2]; T1 is binary over [2,3].
        // 3-var dense: satisfied set is arbitrary but nonempty.
        let t0 = vec![false, true, true, true, true, true, true, true]; // not all-zero
        let or2 = vec![false, true, true, true];
        let cn = setup_problem(4, vec![vec![0, 1, 2], vec![2, 3]], vec![t0, or2]);
        let doms = vec![DomainMask::BOTH; 4];
        // hard_only via the private impl: only the degree-3 tensor is collected.
        let r = k_neighboring_impl(&cn, &doms, 0, 10, 2, true);
        assert_eq!(r.tensors, vec![0]);
    }
}
