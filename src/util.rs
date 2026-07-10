use crate::ct::TableMasks;
use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;

/// Tensor ids that still have at least one unfixed variable, ascending. An
/// ACTIVE tensor is one whose scope isn't fully fixed, so it can still
/// constrain the search. Yielded lazily: hot callers (occurrence scoring, the
/// tensor measures) only iterate or count, so materializing a `Vec` per call is
/// pure waste. Port of `utils.jl::get_active_tensors`.
pub fn active_tensors<'a>(
    cn: &'a ConstraintNetwork,
    doms: &'a [DomainMask],
) -> impl Iterator<Item = usize> + 'a {
    cn.tensors
        .iter()
        .enumerate()
        .filter(move |(_, t)| t.var_axes.iter().any(|&v| !doms[v].is_fixed()))
        .map(|(tid, _)| tid)
}

/// `active_tensors` collected — for the rare caller that needs an owned slice.
pub fn get_active_tensors(cn: &ConstraintNetwork, doms: &[DomainMask]) -> Vec<usize> {
    active_tensors(cn, doms).collect()
}

/// Number of unfixed domains. Port of `utils.jl::count_unfixed`.
pub fn count_unfixed(doms: &[DomainMask]) -> usize {
    doms.iter().filter(|d| !d.is_fixed()).count()
}

/// Whether tensor `tid` is ENTAILED under `doms`: after slicing in the fixed
/// axes, every combination of its unfixed vars is satisfying — the tensor
/// constrains nothing and is dead weight for occurrence selection, region
/// growth, and boundary classification. Computed from the static per-(axis,
/// value) support masks
/// (sliced row count == 2^unfixed), so it needs no live CT table and holds in
/// any context. Monotone down a branch: slicing a full table on a newly fixed
/// axis keeps it full, so an entailed tensor stays entailed in the subtree.
pub fn is_entailed(
    cn: &ConstraintNetwork,
    tid: usize,
    doms: &[DomainMask],
    masks: &[TableMasks],
) -> bool {
    let t = &cn.tensors[tid];
    let m = &masks[t.table_idx];
    let unfixed = t.var_axes.iter().filter(|&&v| !doms[v].is_fixed()).count();
    if unfixed == t.var_axes.len() {
        // No fixed axis to slice: entailed iff the table is full.
        return m.n_rows as u64 == 1u64 << unfixed;
    }
    // Sliced row count: AND the fixed axes' (axis, value) support masks. Each
    // mask is tail-clean (no bits beyond n_rows), and at least one fixed axis
    // exists, so the accumulator never counts padding bits.
    let mut count = 0u64;
    for w in 0..m.n_words {
        let mut acc = u64::MAX;
        for (i, &v) in t.var_axes.iter().enumerate() {
            match doms[v] {
                DomainMask::D0 => acc &= m.supports[(i * 2) * m.n_words + w],
                DomainMask::D1 => acc &= m.supports[(i * 2 + 1) * m.n_words + w],
                _ => {}
            }
        }
        count += acc.count_ones() as u64;
    }
    count == 1u64 << unfixed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    #[test]
    fn active_tensors_have_an_unfixed_var() {
        // T0 over [0,1], T1 over [2]. Fix v2 -> T1 inactive, T0 active.
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![2]],
            vec![vec![false, true, true, true], vec![false, true]],
        );
        let doms = vec![DomainMask::BOTH, DomainMask::BOTH, DomainMask::D1];
        assert_eq!(get_active_tensors(&cn, &doms), vec![0]);
    }

    #[test]
    fn entailment_tracks_slicing() {
        // T0 = OR over [0,1] (3/4 rows), T1 = full over [1,2] (4/4 rows).
        let or2 = vec![false, true, true, true];
        let full2 = vec![true, true, true, true];
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![or2, full2]);
        let (masks, _t) = crate::ct::build_tables(&cn);
        let doms = vec![DomainMask::BOTH; 3];
        // Unsliced: only the full table is entailed.
        assert!(!is_entailed(&cn, 0, &doms, &masks));
        assert!(is_entailed(&cn, 1, &doms, &masks));
        // v1 = 1 satisfies the OR: T0 becomes entailed (v0 free in it).
        let mut doms2 = doms.clone();
        doms2[1] = DomainMask::D1;
        assert!(is_entailed(&cn, 0, &doms2, &masks));
        // v1 = 0 leaves the OR forcing v0: 1 row over 1 unfixed var != 2.
        let mut doms3 = doms.clone();
        doms3[1] = DomainMask::D0;
        assert!(!is_entailed(&cn, 0, &doms3, &masks));
    }

    #[test]
    fn count_unfixed_counts_free_vars() {
        let doms = vec![
            DomainMask::D1,
            DomainMask::BOTH,
            DomainMask::D0,
            DomainMask::BOTH,
        ];
        assert_eq!(count_unfixed(&doms), 2);
    }
}
