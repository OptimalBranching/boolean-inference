//! Static, width-aware constraint-network canonicalizer (bounded-width VE).
//!
//! Port of Julia `bounded_ve_canonicalize` (src/preprocessing/canonicalize.jl).
//! Eliminates a variable `v` by joining all tensors incident to `v` and projecting
//! `v` out (boolean ∃/∧), but only if the elimination width `out.len()+1` is
//! `<= budget_b` (and `out.len() <= 32`, the TensorData cap). Eligible variables are
//! removed in weighted-min-fill order. `protected` variables (read-out vars, e.g.
//! factor bits) are never eliminated and survive into the result; their values are
//! read off the result's `orig_to_new`. The elimination width `sc = out.len()+1` is
//! exact for the relational join we perform (the largest intermediate is the relation
//! over `neighbors(v) ∪ {v}`), so no contraction-order optimizer is needed.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::contract::{dense_relation, join_all};
use crate::network::{setup_problem, ConstraintNetwork};

/// A mutable tensor during elimination: axes (in cn's compressed-var space) + dense table.
struct LiveTensor {
    var_axes: Vec<usize>,
    dense: Vec<bool>,
}

/// Sorted-unique union of the incident tensors' vars, minus `v` (the produced axes).
fn out_vars(live: &[LiveTensor], tids: &[usize], v: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &t in tids {
        for &x in &live[t].var_axes {
            if x != v {
                out.push(x);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Active tensor slots incident to `v`.
fn active_incident(v2t: &[Vec<usize>], active: &[bool], v: usize) -> Vec<usize> {
    v2t[v].iter().copied().filter(|&t| active[t]).collect()
}

/// Weighted-min-fill: number of pairs in `out` that do NOT already share an active tensor.
fn fill_count(live: &[LiveTensor], v2t: &[Vec<usize>], active: &[bool], out: &[usize]) -> usize {
    let mut f = 0usize;
    for i in 0..out.len() {
        for j in (i + 1)..out.len() {
            let (a, b) = (out[i], out[j]);
            let share = v2t[a]
                .iter()
                .any(|&t| active[t] && live[t].var_axes.contains(&b));
            if !share {
                f += 1;
            }
        }
    }
    f
}

/// `Some((fill, sc))` if `v` is eligible to eliminate now, else `None`.
fn score(
    live: &[LiveTensor],
    v2t: &[Vec<usize>],
    active: &[bool],
    is_protected: &[bool],
    budget_b: usize,
    v: usize,
) -> Option<(usize, usize)> {
    if is_protected[v] {
        return None;
    }
    let tids = active_incident(v2t, active, v);
    if tids.is_empty() {
        return None;
    }
    let out = out_vars(live, &tids, v);
    let sc = out.len() + 1;
    if out.len() <= 32 && sc <= budget_b {
        Some((fill_count(live, v2t, active, &out), sc))
    } else {
        None
    }
}

/// Reshape `cn` by bucket-eliminating variables within the width `budget_b`, in
/// weighted-min-fill order. `protected` (cn compressed-var ids) are never eliminated.
/// The returned network's `orig_to_new` indexes the same original var ids as `cn`.
pub fn bounded_ve_canonicalize(
    cn: &ConstraintNetwork,
    budget_b: usize,
    protected: &[usize],
) -> ConstraintNetwork {
    let nv = cn.vars.len();

    let mut live: Vec<LiveTensor> = cn
        .tensors
        .iter()
        .map(|t| LiveTensor {
            var_axes: t.var_axes.clone(),
            dense: cn.data(t).dense.clone(),
        })
        .collect();
    let mut active: Vec<bool> = vec![true; live.len()];
    let mut v2t: Vec<Vec<usize>> = cn.v2t.clone();
    let mut is_protected = vec![false; nv];
    for &p in protected {
        if p < nv {
            is_protected[p] = true;
        }
    }

    // Min-heap on (fill, sc) via Reverse; var id is only a stable tie-break carrier.
    let mut heap: BinaryHeap<Reverse<(usize, usize, usize)>> = BinaryHeap::new();
    for v in 0..nv {
        if let Some((fill, sc)) = score(&live, &v2t, &active, &is_protected, budget_b, v) {
            heap.push(Reverse((fill, sc, v)));
        }
    }

    while let Some(Reverse((fill, sc, v))) = heap.pop() {
        // Lazy staleness: a var may have stale heap entries; only act if it is still
        // eligible with the exact (fill, sc) we popped.
        match score(&live, &v2t, &active, &is_protected, budget_b, v) {
            Some((f, s)) if f == fill && s == sc => {}
            _ => continue,
        }

        let tids = active_incident(&v2t, &active, v);
        let out = out_vars(&live, &tids, v);

        // Bucket-contract: join all incident tensors, project `v` out, densify over `out`.
        let rels: Vec<_> = tids
            .iter()
            .map(|&t| dense_relation(&live[t].var_axes, &live[t].dense))
            .collect();
        let joined = join_all(rels);
        let mut dense = vec![false; 1usize << out.len()];
        for &row in &joined.rows {
            let mut cfg = 0usize;
            for (j, &x) in out.iter().enumerate() {
                let pos = joined.vars.binary_search(&x).expect("out var present in join");
                if (row >> pos) & 1 == 1 {
                    cfg |= 1usize << j;
                }
            }
            dense[cfg] = true;
        }

        // Merge in place: reuse the first incident slot, deactivate the rest.
        let keep = tids[0];
        for &t in &tids {
            let axes = live[t].var_axes.clone();
            for x in axes {
                v2t[x].retain(|&tt| tt != t);
            }
        }
        for &t in &tids[1..] {
            active[t] = false;
        }
        live[keep] = LiveTensor {
            var_axes: out.clone(),
            dense,
        };
        for &x in &out {
            v2t[x].push(keep);
        }

        // Re-score the affected neighbors (stale entries are skipped on later pops).
        for &u in &out {
            if let Some((f, s)) = score(&live, &v2t, &active, &is_protected, budget_b, u) {
                heap.push(Reverse((f, s, u)));
            }
        }
    }

    // Finalize: hand surviving tensors to setup_problem for dedup + compression.
    let mut tv: Vec<Vec<usize>> = Vec::new();
    let mut td: Vec<Vec<bool>> = Vec::new();
    for t in 0..live.len() {
        if active[t] {
            tv.push(live[t].var_axes.clone());
            td.push(live[t].dense.clone());
        }
    }
    let new_cn = setup_problem(nv, tv, td);

    // Compose orig->cn (cn.orig_to_new) with cn->new (new_cn.orig_to_new).
    let mut orig_to_new = vec![None; cn.orig_to_new.len()];
    for (orig, &cnid) in cn.orig_to_new.iter().enumerate() {
        if let Some(c) = cnid {
            orig_to_new[orig] = new_cn.orig_to_new[c];
        }
    }
    ConstraintNetwork {
        orig_to_new,
        ..new_cn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;
    use std::collections::HashSet;

    const OR2: [bool; 4] = [false, true, true, true]; // x ∨ y

    /// All satisfying assignments of `cn` projected onto original vars `orig_vars`,
    /// as a set of bitmasks (bit j = orig_vars[j]). Brute force over compressed vars.
    fn solutions_projected(cn: &ConstraintNetwork, orig_vars: &[usize]) -> HashSet<u64> {
        let n = cn.vars.len();
        let mut out = HashSet::new();
        for cfg in 0u64..(1u64 << n) {
            let ok = cn.tensors.iter().all(|t| {
                let mut idx = 0u32;
                for (i, &v) in t.var_axes.iter().enumerate() {
                    if (cfg >> v) & 1 == 1 {
                        idx |= 1 << i;
                    }
                }
                cn.dense(t)[idx as usize]
            });
            if !ok {
                continue;
            }
            let mut key = 0u64;
            for (j, &o) in orig_vars.iter().enumerate() {
                if let Some(c) = cn.orig_to_new[o] {
                    if (cfg >> c) & 1 == 1 {
                        key |= 1u64 << j;
                    }
                }
            }
            out.insert(key);
        }
        out
    }

    #[test]
    fn eliminates_an_unprotected_chain_var() {
        // (x0∨x1)∧(x1∨x2); eliminate x1 (budget 3, x0 and x2 protected as read-out vars).
        // x1 is the interior chain var; x0,x2 are the endpoints we care about reading.
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![OR2.to_vec(), OR2.to_vec()]);
        let out = bounded_ve_canonicalize(&cn, 3, &[0, 2]);
        // x1 is eliminated (unprotected, sc=3<=budget); x0,x2 survive (protected).
        assert!(out.orig_to_new[1].is_none(), "x1 should be eliminated");
        assert!(out.orig_to_new[0].is_some() && out.orig_to_new[2].is_some());
        // Solutions over {x0,x2} preserved: (x0∨x1)∧(x1∨x2) projected to x0,x2
        // allows everything except... brute-force equality is the real check below.
        assert_eq!(
            solutions_projected(&cn, &[0, 2]),
            solutions_projected(&out, &[0, 2]),
        );
    }

    #[test]
    fn protected_var_is_never_eliminated() {
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![OR2.to_vec(), OR2.to_vec()]);
        let out = bounded_ve_canonicalize(&cn, 3, &[1]); // protect x1
        assert!(out.orig_to_new[1].is_some(), "protected x1 must survive");
    }

    #[test]
    fn budget_one_eliminates_nothing() {
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![OR2.to_vec(), OR2.to_vec()]);
        let out = bounded_ve_canonicalize(&cn, 1, &[]);
        // every elimination needs sc = out.len()+1 >= 2 > 1, so all vars survive.
        assert_eq!(out.vars.len(), cn.vars.len());
    }

    #[test]
    fn solutions_preserved_over_protected_with_elimination() {
        // 4-var chain (x0∨x1)∧(x1∨x2)∧(x2∨x3); protect {x0, x3}; budget 3.
        let cn = setup_problem(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![OR2.to_vec(), OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[0, 3]);
        assert!(out.vars.len() < cn.vars.len(), "some vars eliminated");
        assert!(out.orig_to_new[0].is_some() && out.orig_to_new[3].is_some());
        assert_eq!(
            solutions_projected(&cn, &[0, 3]),
            solutions_projected(&out, &[0, 3]),
            "solution set projected to protected vars must be preserved"
        );
    }

    #[test]
    fn produced_tensors_respect_the_budget() {
        let cn = setup_problem(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![OR2.to_vec(), OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[]);
        for t in &out.tensors {
            assert!(t.var_axes.len() <= 3 - 1, "produced arity <= budget_b-1");
        }
    }
}
