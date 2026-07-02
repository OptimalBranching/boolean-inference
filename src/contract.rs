use rustc_hash::FxHashMap;

use crate::domain::DomainMask;
use crate::network::{BoolTensor, ConstraintNetwork};
use crate::propagate::compute_query_masks;
use crate::region::Region;

/// A boolean relation: the set `rows` of satisfying assignments over `vars`,
/// where `vars` is sorted ascending and bit *j* of a row is the value of
/// `vars[j]`. Rows are deduplicated.
#[derive(Clone, Debug)]
pub struct Relation {
    pub vars: Vec<usize>,
    pub rows: Vec<u64>,
}

impl Relation {
    /// Project each row onto `keep` (a subset of `self.vars`, ascending). Rows are
    /// re-encoded over `keep` bit order, then sorted and deduplicated. Every entry
    /// of `keep` must be present in `self.vars`.
    pub fn project(&self, keep: &[usize]) -> Relation {
        let mut rows: Vec<u64> = self
            .rows
            .iter()
            .map(|&row| {
                let mut r = 0u64;
                for (j, &v) in keep.iter().enumerate() {
                    let pos = self
                        .vars
                        .binary_search(&v)
                        .expect("projection var present in relation");
                    if (row >> pos) & 1 == 1 {
                        r |= 1u64 << j;
                    }
                }
                r
            })
            .collect();
        rows.sort_unstable();
        rows.dedup();
        Relation {
            vars: keep.to_vec(),
            rows,
        }
    }
}

/// Slice one tensor against the fixed variables and return the relation over its
/// *unfixed* (free) variables. Port of `contraction.jl::slicing` (relational form).
pub fn tensor_relation(
    cn: &ConstraintNetwork,
    tensor: &BoolTensor,
    doms: &[DomainMask],
) -> Relation {
    // Fixed-bit mask/value over the tensor's axes (reuse the GAC query helper):
    // `m0`/`m1` mark axes fixed to 0/1, so fmask = m0|m1, fval = m1.
    let (m0, m1) = compute_query_masks(doms, &tensor.var_axes);
    let fmask = m0 | m1;
    let fval = m1;

    // Free axes, paired (global var, axis position), sorted by var so bit order
    // is canonical (ascending var id) — matches the `packint` ordering.
    let mut fv: Vec<(usize, usize)> = tensor
        .var_axes
        .iter()
        .enumerate()
        .filter(|&(_, &v)| !doms[v].is_fixed())
        .map(|(pos, &v)| (v, pos))
        .collect();
    fv.sort_unstable_by_key(|&(v, _)| v);
    let vars: Vec<usize> = fv.iter().map(|&(v, _)| v).collect();

    let mut rows: Vec<u64> = Vec::new();
    for &config in cn.support(tensor) {
        if (config & fmask) != fval {
            continue;
        }
        let mut row = 0u64;
        for (j, &(_, pos)) in fv.iter().enumerate() {
            if (config >> pos) & 1 == 1 {
                row |= 1u64 << j;
            }
        }
        rows.push(row);
    }
    rows.sort_unstable();
    rows.dedup();
    Relation { vars, rows }
}

/// The full boolean relation of a dense truth table over `var_axes` (no domain
/// slicing): rows = configs where `dense` is true, re-encoded over `var_axes` SORTED
/// ascending (canonical bit order, matching `tensor_relation`), deduplicated.
pub fn dense_relation(var_axes: &[usize], dense: &[bool]) -> Relation {
    let mut fv: Vec<(usize, usize)> = var_axes
        .iter()
        .enumerate()
        .map(|(pos, &v)| (v, pos))
        .collect();
    fv.sort_unstable_by_key(|&(v, _)| v);
    let vars: Vec<usize> = fv.iter().map(|&(v, _)| v).collect();

    let mut rows: Vec<u64> = Vec::new();
    for (config, &sat) in dense.iter().enumerate() {
        if !sat {
            continue;
        }
        let mut row = 0u64;
        for (j, &(_, pos)) in fv.iter().enumerate() {
            if (config >> pos) & 1 == 1 {
                row |= 1u64 << j;
            }
        }
        rows.push(row);
    }
    rows.sort_unstable();
    rows.dedup();
    Relation { vars, rows }
}

/// The boolean relation of a tensor's sparse `support` (no domain slicing): each
/// support `config` is a bitmask over `var_axes` order; rows are re-encoded over
/// `var_axes` SORTED ascending (canonical bit order, matching `tensor_relation` /
/// `dense_relation`), deduplicated. The sparse counterpart of `dense_relation`.
pub fn support_relation(var_axes: &[usize], support: &[u32]) -> Relation {
    let mut fv: Vec<(usize, usize)> = var_axes
        .iter()
        .enumerate()
        .map(|(pos, &v)| (v, pos))
        .collect();
    fv.sort_unstable_by_key(|&(v, _)| v);
    let vars: Vec<usize> = fv.iter().map(|&(v, _)| v).collect();

    let mut rows: Vec<u64> = Vec::new();
    for &config in support {
        let mut row = 0u64;
        for (j, &(_, pos)) in fv.iter().enumerate() {
            if (config >> pos) & 1 == 1 {
                row |= 1u64 << j;
            }
        }
        rows.push(row);
    }
    rows.sort_unstable();
    rows.dedup();
    Relation { vars, rows }
}

#[inline]
fn shared_count(a: &[usize], b: &[usize]) -> usize {
    a.iter().filter(|v| b.binary_search(v).is_ok()).count()
}

fn sorted_union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = a.iter().chain(b.iter()).copied().collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Project a `row` (over `vars`) onto `sub` (a subset of `vars`), returning a
/// bitmask over `sub` order. `sub` entries must all be present in `vars`.
#[inline]
fn project_key(vars: &[usize], row: u64, sub: &[usize]) -> u64 {
    let mut key = 0u64;
    for (j, &v) in sub.iter().enumerate() {
        let pos = vars.binary_search(&v).expect("projection var present");
        if (row >> pos) & 1 == 1 {
            key |= 1u64 << j;
        }
    }
    key
}

/// Relational join: rows of `a` and `b` that agree on shared variables, merged
/// over `a.vars ∪ b.vars`.
fn join(a: &Relation, b: &Relation) -> Relation {
    let out_vars = sorted_union(&a.vars, &b.vars);
    debug_assert!(
        out_vars.len() <= 64,
        "joined relation exceeds the 64-variable u64 cap"
    );
    let shared: Vec<usize> = a
        .vars
        .iter()
        .copied()
        .filter(|v| b.vars.binary_search(v).is_ok())
        .collect();

    // Precompute, for each output var, where to read its bit from.
    // (from_a, position-within-that-relation).
    let plan: Vec<(bool, usize)> = out_vars
        .iter()
        .map(|&v| match a.vars.binary_search(&v) {
            Ok(pa) => (true, pa),
            Err(_) => (false, b.vars.binary_search(&v).expect("var in a or b")),
        })
        .collect();

    // Bucket b's rows by their shared-variable projection.
    let mut buckets: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for &br in &b.rows {
        buckets
            .entry(project_key(&b.vars, br, &shared))
            .or_default()
            .push(br);
    }

    let mut rows: Vec<u64> = Vec::new();
    for &ar in &a.rows {
        let key = project_key(&a.vars, ar, &shared);
        if let Some(brs) = buckets.get(&key) {
            for &br in brs {
                let mut row = 0u64;
                for (j, &(from_a, pos)) in plan.iter().enumerate() {
                    let bit = if from_a {
                        (ar >> pos) & 1
                    } else {
                        (br >> pos) & 1
                    };
                    if bit == 1 {
                        row |= 1u64 << j;
                    }
                }
                rows.push(row);
            }
        }
    }
    rows.sort_unstable();
    rows.dedup();
    Relation {
        vars: out_vars,
        rows,
    }
}

/// Fold all relations into one. Order-independent; the greedy "most-shared-vars
/// next" pick only avoids needless Cartesian-product intermediates.
pub fn join_all(mut rels: Vec<Relation>) -> Relation {
    debug_assert!(
        !rels.is_empty(),
        "contract_region called with an empty region"
    );
    let mut acc = rels.swap_remove(0);
    while !rels.is_empty() {
        let mut pick = 0usize;
        let mut pick_shared = shared_count(&acc.vars, &rels[0].vars);
        for i in 1..rels.len() {
            let s = shared_count(&acc.vars, &rels[i].vars);
            if s > pick_shared {
                pick = i;
                pick_shared = s;
            }
        }
        let r = rels.swap_remove(pick);
        acc = join(&acc, &r);
    }
    acc
}

/// Join all `rels` on shared variables, then project onto `keep` (a subset of the
/// union of all rels' vars, ascending). The single contraction primitive shared by
/// `contract_region` and `canonicalize`'s VE step. Binary-join internals for now
/// (`join_all`); the signature admits a generic-join kernel later without changing
/// callers. Precondition: `rels` is non-empty (inherited from `join_all`).
pub fn contract(rels: Vec<Relation>, keep: &[usize]) -> Relation {
    join_all(rels).project(keep)
}

/// Contract a region: the satisfiable configurations over its unfixed variables.
/// Port of `contraction.jl::contract_region` + `contract_tensors`.
pub fn contract_region(
    cn: &ConstraintNetwork,
    region: &Region,
    doms: &[DomainMask],
) -> (Vec<u64>, Vec<usize>) {
    let output_vars: Vec<usize> = region
        .vars
        .iter()
        .copied()
        .filter(|&v| !doms[v].is_fixed())
        .collect();
    debug_assert!(
        output_vars.len() <= 64,
        "region config exceeds the 64-variable u64 cap"
    );

    let rels: Vec<Relation> = region
        .tensors
        .iter()
        .map(|&tid| tensor_relation(cn, &cn.tensors[tid], doms))
        .collect();
    let contracted = contract(rels, &output_vars);
    (contracted.rows, output_vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    const OR2: [bool; 4] = [false, true, true, true]; // x ∨ y

    // Brute-force reference: all configs over `vars` (ascending) satisfying every
    // region tensor, encoded as a bitmask over `vars` order.
    fn brute(cn: &ConstraintNetwork, region: &Region, doms: &[DomainMask]) -> Vec<u64> {
        let vars: Vec<usize> = region
            .vars
            .iter()
            .copied()
            .filter(|&v| !doms[v].is_fixed())
            .collect();
        let mut out = Vec::new();
        for cfg in 0u64..(1u64 << vars.len()) {
            // assignment of region vars
            let val = |v: usize| -> u32 {
                if let Some(pos) = vars.iter().position(|&x| x == v) {
                    ((cfg >> pos) & 1) as u32
                } else {
                    // fixed var
                    match doms[v] {
                        DomainMask::D1 => 1,
                        _ => 0,
                    }
                }
            };
            let ok = region.tensors.iter().all(|&tid| {
                let t = &cn.tensors[tid];
                let mut idx = 0u32;
                for (i, &v) in t.var_axes.iter().enumerate() {
                    idx |= val(v) << i;
                }
                cn.is_sat(t, idx)
            });
            if ok {
                out.push(cfg);
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn single_tensor_region_is_its_support() {
        let cn = setup_problem(2, vec![vec![0, 1]], vec![OR2.to_vec()]);
        let region = Region {
            id: 0,
            tensors: vec![0],
            vars: vec![0, 1],
        };
        let doms = vec![DomainMask::BOTH; 2];
        let (configs, output_vars) = contract_region(&cn, &region, &doms);
        assert_eq!(output_vars, vec![0, 1]);
        assert_eq!(configs, vec![1, 2, 3]); // OR support: all but (0,0)
    }

    #[test]
    fn two_tensor_join_matches_bruteforce() {
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![OR2.to_vec(), OR2.to_vec()],
        );
        let region = Region {
            id: 1,
            tensors: vec![0, 1],
            vars: vec![0, 1, 2],
        };
        let doms = vec![DomainMask::BOTH; 3];
        let (configs, output_vars) = contract_region(&cn, &region, &doms);
        assert_eq!(output_vars, vec![0, 1, 2]);
        assert_eq!(configs, brute(&cn, &region, &doms));
        assert_eq!(configs, vec![2, 3, 5, 6, 7]); // (x0∨x1)∧(x1∨x2)
    }

    #[test]
    fn fixed_var_is_sliced_out_of_output() {
        // Fix v0 = 1; the OR over [0,1] is then satisfied for any v1.
        let cn = setup_problem(2, vec![vec![0, 1]], vec![OR2.to_vec()]);
        let region = Region {
            id: 1,
            tensors: vec![0],
            vars: vec![0, 1],
        };
        let doms = vec![DomainMask::D1, DomainMask::BOTH];
        let (configs, output_vars) = contract_region(&cn, &region, &doms);
        assert_eq!(output_vars, vec![1]); // v0 fixed, dropped from output
        assert_eq!(configs, vec![0, 1]); // v1 free either way
    }

    #[test]
    fn dense_relation_reencodes_unsorted_axes() {
        // Tensor over var_axes = [2, 0] (UNSORTED), dense over (bit0=v2, bit1=v0).
        // dense true at config 0b10 (v2=0,v0=1) and 0b01 (v2=1,v0=0).
        // Relation must be over sorted vars [0, 2] with rows re-encoded:
        //   (v0=1,v2=0) -> bit0(v0)=1,bit1(v2)=0 -> 0b01 = 1
        //   (v0=0,v2=1) -> bit0(v0)=0,bit1(v2)=1 -> 0b10 = 2
        let dense = vec![false, true, true, false]; // idx: 00,01,10,11 over (v2,v0)
        let rel = dense_relation(&[2, 0], &dense);
        assert_eq!(rel.vars, vec![0, 2]);
        let mut rows = rel.rows.clone();
        rows.sort_unstable();
        assert_eq!(rows, vec![1u64, 2u64]);
    }

    #[test]
    fn relation_project_reencodes_and_dedups() {
        // vars [0,1,2], bit j = vars[j]: rows encode (v0,v1,v2).
        //   0b011 -> v0=1,v1=1,v2=0 ; 0b111 -> all 1 ; 0b101 -> v0=1,v1=0,v2=1
        let rel = Relation {
            vars: vec![0, 1, 2],
            rows: vec![0b011, 0b111, 0b101],
        };
        // Project onto [0,2] (new bit0=v0, bit1=v2):
        //   0b011 -> (v0=1,v2=0)=0b01 ; 0b111 -> (1,1)=0b11 ; 0b101 -> (1,1)=0b11 (dup)
        let p = rel.project(&[0, 2]);
        assert_eq!(p.vars, vec![0, 2]);
        assert_eq!(p.rows, vec![0b01, 0b11]);
    }

    #[test]
    fn contract_matches_join_all_then_project() {
        // (x0∨x1) over [0,1] and (x1∨x2) over [1,2]; support {1,2,3} each.
        let a = Relation { vars: vec![0, 1], rows: vec![1, 2, 3] };
        let b = Relation { vars: vec![1, 2], rows: vec![1, 2, 3] };
        let keep = vec![0, 2];
        let got = contract(vec![a.clone(), b.clone()], &keep);
        // Reference: join then hand-project (guards the extraction).
        let want = join_all(vec![a, b]).project(&keep);
        assert_eq!(got.vars, want.vars);
        assert_eq!(got.rows, want.rows);
        // Concrete: (x0∨x1)∧(x1∨x2) projected to (x0,x2) allows all four configs.
        assert_eq!(got.vars, vec![0, 2]);
        assert_eq!(got.rows, vec![0, 1, 2, 3]);
    }
}
