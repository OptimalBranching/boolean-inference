use rustc_hash::{FxHashMap, FxHashSet};

use crate::contract::{join, tensor_relation, Relation};
use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;

/// How many top-ranked frontier tensors get an exact tentative join per growth
/// step. Ranking is a cheap proxy (shared vars, sliced-support size); the exact
/// join is the expensive part, so it is bounded.
const JOIN_CANDIDATES: usize = 3;

#[derive(Clone, Debug)]
pub struct Region {
    /// Focus variable the region was grown from.
    pub id: usize,
    /// Region tensor ids, ascending.
    pub tensors: Vec<usize>,
    /// Region variable ids, ascending. Invariant: exactly the joined relation's
    /// `vars` — every var of every region tensor that is unfixed under the doms
    /// the region was grown at. The boundary-grouping lemma in `table.rs` and
    /// the config bit-encoding both depend on this.
    pub vars: Vec<usize>,
}

/// Region vars with at least one incident tensor OUTSIDE the region — the
/// interface through which the rest of the network constrains the region.
/// Computed against the full static `v2t` (tensors whose other vars are
/// currently fixed still count): the grouping lemma in `table.rs` needs every
/// var that ANY external tensor can see to be classified as boundary.
/// Complement (region vars with all tensors inside) = interior vars.
pub fn boundary_vars(cn: &ConstraintNetwork, region: &Region) -> Vec<usize> {
    region
        .vars
        .iter()
        .copied()
        .filter(|&v| {
            cn.v2t[v]
                .iter()
                .any(|&t| region.tensors.binary_search(&t).is_err())
        })
        .collect()
}

/// Grow a region around `focus` under the CURRENT `doms`, budgeted by the row
/// count of its joined relation — the size OB's GreedyMerge is quadratic in —
/// rather than by hops or tensor count. Returns the region and its joined
/// relation: `region.vars == relation.vars` (ascending) and `relation.rows` are
/// exactly the region's satisfiable configs over those vars, doms-sliced.
///
/// Growth: seed with the focus var's cheapest (fewest sliced rows) incident
/// tensor unconditionally, then repeatedly rank the frontier (tensors sharing
/// an unfixed var with the region) by a cheap proxy, exact-join the top few
/// candidates, and absorb the one yielding the fewest rows — while the result
/// stays within `max_rows` rows and 64 vars (the u64-config hard cap; enforced
/// here at runtime, `join` only debug_asserts it).
pub fn grow_region(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    focus: usize,
    max_rows: usize,
) -> (Region, Relation) {
    debug_assert!(!doms[focus].is_fixed(), "focus variable must be unfixed");

    // Per-call memo of doms-sliced tensor relations.
    let mut rels: FxHashMap<usize, Relation> = FxHashMap::default();
    let rel_of = |tid: usize, rels: &mut FxHashMap<usize, Relation>| {
        rels.entry(tid)
            .or_insert_with(|| tensor_relation(cn, &cn.tensors[tid], doms))
            .clone()
    };

    // Seed: cheapest incident tensor, included unconditionally (a region must
    // hold >= 1 tensor; an over-budget single tensor still beats no region).
    let seed = cn.v2t[focus]
        .iter()
        .copied()
        .min_by_key(|&tid| (rel_of(tid, &mut rels).rows.len(), tid))
        .expect("an unfixed var is incident to at least one tensor");
    let mut in_region: FxHashSet<usize> = FxHashSet::default();
    in_region.insert(seed);
    let mut tensors = vec![seed];
    let mut acc = rel_of(seed, &mut rels);

    loop {
        // Frontier: tensors incident to a region var, not yet absorbed.
        let mut frontier: Vec<usize> = Vec::new();
        let mut seen: FxHashSet<usize> = FxHashSet::default();
        for &v in &acc.vars {
            for &tid in &cn.v2t[v] {
                if !in_region.contains(&tid) && seen.insert(tid) {
                    frontier.push(tid);
                }
            }
        }
        if frontier.is_empty() {
            break;
        }

        // Cheap ranking: most shared unfixed vars first (tight joins grow rows
        // least), then smallest sliced support, then tid for determinism.
        // Scores are computed once per tensor, then sorted.
        let mut scored: Vec<(usize, usize, usize)> = frontier
            .into_iter()
            .map(|tid| {
                let shared = cn.tensors[tid]
                    .var_axes
                    .iter()
                    .filter(|&&v| acc.vars.binary_search(&v).is_ok())
                    .count();
                let rows = rel_of(tid, &mut rels).rows.len();
                (usize::MAX - shared, rows, tid)
            })
            .collect();
        scored.sort_unstable();

        // Exact-join down the ranked list: pick the min-row join among the top
        // few, but if none of those fits the budget keep scanning first-fit, so
        // growth is MAXIMAL — it stops only when NO frontier tensor fits. Every
        // absorbed tensor conditions the table (rows can only stay or shrink on
        // shared vars), so maximality directly sharpens the branching rule.
        let mut best: Option<(usize, Relation)> = None;
        for (scanned, &(_, _, tid)) in scored.iter().enumerate() {
            if best.is_some() && scanned >= JOIN_CANDIDATES {
                break;
            }
            let rel = rel_of(tid, &mut rels);
            // Hard 64-var cap BEFORE joining: u64 rows silently corrupt past it.
            let new_vars = rel
                .vars
                .iter()
                .filter(|&&v| acc.vars.binary_search(&v).is_err())
                .count();
            if acc.vars.len() + new_vars > 64 {
                continue;
            }
            let joined = join(&acc, &rel);
            if joined.rows.len() > max_rows {
                continue;
            }
            if best
                .as_ref()
                .map(|(_, b)| joined.rows.len() < b.rows.len())
                .unwrap_or(true)
            {
                best = Some((tid, joined));
            }
        }
        match best {
            Some((tid, joined)) => {
                in_region.insert(tid);
                tensors.push(tid);
                acc = joined;
            }
            None => break, // nothing on the frontier fits: the region is done
        }
    }

    tensors.sort_unstable();
    (
        Region {
            id: focus,
            tensors,
            vars: acc.vars.clone(),
        },
        acc,
    )
}

/// Per-focus-variable cache of the region grown at the ROOT `initial_doms`
/// (row-budgeted, maximal) and its satisfiable configs. Regions are grown once
/// and reused down the whole search tree; at depth, `table.rs` conditions the
/// cached configs on the currently-fixed vars — which is what keeps deep
/// branching tables small at zero contraction cost. ENCODING INVARIANT: cached
/// configs are bit-encoded over `region.vars` (ascending), all of which are
/// unfixed at `initial_doms`; conditioning at depth relies on this. Do NOT
/// rebuild the cache from non-root doms.
pub struct RegionCache {
    pub initial_doms: Vec<DomainMask>,
    pub var_to_region: Vec<Option<Region>>,
    pub var_to_configs: Vec<Option<Vec<u64>>>,
    pub max_rows: usize,
}

impl RegionCache {
    pub fn new(cn: &ConstraintNetwork, doms: &[DomainMask], max_rows: usize) -> RegionCache {
        let n = cn.vars.len();
        RegionCache {
            initial_doms: doms.to_vec(),
            var_to_region: (0..n).map(|_| None).collect(),
            var_to_configs: (0..n).map(|_| None).collect(),
            max_rows,
        }
    }

    /// Lazily grow the region for `var_id` (at `initial_doms`) once.
    pub fn ensure_region(&mut self, cn: &ConstraintNetwork, var_id: usize) {
        if self.var_to_region[var_id].is_none() {
            let (region, rel) = grow_region(cn, &self.initial_doms, var_id, self.max_rows);
            self.var_to_region[var_id] = Some(region);
            self.var_to_configs[var_id] = Some(rel.rows);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::contract_region;
    use crate::domain::DomainMask;
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
    fn boundary_vars_classifies_interface_vs_interior() {
        let cn = chain(); // T0[0,1] T1[1,2] T2[2,3] T3[3,4]
                          // Whole chain: every incident tensor is inside -> no boundary.
        let full = Region {
            id: 2,
            tensors: vec![0, 1, 2, 3],
            vars: vec![0, 1, 2, 3, 4],
        };
        assert!(boundary_vars(&cn, &full).is_empty());
        // Middle tensor only: both its vars touch tensors outside.
        let mid = Region {
            id: 1,
            tensors: vec![1],
            vars: vec![1, 2],
        };
        assert_eq!(boundary_vars(&cn, &mid), vec![1, 2]);
        // Left half {T0,T1}: vars 0,1 are interior, var 2 touches T2 outside.
        let left = Region {
            id: 0,
            tensors: vec![0, 1],
            vars: vec![0, 1, 2],
        };
        assert_eq!(boundary_vars(&cn, &left), vec![2]);
    }

    #[test]
    fn generous_budget_absorbs_the_whole_chain() {
        let cn = chain();
        let doms = vec![DomainMask::BOTH; 5];
        let (region, rel) = grow_region(&cn, &doms, 2, 1 << 10);
        assert_eq!(region.tensors, vec![0, 1, 2, 3]);
        assert_eq!(region.vars, vec![0, 1, 2, 3, 4]);
        assert_eq!(region.vars, rel.vars);
        // The relation is exactly the region's satisfiable configs.
        let (configs, output_vars) = contract_region(&cn, &region, &doms);
        assert_eq!(rel.vars, output_vars);
        assert_eq!(rel.rows, configs);
    }

    #[test]
    fn budget_stops_growth() {
        let cn = chain();
        let doms = vec![DomainMask::BOTH; 5];
        // A single OR has 3 rows; joining a second gives 5 rows over 3 vars.
        // Budget 3 admits the seed only.
        let (region, rel) = grow_region(&cn, &doms, 2, 3);
        assert_eq!(region.tensors.len(), 1);
        assert!(rel.rows.len() <= 3);
        assert_eq!(region.vars, rel.vars);
        // Budget 5 admits exactly one join.
        let (region5, rel5) = grow_region(&cn, &doms, 2, 5);
        assert_eq!(region5.tensors.len(), 2);
        assert_eq!(rel5.rows.len(), 5);
    }

    #[test]
    fn fixed_vars_are_sliced_out_of_the_region() {
        let cn = chain();
        // Fix var 1 = 1: T0 and T1 are satisfied for any value of vars 0/2, and
        // var 1 must not appear in any grown region.
        let mut doms = vec![DomainMask::BOTH; 5];
        doms[1] = DomainMask::D1;
        let (region, rel) = grow_region(&cn, &doms, 2, 1 << 10);
        assert!(!region.vars.contains(&1));
        assert_eq!(region.vars, rel.vars);
        // Configs match the reference contraction under the same doms.
        let (configs, output_vars) = contract_region(&cn, &region, &doms);
        assert_eq!(rel.vars, output_vars);
        assert_eq!(rel.rows, configs);
    }

    #[test]
    fn grown_relation_matches_reference_contraction_randomized() {
        // Deterministic xorshift; random small networks + random fixings.
        fn next(s: &mut u64) -> u64 {
            let mut x = *s;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = x;
            x
        }
        for seed in 1u64..=200 {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let n_vars = 3 + (next(&mut s) % 4) as usize; // 3..=6
            let n_tensors = 2 + (next(&mut s) % 4) as usize; // 2..=5
            let mut scopes = Vec::new();
            let mut dense = Vec::new();
            for _ in 0..n_tensors {
                let arity = 2 + (next(&mut s) % 2) as usize;
                let mut vs: Vec<usize> = Vec::new();
                while vs.len() < arity {
                    let v = (next(&mut s) % n_vars as u64) as usize;
                    if !vs.contains(&v) {
                        vs.push(v);
                    }
                }
                let rows = 1usize << arity;
                let mut sup = vec![false; rows];
                let mut any = false;
                for r in sup.iter_mut() {
                    if next(&mut s) % 100 < 70 {
                        *r = true;
                        any = true;
                    }
                }
                if !any {
                    sup[(next(&mut s) as usize) % rows] = true;
                }
                scopes.push(vs);
                dense.push(sup);
            }
            let cn = setup_problem(n_vars, scopes, dense);
            let n_cvars = cn.vars.len();
            let mut doms = vec![DomainMask::BOTH; n_cvars];
            for v in 0..n_cvars {
                if next(&mut s) % 4 == 0 {
                    doms[v] = if next(&mut s) & 1 == 1 {
                        DomainMask::D1
                    } else {
                        DomainMask::D0
                    };
                }
            }
            let focus = match (0..n_cvars).find(|&v| !doms[v].is_fixed()) {
                Some(v) => v,
                None => continue,
            };
            let max_rows = 1 + (next(&mut s) % 32) as usize;
            let (region, rel) = grow_region(&cn, &doms, focus, max_rows);
            assert!(!region.tensors.is_empty(), "seed {seed}: empty region");
            assert_eq!(region.vars, rel.vars, "seed {seed}: vars mismatch");
            // Budget respected EXCEPT possibly by the forced seed tensor.
            if region.tensors.len() > 1 {
                assert!(
                    rel.rows.len() <= max_rows,
                    "seed {seed}: budget exceeded after a join"
                );
            }
            // The relation equals the reference contraction of the same region.
            let (configs, output_vars) = contract_region(&cn, &region, &doms);
            assert_eq!(rel.vars, output_vars, "seed {seed}");
            assert_eq!(rel.rows, configs, "seed {seed}");
        }
    }
}
