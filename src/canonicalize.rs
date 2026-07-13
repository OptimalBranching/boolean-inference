//! Static, width-aware constraint-network canonicalizer (bounded-width VE).
//!
//! Port of Julia `bounded_ve_canonicalize` (src/preprocessing/canonicalize.jl).
//! Eliminates a variable `v` by joining all tensors incident to `v` and projecting
//! `v` out (boolean ∃/∧), but only if the elimination width `out.len()+1` is
//! `<= budget_b` (and `out.len() <= 32`, the TruthTable cap). Eligible variables are
//! removed in weighted-min-fill order. `protected` variables (read-out vars, e.g.
//! factor bits) are never eliminated and survive into the result; their values are
//! read off the result's `orig_to_new`. The elimination width `sc = out.len()+1` is
//! exact for the relational join we perform (the largest intermediate is the relation
//! over `neighbors(v) ∪ {v}`), so no contraction-order optimizer is needed.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::contract::{
    join_all, setup_from_relations, support_relation, wjoin_all, Relation, WRelation,
};
use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;
use crate::semiring::Weight;

/// One variable elimination, recorded for model reconstruction: `rel` is the
/// PRE-projection join of all tensors incident to `var` at elimination time,
/// over `out ∪ {var}` in the INPUT network's (compressed) id space — ids are
/// stable during VE; remapping happens only in the final assembly.
pub struct ElimStep {
    pub var: usize,
    pub rel: Relation,
}

/// Result of `bounded_ve_canonicalize`: the reshaped network `cn`, plus a
/// `model` reconstructor that lifts a solve of `cn` back to a full model of the
/// input. The circuit read-out path uses only `cn`; the DIMACS path uses both.
pub struct Canonicalized {
    pub cn: ConstraintNetwork,
    pub model: ModelReconstructor,
}

/// Turns a solve over the canonicalized network's ids into a full model over the
/// INPUT network's (compressed) var ids. Owns the survivor map (input-cn id ->
/// canonicalized id, `None` if eliminated/compressed away) and the elimination
/// stack, so callers never handle those id-spaces by hand.
pub struct ModelReconstructor {
    cn_to_new: Vec<Option<usize>>,
    elim: Vec<ElimStep>,
}

impl ModelReconstructor {
    /// Reconstruct the full model (indexed by input-cn ids) from `solution`
    /// over the canonicalized network's ids. Survivors are read through the
    /// survivor map; eliminated vars are then recovered by replaying the
    /// elimination stack in REVERSE — at each step every out-var of `rel` is
    /// already assigned (it survived VE or was eliminated later, hence replayed
    /// earlier), so a row consistent with the assigned bits exists (the solution
    /// restricted to the out-vars lies in `rel`'s projection) and the eliminated
    /// var is read off it. Pass `&[]` when the canonicalized network is empty.
    pub fn reconstruct(&self, solution: &[DomainMask]) -> Vec<Option<bool>> {
        let mut assignment: Vec<Option<bool>> = vec![None; self.cn_to_new.len()];
        for (c, slot) in self.cn_to_new.iter().enumerate() {
            if let Some(nid) = slot {
                assignment[c] = solution[*nid].value();
            }
        }
        for step in self.elim.iter().rev() {
            let rel = &step.rel;
            let vpos = rel
                .vars
                .binary_search(&step.var)
                .expect("eliminated var is in its own elimination relation");
            let mut mask = 0u64;
            let mut val = 0u64;
            for (j, &v) in rel.vars.iter().enumerate() {
                if v == step.var {
                    continue;
                }
                if let Some(b) = assignment[v] {
                    mask |= 1u64 << j;
                    if b {
                        val |= 1u64 << j;
                    }
                }
            }
            let row = rel.rows.iter().copied().find(|&r| (r & mask) == val);
            debug_assert!(row.is_some(), "a consistent elimination row must exist");
            assignment[step.var] = Some(row.map(|r| (r >> vpos) & 1 == 1).unwrap_or(false));
        }
        assignment
    }
}

/// Sorted-unique union of the incident tensors' vars, minus `v` (the produced axes).
fn out_vars(live: &[Relation], tids: &[usize], v: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &t in tids {
        for &x in &live[t].vars {
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
fn fill_count(live: &[Relation], v2t: &[Vec<usize>], active: &[bool], out: &[usize]) -> usize {
    let mut f = 0usize;
    for i in 0..out.len() {
        for j in (i + 1)..out.len() {
            let (a, b) = (out[i], out[j]);
            let share = v2t[a]
                .iter()
                .any(|&t| active[t] && live[t].vars.contains(&b));
            if !share {
                f += 1;
            }
        }
    }
    f
}

/// `Some((fill, sc))` if `v` is eligible to eliminate now, else `None`.
fn score(
    live: &[Relation],
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
///
/// This is the INITIALIZATION contract: a one-time global rewrite preserving
/// the solution set over surviving vars (all solutions over eliminated vars are
/// recoverable via `reconstruct_eliminated`). Region construction during
/// branching, by contrast, only READS the network. Returns `None` when an
/// elimination proves the instance UNSAT (an incident join with no rows).
pub fn bounded_ve_canonicalize(
    cn: &ConstraintNetwork,
    budget_b: usize,
    protected: &[usize],
) -> Option<Canonicalized> {
    let nv = cn.n_vars;
    let mut elim: Vec<ElimStep> = Vec::new();

    let mut live: Vec<Relation> = cn
        .tensors
        .iter()
        .map(|t| support_relation(&t.var_axes, cn.support(t)))
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

        // Bucket-contract: join incident tensors, record the pre-projection
        // relation for model reconstruction, then project v out. An empty join
        // means no assignment satisfies the bucket — the instance is UNSAT.
        let incident: Vec<Relation> = tids.iter().map(|&t| live[t].clone()).collect();
        let joined = join_all(incident);
        if joined.rows.is_empty() {
            return None;
        }
        let merged = joined.project(&out);
        elim.push(ElimStep {
            var: v,
            rel: joined,
        });

        // Merge in place: reuse the first incident slot, deactivate the rest.
        let keep = tids[0];
        for &t in &tids {
            let axes = live[t].vars.clone();
            for x in axes {
                v2t[x].retain(|&tt| tt != t);
            }
        }
        for &t in &tids[1..] {
            active[t] = false;
        }
        live[keep] = merged;
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

    // Finalize: hand surviving relations to setup_from_relations for dedup +
    // compression — no dense table is ever materialized. A surviving relation
    // with no rows is an input contradiction (empty-support tensor) => UNSAT;
    // 0-ary survivors (a fully-eliminated connected component) are tautologies
    // by now and are dropped rather than assembled.
    let mut surviving: Vec<Relation> = Vec::new();
    for (rel, keep) in live.into_iter().zip(active) {
        if !keep {
            continue;
        }
        if rel.rows.is_empty() {
            return None; // an input contradiction (empty-support tensor)
        }
        if !rel.vars.is_empty() {
            surviving.push(rel); // drop 0-ary tautologies rather than assemble them
        }
    }
    let new_cn = setup_from_relations(nv, surviving);

    // `new_cn.orig_to_new` is indexed by INPUT-cn ids (setup_from_relations
    // treats them as its "orig") — exactly the survivor map reconstruction needs.
    let cn_to_new = new_cn.orig_to_new.clone();

    // Compose orig->cn (cn.orig_to_new) with cn->new (new_cn.orig_to_new).
    let mut orig_to_new = vec![None; cn.orig_to_new.len()];
    for (orig, &cnid) in cn.orig_to_new.iter().enumerate() {
        if let Some(c) = cnid {
            orig_to_new[orig] = new_cn.orig_to_new[c];
        }
    }
    Some(Canonicalized {
        cn: ConstraintNetwork {
            orig_to_new,
            ..new_cn
        },
        model: ModelReconstructor { cn_to_new, elim },
    })
}

/// Result of a WEIGHTED bounded VE (`bounded_ve_canonicalize_weighted`), the
/// counting analogue of `Canonicalized`. `surviving` are the reshaped WEIGHTED
/// relations over the INPUT (compressed) var ids — variable elimination here
/// MARGINALIZES (sums the eliminated var's multiplicities), so the surviving
/// weights already carry the count contribution of every eliminated variable.
/// `scalar` is the running global multiplier: the total weight of every
/// fully-eliminated (0-ary) connected component, folded out of `surviving`.
///
/// Counting invariant (the VE-invariance test's contract):
/// `models(input) == scalar × Σ_{σ over surviving vars} ∏_rel weight_rel(σ)`
/// — VE changes neither side. No `ModelReconstructor` is produced: counting
/// wants the number of models, not a witness, so the elimination stack is not
/// retained.
pub struct WeightedCanonicalized<W> {
    /// Surviving weighted relations over the input cn's (compressed) var ids.
    pub surviving: Vec<WRelation<W>>,
    /// Global multiplier: product of every fully-eliminated component's total.
    pub scalar: W,
    /// The input cn's variable count (the id space `surviving` ranges over).
    pub n_vars: usize,
}

/// Weighted-min-fill over WEIGHTED relations: number of `out` pairs that do NOT
/// already share an active relation. Structural only (reads `.vars`), identical
/// to the boolean `fill_count`; kept separate so the boolean VE path stays
/// untouched (a hard constraint of the counting work).
fn wfill_count<W>(
    live: &[WRelation<W>],
    v2t: &[Vec<usize>],
    active: &[bool],
    out: &[usize],
) -> usize {
    let mut f = 0usize;
    for i in 0..out.len() {
        for j in (i + 1)..out.len() {
            let (a, b) = (out[i], out[j]);
            let share = v2t[a]
                .iter()
                .any(|&t| active[t] && live[t].vars.contains(&b));
            if !share {
                f += 1;
            }
        }
    }
    f
}

/// Sorted-unique union of the incident weighted relations' vars, minus `v`.
fn wout_vars<W>(live: &[WRelation<W>], tids: &[usize], v: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &t in tids {
        for &x in &live[t].vars {
            if x != v {
                out.push(x);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// `Some((fill, sc))` if `v` is eligible to eliminate now, else `None`. Counting
/// mode has NO protected variables (the decision engine protects read-out bits;
/// a counter reads nothing back), so `is_protected` is gone — every variable
/// within the width budget is fair game.
fn wscore<W>(
    live: &[WRelation<W>],
    v2t: &[Vec<usize>],
    active: &[bool],
    budget_b: usize,
    v: usize,
) -> Option<(usize, usize)> {
    let tids = active_incident(v2t, active, v);
    if tids.is_empty() {
        return None;
    }
    let out = wout_vars(live, &tids, v);
    let sc = out.len() + 1;
    if out.len() <= 32 && sc <= budget_b {
        Some((wfill_count(live, v2t, active, &out), sc))
    } else {
        None
    }
}

/// WEIGHTED bounded-width variable elimination — the counting analogue of
/// `bounded_ve_canonicalize`. Eliminates every variable eligible within the
/// width `budget_b` in weighted-min-fill order, joining incident WEIGHTED
/// relations (product of multiplicities) and projecting the eliminated variable
/// out by SUMMING its rows' weights (marginalization, not existence-projection).
///
/// Because eliminating a variable free in its bucket sums the two value copies,
/// its `free_factor` (×2 for plain counting) falls out of the weighted project
/// with no special case. A fully-eliminated connected component collapses to a
/// 0-ary relation whose total weight is folded into `scalar` rather than kept.
/// Returns `None` when an elimination proves the instance UNSAT (an empty
/// bucket) — the count is then 0.
///
/// This is the counting front-end's preprocessing rewrite (M2.1), validated by
/// the VE-invariance property test (count unchanged at every budget). Its
/// residual survivors + scalar are handed to `WeightedNetwork` and counted by
/// `bbcount`; the search still runs on the 0/1 support skeleton, weights consumed
/// only at the counting sites (counting design doc §2 / §7 blocker 1).
pub fn bounded_ve_canonicalize_weighted<W: Weight>(
    cn: &ConstraintNetwork,
    budget_b: usize,
) -> Option<WeightedCanonicalized<W>> {
    let live: Vec<WRelation<W>> = cn
        .tensors
        .iter()
        .map(|t| WRelation::from_relation(&support_relation(&t.var_axes, cn.support(t))))
        .collect();
    bounded_ve_canonicalize_weighted_rels(cn.n_vars, live, budget_b)
}

/// Weighted bounded VE seeded from arbitrary WEIGHTED relations over `n_vars`
/// (ids `0..n_vars`) — the counting FRONT-END entry (M2.1/M2.2). Unlike
/// `bounded_ve_canonicalize_weighted`, which lifts every 0/1 tensor to weight
/// `one()`, this accepts input relations that already carry non-unit weights, so
/// the M2.2 literal-weight tensors (`w(v), w(¬v)` per variable) enter and are
/// folded exactly like any other factor. The 0/1 support skeleton is recovered
/// from each relation's configs; weights ride along and MARGINALIZE (sum) on
/// projection. Same width budget, same UNSAT short-circuit as the boolean VE.
pub fn bounded_ve_canonicalize_weighted_rels<W: Weight>(
    n_vars: usize,
    input: Vec<WRelation<W>>,
    budget_b: usize,
) -> Option<WeightedCanonicalized<W>> {
    let nv = n_vars;
    let mut live: Vec<WRelation<W>> = input;
    let mut active: Vec<bool> = vec![true; live.len()];
    // Variable -> incident relation indices, rebuilt from the input relations'
    // vars (the input need not be an assembled network's `v2t`).
    let mut v2t: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (t, rel) in live.iter().enumerate() {
        for &v in &rel.vars {
            v2t[v].push(t);
        }
    }
    let mut scalar = W::one();

    let mut heap: BinaryHeap<Reverse<(usize, usize, usize)>> = BinaryHeap::new();
    for v in 0..nv {
        if let Some((fill, sc)) = wscore(&live, &v2t, &active, budget_b, v) {
            heap.push(Reverse((fill, sc, v)));
        }
    }

    while let Some(Reverse((fill, sc, v))) = heap.pop() {
        match wscore(&live, &v2t, &active, budget_b, v) {
            Some((f, s)) if f == fill && s == sc => {}
            _ => continue, // stale heap entry
        }
        let tids = active_incident(&v2t, &active, v);
        let out = wout_vars(&live, &tids, v);

        // Bucket-contract: join incident weighted relations, then project the
        // eliminated var out with weight-SUM. An empty join is UNSAT.
        let incident: Vec<WRelation<W>> = tids.iter().map(|&t| live[t].clone()).collect();
        let joined = wjoin_all(incident);
        if joined.rows.is_empty() {
            return None;
        }
        let merged = joined.project(&out);

        let keep = tids[0];
        for &t in &tids {
            let axes = live[t].vars.clone();
            for x in axes {
                v2t[x].retain(|&tt| tt != t);
            }
        }
        for &t in &tids[1..] {
            active[t] = false;
        }
        live[keep] = merged;
        for &x in &out {
            v2t[x].push(keep);
        }

        for &u in &out {
            if let Some((f, s)) = wscore(&live, &v2t, &active, budget_b, u) {
                heap.push(Reverse((f, s, u)));
            }
        }
    }

    // Finalize: an empty-support survivor is a contradiction (count 0); a 0-ary
    // survivor (a fully-eliminated component) is a scalar folded into the global
    // multiplier — the counting analogue of the boolean path's "drop tautology".
    let mut surviving: Vec<WRelation<W>> = Vec::new();
    for (rel, keep) in live.into_iter().zip(active) {
        if !keep {
            continue;
        }
        if rel.rows.is_empty() {
            return None;
        }
        if rel.vars.is_empty() {
            scalar = scalar.mul(&rel.total());
        } else {
            surviving.push(rel);
        }
    }
    Some(WeightedCanonicalized {
        surviving,
        scalar,
        n_vars: nv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;
    use crate::semiring::{BigCount, Weight};
    use std::collections::HashSet;

    const OR2: [bool; 4] = [false, true, true, true]; // x ∨ y

    /// All satisfying assignments of `cn` projected onto original vars `orig_vars`,
    /// as a set of bitmasks (bit j = orig_vars[j]). Brute force over compressed vars.
    fn solutions_projected(cn: &ConstraintNetwork, orig_vars: &[usize]) -> HashSet<u64> {
        let n = cn.n_vars;
        let mut out = HashSet::new();
        for cfg in 0u64..(1u64 << n) {
            let ok = cn.tensors.iter().all(|t| {
                let mut idx = 0u32;
                for (i, &v) in t.var_axes.iter().enumerate() {
                    if (cfg >> v) & 1 == 1 {
                        idx |= 1 << i;
                    }
                }
                cn.is_sat(t, idx)
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
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[0, 2])
            .expect("SAT-preserving")
            .cn;
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
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[1])
            .expect("SAT-preserving")
            .cn; // protect x1
        assert!(out.orig_to_new[1].is_some(), "protected x1 must survive");
    }

    #[test]
    fn budget_one_eliminates_nothing() {
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 1, &[])
            .expect("SAT-preserving")
            .cn;
        // every elimination needs sc = out.len()+1 >= 2 > 1, so all vars survive.
        assert_eq!(out.n_vars, cn.n_vars);
    }

    #[test]
    fn solutions_preserved_over_protected_with_elimination() {
        // 4-var chain (x0∨x1)∧(x1∨x2)∧(x2∨x3); protect {x0, x3}; budget 3.
        let cn = setup_problem(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![OR2.to_vec(), OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[0, 3])
            .expect("SAT-preserving")
            .cn;
        assert!(out.n_vars < cn.n_vars, "some vars eliminated");
        assert!(out.orig_to_new[0].is_some() && out.orig_to_new[3].is_some());
        assert_eq!(
            solutions_projected(&cn, &[0, 3]),
            solutions_projected(&out, &[0, 3]),
            "solution set projected to protected vars must be preserved"
        );
    }

    #[test]
    fn contradictory_instance_returns_none() {
        // (x0) ∧ (¬x0): eliminating x0 joins the two unit relations to an empty
        // bucket -> UNSAT short-circuit.
        let cn = setup_problem(
            1,
            vec![vec![0], vec![0]],
            vec![vec![false, true], vec![true, false]],
        );
        assert!(bounded_ve_canonicalize(&cn, 3, &[]).is_none());
    }

    #[test]
    fn reconstruction_extends_solutions_to_eliminated_vars() {
        // Random small networks: eliminate aggressively, brute-force the
        // canonicalized network, reconstruct, and check the FULL assignment
        // satisfies the ORIGINAL network. Deterministic xorshift.
        fn next(s: &mut u64) -> u64 {
            let mut x = *s;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = x;
            x
        }
        for seed in 1u64..=300 {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let n_vars = 3 + (next(&mut s) % 4) as usize; // 3..=6
            let n_tensors = 2 + (next(&mut s) % 4) as usize; // 2..=5
            let mut scopes = Vec::new();
            let mut dense = Vec::new();
            for _ in 0..n_tensors {
                let arity = 1 + (next(&mut s) % 3) as usize; // 1..=3
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
                    if next(&mut s) % 100 < 60 {
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
            let n_cn = cn.n_vars;
            let budget = 2 + (next(&mut s) % 4) as usize; // 2..=5

            // Ground truth: is the original network satisfiable at all?
            let orig_sat = (0u64..(1u64 << n_cn)).any(|cfg| satisfies(&cn, cfg));

            let canon = match bounded_ve_canonicalize(&cn, budget, &[]) {
                Some(c) => c,
                None => {
                    assert!(!orig_sat, "seed {seed}: VE said UNSAT but instance is SAT");
                    continue;
                }
            };
            // Brute-force a model of the canonicalized network (if any).
            let nk = canon.cn.n_vars;
            let model = (0u64..(1u64 << nk)).find(|&cfg| satisfies(&canon.cn, cfg));
            match model {
                None => {
                    assert!(!orig_sat, "seed {seed}: post-VE UNSAT but instance is SAT");
                }
                Some(cfg) => {
                    assert!(orig_sat, "seed {seed}: post-VE SAT but instance is UNSAT");
                    // Encode the brute-forced model as a DomainMask solution and
                    // lift it back through the reconstructor.
                    let solution: Vec<DomainMask> = (0..nk)
                        .map(|i| {
                            if (cfg >> i) & 1 == 1 {
                                DomainMask::D1
                            } else {
                                DomainMask::D0
                            }
                        })
                        .collect();
                    let assignment = canon.model.reconstruct(&solution);
                    let mut full = 0u64;
                    for (c, a) in assignment.iter().enumerate() {
                        if a.unwrap_or(false) {
                            full |= 1u64 << c;
                        }
                    }
                    assert!(
                        satisfies(&cn, full),
                        "seed {seed}: reconstructed assignment violates the original network"
                    );
                }
            }
        }
    }

    /// Does `cfg` (bit v = value of compressed var v) satisfy every tensor of `cn`?
    fn satisfies(cn: &ConstraintNetwork, cfg: u64) -> bool {
        cn.tensors.iter().all(|t| {
            let mut idx = 0u32;
            for (i, &v) in t.var_axes.iter().enumerate() {
                if (cfg >> v) & 1 == 1 {
                    idx |= 1 << i;
                }
            }
            cn.is_sat(t, idx)
        })
    }

    /// Weighted count of `surviving` × `scalar`: Σ over assignments of the
    /// survivor vars of ∏ matching row weights, all times the global scalar. The
    /// brute-force oracle for weighted VE (marginalization done right).
    fn weighted_ve_count(wc: &WeightedCanonicalized<BigCount>) -> BigCount {
        let mut svars: Vec<usize> = wc
            .surviving
            .iter()
            .flat_map(|r| r.vars.iter().copied())
            .collect();
        svars.sort_unstable();
        svars.dedup();
        assert!(svars.len() <= 20, "brute survivor enumeration capped");
        let mut total = BigCount::zero();
        for cfg in 0u64..(1u64 << svars.len()) {
            let mut w = BigCount::one();
            for rel in &wc.surviving {
                let mut key = 0u64;
                for (j, &v) in rel.vars.iter().enumerate() {
                    let pos = svars.iter().position(|&x| x == v).unwrap();
                    if (cfg >> pos) & 1 == 1 {
                        key |= 1u64 << j;
                    }
                }
                match rel.rows.iter().find(|(c, _)| *c == key) {
                    Some((_, rw)) => w = w.mul(rw),
                    None => {
                        w = BigCount::zero();
                        break;
                    }
                }
            }
            total.add(&w);
        }
        wc.scalar.mul(&total)
    }

    #[test]
    fn weighted_ve_preserves_the_model_count_at_every_budget() {
        // The VE-invariance guarantee (§5/§7): the exact model count is unchanged
        // by weighted bounded VE, at any width budget. For 300 random small
        // networks and budgets 2..=6, the weighted survivor count × scalar must
        // equal the brute-force model count of the ORIGINAL network. This is what
        // proves weighted projection (sum), weighted join (multiply), the 0-ary
        // scalar fold, and the eliminated-free-var doubling are all correct.
        fn next(s: &mut u64) -> u64 {
            let mut x = *s;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = x;
            x
        }
        for seed in 1u64..=300 {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let n_vars = 3 + (next(&mut s) % 4) as usize; // 3..=6
            let n_tensors = 2 + (next(&mut s) % 4) as usize; // 2..=5
            let mut scopes = Vec::new();
            let mut dense = Vec::new();
            for _ in 0..n_tensors {
                let arity = 1 + (next(&mut s) % 3) as usize; // 1..=3
                let mut vs: Vec<usize> = Vec::new();
                while vs.len() < arity {
                    let v = (next(&mut s) % n_vars as u64) as usize;
                    if !vs.contains(&v) {
                        vs.push(v);
                    }
                }
                let rows = 1usize << arity;
                let mut sup = vec![false; rows];
                for r in sup.iter_mut() {
                    if next(&mut s) % 100 < 60 {
                        *r = true;
                    }
                }
                scopes.push(vs);
                dense.push(sup);
            }
            let cn = setup_problem(n_vars, scopes, dense);
            let n_cn = cn.n_vars;
            let want: u64 = (0u64..(1u64 << n_cn))
                .filter(|&c| satisfies(&cn, c))
                .count() as u64;
            let want = BigCount(want.into());
            for budget in 2usize..=6 {
                let got = match bounded_ve_canonicalize_weighted::<BigCount>(&cn, budget) {
                    Some(wc) => weighted_ve_count(&wc),
                    None => BigCount(0u32.into()), // VE proved UNSAT
                };
                assert_eq!(
                    got, want,
                    "seed {seed}, budget {budget}: weighted VE count != model count"
                );
            }
        }
    }

    #[test]
    fn produced_tensors_respect_the_budget() {
        let cn = setup_problem(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![OR2.to_vec(), OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[])
            .expect("SAT-preserving")
            .cn;
        for t in &out.tensors {
            assert!(t.var_axes.len() <= 3 - 1, "produced arity <= budget_b-1");
        }
    }
}
