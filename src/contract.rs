use rustc_hash::FxHashMap;

use crate::domain::DomainMask;
use crate::network::{assemble, Constraint, ConstraintNetwork};
use crate::propagate::compute_query_masks;
use crate::region::Region;
use crate::semiring::Weight;

/// A WEIGHTED boolean relation: the set of satisfying assignments over `vars`
/// (ascending, bit *j* of a config = value of `vars[j]`), each carrying a
/// semiring `weight`. This is the counting analogue of `Relation`: the boolean
/// `Relation` records only WHICH configs satisfy; a `WRelation` also records HOW
/// MUCH each contributes, so that eliminating a variable can MARGINALIZE it
/// (sum the eliminated variable's multiplicities) instead of merely projecting
/// its existence away. `rows` is kept sorted by config and deduplicated: exactly
/// one weight per surviving config.
#[derive(Clone, Debug)]
pub struct WRelation<W> {
    pub vars: Vec<usize>,
    pub rows: Vec<(u64, W)>,
}

impl<W: Weight> WRelation<W> {
    /// Lift a boolean `Relation` to a weighted one, giving every satisfying
    /// config weight `one()` — the initial multiplicity of an original 0/1
    /// constraint (each of its rows stands for exactly one local assignment).
    pub fn from_relation(rel: &Relation) -> WRelation<W> {
        WRelation {
            vars: rel.vars.clone(),
            rows: rel.rows.iter().map(|&r| (r, W::one())).collect(),
        }
    }

    /// Project onto `keep` (a subset of `self.vars`, ascending), SUMMING the
    /// weights of every row that collapses to the same kept config. This is the
    /// single semantic pivot from the boolean `Relation::project` (which dedups
    /// and DROPS multiplicity): summing is what turns existence-projection into
    /// counting-marginalization. Eliminating variable `v` is `join`-then-project
    /// onto `vars \ {v}`, so an eliminated variable free in its bucket doubles
    /// the surviving weights — `free_factor` emerges here without a special case.
    pub fn project(&self, keep: &[usize]) -> WRelation<W> {
        // Group-by-sum on the kept key. FxHashMap keeps this O(rows); a final
        // sort restores the ascending-config invariant.
        let mut acc: FxHashMap<u64, W> = FxHashMap::default();
        for (row, w) in &self.rows {
            let key = project_key(&self.vars, *row, keep);
            acc.entry(key)
                .and_modify(|e| e.add(w))
                .or_insert_with(|| w.clone());
        }
        let mut rows: Vec<(u64, W)> = acc.into_iter().collect();
        rows.sort_unstable_by_key(|&(c, _)| c);
        WRelation {
            vars: keep.to_vec(),
            rows,
        }
    }

    /// The total weight over ALL configs — the scalar a fully-eliminated (0-ary)
    /// relation contributes to the global multiplier.
    pub fn total(&self) -> W {
        let mut acc = W::zero();
        for (_, w) in &self.rows {
            acc.add(w);
        }
        acc
    }
}

/// Weighted relational join: rows of `a` and `b` that agree on shared variables,
/// merged over `a.vars ∪ b.vars`, with the two rows' weights MULTIPLIED (a joint
/// assignment's multiplicity is the product of the two local multiplicities).
/// The boolean `join` unions supports; this one additionally carries the product
/// weight. Output configs are unique (each projects back to exactly one
/// `(a-row, b-row)` pair, every var of each side surviving into the output), so
/// no weight-summing dedup is needed here — that happens only in `project`.
pub fn wjoin<W: Weight>(a: &WRelation<W>, b: &WRelation<W>) -> WRelation<W> {
    let out_vars = sorted_union(&a.vars, &b.vars);
    debug_assert!(
        out_vars.len() <= 64,
        "joined relation exceeds the 64-variable u64 cap"
    );
    let mut scatter_a: Vec<(usize, usize)> = Vec::with_capacity(a.vars.len());
    let mut scatter_b: Vec<(usize, usize)> = Vec::new();
    let mut key_pos_a: Vec<usize> = Vec::new();
    let mut key_pos_b: Vec<usize> = Vec::new();
    for (out_pos, &v) in out_vars.iter().enumerate() {
        match (a.vars.binary_search(&v), b.vars.binary_search(&v)) {
            (Ok(pa), Ok(pb)) => {
                scatter_a.push((pa, out_pos));
                key_pos_a.push(pa);
                key_pos_b.push(pb);
            }
            (Ok(pa), Err(_)) => scatter_a.push((pa, out_pos)),
            (Err(_), Ok(pb)) => scatter_b.push((pb, out_pos)),
            (Err(_), Err(_)) => unreachable!("out var comes from a or b"),
        }
    }
    // Bucket b by shared key, keeping each row PRE-SCATTERED with its weight.
    let mut buckets: FxHashMap<u64, Vec<(u64, W)>> = FxHashMap::default();
    for (br, bw) in &b.rows {
        buckets
            .entry(key_at(*br, &key_pos_b))
            .or_default()
            .push((scatter(*br, &scatter_b), bw.clone()));
    }
    let mut rows: Vec<(u64, W)> = Vec::new();
    for (ar, aw) in &a.rows {
        if let Some(brs) = buckets.get(&key_at(*ar, &key_pos_a)) {
            let scattered_a = scatter(*ar, &scatter_a);
            for (sb, bw) in brs {
                rows.push((scattered_a | sb, aw.mul(bw)));
            }
        }
    }
    rows.sort_unstable_by_key(|&(c, _)| c);
    debug_assert!(
        rows.windows(2).all(|w| w[0].0 < w[1].0),
        "weighted join output configs repeat (join must not need weight-dedup)"
    );
    WRelation {
        vars: out_vars,
        rows,
    }
}

/// Fold all weighted relations into one via `wjoin`, greedily picking the
/// most-shared-vars partner next (same order heuristic as boolean `join_all`,
/// only to avoid needless Cartesian intermediates). Precondition: non-empty.
pub fn wjoin_all<W: Weight>(mut rels: Vec<WRelation<W>>) -> WRelation<W> {
    debug_assert!(!rels.is_empty(), "wjoin_all requires at least one relation");
    let mut acc = rels.swap_remove(0);
    while !rels.is_empty() {
        let mut pick = 0usize;
        let mut pick_shared = shared_count(&acc.vars, &rels[0].vars);
        for (i, r) in rels.iter().enumerate().skip(1) {
            let s = shared_count(&acc.vars, &r.vars);
            if s > pick_shared {
                pick = i;
                pick_shared = s;
            }
        }
        let r = rels.swap_remove(pick);
        acc = wjoin(&acc, &r);
    }
    acc
}

/// A boolean relation: the set `rows` of satisfying assignments over `vars`,
/// where `vars` is sorted ascending and bit *j* of a row is the value of
/// `vars[j]`. Rows are sorted ascending and deduplicated.
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
            .map(|&row| project_key(&self.vars, row, keep))
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
    tensor: &Constraint,
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

/// The boolean relation of a tensor's sparse `support` (no domain slicing): each
/// support `config` is a bitmask over `var_axes` order; rows are re-encoded over
/// `var_axes` SORTED ascending (canonical bit order, matching `tensor_relation`),
/// deduplicated.
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

/// Scatter `row`'s bits from `src` positions to `dst` positions: for each
/// `(src, dst)` pair, bit `src` of `row` lands at bit `dst` of the result.
#[inline]
fn scatter(row: u64, map: &[(usize, usize)]) -> u64 {
    let mut out = 0u64;
    for &(src, dst) in map {
        out |= ((row >> src) & 1) << dst;
    }
    out
}

/// Project `row`'s bits at `positions` (within its own relation) into a
/// packed key, in `positions` order.
#[inline]
fn key_at(row: u64, positions: &[usize]) -> u64 {
    let mut key = 0u64;
    for (j, &pos) in positions.iter().enumerate() {
        key |= ((row >> pos) & 1) << j;
    }
    key
}

/// Relational join: rows of `a` and `b` that agree on shared variables, merged
/// over `a.vars ∪ b.vars`. Callers must ensure `|a.vars ∪ b.vars| <= 64`
/// (checked only by debug_assert here — `grow_region` enforces it up front).
pub(crate) fn join(a: &Relation, b: &Relation) -> Relation {
    join_bounded(a, b, usize::MAX).expect("unbounded join cannot abort")
}

/// `join`, but abort with `None` as soon as the output exceeds `cap` rows —
/// the budget check `grow_region` needs, paid at cap+1 rows instead of after
/// materializing (and sorting) the full product. The output of a join never
/// contains duplicates — every output row projects back to exactly one
/// `(a-row, b-row)` pair because each side's vars are all present in the
/// output — so the row count is exact and no dedup pass exists.
pub(crate) fn join_bounded(a: &Relation, b: &Relation, cap: usize) -> Option<Relation> {
    let out_vars = sorted_union(&a.vars, &b.vars);
    debug_assert!(
        out_vars.len() <= 64,
        "joined relation exceeds the 64-variable u64 cap"
    );
    // Position plans, computed once: where each side's bits land in the
    // output, and where the shared key bits sit within each side.
    let mut scatter_a: Vec<(usize, usize)> = Vec::with_capacity(a.vars.len());
    let mut scatter_b: Vec<(usize, usize)> = Vec::new(); // b-only vars: shared bits come via a
    let mut key_pos_a: Vec<usize> = Vec::new();
    let mut key_pos_b: Vec<usize> = Vec::new();
    for (out_pos, &v) in out_vars.iter().enumerate() {
        match (a.vars.binary_search(&v), b.vars.binary_search(&v)) {
            (Ok(pa), Ok(pb)) => {
                scatter_a.push((pa, out_pos));
                key_pos_a.push(pa);
                key_pos_b.push(pb);
            }
            (Ok(pa), Err(_)) => scatter_a.push((pa, out_pos)),
            (Err(_), Ok(pb)) => scatter_b.push((pb, out_pos)),
            (Err(_), Err(_)) => unreachable!("out var comes from a or b"),
        }
    }

    // Bucket b's rows by shared key, storing each row PRE-SCATTERED onto its
    // b-only output positions — the inner loop below is then a single OR.
    let mut buckets: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for &br in &b.rows {
        buckets
            .entry(key_at(br, &key_pos_b))
            .or_default()
            .push(scatter(br, &scatter_b));
    }

    let mut rows: Vec<u64> = Vec::new();
    for &ar in &a.rows {
        if let Some(brs) = buckets.get(&key_at(ar, &key_pos_a)) {
            if rows.len() + brs.len() > cap {
                return None;
            }
            let scattered_a = scatter(ar, &scatter_a);
            for &sb in brs {
                rows.push(scattered_a | sb);
            }
        }
    }
    rows.sort_unstable();
    debug_assert!(
        rows.windows(2).all(|w| w[0] < w[1]),
        "join output rows repeat"
    );
    Some(Relation {
        vars: out_vars,
        rows,
    })
}

/// Fold all relations into one. Order-independent; the greedy "most-shared-vars
/// next" pick only avoids needless Cartesian-product intermediates.
pub fn join_all(mut rels: Vec<Relation>) -> Relation {
    debug_assert!(!rels.is_empty(), "join_all requires at least one relation");
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
/// union of all rels' vars, ascending) — `join_all` followed by `project`. The
/// live callers that need the pre-projection relation (`region::grow_region`,
/// `canonicalize`'s VE step) call `join`/`join_all` directly and project
/// themselves; this fused wrapper now backs `contract_region` and the contract
/// tests. Precondition: `rels` is non-empty (inherited from `join_all`).
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

/// A WEIGHTED constraint network: the 0/1 SUPPORT skeleton `cn` that CT /
/// region growth / propagation run on UNCHANGED, plus a per-tensor row-weight
/// vector read only at the counting CONSUMPTION sites (leaf/branch fold,
/// closed-region Σ, constant-tensor check). `weights[tid][k]` is the semiring
/// weight of `cn.tensors[tid]`'s `k`-th support config — the two stay aligned
/// because `assemble` preserves tensor order and never reorders a tensor's
/// support (its var-axis remap is monotone, so support bit positions are fixed).
///
/// Blocker 6 ("dedup key must include weights") is satisfied structurally: the
/// flyweight dedup only shares the SUPPORT storage (`TruthTable`), while weights
/// live per-TENSOR here — so two tensors with identical support but different
/// weights share nothing weight-bearing and no weighted factor can be dropped
/// (regression: `weighted_network_keeps_per_tensor_weights_under_support_dedup`).
pub struct WeightedNetwork<W> {
    pub cn: ConstraintNetwork,
    /// `weights[tid]` aligns 1:1 with `cn.tensors[tid]`'s support.
    pub weights: Vec<Vec<W>>,
}

impl<W: Weight> WeightedNetwork<W> {
    /// Build the support skeleton + aligned weights from WEIGHTED relations
    /// (typically a weighted-VE residual, `WeightedCanonicalized::surviving`).
    /// Tensor `i` of the result is relation `i` — `setup_from_relations` keeps
    /// order — so `weights[i]` is just relation `i`'s row weights in order.
    /// Vars appearing in no surviving relation are compressed out of `cn`; they
    /// were ELIMINATED by VE (their multiplicity already folded into the weights
    /// / scalar), NOT free, so no `free_factor` is owed for them.
    pub fn from_relations(var_num: usize, rels: Vec<WRelation<W>>) -> WeightedNetwork<W> {
        let bool_rels: Vec<Relation> = rels
            .iter()
            .map(|r| Relation {
                vars: r.vars.clone(),
                rows: r.rows.iter().map(|&(c, _)| c).collect(),
            })
            .collect();
        let cn = setup_from_relations(var_num, bool_rels);
        let weights: Vec<Vec<W>> = rels
            .into_iter()
            .map(|r| r.rows.into_iter().map(|(_, w)| w).collect())
            .collect();
        debug_assert_eq!(cn.tensors.len(), weights.len(), "one weight vec per tensor");
        WeightedNetwork { cn, weights }
    }

    /// Unit weights for a plain 0/1 network: every support config weight `one()`.
    /// The bridge that lets the weighted counting engine run an UNWEIGHTED
    /// instance (weights all 1 ⇒ every fold/Σ is a no-op) — how the M1 model
    /// counting tests and the budget-0 path exercise the new engine.
    pub fn unit(cn: &ConstraintNetwork) -> Vec<Vec<W>> {
        cn.tensors
            .iter()
            .map(|t| vec![W::one(); cn.support(t).len()])
            .collect()
    }
}

/// Lift every 0/1 tensor of `cn` to a weight-`one()` WEIGHTED relation over `cn`'s
/// (compressed) var ids — the UNWEIGHTED counting front-end's input to
/// `count_with_ve` (weights all 1 ⇒ the count is the plain model count).
pub fn unit_weighted_relations<W: Weight>(cn: &ConstraintNetwork) -> Vec<WRelation<W>> {
    cn.tensors
        .iter()
        .map(|t| WRelation::from_relation(&support_relation(&t.var_axes, cn.support(t))))
        .collect()
}

/// Lift `cn`'s tensors to weight-`one()` WEIGHTED relations over the ORIGINAL
/// (un-compressed) variable ids, relabelling through `cn.orig_to_new`. Weighted
/// counting front-ends that also add a per-original-variable literal-weight tensor
/// need the constraint relations in the SAME (original) id space, so no variable
/// is compressed away before its literal weight is folded in.
pub fn original_id_weighted_relations<W: Weight>(cn: &ConstraintNetwork) -> Vec<WRelation<W>> {
    let mut new_to_orig = vec![0usize; cn.n_vars];
    for (o, &c) in cn.orig_to_new.iter().enumerate() {
        if let Some(ci) = c {
            new_to_orig[ci] = o;
        }
    }
    cn.tensors
        .iter()
        .map(|t| {
            // Compression is monotone, so mapping each axis back yields ascending
            // original ids — support bit order (and thus the configs) is unchanged.
            let orig_vars: Vec<usize> = t.var_axes.iter().map(|&c| new_to_orig[c]).collect();
            WRelation::from_relation(&support_relation(&orig_vars, cn.support(t)))
        })
        .collect()
}

/// Build a `ConstraintNetwork` from sparse relations — the support-based entry used
/// by `canonicalize` so it never materializes a dense table. Each `Relation`
/// contributes `(rel.vars, rel.rows as u32)`; `rel.rows` must be ascending (the
/// `Relation` invariant), which `assemble`/`from_support` require.
pub fn setup_from_relations(var_num: usize, rels: Vec<Relation>) -> ConstraintNetwork {
    let tensors_in: Vec<(Vec<usize>, Vec<u32>)> = rels
        .into_iter()
        .map(|rel| {
            let support: Vec<u32> = rel.rows.iter().map(|&r| r as u32).collect();
            (rel.vars, support)
        })
        .collect();
    assemble(var_num, tensors_in)
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
    fn support_relation_reencodes_unsorted_axes() {
        // Tensor over var_axes = [2, 0] (UNSORTED); support configs over (bit0=v2, bit1=v0)
        // are {1, 2}: config 0b01 (v2=1,v0=0) and 0b10 (v2=0,v0=1).
        // Relation must be over sorted vars [0, 2] with rows re-encoded:
        //   (v0=0,v2=1) -> bit0(v0)=0,bit1(v2)=1 -> 0b10 = 2
        //   (v0=1,v2=0) -> bit0(v0)=1,bit1(v2)=0 -> 0b01 = 1
        let rel = support_relation(&[2, 0], &[1u32, 2u32]);
        assert_eq!(rel.vars, vec![0, 2]);
        assert_eq!(rel.rows, vec![1u64, 2u64]);
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
    fn setup_from_relations_matches_dense_setup() {
        use crate::network::setup_problem;
        let or2 = vec![false, true, true, true]; // support {1,2,3}
        let dense_cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![or2.clone(), or2.clone()],
        );
        let rels = vec![
            Relation {
                vars: vec![0, 1],
                rows: vec![1, 2, 3],
            },
            Relation {
                vars: vec![1, 2],
                rows: vec![1, 2, 3],
            },
        ];
        let rel_cn = setup_from_relations(3, rels);
        assert_eq!(rel_cn.tensors.len(), dense_cn.tensors.len());
        assert_eq!(rel_cn.truth_tables.len(), dense_cn.truth_tables.len()); // both dedup to 1
        assert_eq!(rel_cn.n_vars, dense_cn.n_vars);
        for t in 0..rel_cn.tensors.len() {
            assert_eq!(
                rel_cn.support(&rel_cn.tensors[t]),
                dense_cn.support(&dense_cn.tensors[t]),
            );
        }
    }

    #[test]
    fn weighted_project_sums_multiplicities_not_dedups() {
        use crate::semiring::{BigCount, Weight};
        // vars [0,1], rows (00,01,10) each weight 1. Project onto [0] SUMS: the
        // two rows with x0=0 (00,01) collapse to x0=0 weight 2; x0=1 (10) → 1.
        // The boolean project would dedup to weight-less {0,1}. Summing is the
        // whole counting pivot.
        let wr: WRelation<BigCount> = WRelation {
            vars: vec![0, 1],
            rows: vec![
                (0b00, BigCount::one()),
                (0b01, BigCount::one()),
                (0b10, BigCount::one()),
            ],
        };
        let p = wr.project(&[0]);
        assert_eq!(p.vars, vec![0]);
        assert_eq!(p.rows[0], (0u64, BigCount(2u32.into())));
        assert_eq!(p.rows[1], (1u64, BigCount::one()));
    }

    #[test]
    fn weighted_factors_are_never_dropped_or_merged() {
        use crate::semiring::BigCount;
        // §7 blocker 6: a weighted factor joined with itself must yield w·w, NOT
        // w (0/1 constraints are idempotent under boolean dedup; weighted ones
        // are NOT — w·w ≠ w whenever w ≠ 1). Two identical-SUPPORT relations with
        // weight-3 rows: the join weight must be 9, proving weights are carried
        // through and no flyweight dedup collapses them.
        let three = BigCount(3u32.into());
        let a: WRelation<BigCount> = WRelation {
            vars: vec![0],
            rows: vec![(1u64, three.clone())],
        };
        let b = a.clone();
        let j = wjoin(&a, &b);
        assert_eq!(j.vars, vec![0]);
        assert_eq!(j.rows, vec![(1u64, BigCount(9u32.into()))]);
        assert_ne!(j.rows[0].1, three, "w·w must differ from w for w ≠ 1");
    }

    #[test]
    fn weighted_join_multiplies_agreeing_rows() {
        use crate::semiring::BigCount;
        // a over [0,1] rows {01:2, 11:1}; b over [1,2] rows {01:3, 11:5} (bit0 of
        // b = var1). Agreement is on var1. 01(a: v0=1,v1=0) joins b rows with
        // v1=0 → b 01 has v1=... let's keep it concrete and simple: single rows.
        let a: WRelation<BigCount> = WRelation {
            vars: vec![0, 1],
            rows: vec![(0b11, BigCount(2u32.into()))], // v0=1,v1=1
        };
        let b: WRelation<BigCount> = WRelation {
            vars: vec![1, 2],
            rows: vec![(0b11, BigCount(3u32.into()))], // v1=1,v2=1
        };
        let j = wjoin(&a, &b); // agree on v1=1 ⇒ v0=1,v1=1,v2=1 weight 2·3=6
        assert_eq!(j.vars, vec![0, 1, 2]);
        assert_eq!(j.rows, vec![(0b111, BigCount(6u32.into()))]);
    }

    #[test]
    fn contract_matches_join_all_then_project() {
        // (x0∨x1) over [0,1] and (x1∨x2) over [1,2]; support {1,2,3} each.
        let a = Relation {
            vars: vec![0, 1],
            rows: vec![1, 2, 3],
        };
        let b = Relation {
            vars: vec![1, 2],
            rows: vec![1, 2, 3],
        };
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
