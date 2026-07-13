use std::sync::Arc;

use crate::adapter::BranchSolver;
use crate::blockmerge::block_merge;
use crate::canonicalize::bounded_ve_canonicalize_weighted_rels;
use crate::contract::{WRelation, WeightedNetwork};
use crate::ct::{apply_masked_assignment, ct_propagate, RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{SolverBuffer, Stats, TnProblem};
use crate::propagate::{
    compute_query_masks, dominate_fixpoint, failed_literal_fixpoint, feasible_configs,
};
use crate::region::grow_region;
use crate::selector::{
    compute_occurrence_scores, occurrence_pool, select_var_most_occurrence, Selector,
    FAILED_LITERAL_POOL,
};
use crate::semiring::Weight;
use crate::trail::Trail;
use crate::util::is_entailed;

/// Result of a solve: the verdict, a full satisfying assignment when `found`,
/// and the search statistics. Port of `problem.jl::Result`.
#[derive(Clone, Debug)]
pub struct Solve {
    pub found: bool,
    pub solution: Vec<DomainMask>,
    pub stats: Stats,
}

struct SearchCtx<'a> {
    cn: &'a Arc<ConstraintNetwork>,
    selector: Selector,
    measure: Measure,
    solver: &'a BranchSolver,
}

/// Branch-and-reduce SAT solve. Port of `branch.jl::bbsat!`, extended with
/// connected-component decomposition: at every node the unfixed vars are split
/// into components of the active constraint graph and solved independently.
pub fn bbsat(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
) -> Solve {
    problem.stats.reset();
    // Split disjoint field borrows for the recursion. `doms`, `tables`, `trail`
    // are threaded by `&mut` and mutated in place under the trail. The trail is
    // the one carried on `problem` (root propagation already used it), so its
    // `epoch` stays monotonic across root propagation and the whole search.
    let ctx = SearchCtx {
        cn: &problem.static_cn,
        selector,
        measure,
        solver,
    };
    let masks = &problem.masks;
    let stats = &mut problem.stats;
    let buffer = &mut problem.buffer;
    let doms = &mut problem.doms;
    let tables = &mut problem.tables;
    let trail = &mut problem.trail;

    let scope: Vec<usize> = (0..doms.len()).filter(|&v| !doms[v].is_fixed()).collect();
    let mark = trail.mark();
    let found = bbsat_rec(&ctx, stats, buffer, doms, masks, tables, trail, &scope);
    if !found {
        // A failing later component leaves earlier components' fixings applied
        // (their success path never restores); unwind to the root state so the
        // UNSAT contract matches the pre-decomposition solver.
        trail.restore_to(mark, doms, tables);
    }
    Solve {
        found,
        // The success path never restores, so `doms` holds the full assignment.
        solution: if found { doms.clone() } else { Vec::new() },
        stats: stats.clone(),
    }
}

/// Solve `scope`'s unfixed vars: split them into connected components of the
/// NON-ENTAILED constraint graph and solve each independently. Components share
/// no CONSTRAINING tensor with another's unfixed vars (entailed tensors couple
/// nothing), so propagation and region growth from one can never narrow
/// another; a satisfying assignment of one component stays valid whatever is
/// chosen in the rest. Hence one failing component refutes the whole scope (no
/// cross-component backtracking), and tree size is the SUM of component trees
/// instead of their product. A subproblem separated from the rest only by dead
/// (entailed) constraints is now its own component — the closed-region
/// shortcut's precondition, guaranteed structurally rather than left to whether
/// region growth swallows the patch.
#[allow(clippy::too_many_arguments)]
fn bbsat_rec(
    ctx: &SearchCtx,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
    scope: &[usize],
) -> bool {
    let mut comps = split_components(ctx.cn, doms, masks, scope);
    if comps.len() > 1 {
        stats.record_split();
        // Fail-fast: smallest component first — an UNSAT component refutes the
        // node, and small ones are the cheapest to refute (or solve).
        comps.sort_unstable_by_key(|c| (c.len(), c[0]));
    }
    // Empty `comps` (scope fully fixed) is the SAT leaf: the loop is a no-op.
    for comp in &comps {
        if !branch_component(ctx, stats, buffer, doms, masks, tables, trail, comp) {
            return false;
        }
    }
    true
}

/// Branch on one connected component: pick a focus var inside it, compute the
/// region branching rule, and recurse on the component (whose unfixed vars may
/// split further after propagation). Returns whether the component is
/// satisfiable from the current state; on `false` the trail is restored to the
/// call state.
#[allow(clippy::too_many_arguments)]
fn branch_component(
    ctx: &SearchCtx,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
    comp: &[usize],
) -> bool {
    // A component solved earlier in this node's loop may have fixed all of THIS
    // component's vars through the GLOBAL reductions (domination / failed-literal
    // range over every var, not just their own component) on its success path,
    // which is never restored. A fully-fixed, contradiction-free scope is already
    // satisfied — GAC over fully-fixed vars leaves every constraint's live tuple
    // in place, and `branch_component` only runs with `doms[0] != NONE` — so there
    // is nothing to branch. (In release, `findbest`'s free-var fallback would
    // instead emit an empty no-op clause and recurse to the same SAT leaf via one
    // wasted node; this skips it and is what the `findbest` debug_assert guards.)
    if comp.iter().all(|&v| doms[v].is_fixed()) {
        return true;
    }
    // γ is a cutoff/diagnostic signal only (see `findbest`); the solver descent
    // ignores it.
    let (clauses, variables, _gamma) = ctx.selector.findbest(
        ctx.cn,
        doms,
        buffer,
        ctx.measure,
        ctx.solver,
        masks,
        tables,
        trail,
        comp,
    );
    let clauses = match clauses {
        Some(c) => c,
        None => return false,
    };

    stats.record_branch(clauses.len() as u64);
    for cl in &clauses {
        stats.record_visit();
        trail.open();
        let m = trail.mark();
        buffer.reset_worklist();
        apply_masked_assignment(ctx.cn, doms, buffer, trail, &variables, cl.mask, cl.val);
        ct_propagate(ctx.cn, doms, masks, tables, buffer, trail);
        if doms[0] != DomainMask::NONE {
            // GAC fixpoint reached: apply the selection-independent reductions
            // before descending — both trailed, undone on restore. Domination
            // (pure-literal generalization) first, then failed-literal probing
            // over the occurrence-ranked pool (forces literals / refutes nodes
            // that GAC + domination miss).
            dominate_fixpoint(ctx.cn, doms, masks, tables, buffer, trail);
            if doms[0] != DomainMask::NONE {
                let pool = occurrence_pool(ctx.cn, doms, buffer, masks, FAILED_LITERAL_POOL);
                failed_literal_fixpoint(ctx.cn, doms, masks, tables, buffer, trail, &pool);
            }
        }
        if doms[0] != DomainMask::NONE
            && bbsat_rec(ctx, stats, buffer, doms, masks, tables, trail, comp)
        {
            return true;
        }
        trail.restore_to(m, doms, tables);
    }
    false
}

/// Connected components of the unfixed vars in `scope` under "shares a
/// NON-ENTAILED tensor" adjacency, each sorted ascending. A tensor connects its
/// unfixed vars only if it still constrains them: fixed vars carry no residual
/// coupling (their value is already sliced into every incident table), and an
/// ENTAILED tensor (every combination of its unfixed vars satisfying) couples
/// nothing — any choice on one side extends to any choice on the other, and
/// entailment is monotone under further fixing, so the sides stay independent
/// down the whole subtree. Skipping entailed tensors is what makes a subproblem
/// the rest of the network only touches through dead constraints its OWN
/// component — the same entailment-aware boundary `boundary_vars`/the
/// closed-region shortcut use, now computed once at the decomposition layer
/// instead of rediscovered when region growth happens to swallow the patch.
/// BFS may pull in connected unfixed vars outside `scope`; they belong to the
/// same subproblem and are included.
fn split_components(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    masks: &[TableMasks],
    scope: &[usize],
) -> Vec<Vec<usize>> {
    let mut comps: Vec<Vec<usize>> = Vec::new();
    let mut var_seen = vec![false; doms.len()];
    let mut tensor_seen = vec![false; cn.tensors.len()];
    for &s in scope {
        if doms[s].is_fixed() || var_seen[s] {
            continue;
        }
        var_seen[s] = true;
        let mut comp = vec![s];
        let mut head = 0usize;
        while head < comp.len() {
            let v = comp[head];
            head += 1;
            for &tid in &cn.v2t[v] {
                if tensor_seen[tid] {
                    continue;
                }
                // Mark seen before the entailment test so it is computed at most
                // once per tensor per call; an entailed tensor creates no edges.
                tensor_seen[tid] = true;
                if is_entailed(cn, tid, doms, masks) {
                    continue;
                }
                for &u in &cn.tensors[tid].var_axes {
                    if !doms[u].is_fixed() && !var_seen[u] {
                        var_seen[u] = true;
                        comp.push(u);
                    }
                }
            }
        }
        comp.sort_unstable();
        comps.push(comp);
    }
    comps
}

// ======================================================================
// Exact WEIGHTED model counting (#CSP / weighted #SAT). A SEPARATE recursion
// from `bbsat` that reuses the same propagation primitives (CT / region growth /
// feasible-config probe) UNCHANGED — those run on the 0/1 SUPPORT skeleton — but
// changes the coordination layer per `docs/design/counting-solver.md`:
//   • branches SUM instead of OR-with-short-circuit (every branch traversed),
//   • components MULTIPLY instead of AND (a zero component zeroes the product),
//   • truly-free (tensorless) vars contribute `free_factor` at a leaf,
//   • each tensor's row WEIGHT is consumed exactly once, at the branch step where
//     the tensor first becomes a CONSTANT function (fully fixed, or full-support
//     with all sliced weights equal): its scalar is folded into that branch
//     (`count_branch`'s delta). Tensors already constant at the root are folded
//     in `bbcount`'s pre-pass; both together give exactly-once accounting.
//   • component splitting / focus selection use the WEIGHTED entailment
//     (`sliced_constant`): a full-support but NON-uniform tensor stays a live
//     coupling edge (design §7 blocker 1) — it never wrongly splits a component.
//   • domination is OFF (count-unsafe); GAC/CT and failed-literal stay ON.
// The weights come from a `WeightedNetwork` (0/1 skeleton + per-tensor row
// weights), typically a weighted-VE residual; the caller multiplies the VE
// scalar back in. The decision path (`bbsat` and everything it calls) is
// untouched.
// ======================================================================

/// Result of a counting solve: the (weighted) model count and the search stats.
#[derive(Clone, Debug)]
pub struct CountSolve<W> {
    pub models: W,
    pub stats: Stats,
}

/// The counting branching arm at a region node (design §3 / §3.1). Both arms are
/// an exact PARTITION of the region's feasible set `S` — counting sums branches,
/// so overlap would double-count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CountBranch {
    /// One branch per feasible config (all region vars fixed). The correctness
    /// baseline; compression ratio is always 1.
    PerConfig,
    /// Partition `S` into perfect subcubes (`blockmerge::block_merge`): a block
    /// leaves coordinates free, so a region tensor often folds a whole block in
    /// one node. Strictly correct via the partition property alone.
    BlockMerge,
}

struct CountCtx<'a, W> {
    cn: &'a Arc<ConstraintNetwork>,
    /// Per-tensor row weights, aligned to `cn.tensors[tid]`'s support (the
    /// `WeightedNetwork::weights` half; see `contract::WeightedNetwork`).
    weights: &'a [Vec<W>],
    /// Region row budget (the branching-table size cap), as in `Selector::max_rows`.
    max_rows: usize,
    /// Which partition arm to apply at region nodes (`PerConfig` default).
    branch: CountBranch,
}

/// Exact WEIGHTED model count of a `WeightedNetwork`'s residual over the semiring
/// `W`. `problem` is `TnProblem::from_network_counting(wn.cn)` (domination-free
/// root setup) and `weights` is `wn.weights`; for an UNWEIGHTED instance pass
/// `WeightedNetwork::unit(&cn)` (all `one()`) and the count is the plain model
/// count. The count is over the network's COMPRESSED variables; any variable the
/// weighted-VE front-end eliminated is already folded into the weights/VE scalar,
/// and any declared-but-tensorless variable is the CALLER's `free_factor` (the
/// example front-ends apply it, §7 blocker 2).
///
/// `max_rows` is the region growth budget (mirrors `Selector::MostOccurrence`).
/// `branch` selects the region partition arm (§3.1): `PerConfig` fixes ALL region
/// vars to one config per branch (natively a partition), `BlockMerge` partitions
/// the feasible set into perfect subcubes. Both are exact partitions of the
/// solution space, so the count is identical; only the branching-tree shape
/// (and the compression stats) differ.
pub fn bbcount<W: Weight>(
    problem: &mut TnProblem,
    weights: &[Vec<W>],
    max_rows: usize,
    branch: CountBranch,
) -> CountSolve<W> {
    problem.stats.reset();
    let ctx = CountCtx {
        cn: &problem.static_cn,
        weights,
        max_rows,
        branch,
    };
    let masks = &problem.masks;
    let stats = &mut problem.stats;
    let buffer = &mut problem.buffer;
    let doms = &mut problem.doms;
    let tables = &mut problem.tables;
    let trail = &mut problem.trail;

    // Pre-pass (§7 blocker 1): tensors already CONSTANT under the root-propagated
    // doms — a full-support-uniform weighted tensor from VE, or any tensor whose
    // vars root propagation fixed to one satisfying row — never TRANSITION to
    // constant during search, so `count_branch`'s delta fold never sees them.
    // Fold each such tensor's scalar exactly once here. (Unweighted: every scalar
    // is `one()`, so the pre-pass is a no-op — no divergence from M1 counts.)
    let mut prepass = W::one();
    for tid in 0..ctx.cn.tensors.len() {
        if let Some(w) = sliced_constant(ctx.cn, ctx.weights, tid, doms) {
            prepass = prepass.mul(&w);
        }
    }

    let scope: Vec<usize> = (0..doms.len()).filter(|&v| !doms[v].is_fixed()).collect();
    let mark = trail.mark();
    let residual = count_scope::<W>(&ctx, stats, buffer, doms, masks, tables, trail, &scope);
    // Counting never keeps a witness, so always unwind to the entry state (any
    // forced fixings applied along the way are dropped).
    trail.restore_to(mark, doms, tables);
    CountSolve {
        models: prepass.mul(&residual),
        stats: stats.clone(),
    }
}

/// The counting FRONT-END (M2.1): weighted bounded VE at width `budget`, then
/// `bbcount` over the residual, with the VE scalar multiplied back in. `rels` are
/// the input weighted relations over ids `0..n_vars` — the 0/1 constraints lifted
/// to weight `one()` plus (M2.2) any literal-weight 1-ary tensors. `budget == 0`
/// disables VE (residual = whole network), so the whole ladder of budgets must
/// return the SAME count (budget-invariance = the M2.1 correctness proof). The
/// returned weight is the count over the `n_vars` var space; a caller with
/// declared-but-tensorless variables multiplies their `free_factor` on top
/// (`examples/count_dimacs.rs`, §7 blocker 2). Returns `zero()` if VE or root
/// propagation refutes the instance.
pub fn count_with_ve<W: Weight>(
    n_vars: usize,
    rels: Vec<WRelation<W>>,
    budget: usize,
    max_rows: usize,
    branch: CountBranch,
) -> (W, Stats) {
    match bounded_ve_canonicalize_weighted_rels::<W>(n_vars, rels, budget) {
        None => (W::zero(), Stats::default()),
        Some(wc) => {
            let wn = WeightedNetwork::from_relations(wc.n_vars, wc.surviving);
            match TnProblem::from_network_counting(wn.cn) {
                Ok(mut p) => {
                    let solve = bbcount::<W>(&mut p, &wn.weights, max_rows, branch);
                    (wc.scalar.mul(&solve.models), solve.stats)
                }
                // Root propagation refuted the residual: the VE scalar cannot
                // resurrect an UNSAT instance.
                Err(_) => (W::zero(), Stats::default()),
            }
        }
    }
}

/// Is tensor `tid` a CONSTANT function under `doms`? Slices its support against
/// the fixed axes and returns `Some(w)` iff the surviving rows are FULL (one per
/// combination of the unfixed axes) AND all carry the same weight `w` — the M2.1
/// counting-mode entailment (design §7 blocker 1). Then the tensor contributes a
/// scalar `w` and couples nothing. A full-support but NON-uniform slice, and any
/// non-full slice, return `None`: the tensor stays a live coupling edge. A
/// fully-fixed satisfying tensor is the 0-ary special case (one row = full),
/// returning its single row's weight — how a resolved tensor is consumed.
fn sliced_constant<W: Weight>(
    cn: &ConstraintNetwork,
    weights: &[Vec<W>],
    tid: usize,
    doms: &[DomainMask],
) -> Option<W> {
    let t = &cn.tensors[tid];
    let (m0, m1) = compute_query_masks(doms, &t.var_axes);
    let fmask = m0 | m1;
    let fval = m1;
    let unfixed = t.var_axes.iter().filter(|&&v| !doms[v].is_fixed()).count();
    let support = cn.support(t);
    let w = &weights[tid];
    let mut count: u64 = 0;
    let mut first: Option<&W> = None;
    for (k, &config) in support.iter().enumerate() {
        if (config & fmask) != fval {
            continue;
        }
        count += 1;
        match first {
            None => first = Some(&w[k]),
            Some(f) if w[k] != *f => return None, // non-uniform ⇒ live edge
            Some(_) => {}
        }
    }
    if count == 1u64 << unfixed {
        Some(first.cloned().unwrap_or_else(W::one))
    } else {
        None
    }
}

/// Incident tensors of `comp` that are LIVE (not constant) under `doms` — the
/// set `count_branch` watches for a transition to constant. Constant incident
/// tensors are omitted: they are already folded (pre-pass or an ancestor branch)
/// and must never be folded again.
fn comp_incident_live<W: Weight>(
    cn: &ConstraintNetwork,
    weights: &[Vec<W>],
    comp: &[usize],
    doms: &[DomainMask],
) -> Vec<usize> {
    let mut inc: Vec<usize> = comp
        .iter()
        .flat_map(|&v| cn.v2t[v].iter().copied())
        .collect();
    inc.sort_unstable();
    inc.dedup();
    inc.retain(|&tid| sliced_constant(cn, weights, tid, doms).is_none());
    inc
}

/// One counting branch: fix `vars`/`mask`/`val`, propagate the count-SAFE
/// reductions, fold every `live_before` tensor this fixing turned CONSTANT (its
/// scalar consumed exactly here), recurse over the shrunken component, and return
/// the branch weight (`zero()` on a contradiction). Opens and restores its own
/// trail scope, so the caller's state is unchanged on return.
#[allow(clippy::too_many_arguments)]
fn count_branch<W: Weight>(
    ctx: &CountCtx<W>,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
    comp: &[usize],
    live_before: &[usize],
    vars: &[usize],
    mask: u64,
    val: u64,
) -> W {
    stats.record_visit();
    trail.open();
    let m = trail.mark();
    buffer.reset_worklist();
    apply_masked_assignment(ctx.cn, doms, buffer, trail, vars, mask, val);
    ct_propagate(ctx.cn, doms, masks, tables, buffer, trail);
    let mut w = W::zero();
    if doms[0] != DomainMask::NONE {
        // Count-SAFE only: failed-literal (forces literals no model can drop),
        // probe pool restricted to THIS component (§7 blocker 5). No domination.
        let pool = component_occurrence_pool(ctx.cn, doms, buffer, masks, comp);
        failed_literal_fixpoint(ctx.cn, doms, masks, tables, buffer, trail, &pool);
    }
    if doms[0] != DomainMask::NONE {
        let mut delta = W::one();
        for &tid in live_before {
            if let Some(cw) = sliced_constant(ctx.cn, ctx.weights, tid, doms) {
                delta = delta.mul(&cw); // newly resolved ⇒ fold once, here
            }
        }
        let sub = count_scope::<W>(ctx, stats, buffer, doms, masks, tables, trail, comp);
        w = delta.mul(&sub);
    }
    trail.restore_to(m, doms, tables);
    w
}

/// Connected components of `scope`'s unfixed vars under the WEIGHTED entailment:
/// a tensor connects its unfixed vars iff it is NOT a constant function
/// (`sliced_constant` is `None`). Mirrors `split_components` (used by the decision
/// path) but swaps boolean `is_entailed` for `sliced_constant`, so a full-support
/// but NON-uniform weighted tensor stays a live edge and never wrongly splits two
/// coupled variables into independent components (design §7 blocker 1). For 0/1
/// (unit-weight) tensors the two tests coincide, so unweighted counts are
/// bit-for-bit the decision path's decomposition.
fn split_components_counting<W: Weight>(
    cn: &ConstraintNetwork,
    weights: &[Vec<W>],
    doms: &[DomainMask],
    scope: &[usize],
) -> Vec<Vec<usize>> {
    let mut comps: Vec<Vec<usize>> = Vec::new();
    let mut var_seen = vec![false; doms.len()];
    let mut tensor_seen = vec![false; cn.tensors.len()];
    for &s in scope {
        if doms[s].is_fixed() || var_seen[s] {
            continue;
        }
        var_seen[s] = true;
        let mut comp = vec![s];
        let mut head = 0usize;
        while head < comp.len() {
            let v = comp[head];
            head += 1;
            for &tid in &cn.v2t[v] {
                if tensor_seen[tid] {
                    continue;
                }
                tensor_seen[tid] = true;
                if sliced_constant(cn, weights, tid, doms).is_some() {
                    continue; // constant tensor couples nothing (its scalar is folded elsewhere)
                }
                for &u in &cn.tensors[tid].var_axes {
                    if !doms[u].is_fixed() && !var_seen[u] {
                        var_seen[u] = true;
                        comp.push(u);
                    }
                }
            }
        }
        comp.sort_unstable();
        comps.push(comp);
    }
    comps
}

/// Weighted count of all assignments to `scope`'s unfixed vars satisfying every
/// constraint. Splits `scope` into connected components of the LIVE (non-constant)
/// constraint graph and MULTIPLIES their counts — components share no live
/// coupling, so their models combine independently. A zero component zeroes the
/// product and short-circuits the siblings. An empty split (scope fully fixed) is
/// the leaf: the empty product `one()`.
#[allow(clippy::too_many_arguments)]
fn count_scope<W: Weight>(
    ctx: &CountCtx<W>,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
    scope: &[usize],
) -> W {
    let mut comps = split_components_counting(ctx.cn, ctx.weights, doms, scope);
    if comps.len() > 1 {
        stats.record_split();
        comps.sort_unstable_by_key(|c| (c.len(), c[0]));
    }
    let mut product = W::one();
    for comp in &comps {
        let w = count_component::<W>(ctx, stats, buffer, doms, masks, tables, trail, comp);
        if w.is_zero() {
            return W::zero(); // a zero component ⇒ zero product; skip siblings
        }
        product = product.mul(&w);
    }
    product
}

/// Weighted count of one connected component. Picks a focus and branches its
/// region per-feasible-config (SUM, no early return), folding each config's
/// newly-resolved tensor weights via `count_branch`. When the boolean occurrence
/// selector finds no focus, either every unfixed var is truly FREE (return the
/// `free_factor` product — the constant tensors are already folded) or the only
/// live tensors are full-support-NON-uniform (invisible to the region machinery,
/// design §7 blocker 1); those still couple, so fall back to a binary split on
/// such a variable. Each branch fixes ≥1 var, so the component strictly shrinks.
#[allow(clippy::too_many_arguments)]
fn count_component<W: Weight>(
    ctx: &CountCtx<W>,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
    comp: &[usize],
) -> W {
    // A sibling component's forced propagation (failed-literal ranges over the
    // whole node, its fixes not yet restored) may have already fixed all of this
    // component's vars. Fully-fixed and contradiction-free ⇒ weight one().
    if comp.iter().all(|&v| doms[v].is_fixed()) {
        return W::one();
    }

    // Tensors LIVE on entry — each consumed once when it later resolves.
    let live_before = comp_incident_live(ctx.cn, ctx.weights, comp, doms);

    match select_var_most_occurrence(ctx.cn, doms, buffer, comp, masks) {
        Some(focus) => {
            // Grow the region fresh at the current doms; keep its GAC-feasible
            // configs, then branch each (a natural partition).
            let (region, rel) = grow_region(ctx.cn, doms, focus, ctx.max_rows, masks);
            let region_vars = region.vars;
            let mut feasible = feasible_configs(
                ctx.cn,
                doms,
                masks,
                tables,
                buffer,
                trail,
                &region_vars,
                &rel.rows,
            );
            if feasible.is_empty() {
                return W::zero();
            }
            feasible.sort_unstable();
            feasible.dedup();
            // Partition invariant (§7 blocker 4): one config per branch, strictly
            // increasing after dedup ⇒ mutually exclusive AND exhaustive.
            debug_assert!(
                feasible.windows(2).all(|w| w[0] < w[1]),
                "counting branches must be a strict partition of feasible configs"
            );
            let full_mask = if region_vars.len() == 64 {
                u64::MAX
            } else {
                (1u64 << region_vars.len()) - 1
            };
            // Region-partition branches: `PerConfig` = one singleton per config
            // (mask = full, val = config); `BlockMerge` = one perfect subcube per
            // block (mask leaves the block's free coords open). Both are exact
            // partitions of `feasible`; the recursion handles any coords a cube
            // leaves free. Compression stats feed the §3.1 predictor.
            let branches: Vec<(u64, u64)> = match ctx.branch {
                CountBranch::PerConfig => feasible.iter().map(|&cfg| (full_mask, cfg)).collect(),
                CountBranch::BlockMerge => block_merge(&feasible, region_vars.len())
                    .into_iter()
                    .map(|c| (c.mask, c.val))
                    .collect(),
            };
            stats.record_branch(branches.len() as u64);
            stats.record_region_partition(branches.len() as u64, feasible.len() as u64);
            let mut total = W::zero();
            for &(mask, val) in &branches {
                let w = count_branch(
                    ctx,
                    stats,
                    buffer,
                    doms,
                    masks,
                    tables,
                    trail,
                    comp,
                    &live_before,
                    &region_vars,
                    mask,
                    val,
                );
                total.add(&w);
            }
            total
        }
        None => {
            // No boolean-visible constraint on any comp var. Fall back only if a
            // full-support-NON-uniform tensor still couples a var (blocker 1).
            let branch_var = comp.iter().copied().find(|&v| {
                !doms[v].is_fixed()
                    && ctx.cn.v2t[v]
                        .iter()
                        .any(|&tid| sliced_constant(ctx.cn, ctx.weights, tid, doms).is_none())
            });
            match branch_var {
                None => {
                    // Truly free vars: their constant incident tensors are already
                    // folded (pre-pass / ancestor), so each contributes free_factor.
                    let mut w = W::one();
                    for &v in comp {
                        if !doms[v].is_fixed() {
                            w = w.mul(&W::free_factor(v));
                        }
                    }
                    w
                }
                Some(v) => {
                    stats.record_branch(2);
                    let mut total = W::zero();
                    for val in [0u64, 1u64] {
                        let w = count_branch(
                            ctx,
                            stats,
                            buffer,
                            doms,
                            masks,
                            tables,
                            trail,
                            comp,
                            &live_before,
                            &[v],
                            1u64,
                            val,
                        );
                        total.add(&w);
                    }
                    total
                }
            }
        }
    }
}

/// The failed-literal probe pool RESTRICTED to `comp` (§7 blocker 5: counting
/// keeps the failed-literal pool component-local for attribution clarity, as
/// sharpSAT/Ganak do — the derivation is count-safe either way, but a global
/// pool blurs which component consumed which variable). Top-`FAILED_LITERAL_POOL`
/// component vars by occurrence score; free (score-0) vars are excluded.
fn component_occurrence_pool(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
    masks: &[TableMasks],
    comp: &[usize],
) -> Vec<usize> {
    compute_occurrence_scores(cn, doms, buffer, masks);
    let mut cands: Vec<usize> = comp
        .iter()
        .copied()
        .filter(|&i| !doms[i].is_fixed() && buffer.occurrence_scores[i] > 0.0)
        .collect();
    cands.sort_by(|&a, &b| {
        buffer.occurrence_scores[b]
            .partial_cmp(&buffer.occurrence_scores[a])
            .expect("finite scores")
    });
    cands.truncate(FAILED_LITERAL_POOL);
    cands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::BranchSolver;
    use crate::dimacs::network_from_dimacs;
    use crate::util::count_unfixed;
    use optimal_branching_core::IPSolver;

    fn satisfies(cn: &ConstraintNetwork, sol: &[DomainMask]) -> bool {
        cn.tensors.iter().all(|t| {
            let mut cfg = 0u32;
            for (i, &v) in t.var_axes.iter().enumerate() {
                if sol[v].value().expect("fully assigned") {
                    cfg |= 1 << i;
                }
            }
            cn.is_sat(t, cfg)
        })
    }

    fn solve_cnf(cnf: &str) -> (Solve, ConstraintNetwork) {
        let cn = network_from_dimacs(cnf).expect("parse");
        let cn_for_check = cn.clone();
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        let s = bbsat(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(IPSolver::default()),
        );
        (s, cn_for_check)
    }

    #[test]
    fn failed_literal_root_solves_an_implication_cycle() {
        // (x1∨x2∨x3)(¬x1∨x2)(¬x2∨x3)(¬x3∨x1): every var is in both polarities
        // so domination fixes nothing, but failed-literal probing does — x1=0
        // cascades (x3=0, x2=0) to falsify the first clause, forcing x1=1,
        // which propagates to the unique solution (1,1,1). Solved at the root
        // by the reductions alone — zero branches. (The genuine branch path is
        // covered by `solves_a_pure_2sat_by_branching`.)
        let (s, cn) = solve_cnf("p cnf 3 4\n1 2 3 0\n-1 2 0\n-2 3 0\n-3 1 0\n");
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
        assert_eq!(s.stats.branching_nodes, 0);
    }

    #[test]
    fn domination_solves_pure_literal_instances_at_the_root() {
        // (x1∨x2∨x3) ∧ (¬x1∨x2) ∧ (¬x2∨x3): x3 is pure-positive; fixing it
        // entails the rest into further pure literals — zero branching nodes.
        let (s, cn) = solve_cnf("p cnf 3 3\n1 2 3 0\n-1 2 0\n-2 3 0\n");
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
        assert_eq!(s.stats.branching_nodes, 0);
    }

    #[test]
    fn proves_an_unsatisfiable_3sat() {
        // All eight 3-literal clauses over {x1,x2,x3} -> UNSAT. The region
        // feasibility probe rules out every local config, so the driver proves
        // UNSAT at the root (findbest -> None) WITHOUT branching — sound (GAC never
        // drops a real solution) and a strength of the region method. Assert only
        // the verdict.
        let cnf = "p cnf 3 8\n\
            1 2 3 0\n1 2 -3 0\n1 -2 3 0\n1 -2 -3 0\n\
            -1 2 3 0\n-1 2 -3 0\n-1 -2 3 0\n-1 -2 -3 0\n";
        let (s, _cn) = solve_cnf(cnf);
        assert!(!s.found);
    }

    /// A 4-clause 2-SAT "cycle" over {a,b,c} with every var in both
    /// polarities: (a∨b)(¬a∨¬b)(b∨c)(¬b∨¬c). No pure literal, GAC prunes
    /// nothing at the root, and it has exactly two solutions (1,0,1)/(0,1,0).
    fn cycle2sat(off: usize) -> String {
        let (a, b, c) = (off + 1, off + 2, off + 3);
        format!("{a} {b} 0\n-{a} -{b} 0\n{b} {c} 0\n-{b} -{c} 0\n")
    }

    #[test]
    fn closed_region_solves_a_small_component_in_one_node() {
        // The whole network joins into a 2-row relation well under the budget:
        // the region is closed, so ONE branch fixes one feasible config.
        let cnf = format!("p cnf 3 4\n{}", cycle2sat(0));
        let (s, cn) = solve_cnf(&cnf);
        assert!(s.found);
        assert!(satisfies(&cn, &s.solution));
        assert_eq!(s.stats.branching_nodes, 1);
        assert_eq!(s.stats.total_visited_nodes, 1);
    }

    #[test]
    fn free_vars_are_fixed_by_root_domination() {
        // One FULL tensor over [0,1]: both vars free — a full table flips
        // everywhere, so domination fixes them at the root. Zero branches.
        let full2 = vec![true, true, true, true];
        let cn = crate::network::setup_problem(2, vec![vec![0, 1]], vec![full2]);
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        assert!(p.is_solved(), "root domination fixes free vars");
        let s = bbsat(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(IPSolver::default()),
        );
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert_eq!(s.stats.branching_nodes, 0);
    }

    #[test]
    fn split_components_partitions_disconnected_vars() {
        // T0[0,1], T1[1,2] | T2[3,4]: two components {0,1,2} and {3,4}.
        let or2 = vec![false, true, true, true];
        let cn = crate::network::setup_problem(
            5,
            vec![vec![0, 1], vec![1, 2], vec![3, 4]],
            vec![or2.clone(), or2.clone(), or2],
        );
        let doms = vec![DomainMask::BOTH; 5];
        let (masks, _t) = crate::ct::build_tables(&cn);
        let comps = split_components(&cn, &doms, &masks, &[0, 1, 2, 3, 4]);
        assert_eq!(comps, vec![vec![0, 1, 2], vec![3, 4]]);
        // Fixing the cut var 1 splits {0,1,2} into {0} and {2}.
        let mut doms2 = doms.clone();
        doms2[1] = DomainMask::D1;
        let comps2 = split_components(&cn, &doms2, &masks, &[0, 1, 2]);
        assert_eq!(comps2, vec![vec![0], vec![2]]);
    }

    #[test]
    fn split_components_is_entailment_aware() {
        // T0[0,1] OR, T1[1,2] FULL (entailed), T2[2,3] OR: vars 0,1 and 2,3 are
        // joined only through the always-satisfied T1, which couples nothing.
        // Entailment-aware splitting must separate {0,1} from {2,3}; the old
        // tensor-adjacency would have merged all four.
        let or2 = vec![false, true, true, true];
        let full2 = vec![true, true, true, true];
        let cn = crate::network::setup_problem(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![or2.clone(), full2, or2],
        );
        let doms = vec![DomainMask::BOTH; 4];
        let (masks, _t) = crate::ct::build_tables(&cn);
        let comps = split_components(&cn, &doms, &masks, &[0, 1, 2, 3]);
        assert_eq!(comps, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn disconnected_sat_instance_splits_and_solves() {
        // Two independent pure-literal-free subproblems; the root must split.
        let cnf = format!("p cnf 6 8\n{}{}", cycle2sat(0), cycle2sat(3));
        let (s, cn) = solve_cnf(&cnf);
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
        assert!(s.stats.component_splits >= 1, "root must split");
    }

    #[test]
    fn unsat_component_refutes_a_disconnected_instance() {
        // Component A = pure-literal-free 2-SAT cycle over {1,2,3} (SAT);
        // component B = all eight 3-literal clauses over {4,5,6} (UNSAT).
        // Root GAC and domination cannot see B's contradiction (each clause
        // alone prunes nothing, and every flip direction is blocked in some
        // clause); the component search must refute B regardless of A.
        let cnf = format!(
            "p cnf 6 12\n{}\
            4 5 6 0\n4 5 -6 0\n4 -5 6 0\n4 -5 -6 0\n\
            -4 5 6 0\n-4 5 -6 0\n-4 -5 6 0\n-4 -5 -6 0\n",
            cycle2sat(0)
        );
        let (s, _cn) = solve_cnf(&cnf);
        assert!(!s.found);
        assert!(s.stats.component_splits >= 1, "root must split");
    }

    #[test]
    fn binary_control_arm_is_complete() {
        // The control selector (plain {v=0, v=1} branching, no region
        // machinery) must reach the same verdicts: SAT on the 2-SAT cycle,
        // UNSAT on the all-clauses instance.
        let solve_bin = |cnf: &str| {
            let cn = network_from_dimacs(cnf).expect("parse");
            let cn_for_check = cn.clone();
            let mut p = TnProblem::from_network(cn).expect("root SAT");
            let s = bbsat(
                &mut p,
                Selector::BinaryOccurrence,
                Measure::NumUnfixedVars,
                &BranchSolver::Ip(IPSolver::default()),
            );
            (s, cn_for_check)
        };
        let (s, cn) = solve_bin(&format!("p cnf 3 4\n{}", cycle2sat(0)));
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
        let (u, _) = solve_bin(
            "p cnf 3 8\n\
            1 2 3 0\n1 2 -3 0\n1 -2 3 0\n1 -2 -3 0\n\
            -1 2 3 0\n-1 2 -3 0\n-1 -2 3 0\n-1 -2 -3 0\n",
        );
        assert!(!u.found);
    }

    #[test]
    fn solves_a_pure_2sat_by_branching() {
        // All binary, both polarities everywhere: no special leaf and no
        // pure literal — the occurrence selector picks a var, the region
        // machinery branches, propagation finishes. Completeness must not
        // depend on any residual-class shortcut.
        let cnf = format!("p cnf 3 4\n{}", cycle2sat(0));
        let (s, cn) = solve_cnf(&cnf);
        assert!(s.found);
        assert!(satisfies(&cn, &s.solution));
        assert!(s.stats.branching_nodes >= 1);
    }

    // ---- Exact model counting (`bbcount`) -----------------------------------

    use crate::semiring::{BigCount, CheckedU128};

    /// Brute-force model count of a (compressed) network: number of full
    /// assignments over `cn.n_vars` satisfying every tensor. The counting oracle.
    fn brute_count(cn: &ConstraintNetwork) -> u128 {
        let n = cn.n_vars;
        assert!(n <= 20, "brute force capped at 2^20");
        let mut count = 0u128;
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
            if ok {
                count += 1;
            }
        }
        count
    }

    /// Count `cn` with `bbcount::<CheckedU128>` at `max_rows` under `branch` (0
    /// models on a root contradiction). No unconstrained-var multiplier: `cn` is
    /// already compressed and we brute-force the same compressed space.
    fn count_cn_branch(cn: &ConstraintNetwork, max_rows: usize, branch: CountBranch) -> u128 {
        // Unweighted: unit weights (all one()) ⇒ the weighted engine counts models.
        match TnProblem::from_network_counting(cn.clone()) {
            Ok(mut p) => {
                let w = crate::contract::WeightedNetwork::<CheckedU128>::unit(&p.static_cn);
                bbcount::<CheckedU128>(&mut p, &w, max_rows, branch)
                    .models
                    .value()
            }
            Err(_) => 0,
        }
    }

    /// Both partition arms must agree (they are exact partitions of the same
    /// space); return their common count, asserting the agreement.
    fn count_cn(cn: &ConstraintNetwork, max_rows: usize) -> u128 {
        let pc = count_cn_branch(cn, max_rows, CountBranch::PerConfig);
        let bm = count_cn_branch(cn, max_rows, CountBranch::BlockMerge);
        assert_eq!(
            pc, bm,
            "perconfig and blockmerge disagree at max_rows {max_rows}"
        );
        pc
    }

    fn count_cnf(cnf: &str, max_rows: usize) -> u128 {
        count_cn(&network_from_dimacs(cnf).expect("parse"), max_rows)
    }

    #[test]
    fn counts_independent_or_clauses() {
        // k independent binary ORs over disjoint var pairs: 3 models each,
        // components MULTIPLY ⇒ 3^k. Guards the component-product path.
        let cnf = "p cnf 6 3\n1 2 0\n3 4 0\n5 6 0\n";
        assert_eq!(count_cnf(cnf, 128), 27); // 3^3
    }

    #[test]
    fn counts_xor_and_parity_relations() {
        // XOR over [0,1] (support {01,10} = [F,T,T,F]): 2 models.
        let xor = vec![false, true, true, false];
        let cn = crate::network::setup_problem(2, vec![vec![0, 1]], vec![xor]);
        assert_eq!(count_cn(&cn, 128), 2);
        // 3-var even-parity relation x0⊕x1⊕x2=0 (support {000,011,101,110}): 4.
        let even3 = vec![true, false, false, true, false, true, true, false];
        let cn3 = crate::network::setup_problem(3, vec![vec![0, 1, 2]], vec![even3]);
        assert_eq!(count_cn(&cn3, 128), 4);
    }

    #[test]
    fn unsat_instance_counts_zero() {
        // All eight 3-literal clauses over 3 vars ⇒ UNSAT ⇒ 0 models.
        let cnf = "p cnf 3 8\n\
            1 2 3 0\n1 2 -3 0\n1 -2 3 0\n1 -2 -3 0\n\
            -1 2 3 0\n-1 2 -3 0\n-1 -2 3 0\n-1 -2 -3 0\n";
        assert_eq!(count_cnf(cnf, 128), 0);
    }

    #[test]
    fn free_vars_contribute_free_factor() {
        // (x0∨x1) over declared 4 compressed vars? Here setup keeps only used
        // vars. A single OR over [0,1] with an extra FULL (entailed) tensor over
        // [2]: var 2 is free ⇒ 3 (OR) × 2 (free) = 6.
        let or2 = vec![false, true, true, true];
        let full1 = vec![true, true];
        let cn = crate::network::setup_problem(3, vec![vec![0, 1], vec![2]], vec![or2, full1]);
        assert_eq!(count_cn(&cn, 128), 6);
    }

    #[test]
    fn count_matches_bruteforce_over_random_small_networks() {
        // The exactly-once accounting proof (§7 blocker 5): 300 random small
        // networks, bbcount vs the 2^n enumeration oracle. Any double-counted or
        // dropped variable/table/branch shows up as a mismatch. Deterministic
        // xorshift, mirroring the canonicalize reconstruction test.
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
            let n_vars = 3 + (next(&mut s) % 6) as usize; // 3..=8
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
            let cn = crate::network::setup_problem(n_vars, scopes, dense);
            let want = brute_count(&cn);
            // Branch-sum invariance: several region budgets reshape the branching
            // tree but must not move the count.
            for &mr in &[1usize, 3, 8, 128] {
                let got = count_cn(&cn, mr);
                assert_eq!(
                    got, want,
                    "seed {seed}, max_rows {mr}: bbcount {got} != brute {want}"
                );
            }
        }
    }

    #[test]
    fn bigcount_counts_a_wide_free_instance_exactly() {
        // 40 unconstrained-but-declared vars ⇒ 2^40 models, well past u32 and a
        // typical "count" width — BigCount must report it exactly. Built as one
        // trivially-true unit-free network: a single FULL tensor over [0,1] (4
        // models over 2 used vars) plus the DIMACS front-end's free multiplier.
        // Here we exercise bbcount's own free_factor path: 40 free vars.
        let full2 = vec![true, true, true, true];
        let cn = crate::network::setup_problem(2, vec![vec![0, 1]], vec![full2]);
        // The two used vars are both free (full tensor) ⇒ 4 models.
        let mut p = TnProblem::from_network_counting(cn).expect("root SAT");
        let w = crate::contract::WeightedNetwork::<BigCount>::unit(&p.static_cn);
        let models = bbcount::<BigCount>(&mut p, &w, 128, CountBranch::PerConfig).models;
        assert_eq!(models.to_decimal(), "4");
    }

    // ---- M2 weighted counting (`count_with_ve`, weighted `bbcount`) ----------

    use crate::contract::{WRelation, WeightedNetwork};
    use crate::semiring::RationalWeight;

    /// Does compressed config `cfg` (bit v = value of var v) satisfy every tensor?
    fn sat_cfg(cn: &ConstraintNetwork, cfg: u64) -> bool {
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

    #[test]
    fn weighted_count_matches_bruteforce_across_budgets() {
        // The M2.1 correctness proof (acceptance gate 3): 300 random small
        // networks with UNNORMALIZED per-variable literal weights (as 1-ary
        // weighted tensors). For every VE budget {0,2..=6} and region budget
        // {3,128}, (VE-scalar × bbcount) must equal the weighted brute-force sum
        // Σ_{σ ⊨ cn} ∏_v w(v = σ_v). Budget-INVARIANCE is the whole point: VE
        // reshapes the residual but never moves the count. Unnormalized weights
        // (not w+w̄=1) make free factors ≠ 1, so a dropped weighted factor shows.
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
            let cn = crate::network::setup_problem(n_vars, scopes, dense);
            let n = cn.n_vars;
            // Unnormalized literal weights per compressed var, integers 1..=4.
            let w0: Vec<i64> = (0..n).map(|_| 1 + (next(&mut s) % 4) as i64).collect();
            let w1: Vec<i64> = (0..n).map(|_| 1 + (next(&mut s) % 4) as i64).collect();

            // Weighted brute force over the compressed space.
            let mut brute = RationalWeight::zero();
            for cfg in 0u64..(1u64 << n) {
                if !sat_cfg(&cn, cfg) {
                    continue;
                }
                let mut w = RationalWeight::one();
                for v in 0..n {
                    let lw = if (cfg >> v) & 1 == 1 { w1[v] } else { w0[v] };
                    w = w.mul(&RationalWeight::int(lw));
                }
                brute.add(&w);
            }

            let base = crate::contract::unit_weighted_relations::<RationalWeight>(&cn);
            for &budget in &[0usize, 2, 3, 4, 5, 6] {
                for &mr in &[3usize, 128] {
                    let mut rels = base.clone();
                    for v in 0..n {
                        rels.push(WRelation {
                            vars: vec![v],
                            rows: vec![
                                (0u64, RationalWeight::int(w0[v])),
                                (1u64, RationalWeight::int(w1[v])),
                            ],
                        });
                    }
                    let (got, _stats) = count_with_ve::<RationalWeight>(
                        n,
                        rels,
                        budget,
                        mr,
                        CountBranch::PerConfig,
                    );
                    assert_eq!(
                        got, brute,
                        "seed {seed}, budget {budget}, max_rows {mr}: weighted count != brute"
                    );
                }
            }
        }
    }

    #[test]
    fn nonuniform_full_support_tensor_is_not_split() {
        // §7 blocker 1: A = XOR over {0,1} (v0≠v1), B = XOR over {2,3}, C = FULL
        // over {1,2} but NON-uniform weight — C couples v1,v2. Boolean entailment
        // would treat the full-support C as entailed and SPLIT {0,1} | {2,3},
        // multiplying marginals (2 × 2 = 4). The weighted `sliced_constant` keeps
        // C a live edge, so the components stay joined and the count is right.
        let one = RationalWeight::int(1);
        let two = RationalWeight::int(2);
        let a = WRelation {
            vars: vec![0, 1],
            rows: vec![(1u64, one.clone()), (2u64, one.clone())], // v0≠v1
        };
        // C over (bit0=v1, bit1=v2): weights 00,01,10 = 1, 11 = 2.
        let c = WRelation {
            vars: vec![1, 2],
            rows: vec![
                (0u64, one.clone()),
                (1u64, one.clone()),
                (2u64, one.clone()),
                (3u64, two.clone()),
            ],
        };
        let b = WRelation {
            vars: vec![2, 3],
            rows: vec![(1u64, one.clone()), (2u64, one.clone())], // v2≠v3
        };
        let wn = WeightedNetwork::from_relations(4, vec![a, c, b]);
        let mut p = TnProblem::from_network_counting(wn.cn).expect("root SAT");
        let got =
            bbcount::<RationalWeight>(&mut p, &wn.weights, 128, CountBranch::PerConfig).models;
        // Each (v1,v2) pairs with exactly one (v0,v3), so Σ = Σ C(v1,v2) = 5.
        assert_eq!(got, RationalWeight::int(5), "coupled count must be 5");
        assert_ne!(
            got,
            RationalWeight::int(4),
            "a wrong split would give 2×2 = 4"
        );
    }

    #[test]
    fn ve_on_counts_declared_but_unused_vars() {
        // §7 blocker 2, second site: `p cnf 5 1 / 1 2 0` = 3 (OR) × 2^3 = 24, and
        // that must hold with VE ON at EVERY budget — VE must not lose the free
        // variables the caller's `2^(declared−used)` multiplier restores.
        use num_bigint::BigUint;
        let cn = network_from_dimacs("p cnf 5 1\n1 2 0\n").expect("parse");
        let n_declared = 5usize;
        for budget in [0usize, 1, 2, 3, 4] {
            let rels = crate::contract::unit_weighted_relations::<BigCount>(&cn);
            let (count, _s) =
                count_with_ve::<BigCount>(cn.n_vars, rels, budget, 128, CountBranch::PerConfig);
            let mult = BigCount(BigUint::from(2u32).pow((n_declared - cn.n_vars) as u32));
            assert_eq!(
                count.mul(&mult).to_decimal(),
                "24",
                "budget {budget}: unused-var count must stay 24 with VE on"
            );
        }
    }

    #[test]
    fn weighted_network_keeps_per_tensor_weights_under_support_dedup() {
        // §7 blocker 6: two tensors with IDENTICAL support {0,1} but DIFFERENT
        // weights. The flyweight shares ONE TruthTable, but the weights live
        // per-tensor, so neither weighted factor is dropped/merged.
        let a = WRelation {
            vars: vec![0],
            rows: vec![
                (0u64, RationalWeight::int(2)),
                (1u64, RationalWeight::int(3)),
            ],
        };
        let b = WRelation {
            vars: vec![1],
            rows: vec![
                (0u64, RationalWeight::int(5)),
                (1u64, RationalWeight::int(7)),
            ],
        };
        let wn = WeightedNetwork::from_relations(2, vec![a, b]);
        assert_eq!(
            wn.cn.truth_tables.len(),
            1,
            "identical support ⇒ one shared TruthTable"
        );
        assert_eq!(
            wn.weights[0],
            vec![RationalWeight::int(2), RationalWeight::int(3)]
        );
        assert_eq!(
            wn.weights[1],
            vec![RationalWeight::int(5), RationalWeight::int(7)]
        );
        // The count uses the right per-tensor weights: (2+3)·(5+7) = 60.
        let mut p = TnProblem::from_network_counting(wn.cn).expect("root SAT");
        let got =
            bbcount::<RationalWeight>(&mut p, &wn.weights, 128, CountBranch::PerConfig).models;
        assert_eq!(got, RationalWeight::int(60));
    }

    #[test]
    fn f64_weighted_count_is_approximately_correct() {
        // The fast `f64` weight path: same coupled instance as the blocker-1 test
        // (exact answer 5), checked within f64 tolerance.
        let one = 1.0f64;
        let a = WRelation {
            vars: vec![0, 1],
            rows: vec![(1u64, one), (2u64, one)],
        };
        let c = WRelation {
            vars: vec![1, 2],
            rows: vec![(0u64, one), (1u64, one), (2u64, one), (3u64, 2.0f64)],
        };
        let b = WRelation {
            vars: vec![2, 3],
            rows: vec![(1u64, one), (2u64, one)],
        };
        let wn = WeightedNetwork::<f64>::from_relations(4, vec![a, c, b]);
        let mut p = TnProblem::from_network_counting(wn.cn).expect("root SAT");
        let got = bbcount::<f64>(&mut p, &wn.weights, 128, CountBranch::PerConfig).models;
        assert!((got - 5.0).abs() < 1e-9, "f64 weighted count {got} != 5");
    }

    #[test]
    fn blockmerge_matches_bruteforce_across_budgets_and_rows() {
        // The BlockMerge strict-correctness proof (acceptance gate 2). 300 random
        // small weighted networks with UNNORMALIZED per-variable literal weights;
        // for BOTH partition arms {PerConfig, BlockMerge}, every VE budget {0,3},
        // and every region budget {3,128}, (VE-scalar × bbcount) must equal the
        // weighted brute-force sum. BlockMerge only reshapes the branching tree
        // via a different partition of the same feasible set, so the count is
        // invariant — any partition bug (overlap ⇒ double count, escape ⇒ over
        // count, gap ⇒ under count) shows as a mismatch. The two arms must also
        // agree with each other.
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
            let cn = crate::network::setup_problem(n_vars, scopes, dense);
            let n = cn.n_vars;
            let w0: Vec<i64> = (0..n).map(|_| 1 + (next(&mut s) % 4) as i64).collect();
            let w1: Vec<i64> = (0..n).map(|_| 1 + (next(&mut s) % 4) as i64).collect();

            let mut brute = RationalWeight::zero();
            for cfg in 0u64..(1u64 << n) {
                if !sat_cfg(&cn, cfg) {
                    continue;
                }
                let mut w = RationalWeight::one();
                for v in 0..n {
                    let lw = if (cfg >> v) & 1 == 1 { w1[v] } else { w0[v] };
                    w = w.mul(&RationalWeight::int(lw));
                }
                brute.add(&w);
            }

            let base = crate::contract::unit_weighted_relations::<RationalWeight>(&cn);
            for &budget in &[0usize, 3] {
                for &mr in &[3usize, 128] {
                    let mk_rels = || {
                        let mut rels = base.clone();
                        for v in 0..n {
                            rels.push(WRelation {
                                vars: vec![v],
                                rows: vec![
                                    (0u64, RationalWeight::int(w0[v])),
                                    (1u64, RationalWeight::int(w1[v])),
                                ],
                            });
                        }
                        rels
                    };
                    let (pc, _) = count_with_ve::<RationalWeight>(
                        n,
                        mk_rels(),
                        budget,
                        mr,
                        CountBranch::PerConfig,
                    );
                    let (bm, _) = count_with_ve::<RationalWeight>(
                        n,
                        mk_rels(),
                        budget,
                        mr,
                        CountBranch::BlockMerge,
                    );
                    assert_eq!(
                        pc, brute,
                        "seed {seed}, budget {budget}, mr {mr}: perconfig != brute"
                    );
                    assert_eq!(
                        bm, brute,
                        "seed {seed}, budget {budget}, mr {mr}: blockmerge != brute"
                    );
                }
            }
        }
    }
}
