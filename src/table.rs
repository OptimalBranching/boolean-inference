use std::sync::Arc;

use optimal_branching_core::{BranchingTable, Clause};
use rustc_hash::FxHashSet;

use crate::adapter::{with_measure_scratch, BranchSolver, MeasureAdapter, RuleProblem};
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::propagate::feasible_configs;
use crate::region::{boundary_vars, RegionCache};
use crate::trail::Trail;
use crate::util::mask_value_u64;

/// Compute the optimal branching rule for `var_id`'s region under the current
/// `doms`. Port of `branchtable.jl::compute_branching_result`.
///
/// The region and its configs come from the ROOT-built `RegionCache` (see its
/// encoding invariant); here they are conditioned on the current fixings: mask
/// filter, GAC-feasibility probe, projection onto the still-unfixed vars. This
/// conditioning is what keeps deep branching tables small — the same cached
/// region yields ever-sharper tables as vars get fixed.
#[allow(clippy::too_many_arguments)]
pub fn compute_branching_result(
    cache: &mut RegionCache,
    cn: &Arc<ConstraintNetwork>,
    doms: &mut [DomainMask],
    buffer: &mut SolverBuffer,
    var_id: usize,
    measure: Measure,
    solver: &BranchSolver,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
) -> (Option<Vec<Clause>>, Vec<usize>) {
    cache.ensure_region(cn, var_id);
    let region = cache.var_to_region[var_id].as_ref().unwrap();
    let region_vars = region.vars.clone();
    // Sorted ascending (region.vars is ascending and filter preserves order).
    let boundary = boundary_vars(cn, region);
    // `cached_configs` is read-only; clone it out so `buffer` can be mutated freely.
    let cached_configs = cache.var_to_configs[var_id].as_ref().unwrap().clone();

    // 1. Keep configs consistent with the currently-fixed region vars (cached
    //    configs are encoded over region_vars, all unfixed at the root — the
    //    RegionCache invariant), then decide GAC-feasibility of the survivors
    //    with a single prefix-sharing trie DFS.
    let (check_mask, check_value) = mask_value_u64(doms, &region_vars);
    let filtered: Vec<u64> = cached_configs
        .iter()
        .copied()
        .filter(|&config| (config & check_mask) == check_value)
        .collect();
    let feasible = feasible_configs(
        cn,
        doms,
        masks,
        tables,
        buffer,
        trail,
        &region_vars,
        &filtered,
    );
    if feasible.is_empty() {
        return (None, region_vars);
    }

    // 2. Drop region vars already fixed; project surviving configs onto the rest.
    let mut unfixed_positions: Vec<usize> = Vec::new();
    let mut unfixed_vars: Vec<usize> = Vec::new();
    for (i, &v) in region_vars.iter().enumerate() {
        if !doms[v].is_fixed() {
            unfixed_positions.push(i);
            unfixed_vars.push(v);
        }
    }
    if unfixed_vars.is_empty() {
        return (None, region_vars);
    }
    let mut projected: Vec<u64> = feasible
        .iter()
        .map(|&config| {
            let mut nc = 0u64;
            for (new_i, &old_i) in unfixed_positions.iter().enumerate() {
                if (config >> old_i) & 1 == 1 {
                    nc |= 1u64 << new_i;
                }
            }
            nc
        })
        .collect();
    projected.sort_unstable();
    projected.dedup();

    // 3. One branching-table group per BOUNDARY-equivalence class of surviving
    //    configs. Two region-feasible configs agreeing on all boundary vars are
    //    interchangeable: interior vars' constraints all lie inside the region
    //    (satisfied by either config), and every external tensor sees only
    //    boundary vars + already-fixed vars. So covering one representative per
    //    class keeps the branch set complete, and the table shrinks from
    //    #configs to #boundary-classes — the size GreedyMerge is quadratic in.
    //    A singleton group is covered iff its representative is, so we keep one
    //    config per class. (Consequence: the solver may return a solution whose
    //    interior bits differ from other equally-valid ones — fine for SAT and
    //    read-out, NOT a basis for enumeration/counting.)
    let mut bmask = 0u64;
    for (i, &v) in unfixed_vars.iter().enumerate() {
        if boundary.binary_search(&v).is_ok() {
            bmask |= 1u64 << i;
        }
    }
    // `projected` is sorted, so representatives (first of each class) and group
    // order are deterministic.
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    let groups: Vec<Vec<u64>> = projected
        .iter()
        .copied()
        .filter(|&c| seen.insert(c & bmask))
        .map(|c| vec![c])
        .collect();
    let table = BranchingTable::new(unfixed_vars.len(), groups);

    // 4. Optimal rule via the unified BranchingRuleSolver entry point. The
    //    framework computes each candidate's measure reduction itself
    //    (apply_branch + measure) and applies the literal-count fallback when the
    //    measure is degenerate, so IPSolver/LPSolver/GreedyMerge/NaiveBranch all
    //    produce the rule through this one call. `apply_branch` uses CT via the
    //    thread-local measure scratch primed here.
    let problem = RuleProblem::new(Arc::clone(cn), Arc::clone(masks), doms.to_vec());
    // Lend the live CT state to the measure scratch so apply_branch propagates
    // with CT instead of the linear rescan. apply_branch restores it to base
    // after every candidate, so `doms`/`tables`/`buffer`/`trail` are unchanged here.
    let result = with_measure_scratch(doms, tables, buffer, trail, || {
        solver.optimal_rule(&problem, &table, &unfixed_vars, &MeasureAdapter(measure))
    })
    .expect("optimal_branching_rule failed on a non-empty branching table");
    (Some(result.optimal_rule.clauses), unfixed_vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::BranchSolver;
    use crate::network::setup_problem;
    use optimal_branching_core::{IPSolver, DNF};

    fn or_network() -> Arc<ConstraintNetwork> {
        let or2 = vec![false, true, true, true];
        Arc::new(setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![or2.clone(), or2],
        ))
    }

    #[test]
    fn branching_result_covers_the_table() {
        let cn = or_network();
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        // Generous budget: the region absorbs the whole network.
        let mut cache = RegionCache::new(&cn, &doms, 1 << 10);
        let (clauses, vars) = compute_branching_result(
            &mut cache,
            &cn,
            &mut doms,
            &mut buf,
            1,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
        );
        assert_eq!(vars, vec![0, 1, 2]);
        let clauses = clauses.expect("a branching rule should exist");
        assert!(!clauses.is_empty());
        // The region spans the whole network, so its boundary is empty and all
        // five feasible configs form ONE class: the rule must cover at least one
        // of them (any-of group semantics).
        let table = BranchingTable::new(3, vec![vec![2, 3, 5, 6, 7]]);
        assert!(table.covered_by(&DNF { clauses }));
    }

    #[test]
    fn interior_var_collapses_table_to_boundary_classes() {
        // Budget 3 keeps only the seed tensor T0 over vars {0,1}.
        // Var 1 also touches T1 -> boundary = [1]; var 0 is interior.
        // T0's feasible configs {01,10,11} split into two boundary classes by
        // bit(var1): {1} (v1=0) and {2,3} (v1=1) -> a 2-group table, and the
        // rule must cover one representative per class.
        let cn = or_network();
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        let mut cache = RegionCache::new(&cn, &doms, 3);
        cache.ensure_region(&cn, 0);
        assert_eq!(cache.var_to_region[0].as_ref().unwrap().tensors, vec![0]);
        let (clauses, vars) = compute_branching_result(
            &mut cache,
            &cn,
            &mut doms,
            &mut buf,
            0,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
        );
        assert_eq!(vars, vec![0, 1]);
        let clauses = clauses.expect("a branching rule should exist");
        // Representatives are the first (ascending) config of each class: 1 and 2.
        let grouped = BranchingTable::new(2, vec![vec![1], vec![2, 3]]);
        assert!(grouped.covered_by(&DNF { clauses }));
    }

    #[test]
    fn no_feasible_config_returns_none() {
        // All eight 3-literal clauses over 3 vars: locally every config of the
        // grown region is refuted, so the feasibility probe empties the table.
        let mut scopes = Vec::new();
        let mut tabs = Vec::new();
        for miss in 0..8usize {
            scopes.push(vec![0, 1, 2]);
            let mut dense = vec![true; 8];
            dense[miss] = false; // clause forbidding exactly config `miss`
            tabs.push(dense);
        }
        let cn = Arc::new(setup_problem(3, scopes, tabs));
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        let mut cache = RegionCache::new(&cn, &doms, 1 << 10);
        cache.ensure_region(&cn, 0);
        // The join of all eight clauses is already empty.
        assert!(cache.var_to_configs[0].as_ref().unwrap().is_empty());
        let (clauses, vars) = compute_branching_result(
            &mut cache,
            &cn,
            &mut doms,
            &mut buf,
            0,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
        );
        assert!(clauses.is_none());
        assert_eq!(vars, vec![0, 1, 2]); // region vars reported on the no-op path
    }
}
