use std::sync::Arc;

use optimal_branching_core::{BranchingTable, Clause};

use crate::adapter::{with_measure_scratch, BranchSolver, MeasureAdapter, RuleProblem};
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::propagate::feasible_configs;
use crate::region::RegionCache;
use crate::trail::Trail;
use crate::util::mask_value_u64;

/// Compute the optimal branching rule for `var_id`'s region under the current
/// `doms`. Port of `branchtable.jl::compute_branching_result`.
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
    let region_vars = cache.var_to_region[var_id].as_ref().unwrap().vars.clone();
    // `cached_configs` is read-only; clone it out so `buffer` can be mutated freely.
    let cached_configs = cache.var_to_configs[var_id].as_ref().unwrap().clone();

    // 1. Keep configs consistent with the currently-fixed region vars, then
    //    decide GAC-feasibility of the survivors with a single prefix-sharing
    //    trie DFS (feasible_configs) that shares propagation of common prefixes.
    // Cached configs are encoded over the region's UNFIXED-at-initial_doms vars; here we index them over the full region_vars. These coincide only because the cache is built at the root, where no region var is fixed (the no-internal-var invariant — see the Phase 3 plan preamble). Rebuilding the cache at non-root doms would break this.
    let (check_mask, check_value) = mask_value_u64(doms, &region_vars);
    // Configs consistent with the currently-fixed region vars; feasibility is
    // decided by a single prefix-sharing trie DFS instead of one probe per config.
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

    // 3. One branching-table group per surviving config.
    let table = BranchingTable::new(
        unfixed_vars.len(),
        projected.iter().map(|&c| vec![c]).collect(),
    );

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
        let mut cache = RegionCache::new(&cn, &doms, 2, 10);
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
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
        // The returned DNF must cover every feasible config of the table.
        let table = BranchingTable::new(3, vec![vec![2], vec![3], vec![5], vec![6], vec![7]]);
        assert!(table.covered_by(&DNF { clauses }));
    }

    #[test]
    fn no_feasible_config_returns_none() {
        let cn = or_network();
        // Cache built at all-unfixed (full config set), but query with v0=0,v1=0:
        // every cached config has v0 or v1 set, so none survives the mask filter.
        let init = vec![DomainMask::BOTH; 3];
        let mut cache = RegionCache::new(&cn, &init, 2, 10);
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        let mut cur = vec![DomainMask::D0, DomainMask::D0, DomainMask::BOTH];
        let (clauses, vars) = compute_branching_result(
            &mut cache,
            &cn,
            &mut cur,
            &mut buf,
            1,
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
