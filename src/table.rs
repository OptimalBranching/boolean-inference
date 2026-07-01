use std::sync::Arc;

use optimal_branching_core::{BranchingTable, Clause};

use crate::adapter::{BranchSolver, MeasureAdapter, RuleProblem};
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::propagate::probe;
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

    // 1. Keep configs consistent with the currently-fixed region vars AND feasible
    //    under a full GAC probe; cache each feasible config's measure.
    // Cached configs are encoded over the region's UNFIXED-at-initial_doms vars; here we index them over the full region_vars. These coincide only because the cache is built at the root, where no region var is fixed (the no-internal-var invariant — see the Phase 3 plan preamble). Rebuilding the cache at non-root doms would break this.
    let (check_mask, check_value) = mask_value_u64(doms, &region_vars);
    let n = region_vars.len();
    let full_mask: u64 = if n == 0 {
        0
    } else if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    };
    let mut feasible: Vec<u64> = Vec::new();
    for &config in &cached_configs {
        if (config & check_mask) != check_value {
            continue;
        }
        let feasible_here = probe(
            cn,
            doms,
            masks,
            tables,
            buffer,
            trail,
            &region_vars,
            full_mask,
            config,
            |d| d[0] != DomainMask::NONE,
        );
        if feasible_here {
            feasible.push(config);
        }
    }
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
    //    produce the rule through this one call. The result is identical to the
    //    previous direct `minimize_gamma` path for the set-cover solvers, because
    //    `apply_branch == probe` and `MeasureAdapter == measure_core`.
    let problem = RuleProblem::new(
        Arc::clone(cn),
        doms.to_vec(),
        Arc::clone(masks),
        tables.to_vec(),
    );
    let result = solver
        .optimal_rule(&problem, &table, &unfixed_vars, &MeasureAdapter(measure))
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
