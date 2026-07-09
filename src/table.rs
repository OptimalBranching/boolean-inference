use std::sync::Arc;

use optimal_branching_core::{BranchingTable, Clause};

use crate::adapter::{with_measure_scratch, BranchSolver, MeasureAdapter, RuleProblem};
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::propagate::feasible_configs;
use crate::region::{boundary_vars, grow_region};
use crate::trail::Trail;

/// Compute the optimal branching rule for `var_id`'s region under the current
/// `doms`. Port of `branchtable.jl::compute_branching_result`.
///
/// The region is grown FRESH at the current doms (see `grow_region`): its
/// joined relation is already doms-sliced and ranges over unfixed vars only,
/// so the rows feed the feasibility probe directly — no mask filtering or
/// projection. Deep tables are small because deep regions are grown from
/// small sliced relations, not because a cached root region is conditioned.
#[allow(clippy::too_many_arguments)]
pub fn compute_branching_result(
    cn: &Arc<ConstraintNetwork>,
    doms: &mut [DomainMask],
    buffer: &mut SolverBuffer,
    var_id: usize,
    max_rows: usize,
    measure: Measure,
    solver: &BranchSolver,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
) -> (Option<Vec<Clause>>, Vec<usize>) {
    // 1. Grow the region and keep only its GAC-feasible configs, decided with
    //    a single prefix-sharing trie DFS over the (already doms-sliced) rows.
    let (region, rel) = grow_region(cn, doms, var_id, max_rows, masks);
    let closed = boundary_vars(cn, &region, doms, masks).is_empty();
    let region_vars = region.vars;
    let mut feasible = feasible_configs(
        cn,
        doms,
        masks,
        tables,
        buffer,
        trail,
        &region_vars,
        &rel.rows,
    );
    if feasible.is_empty() {
        return (None, region_vars);
    }

    // 2. CLOSED region: no non-entailed external tensor sees any region var,
    //    so the region is a solved subproblem — any feasible config completes
    //    ANY solution of the rest of the network. Fix the first one in a
    //    single branch: enumerating alternatives buys nothing (if the rest
    //    fails under one config it fails under all), and the rule solver's
    //    quadratic merge is skipped entirely.
    if closed {
        let mask = if region_vars.len() == 64 {
            u64::MAX
        } else {
            (1u64 << region_vars.len()) - 1
        };
        return (Some(vec![Clause::new(mask, feasible[0])]), region_vars);
    }

    // 3. One singleton branching-table group per distinct surviving config:
    //    the rule must cover every one of them. Sorted so group order is
    //    deterministic (join output order is not canonical).
    feasible.sort_unstable();
    feasible.dedup();
    let groups: Vec<Vec<u64>> = feasible.iter().map(|&c| vec![c]).collect();
    let table = BranchingTable::new(region_vars.len(), groups);

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
        solver.optimal_rule(&problem, &table, &region_vars, &MeasureAdapter(measure))
    })
    .expect("optimal_branching_rule failed on a non-empty branching table");
    (Some(result.optimal_rule.clauses), region_vars)
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
    fn closed_region_fixes_one_feasible_config() {
        let cn = or_network();
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        // Generous budget: the region absorbs the whole network, so it is
        // CLOSED (no external tensor) and the shortcut must return a single
        // full-mask clause pinning one feasible config — not a covering rule.
        let (clauses, vars) = compute_branching_result(
            &cn,
            &mut doms,
            &mut buf,
            1,
            1 << 10,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
        );
        assert_eq!(vars, vec![0, 1, 2]);
        let clauses = clauses.expect("a branching rule should exist");
        assert_eq!(clauses.len(), 1, "closed region: one branch, no retry");
        assert_eq!(clauses[0].mask, 0b111, "every region var is fixed");
        // The pinned config is one of the five feasible ones.
        assert!([2u64, 3, 5, 6, 7].contains(&clauses[0].val));
    }

    #[test]
    fn every_distinct_config_is_its_own_group() {
        // Budget 3 keeps only the seed tensor T0 over vars {0,1}.
        // T0's feasible configs {01,10,11} = {1,2,3} each form a singleton
        // group, and the rule must cover all three.
        let cn = or_network();
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let masks = Arc::new(masks);
        let mut trail = Trail::new();
        let (clauses, vars) = compute_branching_result(
            &cn,
            &mut doms,
            &mut buf,
            0,
            3,
            Measure::NumUnfixedVars,
            &BranchSolver::Ip(IPSolver::default()),
            &masks,
            &mut tables,
            &mut trail,
        );
        assert_eq!(vars, vec![0, 1]);
        let clauses = clauses.expect("a branching rule should exist");
        let table = BranchingTable::new(2, vec![vec![1], vec![2], vec![3]]);
        assert!(table.covered_by(&DNF { clauses }));
    }

    #[test]
    fn no_feasible_config_returns_none() {
        // All eight 3-literal clauses over 3 vars: locally every config of the
        // grown region is refuted, so the grown relation is already empty and
        // the branching result is a no-op.
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
        let (clauses, vars) = compute_branching_result(
            &cn,
            &mut doms,
            &mut buf,
            0,
            1 << 10,
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
