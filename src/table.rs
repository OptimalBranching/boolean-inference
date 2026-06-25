use optimal_branching_core::{minimize_gamma, BranchingTable, Clause, SetCoverSolver};

use crate::domain::DomainMask;
use crate::measure::{measure_core, Measure};
use crate::network::ConstraintNetwork;
use crate::problem::{has_contradiction, SolverBuffer};
use crate::propagate::probe_assignment;
use crate::region::RegionCache;
use crate::util::mask_value_u64;

/// Measure reduction of applying `clause` over `vars`, memoized in
/// `buffer.branching_cache`. Port of `branch.jl::size_reduction`.
pub fn size_reduction(
    cn: &ConstraintNetwork,
    buffer: &mut SolverBuffer,
    doms: &[DomainMask],
    m: Measure,
    clause: Clause,
    vars: &[usize],
) -> f64 {
    let before = measure_core(cn, doms, m);
    let after = if let Some(&cached) = buffer.branching_cache.get(&clause) {
        cached
    } else {
        let scratch = probe_assignment(cn, buffer, doms, vars, clause.mask, clause.val);
        debug_assert!(
            !has_contradiction(scratch),
            "probing a table clause must not contradict"
        );
        let mv = measure_core(cn, scratch, m); // last use of `scratch` before the mutable insert
        buffer.branching_cache.insert(clause, mv);
        mv
    };
    before - after
}

/// Compute the optimal branching rule for `var_id`'s region under the current
/// `doms`. Port of `branchtable.jl::compute_branching_result`.
pub fn compute_branching_result(
    cache: &mut RegionCache,
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    buffer: &mut SolverBuffer,
    var_id: usize,
    measure: Measure,
    solver: &impl SetCoverSolver,
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
        let scratch = probe_assignment(cn, buffer, doms, &region_vars, full_mask, config);
        if scratch[0] != DomainMask::NONE {
            let mv = measure_core(cn, scratch, measure); // last use of `scratch`
            buffer
                .branching_cache
                .insert(Clause::new(full_mask, config), mv);
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

    // 4. delta_rho per candidate clause (probe-based + memoized), with the
    //    literal-count fallback (mirrors core::optimal_branching_rule).
    let candidates = table.candidate_clauses();
    let mut delta_rho: Vec<f64> = candidates
        .iter()
        .map(|c| size_reduction(cn, buffer, doms, measure, c.clause, &unfixed_vars))
        .collect();
    if delta_rho.iter().all(|&d| d <= 0.0) {
        for (i, c) in candidates.iter().enumerate() {
            delta_rho[i] = c.clause.len() as f64;
        }
    }

    // 5. Set-cover-optimal rule via the core's minimize_gamma (NOT optimal_branching_rule).
    let result = minimize_gamma(&table, &candidates, &delta_rho, solver)
        .expect("minimize_gamma failed on a non-empty branching table");
    (Some(result.optimal_rule.clauses), unfixed_vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;
    use optimal_branching_core::{IPSolver, DNF};

    fn or_network() -> ConstraintNetwork {
        let or2 = vec![false, true, true, true];
        setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![or2.clone(), or2])
    }

    #[test]
    fn size_reduction_is_before_minus_after() {
        let cn = or_network();
        let doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        // Before: 3 unfixed vars. Probe x1=1: (x0∨x1) and (x1∨x2) both satisfied,
        // nothing else forced -> after = 2 unfixed (x0,x2 free, x1 fixed).
        let cl = Clause::new(0b1, 0b1); // over vars [1]: set var1 = 1
        let r = size_reduction(&cn, &mut buf, &doms, Measure::NumUnfixedVars, cl, &[1]);
        assert_eq!(r, 1.0);
        // Cached now.
        assert!(buf.branching_cache.contains_key(&cl));
    }

    #[test]
    fn branching_result_covers_the_table() {
        let cn = or_network();
        let doms = vec![DomainMask::BOTH; 3];
        let mut cache = RegionCache::new(&cn, &doms, 2, 10);
        let mut buf = SolverBuffer::new(&cn);
        let (clauses, vars) = compute_branching_result(
            &mut cache,
            &cn,
            &doms,
            &mut buf,
            1,
            Measure::NumUnfixedVars,
            &IPSolver::default(),
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
        let cur = vec![DomainMask::D0, DomainMask::D0, DomainMask::BOTH];
        let (clauses, vars) = compute_branching_result(
            &mut cache,
            &cn,
            &cur,
            &mut buf,
            1,
            Measure::NumUnfixedVars,
            &IPSolver::default(),
        );
        assert!(clauses.is_none());
        assert_eq!(vars, vec![0, 1, 2]); // region vars reported on the no-op path
    }
}
