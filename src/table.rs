use std::sync::Arc;
use std::time::Instant;

use optimal_branching_core::{BranchingTable, Clause, NaiveBranch, OptimalBranchingResult, DNF};

use crate::adapter::{with_measure_scratch, BranchSolver, MeasureAdapter, RuleProblem};
use crate::cdcl::CdclPropagator;
use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::propagate::feasible_configs;
use crate::region::{boundary_vars, grow_region_with_closure};
use crate::trail::Trail;

/// One counterfactual rule evaluated at exactly the selected rule's residual
/// state. This intentionally records only geometry/cost, not clause values: the
/// latter would make traces scale with the configuration table again.
#[derive(Clone, Debug, PartialEq)]
pub struct RuleEvaluationDiagnostics {
    pub branches: usize,
    pub decision_literals: usize,
    pub branching_vector: Vec<f64>,
    pub gamma: Option<f64>,
    pub solver_ns: u64,
}

/// Counterfactuals needed to separate region semantics from rule optimization.
/// `binary` branches both ways on the shared focus variable. `naive` emits one
/// full assignment per row of the exact same probe-surviving region table that
/// the selected solver receives.
#[derive(Clone, Debug, PartialEq)]
pub struct SameStateReplayDiagnostics {
    pub binary: RuleEvaluationDiagnostics,
    pub naive: RuleEvaluationDiagnostics,
}

/// Raw, per-node evidence for the region-to-rule mechanism.  These are kept as
/// counts (rather than a pre-computed "compression score") so experiments can
/// test competing explanations without changing the solver or trace schema.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionRuleDiagnostics {
    /// Highest-occurrence variable from which the region was grown.
    pub focus_var: usize,
    /// Number of constraint tensors absorbed into the grown region.
    pub region_tensors: usize,
    /// Number of currently-unfixed variables represented by the joined table.
    pub region_variables: usize,
    /// Region variables still seen by a non-entailed tensor outside the region.
    pub boundary_variables: usize,
    /// Rows in the doms-sliced joined relation before global feasibility probes.
    pub joined_rows: usize,
    /// Distinct rows surviving the global GAC feasibility probes.
    pub feasible_rows: usize,
    /// Distinct rows in the table seen by the rule solver.
    pub branching_rows: usize,
    /// Whether the region has no live connection to the residual network.
    pub closed: bool,
    /// Measure reductions of the selected covering clauses. Empty for a closed
    /// region shortcut or a locally infeasible region, where no rule was solved.
    pub branching_vector: Vec<f64>,
    /// Branching factor of `branching_vector`; unavailable for infeasible or
    /// numerically degenerate rules. Closed one-branch regions have gamma 1.
    pub gamma: Option<f64>,
    /// Present only for instrumented same-state replays. The cuber checks the
    /// selected DNF against every probe-surviving configuration before it can
    /// emit `true`; a failure aborts the experiment instead of writing a trace.
    pub cover_verified: Option<bool>,
    /// Wall-clock stages on the actual selected path. These diagnose whether a
    /// downstream benefit repays region construction and rule synthesis.
    pub region_growth_ns: u64,
    pub feasibility_probe_ns: u64,
    pub rule_solver_ns: u64,
    /// Optional counterfactuals. Disabled in production and only computed when
    /// explicitly requested by the traced cuber.
    pub same_state_replay: Option<SameStateReplayDiagnostics>,
}

/// Rule-selection result shared by the solver and cuber. `diagnostics` is only
/// present for the structure-aware region selector; the binary control arm does
/// not pay for region growth or feasibility probes merely to populate a trace.
#[derive(Clone, Debug)]
pub struct BranchingResult {
    pub clauses: Option<Vec<Clause>>,
    pub variables: Vec<usize>,
    pub diagnostics: Option<RegionRuleDiagnostics>,
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn summarize_rule(result: &OptimalBranchingResult, solver_ns: u64) -> RuleEvaluationDiagnostics {
    RuleEvaluationDiagnostics {
        branches: result.optimal_rule.clauses.len(),
        decision_literals: result
            .optimal_rule
            .clauses
            .iter()
            .map(|clause| clause.mask.count_ones() as usize)
            .sum(),
        branching_vector: result.branching_vector.clone(),
        gamma: result.gamma.is_finite().then_some(result.gamma),
        solver_ns,
    }
}

/// Evaluate binary and configuration-by-configuration controls while the live
/// CT store is lent to the measure scratch. Every `optimal_rule` probe restores
/// that scratch to the same base, so the two results share an identical state.
fn replay_same_state(
    problem: &RuleProblem,
    region_table: &BranchingTable,
    region_vars: &[usize],
    focus_var: usize,
    measure: Measure,
) -> SameStateReplayDiagnostics {
    let naive_solver = BranchSolver::Naive(NaiveBranch);

    let binary_table = BranchingTable::new(1, vec![vec![0], vec![1]]);
    let binary_start = Instant::now();
    let binary = naive_solver
        .optimal_rule(
            problem,
            &binary_table,
            &[focus_var],
            &MeasureAdapter(measure),
        )
        .expect("binary same-state replay failed");
    let binary = summarize_rule(&binary, elapsed_ns(binary_start));

    let naive_start = Instant::now();
    let naive = naive_solver
        .optimal_rule(problem, region_table, region_vars, &MeasureAdapter(measure))
        .expect("naive same-state replay failed");
    let naive = summarize_rule(&naive, elapsed_ns(naive_start));

    SameStateReplayDiagnostics { binary, naive }
}

/// Compute the optimal branching rule for `var_id`'s region under the current
/// `doms`. Port of `branchtable.jl::compute_branching_result`.
///
/// The region is grown FRESH at the current doms (see `grow_region`): its
/// joined relation is already doms-sliced and ranges over unfixed vars only,
/// so the rows feed the feasibility probe directly — no mask filtering is
/// needed. Deep tables are small because deep regions are grown from small
/// sliced relations, not because a cached root region is conditioned.
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
    collect_diagnostics: bool,
    replay_diagnostics: bool,
) -> BranchingResult {
    compute_branching_result_with_cdcl(
        cn,
        doms,
        buffer,
        var_id,
        max_rows,
        measure,
        solver,
        masks,
        tables,
        trail,
        None,
        &[],
        collect_diagnostics,
        replay_diagnostics,
    )
}

/// CDCL-scored form of [`compute_branching_result`]. Region growth and global
/// feasibility remain native/CT; only the many hypothetical `apply_branch`
/// probes performed by the rule optimizer use assumption-only CDCL BCP.
#[allow(clippy::too_many_arguments)]
pub fn compute_branching_result_with_cdcl(
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
    cdcl: Option<&CdclPropagator>,
    cdcl_decisions: &[(usize, bool)],
    collect_diagnostics: bool,
    replay_diagnostics: bool,
) -> BranchingResult {
    debug_assert!(!replay_diagnostics || collect_diagnostics);
    // 1. Grow the region and keep only its GAC-feasible configs, decided with
    //    a single prefix-sharing trie DFS over the (already doms-sliced) rows.
    let region_start = collect_diagnostics.then(Instant::now);
    let (region, rel, closed) = grow_region_with_closure(cn, doms, var_id, max_rows, masks);
    let region_growth_ns = region_start.map(elapsed_ns).unwrap_or(0);
    // Growth already knows whether the live frontier is empty. Enumerate the
    // exact boundary only for trace diagnostics; production search pays no
    // second incidence scan and allocates no boundary vector.
    let boundary_variables = if collect_diagnostics {
        boundary_vars(cn, &region, doms, masks).len()
    } else {
        0
    };
    debug_assert!(!collect_diagnostics || closed == (boundary_variables == 0));
    let region_tensors = region.tensors.len();
    let region_vars = region.vars;
    let joined_rows = rel.rows.len();
    let feasibility_start = collect_diagnostics.then(Instant::now);
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
    let feasibility_probe_ns = feasibility_start.map(elapsed_ns).unwrap_or(0);
    if feasible.is_empty() {
        return BranchingResult {
            clauses: None,
            diagnostics: collect_diagnostics.then(|| RegionRuleDiagnostics {
                focus_var: var_id,
                region_tensors,
                region_variables: region_vars.len(),
                boundary_variables,
                joined_rows,
                feasible_rows: 0,
                branching_rows: 0,
                closed,
                branching_vector: Vec::new(),
                gamma: None,
                cover_verified: None,
                region_growth_ns,
                feasibility_probe_ns,
                rule_solver_ns: 0,
                same_state_replay: None,
            }),
            variables: region_vars,
        };
    }

    // A branching-table group represents a configuration, not a duplicate join
    // witness. Canonicalize before both the closed shortcut and rule solving so
    // the diagnostic count is exactly the semantic state count being covered.
    feasible.sort_unstable();
    feasible.dedup();
    let feasible_rows = feasible.len();

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
        let same_state_replay = if replay_diagnostics {
            let groups: Vec<Vec<u64>> = feasible.iter().map(|&c| vec![c]).collect();
            let table = BranchingTable::new(region_vars.len(), groups);
            let problem = rule_problem(cn, masks, doms, cdcl, cdcl_decisions);
            Some(with_measure_scratch(doms, tables, buffer, trail, || {
                replay_same_state(&problem, &table, &region_vars, var_id, measure)
            }))
        } else {
            None
        };
        return BranchingResult {
            clauses: Some(vec![Clause::new(mask, feasible[0])]),
            diagnostics: collect_diagnostics.then(|| RegionRuleDiagnostics {
                focus_var: var_id,
                region_tensors,
                region_variables: region_vars.len(),
                boundary_variables,
                joined_rows,
                feasible_rows,
                branching_rows: feasible_rows,
                closed,
                branching_vector: Vec::new(),
                gamma: Some(1.0),
                cover_verified: None,
                region_growth_ns,
                feasibility_probe_ns,
                rule_solver_ns: 0,
                same_state_replay,
            }),
            variables: region_vars,
        };
    }

    // 3. Build the rule over all surviving region configurations.
    let groups: Vec<Vec<u64>> = feasible.iter().map(|&config| vec![config]).collect();
    let table = BranchingTable::new(region_vars.len(), groups);

    // 4. Optimal rule via the unified BranchingRuleSolver entry point. The
    //    framework computes each candidate's measure reduction itself
    //    (apply_branch + measure) and applies the literal-count fallback when the
    //    measure is degenerate, so IPSolver/LPSolver/GreedyMerge/NaiveBranch all
    //    produce the rule through this one call. `apply_branch` uses the selected
    //    CDCL or CT propagation backend.
    let problem = rule_problem(cn, masks, doms, cdcl, cdcl_decisions);
    // Keep CT scratch primed for the CT backend and replay path. CDCL candidate
    // calls ignore it. Either way, `doms`/`tables`/`buffer`/`trail` are unchanged.
    let (result, rule_solver_ns, same_state_replay) =
        with_measure_scratch(doms, tables, buffer, trail, || {
            let rule_start = collect_diagnostics.then(Instant::now);
            let result = solver
                .optimal_rule(&problem, &table, &region_vars, &MeasureAdapter(measure))
                .expect("optimal_branching_rule failed on a non-empty branching table");
            let rule_solver_ns = rule_start.map(elapsed_ns).unwrap_or(0);
            let replay = replay_diagnostics
                .then(|| replay_same_state(&problem, &table, &region_vars, var_id, measure));
            (result, rule_solver_ns, replay)
        });
    let gamma = result.gamma.is_finite().then_some(result.gamma);
    let cover_verified = replay_diagnostics.then(|| {
        let covered = table.covered_by(&DNF {
            clauses: result.optimal_rule.clauses.clone(),
        });
        assert!(
            covered,
            "selected branching rule does not cover the probe-surviving table"
        );
        true
    });
    let diagnostics = if collect_diagnostics {
        Some(RegionRuleDiagnostics {
            focus_var: var_id,
            region_tensors,
            region_variables: region_vars.len(),
            boundary_variables,
            joined_rows,
            feasible_rows,
            branching_rows: feasible_rows,
            closed,
            branching_vector: result.branching_vector,
            gamma,
            cover_verified,
            region_growth_ns,
            feasibility_probe_ns,
            rule_solver_ns,
            same_state_replay,
        })
    } else {
        None
    };
    BranchingResult {
        clauses: Some(result.optimal_rule.clauses),
        diagnostics,
        variables: region_vars,
    }
}

fn rule_problem(
    cn: &Arc<ConstraintNetwork>,
    masks: &Arc<Vec<TableMasks>>,
    doms: &[DomainMask],
    cdcl: Option<&CdclPropagator>,
    cdcl_decisions: &[(usize, bool)],
) -> RuleProblem {
    let problem = RuleProblem::new(Arc::clone(cn), Arc::clone(masks), doms.to_vec());
    match cdcl {
        Some(cdcl) => problem.with_cdcl(cdcl.clone(), cdcl_decisions.to_vec()),
        None => problem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::BranchSolver;
    use crate::network::setup_problem;
    use optimal_branching_core::{complexity_bv, IPSolver, DNF};

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
        let result = compute_branching_result(
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
            true,
            true,
        );
        assert_eq!(result.variables, vec![0, 1, 2]);
        let diagnostics = result.diagnostics.expect("region diagnostics");
        assert!(diagnostics.closed);
        assert_eq!(diagnostics.boundary_variables, 0);
        assert_eq!(diagnostics.feasible_rows, 5);
        assert_eq!(diagnostics.branching_rows, 5);
        assert_eq!(diagnostics.gamma, Some(1.0));
        let replay = diagnostics.same_state_replay.expect("closed replay");
        assert_eq!(replay.binary.branches, 2);
        assert_eq!(replay.naive.branches, 5);
        let clauses = result.clauses.expect("a branching rule should exist");
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
        let result = compute_branching_result(
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
            true,
            true,
        );
        assert_eq!(result.variables, vec![0, 1]);
        let diagnostics = result.diagnostics.expect("region diagnostics");
        assert!(!diagnostics.closed);
        assert_eq!(diagnostics.region_tensors, 1);
        assert_eq!(diagnostics.region_variables, 2);
        assert_eq!(diagnostics.boundary_variables, 1);
        assert_eq!(diagnostics.joined_rows, 3);
        assert_eq!(diagnostics.feasible_rows, 3);
        assert_eq!(diagnostics.branching_rows, 3);
        assert!(!diagnostics.branching_vector.is_empty());
        assert!(diagnostics.gamma.is_some());
        let replay = diagnostics.same_state_replay.expect("open replay");
        assert_eq!(replay.binary.branches, 2);
        assert_eq!(replay.naive.branches, 3);
        assert_eq!(replay.naive.decision_literals, 6);
        assert_eq!(replay.naive.branching_vector.len(), 3);
        assert!(
            (diagnostics.gamma.unwrap() - complexity_bv(&diagnostics.branching_vector)).abs()
                < 1e-12
        );
        let clauses = result.clauses.expect("a branching rule should exist");
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
        let result = compute_branching_result(
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
            true,
            true,
        );
        assert!(result.clauses.is_none());
        assert_eq!(result.variables, vec![0, 1, 2]); // region vars reported on the no-op path
        let diagnostics = result.diagnostics.expect("region diagnostics");
        assert_eq!(diagnostics.feasible_rows, 0);
        assert!(diagnostics.gamma.is_none());
        assert!(diagnostics.same_state_replay.is_none());
    }
}
