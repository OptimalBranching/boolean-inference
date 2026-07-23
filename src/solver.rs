use std::sync::Arc;

use crate::adapter::BranchSolver;
use crate::ct::{apply_masked_assignment, ct_propagate, RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{SolverBuffer, Stats, TnProblem};
use crate::propagate::{dominate_fixpoint, failed_literal_fixpoint};
use crate::selector::{occurrence_pool, Selector, FAILED_LITERAL_POOL};
use crate::trail::Trail;

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

/// Branch-and-reduce SAT solve. Port of `branch.jl::bbsat!`.
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
        // The failed search still applied selection-independent reductions and
        // branch fixings at the root node (its per-branch restores unwind the
        // subtree, not the work done before/around the first branch); unwind to
        // the root state so the UNSAT contract leaves `doms` untouched.
        trail.restore_to(mark, doms, tables);
    }
    Solve {
        found,
        // The success path never restores, so `doms` holds the full assignment.
        solution: if found { doms.clone() } else { Vec::new() },
        stats: stats.clone(),
    }
}

/// Recursively solve `scope`'s unfixed vars by region branching. A fully-fixed,
/// contradiction-free `scope` is the SAT leaf: GAC over fully-fixed vars leaves
/// every constraint's live tuple in place, so there is nothing left to branch
/// and the node is already satisfied. (In release, `findbest`'s free-var
/// fallback would otherwise emit an empty no-op clause and recurse to the same
/// leaf via one wasted node; this early return skips it and is what the
/// `findbest` debug_assert guards.) Otherwise pick a focus var, compute the
/// region branching rule, and try each branch — propagating and applying the
/// selection-independent reductions before descending. Returns whether `scope`
/// is satisfiable from the current state; on `false` the trail is restored to
/// the call state.
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
    if scope.iter().all(|&v| doms[v].is_fixed()) {
        return true;
    }
    let selection = ctx.selector.findbest(
        ctx.cn,
        doms,
        buffer,
        ctx.measure,
        ctx.solver,
        masks,
        tables,
        trail,
        scope,
        None,
        &[],
        false,
    );
    let clauses = match selection.clauses {
        Some(c) => c,
        None => return false,
    };
    let variables = selection.variables;

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
            && bbsat_rec(ctx, stats, buffer, doms, masks, tables, trail, scope)
        {
            return true;
        }
        trail.restore_to(m, doms, tables);
    }
    false
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
    fn disconnected_sat_instance_solves() {
        // Two independent pure-literal-free subproblems joined into one search
        // tree; the solver must still find a full satisfying assignment.
        let cnf = format!("p cnf 6 8\n{}{}", cycle2sat(0), cycle2sat(3));
        let (s, cn) = solve_cnf(&cnf);
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
    }

    #[test]
    fn unsat_component_refutes_a_disconnected_instance() {
        // Subproblem A = pure-literal-free 2-SAT cycle over {1,2,3} (SAT);
        // subproblem B = all eight 3-literal clauses over {4,5,6} (UNSAT).
        // Root GAC and domination cannot see B's contradiction (each clause
        // alone prunes nothing, and every flip direction is blocked in some
        // clause); the search must refute B regardless of A.
        let cnf = format!(
            "p cnf 6 12\n{}\
            4 5 6 0\n4 5 -6 0\n4 -5 6 0\n4 -5 -6 0\n\
            -4 5 6 0\n-4 5 -6 0\n-4 -5 6 0\n-4 -5 -6 0\n",
            cycle2sat(0)
        );
        let (s, _cn) = solve_cnf(&cnf);
        assert!(!s.found);
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
}
