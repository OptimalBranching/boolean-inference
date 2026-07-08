use std::sync::Arc;

use crate::adapter::BranchSolver;
use crate::ct::{ct_propagate, enqueue_var_change, RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{SolverBuffer, Stats, TnProblem};
use crate::selector::Selector;
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
/// constraint graph and solve each independently. Components share no tensor
/// with another's unfixed vars, so propagation and region growth from one can
/// never reach another; a satisfying assignment of one component stays valid
/// whatever is chosen in the rest. Hence one failing component refutes the
/// whole scope (no cross-component backtracking), and tree size is the SUM of
/// component trees instead of their product.
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
    let mut comps = split_components(ctx.cn, doms, scope);
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
    let (clauses, variables) = ctx.selector.findbest(
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
        // Apply the clause literals (trailed) and seed the propagation queue.
        buffer.queue.clear();
        for b in buffer.in_queue.iter_mut() {
            *b = false;
        }
        for (i, &var) in variables.iter().enumerate() {
            if (cl.mask >> i) & 1 == 1 {
                let nd = if (cl.val >> i) & 1 == 1 {
                    DomainMask::D1
                } else {
                    DomainMask::D0
                };
                if doms[var] != nd {
                    trail.record_dom(var, doms[var]);
                    doms[var] = nd;
                    enqueue_var_change(ctx.cn, buffer, var);
                }
            }
        }
        ct_propagate(ctx.cn, doms, masks, tables, buffer, trail);
        if doms[0] != DomainMask::NONE
            && bbsat_rec(ctx, stats, buffer, doms, masks, tables, trail, comp)
        {
            return true;
        }
        trail.restore_to(m, doms, tables);
    }
    false
}

/// Connected components of the unfixed vars in `scope` under "shares a tensor"
/// adjacency, each sorted ascending. A tensor connects exactly its unfixed
/// vars: fixed vars carry no residual coupling (their value is already sliced
/// into every incident table). BFS may pull in connected unfixed vars outside
/// `scope`; they belong to the same subproblem and are included.
fn split_components(
    cn: &ConstraintNetwork,
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
    fn solves_a_satisfiable_3sat() {
        // (x1∨x2∨x3) ∧ (¬x1∨x2) ∧ (¬x2∨x3) — satisfiable (e.g. 0,0,1).
        let (s, cn) = solve_cnf("p cnf 3 3\n1 2 3 0\n-1 2 0\n-2 3 0\n");
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
        // The degree-3 clause makes the root non-2-SAT, so the branch path runs.
        assert!(s.stats.branching_nodes >= 1);
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
        let comps = split_components(&cn, &doms, &[0, 1, 2, 3, 4]);
        assert_eq!(comps, vec![vec![0, 1, 2], vec![3, 4]]);
        // Fixing the cut var 1 splits {0,1,2} into {0} and {2}.
        let mut doms2 = doms.clone();
        doms2[1] = DomainMask::D1;
        let comps2 = split_components(&cn, &doms2, &[0, 1, 2]);
        assert_eq!(comps2, vec![vec![0], vec![2]]);
    }

    #[test]
    fn disconnected_sat_instance_splits_and_solves() {
        // Two independent 3-var subproblems; the root must split.
        let cnf = "p cnf 6 4\n1 2 3 0\n-1 2 0\n4 5 6 0\n-4 5 0\n";
        let (s, cn) = solve_cnf(cnf);
        assert!(s.found);
        assert_eq!(count_unfixed(&s.solution), 0);
        assert!(satisfies(&cn, &s.solution));
        assert!(s.stats.component_splits >= 1, "root must split");
    }

    #[test]
    fn unsat_component_refutes_a_disconnected_instance() {
        // Component A = (x1∨x2) is trivially SAT; component B = all eight
        // 3-literal clauses over {x3,x4,x5} is UNSAT. Root GAC cannot see B's
        // contradiction (each clause alone prunes nothing); the component
        // search must refute B regardless of A's assignment.
        let cnf = "p cnf 5 9\n1 2 0\n\
            3 4 5 0\n3 4 -5 0\n3 -4 5 0\n3 -4 -5 0\n\
            -3 4 5 0\n-3 4 -5 0\n-3 -4 5 0\n-3 -4 -5 0\n";
        let (s, _cn) = solve_cnf(cnf);
        assert!(!s.found);
        assert!(s.stats.component_splits >= 1, "root must split");
    }

    #[test]
    fn solves_a_pure_2sat_by_branching() {
        // All binary: no special leaf — the occurrence selector picks a var,
        // the region machinery branches, propagation finishes. Completeness
        // must not depend on any residual-class shortcut.
        let (s, cn) = solve_cnf("p cnf 3 3\n1 2 0\n-2 3 0\n-1 3 0\n");
        assert!(s.found);
        assert!(satisfies(&cn, &s.solution));
        assert!(s.stats.branching_nodes >= 1);
    }
}
