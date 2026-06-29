use std::sync::Arc;

use crate::adapter::BranchSolver;
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{SolverBuffer, Stats, TnProblem};
use crate::propagate::probe_assignment;
use crate::region::RegionCache;
use crate::selector::Selector;
use crate::twosat::solve_2sat;
use crate::util::{count_unfixed, is_two_sat};

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
    let (k, max_tensors) = selector.k_max();
    let mut cache = RegionCache::new(&problem.static_cn, &problem.doms, k, max_tensors);
    let doms0 = problem.doms.clone();

    // Split disjoint field borrows for the recursion.
    let ctx = SearchCtx {
        cn: &problem.static_cn,
        selector,
        measure,
        solver,
    };
    let stats = &mut problem.stats;
    let buffer = &mut problem.buffer;
    bbsat_rec(&ctx, &mut cache, stats, buffer, doms0)
}

fn bbsat_rec(
    ctx: &SearchCtx,
    cache: &mut RegionCache,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: Vec<DomainMask>,
) -> Solve {
    if count_unfixed(&doms) == 0 {
        return Solve {
            found: true,
            solution: doms,
            stats: stats.clone(),
        };
    }

    if is_two_sat(ctx.cn, &doms) {
        return match solve_2sat(ctx.cn, &doms) {
            Some(sol) => Solve {
                found: true,
                solution: sol,
                stats: stats.clone(),
            },
            None => Solve {
                found: false,
                solution: Vec::new(),
                stats: stats.clone(),
            },
        };
    }

    let (clauses, variables) =
        ctx.selector
            .findbest(cache, ctx.cn, &doms, buffer, ctx.measure, ctx.solver);
    let clauses = match clauses {
        Some(c) => c,
        None => {
            return Solve {
                found: false,
                solution: Vec::new(),
                stats: stats.clone(),
            }
        }
    };

    stats.record_branch(clauses.len() as u64);
    for cl in &clauses {
        stats.record_visit();
        let scratch = probe_assignment(ctx.cn, buffer, &doms, &variables, cl.mask, cl.val);
        let sub = scratch.to_vec(); // = Julia's copy(subproblem_doms); scratch is reused
        let result = bbsat_rec(ctx, cache, stats, buffer, sub);
        if result.found {
            return result;
        }
    }
    Solve {
        found: false,
        solution: Vec::new(),
        stats: stats.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimacs::network_from_dimacs;
    use crate::adapter::BranchSolver;
    use optimal_branching_core::IPSolver;

    fn satisfies(cn: &ConstraintNetwork, sol: &[DomainMask]) -> bool {
        cn.tensors.iter().all(|t| {
            let mut cfg = 0u32;
            for (i, &v) in t.var_axes.iter().enumerate() {
                if sol[v].value().expect("fully assigned") {
                    cfg |= 1 << i;
                }
            }
            cn.dense(t)[cfg as usize]
        })
    }

    fn solve_cnf(cnf: &str) -> (Solve, ConstraintNetwork) {
        let cn = network_from_dimacs(cnf).expect("parse");
        let cn_for_check = cn.clone();
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        let s = bbsat(
            &mut p,
            Selector::MostOccurrence {
                k: 1,
                max_tensors: 2,
            },
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
    fn solves_a_pure_2sat_via_the_leaf() {
        // All binary -> handled entirely by the 2-SAT leaf, no branching.
        let (s, cn) = solve_cnf("p cnf 3 3\n1 2 0\n-2 3 0\n-1 3 0\n");
        assert!(s.found);
        assert!(satisfies(&cn, &s.solution));
        assert_eq!(s.stats.branching_nodes, 0);
    }
}
