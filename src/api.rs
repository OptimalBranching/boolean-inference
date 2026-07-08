use crate::adapter::BranchSolver;
use crate::canonicalize::bounded_ve_canonicalize;
use crate::dimacs::{network_from_dimacs, DimacsError};
use crate::measure::Measure;
use crate::problem::TnProblem;
use crate::selector::Selector;
use crate::solver::bbsat;
use optimal_branching_core::GreedyMerge;

/// Default width budget for the initialization VE pass on the DIMACS path.
const DEFAULT_VE_BUDGET: usize = 6;

/// SAT verdict plus, when satisfiable, a full assignment over the original
/// DIMACS variables. `assignment[i]` is the value of DIMACS variable `i+1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Solution {
    Sat(Vec<bool>),
    Unsat,
}

/// Solve a DIMACS CNF with the default strategy (mirrors `interface.jl`'s
/// `solve_sat_problem` default: `MostOccurrence` + `NumUnfixedVars` +
/// `GreedyMerge`), with the default initialization VE budget.
pub fn solve_dimacs(cnf: &str) -> Result<Solution, DimacsError> {
    solve_dimacs_with(cnf, DEFAULT_VE_BUDGET)
}

/// Solve a DIMACS CNF. Initialization: parse -> bounded-width VE (no protected
/// vars — eliminated vars are reconstructed after the solve) -> root GAC
/// propagation -> branch-and-reduce. The returned model covers ALL original
/// DIMACS variables.
pub fn solve_dimacs_with(cnf: &str, ve_budget: usize) -> Result<Solution, DimacsError> {
    let cn = network_from_dimacs(cnf)?;
    // `orig_to_new.len()` is the declared variable count (one slot per original var).
    let n_orig = cn.orig_to_new.len();
    let orig_to_cn = cn.orig_to_new.clone();

    let canon = match bounded_ve_canonicalize(&cn, ve_budget, &[]) {
        Some(c) => c,
        None => return Ok(Solution::Unsat), // VE hit an empty bucket join
    };

    // Solve the canonicalized network (skip the solver when VE consumed
    // everything), then lift the result back to a full model over input-cn ids —
    // survivor mapping + eliminated-var replay live in `canon.model`.
    let assignment_cn: Vec<Option<bool>> = if canon.cn.vars.is_empty() {
        canon.model.reconstruct(&[])
    } else {
        let mut problem = match TnProblem::from_network(canon.cn) {
            Ok(p) => p,
            Err(_) => return Ok(Solution::Unsat), // root propagation found a contradiction
        };
        let result = bbsat(
            &mut problem,
            Selector::MostOccurrence { max_rows: 512 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
        );
        if !result.found {
            return Ok(Solution::Unsat);
        }
        canon.model.reconstruct(&result.solution)
    };

    // Map input-cn ids back onto original DIMACS vars. A variable compressed out
    // pre-VE (appears in no clause) or left unconstrained is free -> default false.
    let mut assignment = vec![false; n_orig];
    for (orig, slot) in orig_to_cn.iter().enumerate() {
        if let Some(cnid) = slot {
            assignment[orig] = assignment_cn[*cnid].unwrap_or(false);
        }
    }
    Ok(Solution::Sat(assignment))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimacs::parse_dimacs;

    fn model_satisfies(assignment: &[bool], clauses: &[Vec<i64>]) -> bool {
        clauses.iter().all(|c| {
            c.iter().any(|&lit| {
                let v = lit.unsigned_abs() as usize - 1;
                if lit > 0 {
                    assignment[v]
                } else {
                    !assignment[v]
                }
            })
        })
    }

    #[test]
    fn sat_instance_returns_a_valid_model() {
        let cnf = "p cnf 3 3\n1 2 3 0\n-1 2 0\n-2 3 0\n";
        let (nvars, clauses) = parse_dimacs(cnf).unwrap();
        match solve_dimacs(cnf).unwrap() {
            Solution::Sat(a) => {
                assert_eq!(a.len(), nvars);
                assert!(model_satisfies(&a, &clauses));
            }
            Solution::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn unsat_instance_returns_unsat() {
        // All eight 3-literal clauses over {x1,x2,x3} -> UNSAT.
        let cnf = "p cnf 3 8\n\
            1 2 3 0\n1 2 -3 0\n1 -2 3 0\n1 -2 -3 0\n\
            -1 2 3 0\n-1 2 -3 0\n-1 -2 3 0\n-1 -2 -3 0\n";
        assert_eq!(solve_dimacs(cnf).unwrap(), Solution::Unsat);
    }

    #[test]
    fn root_contradiction_is_unsat() {
        // (x1) ∧ (¬x1): initial propagation contradicts before any search.
        assert_eq!(
            solve_dimacs("p cnf 1 2\n1 0\n-1 0\n").unwrap(),
            Solution::Unsat
        );
    }

    #[test]
    fn free_variable_defaults_false() {
        // var 2 appears in no clause -> compressed out -> free -> default false.
        // (x1) forces x1=true.
        let cnf = "p cnf 2 1\n1 0\n";
        match solve_dimacs(cnf).unwrap() {
            Solution::Sat(a) => {
                assert_eq!(a, vec![true, false]);
            }
            Solution::Unsat => panic!("expected SAT"),
        }
    }
}
