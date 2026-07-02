use crate::adapter::BranchSolver;
use crate::dimacs::{network_from_dimacs, DimacsError};
use crate::measure::Measure;
use crate::problem::TnProblem;
use crate::selector::Selector;
use crate::solver::bbsat;
use optimal_branching_core::GreedyMerge;

/// SAT verdict plus, when satisfiable, a full assignment over the original
/// DIMACS variables. `assignment[i]` is the value of DIMACS variable `i+1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Solution {
    Sat(Vec<bool>),
    Unsat,
}

/// Solve a DIMACS CNF with the default strategy (mirrors `interface.jl`'s
/// `solve_sat_problem` default: `MostOccurrence(1,2)` + `NumUnfixedVars` +
/// `GreedyMerge`).
pub fn solve_dimacs(cnf: &str) -> Result<Solution, DimacsError> {
    let cn = network_from_dimacs(cnf)?;
    // `orig_to_new.len()` is the declared variable count (one slot per original var).
    let n_orig = cn.orig_to_new.len();
    let orig_to_new = cn.orig_to_new.clone();

    let mut problem = match TnProblem::from_network(cn) {
        Ok(p) => p,
        Err(_) => return Ok(Solution::Unsat), // root propagation found a contradiction
    };

    let result = bbsat(
        &mut problem,
        Selector::MostOccurrence {
            k: 1,
            max_tensors: 2,
        },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    if !result.found {
        return Ok(Solution::Unsat);
    }

    // Map the compressed internal assignment back onto original DIMACS vars.
    // A variable compressed out (appears in no clause) is free -> default false.
    let mut assignment = vec![false; n_orig];
    for (orig, slot) in orig_to_new.iter().enumerate() {
        if let Some(nid) = slot {
            assignment[orig] = result.solution[*nid]
                .value()
                .expect("a solved var is fixed");
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
