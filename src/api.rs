use crate::adapter::BranchSolver;
use crate::circuit::{network_from_circuit_sat, CircuitError};
use crate::measure::Measure;
use crate::problem::TnProblem;
use crate::selector::Selector;
use crate::solver::bbsat;
use optimal_branching_core::GreedyMerge;

/// SAT verdict plus, when satisfiable, a named assignment over the CircuitSAT
/// variables. Assignments follow the input's `variables` order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Solution {
    Sat(Vec<(String, bool)>),
    Unsat,
}

/// Solve a structure-preserving CircuitSAT document with the default region
/// branching strategy.
pub fn solve_circuit_sat(json: &str) -> Result<Solution, CircuitError> {
    let cp = network_from_circuit_sat(json)?;
    let mut problem = match TnProblem::from_network(cp.network.clone()) {
        Ok(problem) => problem,
        Err(_) => return Ok(Solution::Unsat),
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

    let mut assignment: Vec<_> = cp
        .name_to_orig
        .iter()
        .map(|(name, &orig)| {
            let value = cp.network.orig_to_new[orig]
                .and_then(|cid| result.solution[cid].value())
                .unwrap_or(false);
            (orig, name.clone(), value)
        })
        .collect();
    assignment.sort_unstable_by_key(|(orig, _, _)| *orig);

    Ok(Solution::Sat(
        assignment
            .into_iter()
            .map(|(_, name, value)| (name, value))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sat_instance_returns_a_named_model() {
        let json = r#"{
            "variables": ["a", "b", "c"],
            "circuit": { "assignments": [
                { "outputs": ["c"], "expr": { "op": { "And": [
                    { "op": { "Var": "a" } },
                    { "op": { "Var": "b" } }
                ] } } }
            ] }
        }"#;
        match solve_circuit_sat(json).unwrap() {
            Solution::Sat(assignment) => {
                let value = |name| assignment.iter().find(|(n, _)| n == name).unwrap().1;
                assert_eq!(value("c"), value("a") && value("b"));
            }
            Solution::Unsat => panic!("expected SAT"),
        }
    }

    #[test]
    fn contradictory_gates_are_unsat() {
        let json = r#"{
            "variables": ["x"],
            "circuit": { "assignments": [
                { "outputs": ["x"], "expr": { "op": { "Const": true } } },
                { "outputs": ["x"], "expr": { "op": { "Const": false } } }
            ] }
        }"#;
        assert_eq!(solve_circuit_sat(json).unwrap(), Solution::Unsat);
    }

    #[test]
    fn unused_variables_default_false() {
        let json = r#"{
            "variables": ["x", "unused"],
            "circuit": { "assignments": [
                { "outputs": ["x"], "expr": { "op": { "Const": true } } }
            ] }
        }"#;
        assert_eq!(
            solve_circuit_sat(json).unwrap(),
            Solution::Sat(vec![("x".into(), true), ("unused".into(), false)])
        );
    }
}
