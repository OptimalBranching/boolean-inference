use boolean_inference::adapter::BranchSolver;
use boolean_inference::canonicalize::bounded_ve_canonicalize;
use boolean_inference::circuit::{network_from_circuit_sat, CircuitProblem};
use boolean_inference::measure::Measure;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use boolean_inference::solver::bbsat;
use optimal_branching_core::GreedyMerge;

#[test]
fn factoring_22x22_node_counts() {
    let json = include_str!("fixtures/factoring_22x22.circuitsat.json");
    let cp = network_from_circuit_sat(json).expect("load");
    // protect p1..p22, q1..q22 across bounded-VE (budget_B = 10)
    let mut protected = Vec::new();
    for pfx in ["p", "q"] {
        for i in 1..=22 {
            if let Some(&orig) = cp.name_to_orig.get(&format!("{pfx}{i}")) {
                if let Some(c) = cp.network.orig_to_new[orig] {
                    protected.push(c);
                }
            }
        }
    }
    let cn2 = bounded_ve_canonicalize(&cp.network, 10, &protected);
    let cp2 = CircuitProblem {
        network: cn2,
        name_to_orig: cp.name_to_orig,
    };
    let mut problem = TnProblem::from_network(cp2.network.clone()).expect("root SAT");
    let solve = bbsat(
        &mut problem,
        Selector::DiffLookahead {
            k: 1,
            max_tensors: 2,
            pool: 16,
        },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    assert!(solve.found);
    assert_eq!(
        solve.stats.branching_nodes, 19761,
        "branching nodes must match pre-CT baseline"
    );
    assert_eq!(
        solve.stats.total_visited_nodes, 45322,
        "visited nodes must match pre-CT baseline"
    );
}
