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
    let cn2 = bounded_ve_canonicalize(&cp.network, 10, &protected)
        .expect("factoring instance is SAT")
        .cn;
    let cp2 = CircuitProblem {
        network: cn2,
        name_to_orig: cp.name_to_orig,
    };
    let mut problem = TnProblem::from_network(cp2.network.clone()).expect("root SAT");
    let solve = bbsat(
        &mut problem,
        Selector::DiffLookahead {
            max_rows: 512,
            pool: 16,
        },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    assert!(solve.found);
    // Baseline re-pinned for the region redesign (boundary-grouped branching
    // tables + row-budgeted maximal root regions, max_rows=512); the previous
    // pins were 19761/45322 with k-hop regions (k=1, max_tensors=2) — the
    // redesign shrinks the search tree by ~28%.
    assert_eq!(
        solve.stats.branching_nodes, 14259,
        "branching nodes must match the region-redesign baseline"
    );
    assert_eq!(
        solve.stats.total_visited_nodes, 42273,
        "visited nodes must match the region-redesign baseline"
    );
}
