//! Search-tree pins for the shipped default config (MostOccurrence, VE budget
//! 10, max_rows = 128) across FIVE independent semiprime factoring instances —
//! multiple instances so a change that overfits one fixture cannot pass. The
//! baselines encode the fresh-region architecture (A13): regions grown at the
//! current doms per node, no region cache, budget-bounded scatter join.
//!
//! Re-pin deliberately (with a runscribe record) when the search strategy
//! changes; never to make a red test green.

use boolean_inference::adapter::BranchSolver;
use boolean_inference::canonicalize::bounded_ve_canonicalize;
use boolean_inference::circuit::{network_from_circuit_sat, CircuitProblem};
use boolean_inference::measure::Measure;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use boolean_inference::solver::bbsat;
use optimal_branching_core::GreedyMerge;

/// Solve a factoring CircuitSAT fixture with the default strategy and return
/// (branching_nodes, total_visited_nodes).
fn solve_fixture(json: &str, bits: usize) -> (u64, u64) {
    let cp = network_from_circuit_sat(json).expect("load");
    // Protect p1..p{bits}, q1..q{bits} across bounded-VE (budget_B = 10).
    let mut protected = Vec::new();
    for pfx in ["p", "q"] {
        for i in 1..=bits {
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
        Selector::MostOccurrence { max_rows: 128 },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    assert!(solve.found, "factoring instance must be SAT");
    (solve.stats.branching_nodes, solve.stats.total_visited_nodes)
}

/// Assert each `(fixture, bits, branching, visited)` pin still holds.
fn check_ladder(ladder: &[(&str, usize, u64, u64)]) {
    for &(json, bits, branch, visited) in ladder {
        let (b, v) = solve_fixture(json, bits);
        assert_eq!(
            (b, v),
            (branch, visited),
            "{bits}x{bits}: node counts must match the fresh-region baseline"
        );
    }
}

// Pins from runscribe C3 (`flmost-f*` runs, 2026-07-09): MostOccurrence region
// branching with the selection-independent failed-literal reduce (extracted from
// the retired DiffLookahead selector) applied at every node and the root,
// alongside GAC + domination. (fixture, factor bits, branching, visited).

/// Fast pins (12/16/18) — the always-on CI regression guard. Runs in seconds
/// under `--release`; keep the heavy 20/22 solves in the `#[ignore]`d test below.
#[test]
fn factoring_ladder_node_counts() {
    check_ladder(&[
        (
            include_str!("fixtures/factoring_12x12.circuitsat.json"),
            12,
            4,
            33,
        ),
        (
            include_str!("fixtures/factoring_16x16.circuitsat.json"),
            16,
            226,
            2493,
        ),
        (
            include_str!("fixtures/factoring_18x18.circuitsat.json"),
            18,
            914,
            9832,
        ),
    ]);
}

/// Heavy pins (20/22) — real multi-second solves, too slow for every CI run.
/// Run on demand: `cargo test --release -- --ignored`.
#[test]
#[ignore = "heavy 20x20/22x22 solves; run with `cargo test --release -- --ignored`"]
fn factoring_ladder_node_counts_large() {
    check_ladder(&[
        (
            include_str!("fixtures/factoring_20x20.circuitsat.json"),
            20,
            7382,
            39358,
        ),
        (
            include_str!("fixtures/factoring_22x22.circuitsat.json"),
            22,
            479,
            7035,
        ),
    ]);
}
