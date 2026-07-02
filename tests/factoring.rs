//! End-to-end: a real factoring instance, fed STRUCTURALLY (circuit gates, not
//! flattened CNF). The CircuitSAT fixture is produced by problem-reductions'
//! `pred create Factoring --target 15 --m 4 --n 4 | pred reduce - --to CircuitSAT`.
//! We load the gate-level network, solve it, decode the factor bits, and check
//! the product — proving the solver works on structured instances, which is
//! where the region-contraction method is supposed to have an edge over CNF.

use boolean_inference::adapter::BranchSolver;
use boolean_inference::circuit::{network_from_circuit_sat, CircuitProblem};
use boolean_inference::domain::DomainMask;
use boolean_inference::measure::Measure;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use boolean_inference::solver::bbsat;
use optimal_branching_core::GreedyMerge;

/// Decode a little-endian factor (wire `{prefix}1` = bit 0) from a solution.
fn decode(cp: &CircuitProblem, sol: &[DomainMask], prefix: &str, bits: usize) -> u64 {
    let mut v = 0u64;
    for i in 0..bits {
        let name = format!("{prefix}{}", i + 1);
        if cp.wire_value(sol, &name) == Some(true) {
            v |= 1 << i;
        }
    }
    v
}

#[test]
fn factoring_15_solves_and_decodes() {
    let json = include_str!("fixtures/factoring_15.circuitsat.json");
    let cp = network_from_circuit_sat(json).expect("load CircuitSAT");

    let mut problem = TnProblem::from_network(cp.network.clone()).expect("root SAT");
    let solve = bbsat(
        &mut problem,
        Selector::MostOccurrence {
            k: 1,
            max_tensors: 2,
        },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );

    assert!(solve.found, "N=15 is composite, must be SAT");
    let p = decode(&cp, &solve.solution, "p", 4);
    let q = decode(&cp, &solve.solution, "q", 4);
    assert_eq!(p * q, 15, "decoded factors {p} * {q} must equal 15");
    assert!(p >= 1 && q >= 1, "factors must be positive ({p}, {q})");
}

#[test]
fn factoring_15_solves_after_canonicalize() {
    use boolean_inference::canonicalize::bounded_ve_canonicalize;
    use boolean_inference::circuit::CircuitProblem;

    let json = include_str!("fixtures/factoring_15.circuitsat.json");
    let cp = network_from_circuit_sat(json).expect("load CircuitSAT");
    let raw_vars = cp.network.vars.len();

    // Protect the 4-bit factor wires p1..p4, q1..q4 (compressed ids).
    let mut protected = Vec::new();
    for prefix in ["p", "q"] {
        for i in 1..=4 {
            let name = format!("{prefix}{i}");
            if let Some(&orig) = cp.name_to_orig.get(&name) {
                if let Some(c) = cp.network.orig_to_new[orig] {
                    protected.push(c);
                }
            }
        }
    }
    assert!(!protected.is_empty(), "factor-bit wires must be present");

    let cn2 = bounded_ve_canonicalize(&cp.network, 10, &protected);
    assert!(
        cn2.vars.len() < raw_vars,
        "canonicalization must shrink the branch set"
    );

    let cp2 = CircuitProblem {
        network: cn2,
        name_to_orig: cp.name_to_orig,
    };
    let mut problem = TnProblem::from_network(cp2.network.clone()).expect("root SAT");
    let solve = bbsat(
        &mut problem,
        Selector::MostOccurrence {
            k: 1,
            max_tensors: 2,
        },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    assert!(solve.found, "canonicalized N=15 must be SAT");
    let p = decode(&cp2, &solve.solution, "p", 4);
    let q = decode(&cp2, &solve.solution, "q", 4);
    assert_eq!(
        p * q,
        15,
        "decoded factors {p} * {q} must equal 15 after VE"
    );
}
