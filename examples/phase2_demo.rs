// Phase 2 demonstration: concrete SAT instances in, deduced assignments out.
// Run with:  cargo run --example phase2_demo
//
// Everything printed here is computed live by the propagation engine — nothing
// is hand-written. The point is to *see* the correctness on small instances.

use boolean_inference::dimacs::network_from_dimacs;
use boolean_inference::domain::DomainMask;
use boolean_inference::network::ConstraintNetwork;
use boolean_inference::problem::{has_contradiction, SolverBuffer, TnProblem};
use boolean_inference::propagate::probe_assignment;

/// Render a domain vector as `x1=1  x2=0  x3=?` (? = still free, ∅ = contradiction).
fn fmt(doms: &[DomainMask]) -> String {
    doms.iter()
        .enumerate()
        .map(|(i, d)| {
            let v = match d.value() {
                Some(true) => "1",
                Some(false) => "0",
                None if *d == DomainMask::NONE => "∅",
                None => "?",
            };
            format!("x{}={}", i + 1, v)
        })
        .collect::<Vec<_>>()
        .join("  ")
}

/// Show what initial propagation deduces from the clauses alone (no assumptions).
fn initial(title: &str, clauses_human: &str, cnf: &str) {
    println!("== {title} ==");
    println!("clauses : {clauses_human}");
    let cn = network_from_dimacs(cnf).expect("parse");
    match TnProblem::from_network(cn) {
        Ok(p) => println!(
            "deduced : {}   ({} of {} still free)",
            fmt(&p.doms),
            p.count_unfixed(),
            p.doms.len()
        ),
        Err(_) => println!("deduced : UNSAT — the clauses contradict each other outright"),
    }
    println!();
}

/// Probe both polarities of one variable from the post-propagation base, and
/// show the cascade (or contradiction) each assumption produces.
fn probe_both(title: &str, clauses_human: &str, cnf: &str, var_1based: usize) {
    println!("== {title} ==");
    println!("clauses : {clauses_human}");
    let cn = network_from_dimacs(cnf).expect("parse");
    let p = TnProblem::from_network(cn).expect("root should be SAT");
    println!("base    : {}", fmt(&p.doms));
    let vid = var_1based - 1;
    for (label, bit) in [("=1", 1u64), ("=0", 0u64)] {
        let mut buf = SolverBuffer::new(&p.static_cn);
        let res = probe_assignment(&p.static_cn, &mut buf, &p.doms, &[vid], 1, bit);
        if has_contradiction(res) {
            println!("assume x{var_1based}{label} : CONTRADICTION — this branch is UNSAT");
        } else {
            println!("assume x{var_1based}{label} : {}", fmt(res));
        }
    }
    println!();
}

fn main() {
    println!("\nboolean-inference · Phase 2 live demo\n");

    // 1. A unit clause kicks off a chain: x1 -> x2 -> x3, all forced to 1.
    initial(
        "Unit-propagation chain",
        "(x1) AND (¬x1 ∨ x2) AND (¬x2 ∨ x3)",
        "p cnf 3 3\n1 0\n-1 2 0\n-2 3 0\n",
    );

    // 2. Underconstrained: one wide clause forces nothing — engine must NOT over-deduce.
    initial(
        "Underconstrained (nothing is forced)",
        "(x1 ∨ x2 ∨ x3)",
        "p cnf 3 1\n1 2 3 0\n",
    );

    // 3. Direct contradiction at the root.
    initial(
        "Outright contradiction",
        "(x1) AND (¬x1)",
        "p cnf 1 2\n1 0\n-1 0\n",
    );

    // 4. Two clauses encode x1 = x2; one more makes it a forced value once probed.
    initial(
        "Equivalence x1 = x2, plus unit (x1)",
        "(¬x1 ∨ x2) AND (x1 ∨ ¬x2) AND (x1)",
        "p cnf 2 3\n-1 2 0\n1 -2 0\n1 0\n",
    );

    // 5. XOR: nothing forced at root, but each assumption forces the other variable.
    probe_both(
        "XOR  x1 ⊕ x2  (what-if both ways)",
        "(x1 ∨ x2) AND (¬x1 ∨ ¬x2)",
        "p cnf 2 2\n1 2 0\n-1 -2 0\n",
        1,
    );

    // 6. Failed literal: assuming x1=1 cascades into a contradiction, so x1 must be 0.
    probe_both(
        "Failed literal  (x1=1 is impossible)",
        "(¬x1 ∨ x2) AND (¬x1 ∨ x3) AND (¬x2 ∨ ¬x3)",
        "p cnf 3 3\n-1 2 0\n-1 3 0\n-2 -3 0\n",
        1,
    );
}
