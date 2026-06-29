//! Solve a CircuitSAT instance — problem-reductions' `pred reduce --to
//! CircuitSAT` JSON — on the STRUCTURED constraint network (gates preserved, not
//! flattened to CNF). Prints the verdict, search stats, wall time, and, for a
//! factoring instance, the decoded factors.
//!
//! Usage:  solve_circuit <circuitsat.json> [factor_bits]
//! e.g.    cargo run --release --example solve_circuit -- fact.json 12

use std::time::Instant;

use boolean_inference::adapter::BranchSolver;
use boolean_inference::circuit::{network_from_circuit_sat, CircuitProblem};
use boolean_inference::domain::DomainMask;
use boolean_inference::measure::Measure;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use boolean_inference::solver::bbsat;
use optimal_branching_core::GreedyMerge;

fn decode(cp: &CircuitProblem, sol: &[DomainMask], prefix: &str, bits: usize) -> u64 {
    let mut v = 0u64;
    for i in 0..bits {
        if cp.wire_value(sol, &format!("{prefix}{}", i + 1)) == Some(true) {
            v |= 1 << i;
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: solve_circuit <circuitsat.json> [factor_bits]");
        std::process::exit(2);
    }
    let json = std::fs::read_to_string(&args[1]).expect("read JSON");
    let bits: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let cp = network_from_circuit_sat(&json).expect("load CircuitSAT");
    let n_vars = cp.network.vars.len();
    let n_tensors = cp.network.tensors.len();

    let t0 = Instant::now();
    let mut problem = match TnProblem::from_network(cp.network.clone()) {
        Ok(p) => p,
        Err(_) => {
            println!("vars={n_vars} tensors={n_tensors} UNSAT (root contradiction)");
            return;
        }
    };
    let solve = bbsat(
        &mut problem,
        Selector::MostOccurrence {
            k: 1,
            max_tensors: 2,
        },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    let dt = t0.elapsed();

    println!(
        "vars={n_vars} tensors={n_tensors} found={} branching_nodes={} visited={} time={:.3}s",
        solve.found, solve.stats.branching_nodes, solve.stats.total_visited_nodes, dt.as_secs_f64()
    );
    if solve.found && bits > 0 {
        let p = decode(&cp, &solve.solution, "p", bits);
        let q = decode(&cp, &solve.solution, "q", bits);
        println!("p={p} q={q} product={} (check {})", p * q, if p * q == 0 { "n/a" } else { "p*q" });
    }
}
