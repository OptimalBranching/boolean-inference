//! Exact model counting / weighted counting CLI — the counting engine's
//! official entry point. Eats any supported instance format (dispatch on
//! extension; see `boolean_inference::instance`):
//!   .csp          native constraint tables (+ optional `w` weight lines)
//!   .cnf/.dimacs  DIMACS CNF (+ MCC 2021 `c p weight` lines)
//!   .json         canonical wcn-1, or problem-reductions CircuitSAT
//!
//! Pipeline: weighted bounded VE at width `budget`, then region-branching
//! search over the residual (`perconfig` = one branch per feasible config;
//! `blockmerge` = perfect-subcube partition). Any budget returns the SAME
//! count; the knob only moves time.
//!
//! Usage:  count <instance> [budget=16] [max_rows=128] [perconfig|blockmerge]

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use boolean_inference::instance::load_instance;
use boolean_inference::solver::CountBranch;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: count <instance.{{csp,cnf,json}}> [budget] [max_rows] [perconfig|blockmerge]"
        );
        return ExitCode::from(2);
    }
    let budget: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let max_rows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let branch = match args.get(4).map(String::as_str) {
        None | Some("perconfig") => CountBranch::PerConfig,
        Some("blockmerge") => CountBranch::BlockMerge,
        Some(other) => {
            eprintln!("unknown strategy {other:?} (use perconfig|blockmerge)");
            return ExitCode::from(2);
        }
    };

    let inst = match load_instance(Path::new(&args[1])) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let t0 = Instant::now();
    let (models, stats) = match inst.count(budget, max_rows, branch) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let dt = t0.elapsed();
    println!(
        "vars={} tensors={} models={models} budget={budget} branching_nodes={} visited={} compress={:.3} time={:.3}s",
        inst.n_vars,
        inst.tensors.len(),
        stats.branching_nodes,
        stats.total_visited_nodes,
        stats.compression_ratio(),
        dt.as_secs_f64()
    );
    ExitCode::SUCCESS
}
