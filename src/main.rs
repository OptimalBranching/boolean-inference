use std::io::Read;
use std::process::ExitCode;

use boolean_inference::api::{solve_circuit_sat, Solution};

/// CircuitSAT solver CLI. Reads structure-preserving CircuitSAT JSON from the
/// file given as the first argument, or from stdin when no file is given.
fn main() -> ExitCode {
    let input = match std::env::args().nth(1) {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read {path}: {e}");
                return ExitCode::from(2);
            }
        },
        None => {
            let mut s = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut s) {
                eprintln!("error: cannot read stdin: {e}");
                return ExitCode::from(2);
            }
            s
        }
    };

    match solve_circuit_sat(&input) {
        Err(e) => {
            eprintln!("error: invalid CircuitSAT: {e}");
            ExitCode::from(2)
        }
        Ok(Solution::Unsat) => {
            println!("s UNSATISFIABLE");
            ExitCode::from(20)
        }
        Ok(Solution::Sat(assignment)) => {
            println!("s SATISFIABLE");
            let values = assignment
                .into_iter()
                .map(|(name, value)| format!("{name}={}", u8::from(value)))
                .collect::<Vec<_>>()
                .join(" ");
            println!("v {values}");
            ExitCode::from(10)
        }
    }
}
