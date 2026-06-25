use std::io::Read;
use std::process::ExitCode;

use boolean_inference::api::{solve_dimacs, Solution};

/// Minimal DIMACS CNF solver CLI. Reads from the file given as the first
/// argument, or from stdin if none. SAT-Competition output + exit codes
/// (10 = SAT, 20 = UNSAT, 2 = error).
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

    match solve_dimacs(&input) {
        Err(e) => {
            eprintln!("error: invalid DIMACS: {e:?}");
            ExitCode::from(2)
        }
        Ok(Solution::Unsat) => {
            println!("s UNSATISFIABLE");
            ExitCode::from(20)
        }
        Ok(Solution::Sat(assignment)) => {
            println!("s SATISFIABLE");
            let mut line = String::from("v");
            for (i, &val) in assignment.iter().enumerate() {
                let lit = (i as i64 + 1) * if val { 1 } else { -1 };
                line.push_str(&format!(" {lit}"));
            }
            line.push_str(" 0");
            println!("{line}");
            ExitCode::from(10)
        }
    }
}
