//! Normalize any supported instance format into the canonical wcn-1 JSON
//! (see `boolean_inference::instance`) — the converter that lets the bank
//! store one format while generators keep emitting whatever is natural.
//!
//! Usage:  to_wcn <instance.{csp,cnf,json}> [out.json]
//! With no output path, writes the JSON to stdout.

use std::path::Path;
use std::process::ExitCode;

use boolean_inference::instance::load_instance;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: to_wcn <instance.{{csp,cnf,json}}> [out.json]");
        return ExitCode::from(2);
    }
    let inst = match load_instance(Path::new(&args[1])) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let json = inst.to_json();
    match args.get(2) {
        Some(out) => {
            if let Err(e) = std::fs::write(out, json + "\n") {
                eprintln!("error: cannot write {out}: {e}");
                return ExitCode::from(2);
            }
        }
        None => println!("{json}"),
    }
    ExitCode::SUCCESS
}
