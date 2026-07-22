//! Audit every emitted cube under GAC in a native relation or DIMACS network.

use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use boolean_inference::circuit::network_from_circuit_sat;
use boolean_inference::csp::network_from_csp;
use boolean_inference::ct::{apply_masked_assignment, ct_propagate};
use boolean_inference::dimacs::network_from_dimacs;
use boolean_inference::domain::DomainMask;
use boolean_inference::network::ConstraintNetwork;
use boolean_inference::problem::TnProblem;
use boolean_inference::residual::diagnose;

struct Args {
    input: PathBuf,
    frontier: PathBuf,
    output: PathBuf,
    arm: String,
}

fn usage() -> &'static str {
    "usage: residual_audit <instance.(json|cnf|csp)> <frontier.icnf> -o <audit.jsonl> [--arm <name>]"
}

fn parse_args() -> Result<Args, String> {
    let values: Vec<String> = env::args().skip(1).collect();
    if values.len() < 4 {
        return Err(usage().to_string());
    }
    let mut positional = Vec::new();
    let mut output = None;
    let mut arm = "unknown".to_string();
    let mut index = 0usize;
    while index < values.len() {
        match values[index].as_str() {
            "-o" | "--output" => {
                index += 1;
                output = Some(
                    values
                        .get(index)
                        .ok_or_else(|| "missing output path".to_string())?
                        .into(),
                );
            }
            "--arm" => {
                index += 1;
                arm = values
                    .get(index)
                    .ok_or_else(|| "missing arm name".to_string())?
                    .clone();
            }
            option if option.starts_with('-') => {
                return Err(format!("unknown option: {option}\n{}", usage()));
            }
            value => positional.push(PathBuf::from(value)),
        }
        index += 1;
    }
    if positional.len() != 2 {
        return Err(usage().to_string());
    }
    Ok(Args {
        input: positional.remove(0),
        frontier: positional.remove(0),
        output: output.ok_or_else(|| "missing -o output path".to_string())?,
        arm,
    })
}

fn load_network(path: &Path) -> Result<(ConstraintNetwork, &'static str), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("json") => network_from_circuit_sat(&text)
            .map(|problem| (problem.network, "circuit-sat"))
            .map_err(|error| format!("parse CircuitSAT {}: {error}", path.display())),
        Some("cnf") => network_from_dimacs(&text)
            .map(|network| (network, "dimacs"))
            .map_err(|error| format!("parse DIMACS {}: {error}", path.display())),
        Some("csp") => network_from_csp(&text)
            .map(|network| (network, "extensional-csp"))
            .map_err(|error| format!("parse CSP {}: {error}", path.display())),
        _ => Err(format!("unsupported input extension: {}", path.display())),
    }
}

fn read_cubes(path: &Path) -> Result<Vec<Vec<i64>>, String> {
    let file = File::open(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut cubes = Vec::new();
    for (offset, line) in BufReader::new(file).lines().enumerate() {
        let line_number = offset + 1;
        let line = line.map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() || fields[0] == "c" {
            continue;
        }
        if fields[0] != "a" || fields.last() != Some(&"0") {
            return Err(format!(
                "{}:{line_number}: expected 'a <literals> 0'",
                path.display()
            ));
        }
        let literals = fields[1..fields.len() - 1]
            .iter()
            .map(|field| {
                field.parse::<i64>().map_err(|_| {
                    format!("{}:{line_number}: invalid literal {field}", path.display())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut variables = HashSet::with_capacity(literals.len());
        for &literal in &literals {
            if literal == 0 || !variables.insert(literal.unsigned_abs()) {
                return Err(format!(
                    "{}:{line_number}: zero or duplicate variable in cube",
                    path.display()
                ));
            }
        }
        cubes.push(literals);
    }
    if cubes.is_empty() {
        return Err(format!("{}: empty frontier", path.display()));
    }
    Ok(cubes)
}

fn fixed_assignment_signature(problem: &TnProblem) -> (usize, String, String) {
    let originals = problem.static_cn.orig_to_new.len();
    let words = originals.div_ceil(64);
    let mut fixed_mask = vec![0u64; words];
    let mut value_mask = vec![0u64; words];
    let mut fixed_original_variables = 0usize;
    for (original, mapped) in problem.static_cn.orig_to_new.iter().enumerate() {
        let Some(variable) = mapped else {
            continue;
        };
        let Some(value) = problem.doms[*variable].value() else {
            continue;
        };
        fixed_original_variables += 1;
        fixed_mask[original / 64] |= 1u64 << (original % 64);
        if value {
            value_mask[original / 64] |= 1u64 << (original % 64);
        }
    }
    let encode = |values: &[u64]| {
        values
            .iter()
            .map(|value| format!("{value:016x}"))
            .collect::<String>()
    };
    (
        fixed_original_variables,
        encode(&fixed_mask),
        encode(&value_mask),
    )
}

fn run(args: Args) -> Result<(), String> {
    let (network, input_kind) = load_network(&args.input)?;
    let original_variables = network.orig_to_new.len();
    let mut problem = TnProblem::from_network_gac(network)
        .map_err(|error| format!("root GAC failed: {error}"))?;
    let root_fixed_variables = problem
        .doms
        .iter()
        .filter(|domain| domain.is_fixed())
        .count();
    let cubes = read_cubes(&args.frontier)?;
    let mut writer = BufWriter::new(
        File::create(&args.output)
            .map_err(|error| format!("create {}: {error}", args.output.display()))?,
    );

    for (cube_index, literals) in cubes.iter().enumerate() {
        problem.trail.open();
        let mark = problem.trail.mark();
        problem.buffer.reset_worklist();
        let mut explicit_new_variables = 0usize;
        for &literal in literals {
            let original = literal.unsigned_abs() as usize - 1;
            if original >= original_variables {
                return Err(format!(
                    "cube {cube_index}: literal {literal} is out of range"
                ));
            }
            let Some(variable) = problem.static_cn.orig_to_new[original] else {
                return Err(format!(
                    "cube {cube_index}: literal {literal} names a compressed-out variable"
                ));
            };
            let desired = if literal > 0 {
                DomainMask::D1
            } else {
                DomainMask::D0
            };
            match problem.doms[variable] {
                current if current == desired => {}
                DomainMask::BOTH => {
                    explicit_new_variables += 1;
                    apply_masked_assignment(
                        &problem.static_cn,
                        &mut problem.doms,
                        &mut problem.buffer,
                        &mut problem.trail,
                        &[variable],
                        1,
                        usize::from(literal > 0) as u64,
                    );
                }
                current => {
                    return Err(format!(
                        "cube {cube_index}: literal {literal} contradicts root domain {current:?}"
                    ));
                }
            }
        }
        ct_propagate(
            &problem.static_cn,
            &mut problem.doms,
            &problem.masks,
            &mut problem.tables,
            &mut problem.buffer,
            &mut problem.trail,
        );
        let residual = diagnose(
            &problem.static_cn,
            &problem.doms,
            &problem.masks,
            &problem.tables,
        );
        let gac_additional_fixed_variables = residual
            .fixed_variables
            .saturating_sub(root_fixed_variables + explicit_new_variables);
        let (fixed_original_variables, fixed_mask_hex, fixed_value_hex) =
            fixed_assignment_signature(&problem);
        let record = serde_json::json!({
            "schema_version": 2,
            "propagation": "gac-only",
            "arm": args.arm,
            "input_kind": input_kind,
            "cube_index": cube_index,
            "literals": literals,
            "cube_literals": literals.len(),
            "original_variables": original_variables,
            "compressed_variables": problem.static_cn.n_vars,
            "root_fixed_variables": root_fixed_variables,
            "explicit_new_variables": explicit_new_variables,
            "gac_additional_fixed_variables": gac_additional_fixed_variables,
            "fixed_assignment_encoding": "u64-words-in-original-variable-order;bit-0-is-first-variable",
            "fixed_original_variables": fixed_original_variables,
            "fixed_mask_hex": fixed_mask_hex,
            "fixed_value_hex": fixed_value_hex,
            "residual": residual,
        });
        serde_json::to_writer(&mut writer, &record)
            .map_err(|error| format!("serialize audit: {error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("write audit: {error}"))?;
        problem
            .trail
            .restore_to(mark, &mut problem.doms, &mut problem.tables);
    }
    writer
        .flush()
        .map_err(|error| format!("flush {}: {error}", args.output.display()))?;
    eprintln!(
        "audited={} arm={} input_kind={} output={}",
        cubes.len(),
        args.arm,
        input_kind,
        args.output.display()
    );
    Ok(())
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boolean_inference::network::setup_problem;

    #[test]
    fn fixed_signature_preserves_original_variable_and_value() {
        let network = setup_problem(2, vec![vec![0]], vec![vec![false, true]]);
        let problem = TnProblem::from_network_gac(network).unwrap();
        let (fixed, mask, values) = fixed_assignment_signature(&problem);
        assert_eq!(fixed, 1);
        assert_eq!(mask, "0000000000000001");
        assert_eq!(values, "0000000000000001");
    }
}
