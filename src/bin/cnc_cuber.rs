//! Generate Cube-and-Conquer assumptions with the current Rust region cuber.
//!
//! The primary stopping rule is the classical online Cube-and-Conquer
//! difficulty cutoff (`--cc-threshold`). A march-compatible remaining-variable
//! cutoff (`-n`) is retained for controlled ablations.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use boolean_inference::adapter::BranchSolver;
use boolean_inference::circuit::network_from_circuit_sat;
use boolean_inference::cube::{
    generate_cubes_with_cutoff, generate_cubes_with_cutoff_trace, CubeCutoff, CubeNodeKind,
    CubeNodeTrace, CubeRefutationReason,
};
use boolean_inference::dimacs::network_from_dimacs;
use boolean_inference::measure::Measure;
use boolean_inference::network::ConstraintNetwork;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use optimal_branching_core::GreedyMerge;

const USAGE: &str = "usage: cnc_cuber <instance.(json|cnf)> (-n <remaining-vars> | --cc-threshold <difficulty>) -o <cubes.icnf|-> \
     [--selector <region|structure-blind>] [--max-rows <rows>] [--trace <nodes.jsonl>]";

#[derive(Clone, Copy, Debug)]
enum SelectorKind {
    Region,
    StructureBlind,
}

impl SelectorKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "region" => Ok(Self::Region),
            "structure-blind" => Ok(Self::StructureBlind),
            _ => Err(format!(
                "invalid --selector value: {value}; expected region or structure-blind"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Region => "region",
            Self::StructureBlind => "structure-blind",
        }
    }
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    cutoff: CubeCutoff,
    selector: SelectorKind,
    max_rows: usize,
    trace: Option<PathBuf>,
}

enum Command {
    Help,
    Run(Args),
}

fn take_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_args() -> Result<Command, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut input = None;
    let mut output = None;
    let mut cutoff_vars = None;
    let mut cc_threshold = None;
    let mut max_rows = 512usize;
    let mut selector = SelectorKind::Region;
    let mut trace = None;
    let mut i = 0usize;

    while i < raw.len() {
        match raw[i].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-n" => {
                let value = take_value(&raw, &mut i, "-n")?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid -n value: {value}"))?;
                cutoff_vars = Some(
                    NonZeroUsize::new(parsed)
                        .ok_or_else(|| "-n must be greater than zero".to_string())?,
                );
            }
            "--cc-threshold" => {
                let value = take_value(&raw, &mut i, "--cc-threshold")?;
                cc_threshold = Some(
                    value
                        .parse::<u128>()
                        .map_err(|_| format!("invalid --cc-threshold value: {value}"))?,
                );
            }
            "-o" => output = Some(take_value(&raw, &mut i, "-o")?),
            "--trace" => trace = Some(take_value(&raw, &mut i, "--trace")?),
            "--selector" => {
                selector = SelectorKind::parse(&take_value(&raw, &mut i, "--selector")?)?;
            }
            "--max-rows" => {
                let value = take_value(&raw, &mut i, "--max-rows")?;
                max_rows = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --max-rows value: {value}"))?;
                if max_rows == 0 {
                    return Err("--max-rows must be greater than zero".into());
                }
            }
            option if option.starts_with('-') => {
                return Err(format!("unknown option: {option}"));
            }
            value if input.is_none() => input = Some(value.to_owned()),
            value => return Err(format!("unexpected positional argument: {value}")),
        }
        i += 1;
    }

    let cutoff = match (cutoff_vars, cc_threshold) {
        (Some(n), None) => CubeCutoff::RemainingVars(n),
        (None, Some(threshold)) => CubeCutoff::CcDifficulty(threshold),
        (None, None) => return Err("missing -n or --cc-threshold cutoff".to_string()),
        (Some(_), Some(_)) => {
            return Err("-n and --cc-threshold are mutually exclusive".to_string())
        }
    };
    Ok(Command::Run(Args {
        input: PathBuf::from(input.ok_or_else(|| "missing input instance".to_string())?),
        output: PathBuf::from(output.ok_or_else(|| "missing -o output".to_string())?),
        cutoff,
        selector,
        max_rows,
        trace: trace.map(PathBuf::from),
    }))
}

fn load_network(path: &Path) -> Result<ConstraintNetwork, String> {
    let display = path.display();
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {display}: {e}"))?;
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("json") => network_from_circuit_sat(&text)
            .map(|problem| problem.network)
            .map_err(|e| format!("parse CircuitSAT {display}: {e}")),
        Some("cnf") => {
            network_from_dimacs(&text).map_err(|e| format!("parse DIMACS {display}: {e}"))
        }
        _ => Err(format!("unsupported input extension: {display}")),
    }
}

fn output_writer(path: &Path) -> Result<Box<dyn Write>, String> {
    if path == Path::new("-") {
        Ok(Box::new(BufWriter::new(io::stdout())))
    } else {
        File::create(path)
            .map(|file| Box::new(BufWriter::new(file)) as Box<dyn Write>)
            .map_err(|e| format!("create {}: {e}", path.display()))
    }
}

fn node_kind(kind: CubeNodeKind) -> &'static str {
    match kind {
        CubeNodeKind::Branch => "branch",
        CubeNodeKind::Cutoff => "cutoff",
        CubeNodeKind::Refuted => "refuted",
        CubeNodeKind::Sat => "sat",
    }
}

fn refutation_reason(reason: CubeRefutationReason) -> &'static str {
    match reason {
        CubeRefutationReason::RootPropagation => "root-propagation-contradiction",
        CubeRefutationReason::SelectorNoFeasibleConfig => "selector-no-feasible-config",
        CubeRefutationReason::BranchPropagation => "branch-propagation-contradiction",
    }
}

fn write_trace_node(
    writer: &mut dyn Write,
    node: CubeNodeTrace,
    new_to_orig: &[usize],
) -> Result<(), String> {
    let literals: Vec<i64> = node
        .decisions
        .iter()
        .map(|&(compressed, value)| {
            let variable = (new_to_orig[compressed] + 1) as i64;
            if value {
                variable
            } else {
                -variable
            }
        })
        .collect();
    let variables: Vec<usize> = node
        .variables
        .iter()
        .map(|&compressed| new_to_orig[compressed] + 1)
        .collect();
    let clauses: Vec<_> = node
        .clauses
        .iter()
        .map(|clause| serde_json::json!({"mask": clause.mask, "value": clause.value}))
        .collect();
    let record = serde_json::json!({
        "schema_version": 1,
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "child_index": node.child_index,
        "depth": node.depth,
        "kind": node_kind(node.kind),
        "refutation_reason": node.refutation_reason.map(refutation_reason),
        "literals": literals,
        "sigma_dec": node.sigma_dec,
        "sigma_all": node.sigma_all,
        "freevars": node.freevars,
        "rule_variables": variables,
        "rule_clauses": clauses,
    });
    serde_json::to_writer(&mut *writer, &record)
        .map_err(|error| format!("serialize trace: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("write trace: {error}"))
}

fn run(args: Args) -> Result<(), String> {
    if args.trace.as_deref() == Some(args.output.as_path()) {
        return Err("--trace must differ from the cube output path".into());
    }
    let network = load_network(&args.input)?;
    let nvars = network.n_vars;

    // Cubes store compressed variable ids. Convert them back to the original
    // DIMACS/CircuitSAT numbering when writing assumptions.
    let mut new_to_orig = vec![usize::MAX; nvars];
    for (original, compressed) in network.orig_to_new.iter().enumerate() {
        if let Some(compressed) = compressed {
            new_to_orig[*compressed] = original;
        }
    }
    if new_to_orig.contains(&usize::MAX) {
        return Err("constraint network has an incomplete variable map".into());
    }

    let mut writer = output_writer(&args.output)?;
    let mut trace_writer = match &args.trace {
        Some(path) => {
            Some(BufWriter::new(File::create(path).map_err(|error| {
                format!("create {}: {error}", path.display())
            })?))
        }
        None => None,
    };
    let mut problem = match TnProblem::from_network(network) {
        Ok(problem) => problem,
        Err(_) => {
            if let Some(trace_writer) = trace_writer.as_mut() {
                write_trace_node(
                    trace_writer,
                    CubeNodeTrace {
                        node_id: 0,
                        parent_id: None,
                        child_index: None,
                        depth: 0,
                        kind: CubeNodeKind::Refuted,
                        refutation_reason: Some(CubeRefutationReason::RootPropagation),
                        decisions: Vec::new(),
                        sigma_dec: 0,
                        sigma_all: 0,
                        freevars: nvars,
                        variables: Vec::new(),
                        clauses: Vec::new(),
                    },
                    &new_to_orig,
                )?;
                trace_writer
                    .flush()
                    .map_err(|error| format!("flush trace: {error}"))?;
            }
            writer.flush().map_err(|e| format!("flush output: {e}"))?;
            eprintln!(
                "status=UNSAT_AT_ROOT cubes=0 refuted=1 sat_leaves=0 cutoff={:?}",
                args.cutoff
            );
            return Ok(());
        }
    };
    let root_unfixed = problem.count_unfixed();

    let mut emitted = 0usize;
    let mut min_remaining = usize::MAX;
    let mut max_remaining = 0usize;
    let selector = match args.selector {
        SelectorKind::Region => Selector::MostOccurrence {
            max_rows: args.max_rows,
        },
        SelectorKind::StructureBlind => Selector::BinaryOccurrence,
    };
    let solver = BranchSolver::Greedy(GreedyMerge);
    let mut emit = |cube: boolean_inference::cube::Cube| {
        if cube.refuted || cube.sat {
            return Ok(());
        }

        let remaining = nvars - cube.sigma_all;
        let stopped = match args.cutoff {
            CubeCutoff::RemainingVars(n) => remaining < n.get(),
            CubeCutoff::CcDifficulty(threshold) => {
                (cube.sigma_dec as u128).pow(2) * (cube.sigma_all as u128)
                    > threshold * (nvars as u128)
            }
        };
        if !stopped {
            return Err(format!(
                "internal cutoff error: emitted cube does not satisfy {:?}",
                args.cutoff
            ));
        }

        writer
            .write_all(b"a")
            .map_err(|e| format!("write output: {e}"))?;
        for &(compressed, value) in &cube.decisions {
            let literal = (new_to_orig[compressed] + 1) as i64;
            let literal = if value { literal } else { -literal };
            write!(writer, " {literal}").map_err(|e| format!("write output: {e}"))?;
        }
        writer
            .write_all(b" 0\n")
            .map_err(|e| format!("write output: {e}"))?;

        emitted += 1;
        min_remaining = min_remaining.min(remaining);
        max_remaining = max_remaining.max(remaining);
        Ok(())
    };
    let stats = match trace_writer.as_mut() {
        Some(trace_writer) => generate_cubes_with_cutoff_trace(
            &mut problem,
            selector,
            Measure::NumUnfixedVars,
            &solver,
            args.cutoff,
            &mut emit,
            |node| write_trace_node(trace_writer, node, &new_to_orig),
        ),
        None => generate_cubes_with_cutoff(
            &mut problem,
            selector,
            Measure::NumUnfixedVars,
            &solver,
            args.cutoff,
            &mut emit,
        ),
    }?;
    writer.flush().map_err(|e| format!("flush output: {e}"))?;
    if let Some(trace_writer) = trace_writer.as_mut() {
        trace_writer
            .flush()
            .map_err(|error| format!("flush trace: {error}"))?;
    }

    let remaining_range = if emitted == 0 {
        "none".to_string()
    } else {
        format!("{min_remaining}..={max_remaining}")
    };
    eprintln!(
        "status=OK cubes={} refuted={} sat_leaves={} visited={} cutoff={:?} \
         root_unfixed={} remaining_range={} selector={} max_rows={}",
        stats.cubes,
        stats.refuted,
        stats.sat_leaves,
        stats.visited,
        args.cutoff,
        root_unfixed,
        remaining_range,
        args.selector.label(),
        args.max_rows
    );
    if emitted != stats.cubes {
        return Err(format!(
            "internal accounting error: wrote {emitted} cubes, expected {}",
            stats.cubes
        ));
    }
    Ok(())
}

fn main() {
    match parse_args() {
        Ok(Command::Help) => println!("{USAGE}"),
        Ok(Command::Run(args)) => {
            if let Err(message) = run(args) {
                eprintln!("error: {message}");
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
        }
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}
