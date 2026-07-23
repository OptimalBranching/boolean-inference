//! Export a complete cube frontier or stream it into parallel Kissat workers.
//!
//! The primary stopping rule is the classical online Cube-and-Conquer
//! difficulty cutoff (`--cc-threshold`). A march-compatible remaining-variable
//! cutoff (`-n`) is retained for controlled ablations.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use boolean_inference::adapter::BranchSolver;
use boolean_inference::cdcl::CdclPropagator;
use boolean_inference::circuit::network_from_circuit_sat;
use boolean_inference::conquer::{ConquerResult, StreamingConquer};
use boolean_inference::csp::network_from_csp;
use boolean_inference::cube::{
    generate_cubes_configured, generate_cubes_configured_with_trace, CdclIntegrationMode,
    CncSatPolicy, CubeCdclOptions, CubeCutoff, CubeGenerationOptions, CubeNodeKind, CubeNodeTrace,
    CubeRefutationReason,
};
use boolean_inference::dimacs::network_from_dimacs;
use boolean_inference::measure::Measure;
use boolean_inference::network::ConstraintNetwork;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use boolean_inference::tail_greedy::TailGreedyMerge;
use optimal_branching_core::{GreedyMerge, NaiveBranch};

const USAGE: &str =
    "usage: cnc_cuber <instance.(json|cnf|csp)> (-n <remaining-vars> | --cc-threshold <difficulty>) \
     (-o <cubes.icnf|-> | --solve-cnf <base.cnf> --kissat <path> --workers <count>) \
     --branch-solver <greedy|tail-greedy|naive> \
     --measure <vars|tensors|hard-tensors> \
     [--propagation <ct|cdcl|hybrid>] [--propagate-cnf <base.cnf>] \
     [--selector <region|structure-blind>] \
     [--max-rows <rows>] [--trace <nodes.jsonl>] [--trace-replay]";

#[derive(Clone, Copy, Debug)]
enum SelectorKind {
    Region,
    StructureBlind,
}

#[derive(Clone, Copy, Debug)]
enum PropagationKind {
    Ct,
    Cdcl,
    Hybrid,
}

impl PropagationKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ct" => Ok(Self::Ct),
            "cdcl" => Ok(Self::Cdcl),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(format!(
                "invalid --propagation value: {value}; expected ct, cdcl, or hybrid"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Ct => "ct",
            Self::Cdcl => "cdcl",
            Self::Hybrid => "hybrid",
        }
    }
}

fn cdcl_mode_label(propagation: PropagationKind) -> &'static str {
    match propagation {
        PropagationKind::Ct => "off",
        PropagationKind::Cdcl | PropagationKind::Hybrid => "branch-learning",
    }
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

#[derive(Clone, Copy, Debug)]
enum BranchSolverKind {
    Greedy,
    TailGreedy,
    Naive,
}

impl BranchSolverKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "greedy" => Ok(Self::Greedy),
            "tail-greedy" => Ok(Self::TailGreedy),
            "naive" => Ok(Self::Naive),
            _ => Err(format!(
                "invalid --branch-solver value: {value}; expected greedy, tail-greedy, or naive"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::TailGreedy => "tail-greedy",
            Self::Naive => "naive",
        }
    }
}

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    solve_cnf: Option<PathBuf>,
    propagate_cnf: Option<PathBuf>,
    kissat: Option<PathBuf>,
    workers: Option<usize>,
    cutoff: CubeCutoff,
    selector: SelectorKind,
    branch_solver: BranchSolverKind,
    measure: Measure,
    propagation: PropagationKind,
    max_rows: usize,
    trace: Option<PathBuf>,
    trace_replay: bool,
}

enum Command {
    Help,
    Run(Box<Args>),
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
    let mut solve_cnf = None;
    let mut propagate_cnf = None;
    let mut kissat = None;
    let mut workers = None;
    let mut cutoff_vars = None;
    let mut cc_threshold = None;
    let mut max_rows = 512usize;
    let mut selector = SelectorKind::Region;
    let mut branch_solver = None;
    let mut measure = None;
    let mut propagation = PropagationKind::Ct;
    let mut trace = None;
    let mut trace_replay = false;
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
            "--solve-cnf" => solve_cnf = Some(take_value(&raw, &mut i, "--solve-cnf")?),
            "--propagate-cnf" => propagate_cnf = Some(take_value(&raw, &mut i, "--propagate-cnf")?),
            "--kissat" => kissat = Some(take_value(&raw, &mut i, "--kissat")?),
            "--workers" => {
                let value = take_value(&raw, &mut i, "--workers")?;
                let count = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --workers value: {value}"))?;
                if count == 0 {
                    return Err("--workers must be greater than zero".into());
                }
                workers = Some(count);
            }
            "--trace" => trace = Some(take_value(&raw, &mut i, "--trace")?),
            "--trace-replay" => trace_replay = true,
            "--selector" => {
                selector = SelectorKind::parse(&take_value(&raw, &mut i, "--selector")?)?;
            }
            "--branch-solver" => {
                branch_solver = Some(BranchSolverKind::parse(&take_value(
                    &raw,
                    &mut i,
                    "--branch-solver",
                )?)?);
            }
            "--measure" => {
                measure = Some(Measure::parse(&take_value(&raw, &mut i, "--measure")?)?);
            }
            "--propagation" => {
                propagation = PropagationKind::parse(&take_value(&raw, &mut i, "--propagation")?)?;
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
    match (&output, &solve_cnf, &kissat, workers) {
        (Some(_), None, None, None) => {}
        (None, Some(_), Some(_), Some(_)) => {}
        _ => return Err("select either -o, or all of --solve-cnf/--kissat/--workers".to_string()),
    }
    if trace_replay && trace.is_none() {
        return Err("--trace-replay requires --trace".to_string());
    }
    if trace_replay && matches!(selector, SelectorKind::StructureBlind) {
        return Err("--trace-replay requires --selector region".to_string());
    }
    let input = PathBuf::from(input.ok_or_else(|| "missing input instance".to_string())?);
    if matches!(propagation, PropagationKind::Cdcl | PropagationKind::Hybrid)
        && solve_cnf.is_none()
        && propagate_cnf.is_none()
        && input.extension().and_then(|extension| extension.to_str()) != Some("cnf")
    {
        return Err(format!(
            "--propagation {} requires --propagate-cnf, --solve-cnf, or a DIMACS input instance",
            propagation.label()
        ));
    }
    Ok(Command::Run(Box::new(Args {
        input,
        output: output.map(PathBuf::from),
        solve_cnf: solve_cnf.map(PathBuf::from),
        propagate_cnf: propagate_cnf.map(PathBuf::from),
        kissat: kissat.map(PathBuf::from),
        workers,
        cutoff,
        selector,
        branch_solver: branch_solver.ok_or_else(|| {
            "missing --branch-solver (experiments must select it explicitly)".to_string()
        })?,
        measure: measure.ok_or_else(|| {
            "missing --measure (experiments must select it explicitly)".to_string()
        })?,
        propagation,
        max_rows,
        trace: trace.map(PathBuf::from),
        trace_replay,
    })))
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
        Some("csp") => network_from_csp(&text)
            .map_err(|error| format!("parse extensional CSP {display}: {error}")),
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
        CubeRefutationReason::CdclPropagationConflict => "cdcl-propagation-conflict",
    }
}

#[allow(clippy::too_many_arguments)]
fn write_trace_node(
    writer: &mut dyn Write,
    node: CubeNodeTrace,
    new_to_orig: &[usize],
    selector: &str,
    branch_solver: &str,
    measure: &str,
    propagation: &str,
    cdcl_mode: &str,
    input_kind: &str,
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
    let optimized_clauses: Vec<_> = node
        .optimized_clauses
        .iter()
        .map(|clause| serde_json::json!({"mask": clause.mask, "value": clause.value}))
        .collect();
    let rule_diagnostics = node.rule_diagnostics.as_ref().map(|diagnostics| {
        let rule_semantics = if diagnostics.feasible_rows == 0 {
            "local-refutation"
        } else if diagnostics.closed {
            "closed-witness"
        } else {
            "cover"
        };
        let same_state_replay = diagnostics.same_state_replay.as_ref().map(|replay| {
            let evaluation = |value: &boolean_inference::table::RuleEvaluationDiagnostics| {
                serde_json::json!({
                    "branches": value.branches,
                    "decision_literals": value.decision_literals,
                    "branching_vector": value.branching_vector,
                    "gamma": value.gamma,
                    "solver_ns": value.solver_ns,
                })
            };
            serde_json::json!({
                "binary": evaluation(&replay.binary),
                "naive": evaluation(&replay.naive),
            })
        });
        serde_json::json!({
            "rule_semantics": rule_semantics,
            "focus_variable": new_to_orig[diagnostics.focus_var] + 1,
            "region_tensors": diagnostics.region_tensors,
            "region_variables": diagnostics.region_variables,
            "boundary_variables": diagnostics.boundary_variables,
            "joined_rows": diagnostics.joined_rows,
            "feasible_rows": diagnostics.feasible_rows,
            "branching_rows": diagnostics.branching_rows,
            "closed": diagnostics.closed,
            "branching_vector": diagnostics.branching_vector,
            "gamma": diagnostics.gamma,
            "cover_verified": diagnostics.cover_verified,
            "timing_ns": {
                "region_growth": diagnostics.region_growth_ns,
                "feasibility_probe": diagnostics.feasibility_probe_ns,
                "rule_solver": diagnostics.rule_solver_ns,
            },
            "same_state_replay": same_state_replay,
        })
    });
    let record = serde_json::json!({
        "search_semantics": "sat-decision",
        "selector": selector,
        "branch_solver": branch_solver,
        "measure": measure,
        "propagation": propagation,
        "cdcl_mode": cdcl_mode,
        "input_kind": input_kind,
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
        "rule_diagnostics": rule_diagnostics,
        "rule_variables": variables,
        "optimized_rule_clauses": optimized_clauses,
        "rule_clauses": clauses,
        "rule_partition_sources": node.partition_sources,
    });
    serde_json::to_writer(&mut *writer, &record)
        .map_err(|error| format!("serialize trace: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("write trace: {error}"))
}

fn run(args: Args) -> Result<i32, String> {
    if args
        .trace
        .as_ref()
        .is_some_and(|trace| args.output.as_ref() == Some(trace))
    {
        return Err("--trace must differ from the cube output path".into());
    }
    let input_kind = match args
        .input
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("json") => "circuit-sat",
        Some("cnf") => "dimacs",
        Some("csp") => "extensional-csp",
        _ => {
            return Err(format!(
                "unsupported input extension: {}",
                args.input.display()
            ))
        }
    };
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

    let mut writer: Box<dyn Write> = match &args.output {
        Some(output) => output_writer(output)?,
        None => Box::new(io::sink()),
    };
    let mut conquer = match (&args.solve_cnf, &args.kissat, args.workers) {
        (Some(cnf), Some(kissat), Some(workers)) => {
            Some(StreamingConquer::start(cnf, kissat, workers).map_err(|error| error.to_string())?)
        }
        _ => None,
    };
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
                        rule_diagnostics: None,
                        variables: Vec::new(),
                        optimized_clauses: Vec::new(),
                        clauses: Vec::new(),
                        partition_sources: Vec::new(),
                    },
                    &new_to_orig,
                    args.selector.label(),
                    args.branch_solver.label(),
                    args.measure.label(),
                    args.propagation.label(),
                    cdcl_mode_label(args.propagation),
                    input_kind,
                )?;
                trace_writer
                    .flush()
                    .map_err(|error| format!("flush trace: {error}"))?;
            }
            writer.flush().map_err(|e| format!("flush output: {e}"))?;
            eprintln!(
                "status=UNSAT_AT_ROOT cubes=0 refuted=1 sat_leaves=0 cutoff={:?} \
                 selector={} branch_solver={} measure={} propagation={} cdcl_mode={} \
                 max_rows={}",
                args.cutoff,
                args.selector.label(),
                args.branch_solver.label(),
                args.measure.label(),
                args.propagation.label(),
                cdcl_mode_label(args.propagation),
                args.max_rows
            );
            if let Some(conquer) = conquer.take() {
                let summary = conquer.finish(true).map_err(|error| error.to_string())?;
                debug_assert_eq!(summary.result, ConquerResult::Unsat);
                println!("s UNSATISFIABLE");
                return Ok(20);
            }
            return Ok(0);
        }
    };
    let root_unfixed = problem.count_unfixed();
    let cdcl = match args.propagation {
        PropagationKind::Ct => None,
        PropagationKind::Cdcl | PropagationKind::Hybrid => {
            let cnf = args
                .propagate_cnf
                .as_ref()
                .or(args.solve_cnf.as_ref())
                .unwrap_or(&args.input);
            Some(CdclPropagator::from_dimacs_path(cnf, new_to_orig.clone())?)
        }
    };
    let cdcl_integration = match args.propagation {
        PropagationKind::Hybrid => CdclIntegrationMode::HybridCtCandidates,
        PropagationKind::Ct | PropagationKind::Cdcl => CdclIntegrationMode::FullPropagation,
    };
    let sat_policy = if conquer.is_some() {
        CncSatPolicy::StopDecision
    } else {
        CncSatPolicy::CompleteFrontier
    };
    let termination = conquer.as_ref().map(StreamingConquer::termination_signal);

    let mut emitted = 0usize;
    let mut min_remaining = usize::MAX;
    let mut max_remaining = 0usize;
    let selector = match (args.selector, args.trace_replay) {
        (SelectorKind::Region, false) => Selector::MostOccurrence {
            max_rows: args.max_rows,
        },
        (SelectorKind::Region, true) => Selector::MostOccurrenceReplay {
            max_rows: args.max_rows,
        },
        (SelectorKind::StructureBlind, false) => Selector::BinaryOccurrence,
        (SelectorKind::StructureBlind, true) => unreachable!("validated by parse_args"),
    };
    let solver = match args.branch_solver {
        BranchSolverKind::Greedy => BranchSolver::Greedy(GreedyMerge),
        BranchSolverKind::TailGreedy => BranchSolver::TailGreedy(TailGreedyMerge),
        BranchSolverKind::Naive => BranchSolver::Naive(NaiveBranch),
    };
    let mut emit = |cube: boolean_inference::cube::Cube| {
        if cube.refuted {
            return Ok(());
        }

        let leaf_sat = cube.sat;
        if leaf_sat {
            if let Some(conquer) = conquer.as_ref() {
                conquer.mark_sat();
            }
            return Ok(());
        }

        let remaining = nvars - cube.sigma_all;
        if !args.cutoff.stops(cube.sigma_dec, cube.sigma_all, remaining) {
            return Err(format!(
                "internal cutoff error: emitted cube does not satisfy {:?}",
                args.cutoff
            ));
        }

        let mut literals = Vec::with_capacity(cube.decisions.len());
        for &(compressed, value) in &cube.decisions {
            let literal = (new_to_orig[compressed] + 1) as i64;
            let literal = if value { literal } else { -literal };
            literals.push(literal);
        }
        if let Some(conquer) = conquer.as_ref() {
            if !conquer
                .submit(literals)
                .map_err(|error| error.to_string())?
            {
                return Ok(());
            }
        } else {
            writer
                .write_all(b"a")
                .map_err(|e| format!("write output: {e}"))?;
            for literal in literals {
                write!(writer, " {literal}").map_err(|e| format!("write output: {e}"))?;
            }
            writer
                .write_all(b" 0\n")
                .map_err(|e| format!("write output: {e}"))?;
        }

        emitted += 1;
        min_remaining = min_remaining.min(remaining);
        max_remaining = max_remaining.max(remaining);
        Ok(())
    };
    let generation_options = CubeGenerationOptions {
        cutoff: args.cutoff,
        cdcl: cdcl.as_ref().map(|cdcl| CubeCdclOptions {
            propagator: cdcl.clone(),
            integration: cdcl_integration,
        }),
        sat_policy,
        termination,
    };
    let generated = match trace_writer.as_mut() {
        Some(trace_writer) => generate_cubes_configured_with_trace(
            &mut problem,
            selector,
            args.measure,
            &solver,
            generation_options,
            &mut emit,
            |node| {
                write_trace_node(
                    trace_writer,
                    node,
                    &new_to_orig,
                    args.selector.label(),
                    args.branch_solver.label(),
                    args.measure.label(),
                    args.propagation.label(),
                    cdcl_mode_label(args.propagation),
                    input_kind,
                )
            },
        ),
        None => generate_cubes_configured(
            &mut problem,
            selector,
            args.measure,
            &solver,
            generation_options,
            &mut emit,
        ),
    };
    let stopped_during_generation = generated.as_ref().is_ok_and(|stats| stats.stopped_early);
    let cdcl_stats = cdcl.as_ref().map(CdclPropagator::stats);
    let stopped_on_sat = stopped_during_generation;
    let stats = match generated {
        Ok(stats) if stats.stopped_early => None,
        Ok(stats) => Some(stats),
        Err(error) => {
            if let Some(conquer) = conquer.take() {
                let _ = conquer.finish(false);
            }
            return Err(error);
        }
    };
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
    if let Some(stats) = stats {
        eprintln!(
            "status=OK cubes={} refuted={} sat_leaves={} visited={} cutoff={:?} \
             root_unfixed={} remaining_range={} selector={} branch_solver={} measure={} \
             propagation={} cdcl_mode={} max_rows={}",
            stats.cubes,
            stats.refuted,
            stats.sat_leaves,
            stats.visited,
            args.cutoff,
            root_unfixed,
            remaining_range,
            args.selector.label(),
            args.branch_solver.label(),
            args.measure.label(),
            args.propagation.label(),
            cdcl_mode_label(args.propagation),
            args.max_rows
        );
        let expected = stats.cubes + stats.sat_leaves;
        if emitted != expected {
            return Err(format!(
                "internal accounting error: wrote {emitted} cubes, expected {}",
                expected
            ));
        }
    } else {
        eprintln!(
            "status=SAT_EARLY cubes_submitted={} cutoff={:?} selector={} branch_solver={} \
             measure={} propagation={} cdcl_mode={}",
            emitted,
            args.cutoff,
            args.selector.label(),
            args.branch_solver.label(),
            args.measure.label(),
            args.propagation.label(),
            cdcl_mode_label(args.propagation)
        );
    }
    if let Some(stats) = cdcl_stats {
        eprintln!(
            "cdcl propagation_calls={} propagation_conflicts={} assumption_literals={} \
             full_search_calls={} conflicts={} decisions={} propagations={} \
             learned_total={} redundant_current={}",
            stats.propagation_calls,
            stats.propagation_conflicts,
            stats.assumption_literals,
            stats.full_search_calls,
            stats.conflicts,
            stats.decisions,
            stats.propagations,
            stats.total_learned_clauses,
            stats.current_redundant_clauses
        );
    }
    if let Some(conquer) = conquer.take() {
        let summary = conquer
            .finish(!stopped_on_sat)
            .map_err(|error| error.to_string())?;
        eprintln!(
            "streaming submitted={} sat={} unsat={} errors={}",
            summary.submitted, summary.sat, summary.unsat, summary.errors
        );
        return match summary.result {
            ConquerResult::Sat => {
                if let Some(witness) = summary.witness {
                    print!("{witness}");
                } else {
                    println!("s SATISFIABLE");
                }
                Ok(10)
            }
            ConquerResult::Unsat => {
                println!("s UNSATISFIABLE");
                Ok(20)
            }
            ConquerResult::Incomplete => Err("streaming conquer was incomplete".into()),
        };
    }
    Ok(0)
}

fn main() {
    match parse_args() {
        Ok(Command::Help) => println!("{USAGE}"),
        Ok(Command::Run(args)) => match run(*args) {
            Ok(0) => {}
            Ok(code) => std::process::exit(code),
            Err(message) => {
                eprintln!("error: {message}");
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
        },
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}
