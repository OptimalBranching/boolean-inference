//! Cube generation for Cube-and-Conquer: use the region-branching rule as a
//! CUBER. Descend the branch tree and evaluate an online stopping rule after
//! propagation and reductions at each node. The primary rule is the classical
//! Cube-and-Conquer difficulty `D^2(D+I)/N`; a remaining-variable cutoff is
//! retained as an ablation.
//!
//! This is deliberately NOT the full `bbsat` solver: no connected-component
//! decomposition (a cube is a partial assignment to the WHOLE formula, so the
//! cuber branches monolithically to keep one auditable semantic frontier), and every
//! branch is explored to a cube leaf rather than stopping at the first SAT. It
//! reuses the node primitives (`findbest`, propagation, the reductions) so the
//! cubes match what the solver would branch on; only the descent policy differs.
//!
//! Cubes are emitted as DECISION literals only (`sigma_dec`), matching
//! `march_cu`'s assumption-literal convention: the conquer solver re-derives
//! propagation itself. Each cube also carries `sigma_dec`/`sigma_all` so the
//! emitted frontier can be audited.

use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::adapter::BranchSolver;
use crate::ct::{apply_masked_assignment, ct_propagate, RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::Measure;
use crate::network::ConstraintNetwork;
use crate::problem::{SolverBuffer, Stats, TnProblem};
use crate::propagate::{dominate_fixpoint, failed_literal_fixpoint};
use crate::selector::{occurrence_pool, Selector, FAILED_LITERAL_POOL};
use crate::table::RegionRuleDiagnostics;
use crate::termination::TerminationSignal;
use crate::trail::Trail;
use crate::util::count_unfixed;

/// One generated cube: the decision literals as `(var, value)` pairs, plus the
/// two audit counts at the emitting leaf. `refuted` cubes were closed by
/// propagation before reaching the cutoff (locally UNSAT — no conquer needed).
#[derive(Clone, Debug)]
pub struct Cube {
    pub decisions: Vec<(usize, bool)>,
    pub sigma_dec: usize,
    pub sigma_all: usize,
    pub refuted: bool,
    pub sat: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CubeStats {
    pub cubes: usize,
    pub refuted: usize,
    pub sat_leaves: usize,
    /// A decision-mode component found SAT before the frontier was complete.
    pub stopped_early: bool,
    /// Nodes visited (branch decisions applied).
    pub visited: u64,
}

/// Classification of a node in an instrumented cubing tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeNodeKind {
    Branch,
    Cutoff,
    Refuted,
    Sat,
}

/// Why a traced leaf is known to be closed without a conquer cube.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeRefutationReason {
    RootPropagation,
    SelectorNoFeasibleConfig,
    BranchPropagation,
}

/// One branching clause in the bit encoding over `CubeNodeTrace::variables`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceClause {
    pub mask: u64,
    pub value: u64,
}

/// Optional audit record for every node visited while generating a frontier.
/// Variable ids are compressed network ids; frontends that expose original
/// DIMACS ids should map them at serialization time.
#[derive(Clone, Debug)]
pub struct CubeNodeTrace {
    pub node_id: u64,
    pub parent_id: Option<u64>,
    pub child_index: Option<usize>,
    pub depth: usize,
    pub kind: CubeNodeKind,
    pub refutation_reason: Option<CubeRefutationReason>,
    pub decisions: Vec<(usize, bool)>,
    pub sigma_dec: usize,
    pub sigma_all: usize,
    pub freevars: usize,
    /// Raw region-to-rule evidence. Absent at leaves and for the structure-blind
    /// control arm, which deliberately does not run the region machinery.
    pub rule_diagnostics: Option<RegionRuleDiagnostics>,
    pub variables: Vec<usize>,
    /// Clauses selected by the branching-rule optimizer. Diagnostics such as
    /// `branching_vector` and `gamma` describe this cover.
    pub optimized_clauses: Vec<TraceClause>,
    /// Pairwise-disjoint clauses actually traversed by the CnC search.
    pub clauses: Vec<TraceClause>,
    /// For each traversed clause, the optimizer clause from which it was split.
    pub partition_sources: Vec<usize>,
}

struct CubeCtx<'a> {
    cn: &'a Arc<ConstraintNetwork>,
    selector: Selector,
    measure: Measure,
    solver: &'a BranchSolver,
    cutoff: CubeCutoff,
    sat_policy: CncSatPolicy,
    termination: Option<TerminationSignal>,
}

impl CubeCtx<'_> {
    fn should_stop_for_sat(&self) -> bool {
        if self.sat_policy != CncSatPolicy::StopDecision {
            return false;
        }
        self.termination
            .as_ref()
            .is_some_and(TerminationSignal::is_requested)
    }
}

/// Online stopping rule evaluated at each post-reduction search node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeCutoff {
    /// march-compatible rule: stop when `freevars < n`.
    RemainingVars(NonZeroUsize),
    /// Classical CC difficulty: stop when `D^2 * (D + I) / N > threshold`.
    CcDifficulty(u128),
}

/// Whether cube generation is exhaustive or participates in first-answer CnC.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CncSatPolicy {
    /// Preserve a complete exported frontier even when a solver finds SAT.
    #[default]
    CompleteFrontier,
    /// Stop when the native cuber or any conquer worker proves SAT.
    StopDecision,
}

/// Orthogonal generation policies collected in one value.
#[derive(Clone)]
pub struct CubeGenerationOptions {
    pub cutoff: CubeCutoff,
    pub sat_policy: CncSatPolicy,
    pub termination: Option<TerminationSignal>,
}

impl CubeGenerationOptions {
    pub fn new(cutoff: CubeCutoff) -> Self {
        Self {
            cutoff,
            sat_policy: CncSatPolicy::CompleteFrontier,
            termination: None,
        }
    }
}

impl CubeCutoff {
    /// Return whether a post-reduction node satisfies this cutoff.
    pub fn stops(self, sigma_dec: usize, sigma_all: usize, freevars: usize) -> bool {
        match self {
            Self::RemainingVars(n) => freevars < n.get(),
            Self::CcDifficulty(threshold) => {
                let total = sigma_all + freevars;
                (sigma_dec as u128).pow(2) * (sigma_all as u128) > threshold * (total as u128)
            }
        }
    }
}

/// Streaming generation under either supported online stopping rule.
pub fn generate_cubes_with_cutoff<E, F>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    cutoff: CubeCutoff,
    emit: F,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
{
    generate_cubes_configured(
        problem,
        selector,
        measure,
        solver,
        CubeGenerationOptions::new(cutoff),
        emit,
    )
}

/// Primary streaming entry point. A conquer worker can stop native cubing
/// through the shared termination signal.
pub fn generate_cubes_configured<E, F>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    options: CubeGenerationOptions,
    emit: F,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
{
    generate_cubes_impl(
        problem,
        selector,
        measure,
        solver,
        options,
        emit,
        None::<&mut fn(CubeNodeTrace) -> Result<(), E>>,
    )
}

/// Traced generation under either supported online stopping rule.
pub fn generate_cubes_with_cutoff_trace<E, F, T>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    cutoff: CubeCutoff,
    emit: F,
    trace: T,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
    T: FnMut(CubeNodeTrace) -> Result<(), E>,
{
    generate_cubes_configured_with_trace(
        problem,
        selector,
        measure,
        solver,
        CubeGenerationOptions::new(cutoff),
        emit,
        trace,
    )
}

/// Traced form of [`generate_cubes_configured`].
pub fn generate_cubes_configured_with_trace<E, F, T>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    options: CubeGenerationOptions,
    emit: F,
    mut trace: T,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
    T: FnMut(CubeNodeTrace) -> Result<(), E>,
{
    generate_cubes_impl(
        problem,
        selector,
        measure,
        solver,
        options,
        emit,
        Some(&mut trace),
    )
}

fn generate_cubes_impl<E, F, T>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    options: CubeGenerationOptions,
    mut emit: F,
    mut trace: Option<&mut T>,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
    T: FnMut(CubeNodeTrace) -> Result<(), E>,
{
    problem.stats.reset();
    let termination = match (options.sat_policy, options.termination) {
        (CncSatPolicy::StopDecision, None) => Some(TerminationSignal::new()),
        (_, termination) => termination,
    };
    let ctx = CubeCtx {
        cn: &problem.static_cn,
        selector,
        measure,
        solver,
        cutoff: options.cutoff,
        sat_policy: options.sat_policy,
        termination,
    };
    let masks = &problem.masks;
    let stats = &mut problem.stats;
    let buffer = &mut problem.buffer;
    let doms = &mut problem.doms;
    let tables = &mut problem.tables;
    let trail = &mut problem.trail;

    let mut cube_stats = CubeStats::default();
    let mut decisions: Vec<(usize, bool)> = Vec::new();
    let mut next_node_id = 0u64;
    let mark = trail.mark();
    // Root already propagated; if it is already solved or refuted, that is a
    // single (degenerate) cube.
    let result = if ctx.should_stop_for_sat() {
        Ok(())
    } else if doms[0] == DomainMask::NONE {
        if let Some(trace) = trace.as_deref_mut() {
            trace(CubeNodeTrace {
                node_id: 0,
                parent_id: None,
                child_index: None,
                depth: 0,
                kind: CubeNodeKind::Refuted,
                refutation_reason: Some(CubeRefutationReason::RootPropagation),
                decisions: Vec::new(),
                sigma_dec: 0,
                sigma_all: 0,
                freevars: doms.len(),
                rule_diagnostics: None,
                variables: Vec::new(),
                optimized_clauses: Vec::new(),
                clauses: Vec::new(),
                partition_sources: Vec::new(),
            })?;
        }
        emit_cube(
            &mut cube_stats,
            &mut emit,
            Cube {
                decisions: Vec::new(),
                sigma_dec: 0,
                sigma_all: 0,
                refuted: true,
                sat: false,
            },
        )
    } else {
        cube_rec(
            &ctx,
            stats,
            &mut cube_stats,
            buffer,
            doms,
            masks,
            tables,
            trail,
            &mut decisions,
            &mut emit,
            &mut trace,
            &mut next_node_id,
            None,
            None,
            0,
        )
    };
    trail.restore_to(mark, doms, tables);
    result?;

    cube_stats.stopped_early = ctx.should_stop_for_sat();
    cube_stats.visited = stats.total_visited_nodes;
    Ok(cube_stats)
}

fn emit_cube<E, F>(stats: &mut CubeStats, emit: &mut F, cube: Cube) -> Result<(), E>
where
    F: FnMut(Cube) -> Result<(), E>,
{
    if cube.refuted {
        stats.refuted += 1;
    } else if cube.sat {
        stats.sat_leaves += 1;
    } else {
        stats.cubes += 1;
    }
    emit(cube)
}

#[allow(clippy::too_many_arguments)]
fn cube_rec<E, F, T>(
    ctx: &CubeCtx,
    stats: &mut Stats,
    cube_stats: &mut CubeStats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
    decisions: &mut Vec<(usize, bool)>,
    emit: &mut F,
    trace: &mut Option<&mut T>,
    next_node_id: &mut u64,
    parent_id: Option<u64>,
    child_index: Option<usize>,
    depth: usize,
) -> Result<(), E>
where
    F: FnMut(Cube) -> Result<(), E>,
    T: FnMut(CubeNodeTrace) -> Result<(), E>,
{
    if ctx.should_stop_for_sat() {
        return Ok(());
    }
    let node_id = *next_node_id;
    *next_node_id += 1;
    // march_cu -n cutoff on the current post-reduction node. The reductions run
    // before recursion enters this node, so `freevars` observes all implied
    // assignments as well as the explicit branch decisions.
    let sigma_dec = decisions.len();
    let freevars = count_unfixed(doms);
    let sigma_all = doms.len() - freevars;
    if ctx.cutoff.stops(sigma_dec, sigma_all, freevars) {
        if let Some(trace) = trace.as_deref_mut() {
            trace(CubeNodeTrace {
                node_id,
                parent_id,
                child_index,
                depth,
                kind: CubeNodeKind::Cutoff,
                refutation_reason: None,
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                freevars,
                rule_diagnostics: None,
                variables: Vec::new(),
                optimized_clauses: Vec::new(),
                clauses: Vec::new(),
                partition_sources: Vec::new(),
            })?;
        }
        return emit_cube(
            cube_stats,
            emit,
            Cube {
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                refuted: false,
                sat: false,
            },
        );
    }

    let scope: Vec<usize> = (0..doms.len()).filter(|&v| !doms[v].is_fixed()).collect();
    if scope.is_empty() {
        // Fully assigned without hitting the cutoff: a SAT leaf. This branch is
        // unreachable for nonzero static -n, but remains an explicit invariant.
        stats.record_visit();
        if let Some(trace) = trace.as_deref_mut() {
            trace(CubeNodeTrace {
                node_id,
                parent_id,
                child_index,
                depth,
                kind: CubeNodeKind::Sat,
                refutation_reason: None,
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                freevars,
                rule_diagnostics: None,
                variables: Vec::new(),
                optimized_clauses: Vec::new(),
                clauses: Vec::new(),
                partition_sources: Vec::new(),
            })?;
        }
        if ctx.sat_policy == CncSatPolicy::StopDecision {
            ctx.termination
                .as_ref()
                .expect("decision mode always has a termination signal")
                .request();
        }
        return emit_cube(
            cube_stats,
            emit,
            Cube {
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                refuted: false,
                sat: true,
            },
        );
    }

    let selection = ctx.selector.findbest(
        ctx.cn,
        doms,
        buffer,
        ctx.measure,
        ctx.solver,
        masks,
        tables,
        trail,
        &scope,
        trace.is_some(),
    );
    if ctx.should_stop_for_sat() {
        return Ok(());
    }
    let clauses = match selection.clauses {
        // No rule (region proved locally UNSAT): refuted cube.
        None => {
            if let Some(trace) = trace.as_deref_mut() {
                trace(CubeNodeTrace {
                    node_id,
                    parent_id,
                    child_index,
                    depth,
                    kind: CubeNodeKind::Refuted,
                    refutation_reason: Some(CubeRefutationReason::SelectorNoFeasibleConfig),
                    decisions: decisions.clone(),
                    sigma_dec,
                    sigma_all,
                    freevars,
                    rule_diagnostics: selection.diagnostics,
                    variables: selection.variables,
                    optimized_clauses: Vec::new(),
                    clauses: Vec::new(),
                    partition_sources: Vec::new(),
                })?;
            }
            return emit_cube(
                cube_stats,
                emit,
                Cube {
                    decisions: decisions.clone(),
                    sigma_dec,
                    sigma_all,
                    refuted: true,
                    sat: false,
                },
            );
        }
        Some(clauses) => clauses,
    };
    // Optimal-branching rules are set covers: their conjunctions may overlap.
    // That is acceptable for branch-and-reduce, but a CnC frontier must be a
    // partition or the same residual search space can be submitted repeatedly.
    // Subtract earlier cubes from each later cube to obtain an equivalent
    // disjoint DNF before descending.
    let optimized_clauses = clauses;
    let partition = disjointize_clauses_with_sources(&optimized_clauses);
    let clauses = partition
        .iter()
        .map(|(clause, _)| *clause)
        .collect::<Vec<_>>();
    let partition_sources = partition
        .iter()
        .map(|(_, source)| *source)
        .collect::<Vec<_>>();
    let variables = selection.variables;
    let rule_diagnostics = selection.diagnostics;

    if let Some(trace) = trace.as_deref_mut() {
        let trace_optimized_clauses = optimized_clauses
            .iter()
            .map(|clause| TraceClause {
                mask: clause.mask,
                value: clause.val,
            })
            .collect();
        let trace_clauses = clauses
            .iter()
            .map(|clause| TraceClause {
                mask: clause.mask,
                value: clause.val,
            })
            .collect();
        trace(CubeNodeTrace {
            node_id,
            parent_id,
            child_index,
            depth,
            kind: CubeNodeKind::Branch,
            refutation_reason: None,
            decisions: decisions.clone(),
            sigma_dec,
            sigma_all,
            freevars,
            rule_diagnostics,
            variables: variables.clone(),
            optimized_clauses: trace_optimized_clauses,
            clauses: trace_clauses,
            partition_sources: partition_sources.clone(),
        })?;
    }

    for (branch_index, cl) in clauses.iter().enumerate() {
        if ctx.should_stop_for_sat() {
            break;
        }
        stats.record_visit();
        trail.open();
        let mark = trail.mark();
        buffer.reset_worklist();
        // Record this branch's decision literals before applying.
        let decision_base = decisions.len();
        for (i, &var) in variables.iter().enumerate() {
            if (cl.mask >> i) & 1 == 1 {
                decisions.push((var, (cl.val >> i) & 1 == 1));
            }
        }
        apply_masked_assignment(ctx.cn, doms, buffer, trail, &variables, cl.mask, cl.val);
        ct_propagate(ctx.cn, doms, masks, tables, buffer, trail);
        let mut stop_for_sat = ctx.should_stop_for_sat();
        if !stop_for_sat && doms[0] != DomainMask::NONE {
            dominate_fixpoint(ctx.cn, doms, masks, tables, buffer, trail);
        }
        stop_for_sat |= ctx.should_stop_for_sat();
        if !stop_for_sat && doms[0] != DomainMask::NONE {
            let pool = occurrence_pool(ctx.cn, doms, buffer, masks, FAILED_LITERAL_POOL);
            failed_literal_fixpoint(ctx.cn, doms, masks, tables, buffer, trail, &pool);
        }
        stop_for_sat |= ctx.should_stop_for_sat();
        let branch_result = if stop_for_sat {
            Ok(())
        } else if doms[0] == DomainMask::NONE {
            // Branch closed by propagation: refuted cube (no conquer needed).
            let branch_freevars = count_unfixed(doms);
            let child_node_id = *next_node_id;
            *next_node_id += 1;
            if let Some(trace) = trace.as_deref_mut() {
                trace(CubeNodeTrace {
                    node_id: child_node_id,
                    parent_id: Some(node_id),
                    child_index: Some(branch_index),
                    depth: depth + 1,
                    kind: CubeNodeKind::Refuted,
                    refutation_reason: Some(CubeRefutationReason::BranchPropagation),
                    decisions: decisions.clone(),
                    sigma_dec: decisions.len(),
                    sigma_all: doms.len() - branch_freevars,
                    freevars: branch_freevars,
                    rule_diagnostics: None,
                    variables: Vec::new(),
                    optimized_clauses: Vec::new(),
                    clauses: Vec::new(),
                    partition_sources: Vec::new(),
                })?;
            }
            emit_cube(
                cube_stats,
                emit,
                Cube {
                    decisions: decisions.clone(),
                    sigma_dec: decisions.len(),
                    sigma_all: doms.len() - branch_freevars,
                    refuted: true,
                    sat: false,
                },
            )
        } else {
            cube_rec(
                ctx,
                stats,
                cube_stats,
                buffer,
                doms,
                masks,
                tables,
                trail,
                decisions,
                emit,
                trace,
                next_node_id,
                Some(node_id),
                Some(branch_index),
                depth + 1,
            )
        };
        decisions.truncate(decision_base);
        trail.restore_to(mark, doms, tables);
        branch_result?;
        if stop_for_sat || ctx.should_stop_for_sat() {
            break;
        }
    }
    Ok(())
}

/// Convert a DNF cube cover into an equivalent pairwise-disjoint cube cover.
///
/// Clauses are processed in order. Each new cube has the union of all earlier
/// output cubes subtracted from it; subtraction of one conjunction from another
/// uses the standard prefix split of `A ∧ ¬B`.
#[cfg(test)]
fn disjointize_clauses(
    clauses: &[optimal_branching_core::Clause],
) -> Vec<optimal_branching_core::Clause> {
    disjointize_clauses_with_sources(clauses)
        .into_iter()
        .map(|(clause, _)| clause)
        .collect()
}

fn disjointize_clauses_with_sources(
    clauses: &[optimal_branching_core::Clause],
) -> Vec<(optimal_branching_core::Clause, usize)> {
    let mut disjoint = Vec::new();
    for (source, &clause) in clauses.iter().enumerate() {
        let mut pieces = vec![clause];
        for &(covered, _) in &disjoint {
            pieces = pieces
                .into_iter()
                .flat_map(|piece| subtract_clause(piece, covered))
                .collect();
            if pieces.is_empty() {
                break;
            }
        }
        disjoint.extend(pieces.into_iter().map(|piece| (piece, source)));
    }
    disjoint
}

fn subtract_clause(
    minuend: optimal_branching_core::Clause,
    subtrahend: optimal_branching_core::Clause,
) -> Vec<optimal_branching_core::Clause> {
    let shared = minuend.mask & subtrahend.mask;
    if ((minuend.val ^ subtrahend.val) & shared) != 0 {
        return vec![minuend];
    }

    let mut remaining = subtrahend.mask & !minuend.mask;
    if remaining == 0 {
        return Vec::new();
    }

    let mut prefix = minuend;
    let mut pieces = Vec::with_capacity(remaining.count_ones() as usize);
    while remaining != 0 {
        let bit = remaining & remaining.wrapping_neg();
        let required = subtrahend.val & bit;
        pieces.push(optimal_branching_core::Clause::new(
            prefix.mask | bit,
            prefix.val | (required ^ bit),
        ));
        prefix = optimal_branching_core::Clause::new(prefix.mask | bit, prefix.val | required);
        remaining &= !bit;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use crate::dimacs::network_from_dimacs;
    use optimal_branching_core::GreedyMerge;

    fn xor_chain() -> TnProblem {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");
        TnProblem::from_network(cn).expect("root SAT")
    }

    fn n(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test cutoff must be nonzero")
    }

    #[test]
    fn overlapping_branch_cover_is_disjointized_without_changing_union() {
        use optimal_branching_core::Clause;

        // x0=0 and x1=0 overlap on 00*. The third cube also overlaps both.
        let cover = vec![
            Clause::new(0b001, 0),
            Clause::new(0b010, 0),
            Clause::new(0b100, 0b100),
        ];
        let partition = disjointize_clauses(&cover);

        for assignment in 0..8 {
            let covered = cover.iter().any(|clause| clause.covered_by(assignment));
            let partition_count = partition
                .iter()
                .filter(|clause| clause.covered_by(assignment))
                .count();
            assert_eq!(partition_count, usize::from(covered));
        }
    }

    /// A cutoff larger than the root residual emits the empty decision path,
    /// while a cutoff equal to the residual must branch because `-n` is strict.
    #[test]
    fn static_n_controls_the_emitted_frontier() {
        let mut root_cut = xor_chain();
        assert_eq!(root_cut.count_unfixed(), 3);
        let mut root_cubes = Vec::new();
        let root_stats = generate_cubes_with_cutoff(
            &mut root_cut,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeCutoff::RemainingVars(n(4)),
            |cube| {
                root_cubes.push(cube);
                Ok::<(), Infallible>(())
            },
        )
        .expect("infallible callback");
        assert_eq!(root_stats.cubes, 1);
        assert!(root_cubes[0].decisions.is_empty());

        let mut strict = xor_chain();
        let mut strict_cubes = Vec::new();
        let strict_stats = generate_cubes_with_cutoff(
            &mut strict,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeCutoff::RemainingVars(n(3)),
            |cube| {
                strict_cubes.push(cube);
                Ok::<(), Infallible>(())
            },
        )
        .expect("infallible callback");
        assert!(strict_stats.cubes >= 1);
        assert!(strict_cubes
            .iter()
            .filter(|c| !c.refuted && !c.sat)
            .all(|c| !c.decisions.is_empty()));
    }

    #[test]
    fn decision_policy_honors_a_conquer_stop() {
        let signal = TerminationSignal::new();
        signal.request();
        let mut problem = xor_chain();
        let mut cubes = Vec::new();

        let stats = generate_cubes_configured(
            &mut problem,
            Selector::MostOccurrence { max_rows: 1 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeGenerationOptions {
                cutoff: CubeCutoff::RemainingVars(n(3)),
                sat_policy: CncSatPolicy::StopDecision,
                termination: Some(signal),
            },
            |cube| {
                cubes.push(cube);
                Ok::<(), Infallible>(())
            },
        )
        .expect("infallible callback");

        assert!(stats.stopped_early);
        assert!(cubes.is_empty());
    }

    #[test]
    fn cc_difficulty_cutoff_is_evaluated_online() {
        let mut problem = xor_chain();
        let mut cubes = Vec::new();
        let stats = generate_cubes_with_cutoff(
            &mut problem,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeCutoff::CcDifficulty(0),
            |cube| {
                cubes.push(cube);
                Ok::<(), Infallible>(())
            },
        )
        .expect("infallible callback");
        assert!(stats.cubes > 0);
        assert!(cubes
            .iter()
            .filter(|cube| !cube.refuted && !cube.sat)
            .all(|cube| cube.sigma_dec * cube.sigma_all > 0));
    }

    /// march_cu checks its static cutoff before declaring a solved leaf, so a
    /// root-solved instance emits `a 0` at any valid static `-n`.
    #[test]
    fn root_solved_instance_emits_the_empty_cube() {
        let cn = network_from_dimacs("p cnf 1 1\n1 0\n").expect("parse");
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        assert_eq!(p.count_unfixed(), 0);
        let mut cubes = Vec::new();
        let stats = generate_cubes_with_cutoff(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeCutoff::RemainingVars(n(1)),
            |cube| {
                cubes.push(cube);
                Ok::<(), Infallible>(())
            },
        )
        .expect("infallible callback");
        assert_eq!(stats.cubes, 1);
        assert_eq!(stats.sat_leaves, 0);
        assert!(cubes[0].decisions.is_empty());
    }

    #[test]
    fn streaming_callback_error_restores_the_search_state() {
        let mut p = xor_chain();
        let root_doms = p.doms.clone();
        let result = generate_cubes_with_cutoff(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeCutoff::RemainingVars(n(3)),
            |_| Err("stop"),
        );
        assert_eq!(result.unwrap_err(), "stop");
        assert_eq!(p.doms, root_doms);
    }

    #[test]
    fn tracing_preserves_frontier_and_records_a_tree() {
        let mut plain = xor_chain();
        let mut plain_cubes = Vec::new();
        let plain_stats = generate_cubes_with_cutoff(
            &mut plain,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeCutoff::RemainingVars(n(3)),
            |cube| {
                plain_cubes.push(cube);
                Ok::<(), Infallible>(())
            },
        )
        .expect("infallible callback");

        let mut traced = xor_chain();
        let root_doms = traced.doms.clone();
        let mut traced_cubes = Vec::new();
        let mut nodes = Vec::new();
        let traced_stats = generate_cubes_with_cutoff_trace(
            &mut traced,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            CubeCutoff::RemainingVars(n(3)),
            |cube| {
                traced_cubes.push(cube);
                Ok::<(), Infallible>(())
            },
            |node| {
                nodes.push(node);
                Ok::<(), Infallible>(())
            },
        )
        .expect("infallible callbacks");

        assert_eq!(traced.doms, root_doms, "trace run must restore state");
        assert_eq!(plain_stats.cubes, traced_stats.cubes);
        assert_eq!(plain_stats.refuted, traced_stats.refuted);
        assert_eq!(plain_stats.sat_leaves, traced_stats.sat_leaves);
        assert_eq!(plain_stats.visited, traced_stats.visited);
        assert_eq!(plain_cubes.len(), traced_cubes.len());
        for (expected, actual) in plain_cubes.iter().zip(&traced_cubes) {
            assert_eq!(expected.decisions, actual.decisions);
            assert_eq!(expected.sigma_dec, actual.sigma_dec);
            assert_eq!(expected.sigma_all, actual.sigma_all);
            assert_eq!(expected.refuted, actual.refuted);
            assert_eq!(expected.sat, actual.sat);
        }

        assert!(!nodes.is_empty());
        assert_eq!(nodes[0].node_id, 0);
        assert_eq!(nodes[0].parent_id, None);
        for (index, node) in nodes.iter().enumerate() {
            assert_eq!(node.node_id as usize, index, "DFS ids must be contiguous");
            if let Some(parent) = node.parent_id {
                assert!(parent < node.node_id, "parent must precede child");
                assert_eq!(node.depth, nodes[parent as usize].depth + 1);
            }
            assert_eq!(
                node.kind == CubeNodeKind::Branch,
                !node.clauses.is_empty(),
                "only branch nodes carry rule clauses"
            );
        }
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.kind == CubeNodeKind::Cutoff)
                .count(),
            traced_stats.cubes
        );
    }
}
