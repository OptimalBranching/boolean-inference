//! Cube generation for Cube-and-Conquer: use the region-branching rule as a
//! CUBER. Descend the branch tree and evaluate an online stopping rule after
//! propagation and reductions at each node. The primary rule is the classical
//! Cube-and-Conquer difficulty `D^2(D+I)/N`; a remaining-variable cutoff is
//! retained as an ablation.
//!
//! This is deliberately NOT the full `bbsat` solver: no connected-component
//! decomposition (a cube is a partial assignment to the WHOLE formula, so the
//! cuber must branch monolithically to keep cubes a clean partition), and every
//! branch is explored to a cube leaf rather than stopping at the first SAT. It
//! reuses the node primitives (`findbest`, propagation, the reductions) so the
//! cubes match what the solver would branch on; only the descent policy differs.
//!
//! Cubes are emitted as DECISION literals only (`sigma_dec`), matching
//! `march_cu`'s assumption-literal convention: the conquer solver re-derives
//! propagation itself. Each cube also carries `sigma_dec`/`sigma_all` so the
//! emitted frontier can be audited.

use std::convert::Infallible;
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
    pub decisions: Vec<(usize, bool)>,
    pub sigma_dec: usize,
    pub sigma_all: usize,
    pub freevars: usize,
    pub variables: Vec<usize>,
    pub clauses: Vec<TraceClause>,
}

struct CubeCtx<'a> {
    cn: &'a Arc<ConstraintNetwork>,
    selector: Selector,
    measure: Measure,
    solver: &'a BranchSolver,
    cutoff: CubeCutoff,
}

/// Online stopping rule evaluated at each post-reduction search node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeCutoff {
    /// march-compatible rule: stop when `freevars < n`.
    RemainingVars(NonZeroUsize),
    /// Classical CC difficulty: stop when `D^2 * (D + I) / N > threshold`.
    CcDifficulty(u128),
}

impl CubeCutoff {
    fn stops(self, sigma_dec: usize, sigma_all: usize, freevars: usize) -> bool {
        match self {
            Self::RemainingVars(n) => freevars < n.get(),
            Self::CcDifficulty(threshold) => {
                let total = sigma_all + freevars;
                (sigma_dec as u128).pow(2) * (sigma_all as u128) > threshold * (total as u128)
            }
        }
    }
}

/// Generate a cube partition of `problem` using march_cu's static `-n` cutoff:
/// emit the current decision path when fewer than `cutoff_vars` variables remain
/// unfixed. The comparison is strict, exactly as in march_cu's static mode;
/// `cutoff_vars` is nonzero because march_cu reserves zero for dynamic mode.
///
/// The problem's root propagation must already have run (as after
/// `from_network`). Returns the open cubes (to hand to a conquer solver) and
/// generation stats. Refuted and SAT leaves are included in the returned
/// vector, flagged, so callers can audit coverage.
///
/// New experiments should call [`generate_cubes_with_cutoff`] with
/// [`CubeCutoff::CcDifficulty`]. This wrapper remains for compatibility and
/// remaining-variable ablations.
pub fn generate_cubes(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    cutoff_vars: NonZeroUsize,
) -> (Vec<Cube>, CubeStats) {
    let mut cubes = Vec::new();
    let stats = match generate_cubes_with(problem, selector, measure, solver, cutoff_vars, |cube| {
        cubes.push(cube);
        Ok::<(), Infallible>(())
    }) {
        Ok(stats) => stats,
        Err(error) => match error {},
    };
    (cubes, stats)
}

/// Streaming form of [`generate_cubes`]. Each open, refuted, or SAT leaf is
/// passed to `emit` as soon as it is reached, so production cubers need not keep
/// the entire frontier and every cloned decision path in memory.
pub fn generate_cubes_with<E, F>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    cutoff_vars: NonZeroUsize,
    emit: F,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
{
    generate_cubes_with_cutoff(
        problem,
        selector,
        measure,
        solver,
        CubeCutoff::RemainingVars(cutoff_vars),
        emit,
    )
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
    generate_cubes_impl(
        problem,
        selector,
        measure,
        solver,
        cutoff,
        emit,
        None::<&mut fn(CubeNodeTrace) -> Result<(), E>>,
    )
}

/// Streaming cube generation with an additional callback for every tree node.
/// The trace callback observes data already computed by the normal search and
/// must not mutate solver state, so enabling it does not alter the frontier.
pub fn generate_cubes_with_trace<E, F, T>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    cutoff_vars: NonZeroUsize,
    emit: F,
    trace: T,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
    T: FnMut(CubeNodeTrace) -> Result<(), E>,
{
    generate_cubes_with_cutoff_trace(
        problem,
        selector,
        measure,
        solver,
        CubeCutoff::RemainingVars(cutoff_vars),
        emit,
        trace,
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
        cutoff,
        emit,
        Some(&mut trace),
    )
}

fn generate_cubes_impl<E, F, T>(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    cutoff: CubeCutoff,
    mut emit: F,
    mut trace: Option<&mut T>,
) -> Result<CubeStats, E>
where
    F: FnMut(Cube) -> Result<(), E>,
    T: FnMut(CubeNodeTrace) -> Result<(), E>,
{
    problem.stats.reset();
    let ctx = CubeCtx {
        cn: &problem.static_cn,
        selector,
        measure,
        solver,
        cutoff,
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
    let result = if doms[0] == DomainMask::NONE {
        if let Some(trace) = trace.as_deref_mut() {
            trace(CubeNodeTrace {
                node_id: 0,
                parent_id: None,
                child_index: None,
                depth: 0,
                kind: CubeNodeKind::Refuted,
                decisions: Vec::new(),
                sigma_dec: 0,
                sigma_all: 0,
                freevars: doms.len(),
                variables: Vec::new(),
                clauses: Vec::new(),
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
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                freevars,
                variables: Vec::new(),
                clauses: Vec::new(),
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
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                freevars,
                variables: Vec::new(),
                clauses: Vec::new(),
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
                sat: true,
            },
        );
    }

    let (maybe_clauses, variables) = ctx.selector.findbest(
        ctx.cn,
        doms,
        buffer,
        ctx.measure,
        ctx.solver,
        masks,
        tables,
        trail,
        &scope,
    );
    let clauses = match maybe_clauses {
        // No rule (region proved locally UNSAT): refuted cube.
        None => {
            if let Some(trace) = trace.as_deref_mut() {
                trace(CubeNodeTrace {
                    node_id,
                    parent_id,
                    child_index,
                    depth,
                    kind: CubeNodeKind::Refuted,
                    decisions: decisions.clone(),
                    sigma_dec,
                    sigma_all,
                    freevars,
                    variables,
                    clauses: Vec::new(),
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

    if let Some(trace) = trace.as_deref_mut() {
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
            decisions: decisions.clone(),
            sigma_dec,
            sigma_all,
            freevars,
            variables: variables.clone(),
            clauses: trace_clauses,
        })?;
    }

    for (branch_index, cl) in clauses.iter().enumerate() {
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
        if doms[0] != DomainMask::NONE {
            dominate_fixpoint(ctx.cn, doms, masks, tables, buffer, trail);
        }
        if doms[0] != DomainMask::NONE {
            let pool = occurrence_pool(ctx.cn, doms, buffer, masks, FAILED_LITERAL_POOL);
            failed_literal_fixpoint(ctx.cn, doms, masks, tables, buffer, trail, &pool);
        }
        let branch_result = if doms[0] == DomainMask::NONE {
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
                    decisions: decisions.clone(),
                    sigma_dec: decisions.len(),
                    sigma_all: doms.len() - branch_freevars,
                    freevars: branch_freevars,
                    variables: Vec::new(),
                    clauses: Vec::new(),
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
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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

    /// A cutoff larger than the root residual emits the empty decision path,
    /// while a cutoff equal to the residual must branch because `-n` is strict.
    #[test]
    fn static_n_controls_the_emitted_frontier() {
        let mut root_cut = xor_chain();
        assert_eq!(root_cut.count_unfixed(), 3);
        let (root_cubes, root_stats) = generate_cubes(
            &mut root_cut,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            n(4),
        );
        assert_eq!(root_stats.cubes, 1);
        assert!(root_cubes[0].decisions.is_empty());

        let mut strict = xor_chain();
        let (strict_cubes, strict_stats) = generate_cubes(
            &mut strict,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            n(3),
        );
        assert!(strict_stats.cubes >= 1);
        assert!(strict_cubes
            .iter()
            .filter(|c| !c.refuted && !c.sat)
            .all(|c| !c.decisions.is_empty()));
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

    /// march_cu checks its static cutoff before declaring a solved leaf. Lock
    /// that compatibility behavior: a root-solved instance emits `a 0` at any
    /// valid static `-n` rather than being counted as a SAT leaf.
    #[test]
    fn root_solved_instance_emits_the_empty_cube() {
        let cn = network_from_dimacs("p cnf 1 1\n1 0\n").expect("parse");
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        assert_eq!(p.count_unfixed(), 0);
        let (cubes, stats) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            n(1),
        );
        assert_eq!(stats.cubes, 1);
        assert_eq!(stats.sat_leaves, 0);
        assert!(cubes[0].decisions.is_empty());
    }

    #[test]
    fn streaming_callback_error_restores_the_search_state() {
        let mut p = xor_chain();
        let root_doms = p.doms.clone();
        let result = generate_cubes_with(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            n(3),
            |_| Err("stop"),
        );
        assert_eq!(result.unwrap_err(), "stop");
        assert_eq!(p.doms, root_doms);
    }

    #[test]
    fn tracing_preserves_frontier_and_records_a_tree() {
        let mut plain = xor_chain();
        let (plain_cubes, plain_stats) = generate_cubes(
            &mut plain,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            n(3),
        );

        let mut traced = xor_chain();
        let root_doms = traced.doms.clone();
        let mut traced_cubes = Vec::new();
        let mut nodes = Vec::new();
        let traced_stats = generate_cubes_with_trace(
            &mut traced,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            n(3),
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
