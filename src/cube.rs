//! Node SAMPLING for the Phase 0 cutoff study (see `docs/research/cutoff_plan.md`).
//! Descend the region-branching tree exhaustively and emit EVERY internal node
//! as a sample: its decision-literal path (a cube) plus a bundle of candidate
//! cutoff signals. There is deliberately NO cutoff here — the old
//! `|sigma_dec| * |sigma_all| > theta` rule was an arbitrary hand-picked
//! threshold, and it doubled as the sample selector, which biased the very
//! measurement meant to replace it. Instead we conquer each sampled node's
//! residual and correlate the signals against realized difficulty, spanning the
//! full depth range from the root (hardest) to near-leaf (easiest).
//!
//! This is deliberately NOT the full `bbsat` solver: no connected-component
//! decomposition (a cube is a partial assignment to the WHOLE formula, so the
//! sampler branches monolithically), and every branch is explored to a leaf
//! rather than stopping at the first SAT. It reuses the node primitives
//! (`findbest`, propagation, the reductions) so the sampled cubes match what
//! the solver would branch on; only the descent policy differs.
//!
//! Cubes carry DECISION literals only (`sigma_dec`), matching `march_cu`'s
//! assumption-literal convention: the conquer solver re-derives propagation
//! itself. `node_gamma` comes for free — it is the branching factor of the very
//! rule used to descend from this node — so no extra probe is needed. A
//! `max_nodes` cap bounds the sample count on large instances.

use std::sync::Arc;

use crate::adapter::BranchSolver;
use crate::ct::{apply_masked_assignment, ct_propagate, RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::measure::{measure_core, Measure};
use crate::network::ConstraintNetwork;
use crate::problem::{SolverBuffer, Stats, TnProblem};
use crate::propagate::{dominate_fixpoint, failed_literal_fixpoint};
use crate::selector::{occurrence_pool, Selector, FAILED_LITERAL_POOL};
use crate::trail::Trail;

/// Production cutoff: when it fires at a node, the node is emitted as a cube
/// (flagged `boundary`, conquer-only) and descent stops — the same emission
/// path as the `max_depth` cap, so downstream frontier analysis prices it
/// identically. The single mechanism is the resource-aware residual budget
/// (BBTN's ρ in decision form; the same measure march_cu's shipped dynamic
/// mode thresholds on): stop when the leftover piece is small enough for the
/// conquer engine to eat. It reads measures the sampler already computes, so
/// the predicate costs nothing extra per node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cutoff {
    /// Sampling mode: never fire (Phase 0 exhaustive / depth-capped studies).
    None,
    /// THE PRODUCTION RULE: fire when the residual measure drops to budget
    /// `B`. `B` is never hand-set — it comes from the sampling-calibration
    /// loop (`benchmarks/calibrate_cutoff.py`, Zaikin JAIR'23 Alg. 4-5): cut
    /// at candidate thresholds, sample-conquer, pick the argmin of estimated
    /// end-to-end cost. Same protocol drives march_cu's `-n` (same measure),
    /// which is what makes cuber comparisons cutoff-fair.
    ResidualBudget(Measure, usize),
    /// CONTROL ARM ONLY. march_cu's shipped feedback rule (CnC repo
    /// solver.c:544/653/1400): threshold starts at 0, is raised to the
    /// residual measure of any branch our propagation refutes, decays by
    /// 1 − fraction^(depth^downexp) per emission. Measured on f28 (2026-07-11)
    /// and documented in Heule's own tutorial: on hard instances the
    /// refutation onset sits far BELOW the CDCL difficulty cliff, so this rule
    /// cuts pathologically deep — practice uses static thresholds calibrated
    /// by sampling instead. Kept as the ablation/control arm.
    RefutationAdaptive {
        measure: Measure,
        /// march defaults: fraction 0.02, downexp 0.3.
        fraction: f64,
        downexp: f64,
    },
}

/// Mutable runtime state of the cutoff (the adaptive threshold). One per
/// `generate_cubes` call, threaded through the DFS.
pub struct CutoffState {
    threshold: f64,
}

impl Cutoff {
    pub fn march_default(measure: Measure) -> Cutoff {
        Cutoff::RefutationAdaptive {
            measure,
            fraction: 0.02,
            downexp: 0.3,
        }
    }

    fn measure_of(m: Measure, unfixed: usize, active: usize, hard: usize) -> usize {
        match m {
            Measure::NumUnfixedVars => unfixed,
            Measure::NumUnfixedTensors => active,
            Measure::NumHardTensors => hard,
        }
    }

    /// Does the cutoff fire at a node with these residual readings? For the
    /// adaptive rule this also applies the emission decay (march does both at
    /// the cut site).
    fn fires(
        &self,
        state: &mut CutoffState,
        depth: usize,
        unfixed: usize,
        active: usize,
        hard: usize,
    ) -> bool {
        match *self {
            Cutoff::None => false,
            Cutoff::ResidualBudget(m, budget) => {
                Self::measure_of(m, unfixed, active, hard) <= budget
            }
            Cutoff::RefutationAdaptive {
                measure,
                fraction,
                downexp,
            } => {
                let r = Self::measure_of(measure, unfixed, active, hard) as f64;
                if r < state.threshold {
                    let d = depth.max(1) as f64;
                    state.threshold *= 1.0 - fraction.powf(d.powf(downexp));
                    true
                } else {
                    false
                }
            }
        }
    }

    /// A branch at residual reading (unfixed, active, hard) was refuted by the
    /// cuber's own propagation: raise the adaptive threshold to that level
    /// (march solver.c:653 / 1400-1402).
    fn on_refuted(&self, state: &mut CutoffState, unfixed: usize, active: usize, hard: usize) {
        if let Cutoff::RefutationAdaptive { measure, .. } = *self {
            let r = Self::measure_of(measure, unfixed, active, hard) as f64;
            if r > state.threshold {
                state.threshold = r;
            }
        }
    }
}

/// One sampled search node: the decision literals as `(var, value)` pairs, the
/// two path counts, and a bundle of candidate cutoff SIGNALS snapshotted at the
/// node (Phase 0 measurement — see `docs/research/cutoff_plan.md`). Internal
/// (branchable) nodes are the difficulty samples; terminal nodes (SAT leaves,
/// locally-refuted nodes) are also pushed but flagged, so `CubeStats` and
/// callers can audit coverage — the conquer worklist filters on `!refuted &&
/// !sat`, which drops those trivial conquers.
///
/// The signals let the conquer harness correlate each candidate against realized
/// conquer difficulty and pick the one that best predicts it, across instance
/// families, so a resource-aware cutoff can key on it (replacing the discarded
/// `theta` heuristic entirely).
#[derive(Clone, Debug)]
pub struct Cube {
    pub decisions: Vec<(usize, bool)>,
    pub sigma_dec: usize,
    pub sigma_all: usize,
    /// `measure_core(NumUnfixedVars)` at the node.
    pub unfixed_vars: usize,
    /// `measure_core(NumUnfixedTensors)` — active (still-biting) constraints.
    pub active_tensors: usize,
    /// `measure_core(NumHardTensors)` — sum of (degree − 2) over wide tensors.
    pub hard_excess: usize,
    /// `sigma_all` of this node's PARENT (the node one branch decision up), so
    /// the harness can derive last-branch yield `sigma_all - sigma_all_parent`.
    pub sigma_all_parent: usize,
    /// γ of the rule that PRODUCED this node (the parent's `findbest` γ);
    /// `f64::NAN` at the root.
    pub parent_gamma: f64,
    /// γ of THIS node's own `findbest` rule — the branching factor of the rule
    /// used to descend from here. Free (no extra probe): it is the same rule the
    /// sampler branches on. `f64::NAN` at terminal (SAT/refuted) nodes.
    pub node_gamma: f64,
    pub refuted: bool,
    pub sat: bool,
    /// True iff the depth cap stopped expansion HERE: the node is open (has a
    /// branching rule) but its children were not sampled. Downstream frontier
    /// analysis must treat it as conquer-only (no expand option), which keeps
    /// the oracle DP exact on the depth-d prefix — the tractable substitute for
    /// exhaustive sampling at scales where the full tree is out of reach.
    pub boundary: bool,
}

pub struct CubeStats {
    pub cubes: usize,
    pub refuted: usize,
    pub sat_leaves: usize,
    /// Nodes visited (branch decisions applied).
    pub visited: u64,
}

struct CubeCtx<'a> {
    cn: &'a Arc<ConstraintNetwork>,
    selector: Selector,
    measure: Measure,
    solver: &'a BranchSolver,
    /// Backstop for large instances: stop descending once this many rows have
    /// been emitted (internal samples plus terminal-node markers). `None` =
    /// exhaustive (every node in the tree). NOTE: a count cap truncates the
    /// tree mid-subtree (partially expanded interiors), which invalidates
    /// frontier-DP analysis — use `max_depth` for that instead.
    max_nodes: Option<usize>,
    /// Subtree-complete prefix cap: nodes at this branch-step depth are emitted
    /// as `boundary` samples and not expanded. Every non-boundary open node
    /// then has ALL its children in the sample, which is exactly the invariant
    /// the frontier DP needs on instances too large to exhaust.
    max_depth: Option<usize>,
    /// Production stopping rule; fires through the same boundary-emission path
    /// as `max_depth`, so cutoff runs stay valid frontier-DP input.
    cutoff: Cutoff,
    /// Wall-clock guard: once elapsed, every still-open path emits its node as
    /// a boundary cube and unwinds — the frontier stays complete (every
    /// root-leaf path is cut), so a time-capped run is still a sound cube set.
    deadline: Option<std::time::Instant>,
}

/// Sample search nodes from `problem` for the Phase 0 cutoff study: descend the
/// region-branching tree and emit every internal node (no cutoff). The problem's
/// root propagation must already have run (as after `from_network`). Returns the
/// emitted samples (to conquer) and generation stats. `max_nodes` caps the
/// sample count on large instances (`None` = exhaustive). Terminal nodes (SAT
/// leaves, locally-refuted nodes) are counted in stats but not emitted.
pub fn generate_cubes(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    max_nodes: Option<usize>,
    max_depth: Option<usize>,
    cutoff: Cutoff,
    max_seconds: Option<f64>,
) -> (Vec<Cube>, CubeStats) {
    problem.stats.reset();
    let ctx = CubeCtx {
        cn: &problem.static_cn,
        selector,
        measure,
        solver,
        max_nodes,
        max_depth,
        cutoff,
        deadline: max_seconds
            .map(|s| std::time::Instant::now() + std::time::Duration::from_secs_f64(s)),
    };
    let masks = &problem.masks;
    let stats = &mut problem.stats;
    let buffer = &mut problem.buffer;
    let doms = &mut problem.doms;
    let tables = &mut problem.tables;
    let trail = &mut problem.trail;

    let mut out = Vec::new();
    let mut decisions: Vec<(usize, bool)> = Vec::new();
    let mut cutoff_state = CutoffState { threshold: 0.0 };
    let mark = trail.mark();
    // Root already propagated; a refuted root is a trivial UNSAT with no internal
    // node to sample, so we emit nothing. Otherwise descend from the root, whose
    // parent seed is its own fixed-count (last-branch yield 0) and NaN parent γ.
    if doms[0] != DomainMask::NONE {
        let root_sigma_all = doms.iter().filter(|d| d.is_fixed()).count();
        cube_rec(
            &ctx,
            stats,
            buffer,
            doms,
            masks,
            tables,
            trail,
            &mut decisions,
            &mut out,
            &mut cutoff_state,
            root_sigma_all,
            f64::NAN,
            0,
        );
    }
    trail.restore_to(mark, doms, tables);

    let refuted = out.iter().filter(|c| c.refuted).count();
    let sat_leaves = out.iter().filter(|c| c.sat).count();
    let cubes = out.len() - refuted - sat_leaves;
    let stats_out = CubeStats {
        cubes,
        refuted,
        sat_leaves,
        visited: stats.total_visited_nodes,
    };
    (out, stats_out)
}

#[allow(clippy::too_many_arguments)]
fn cube_rec(
    ctx: &CubeCtx,
    stats: &mut Stats,
    buffer: &mut SolverBuffer,
    doms: &mut Vec<DomainMask>,
    masks: &Arc<Vec<TableMasks>>,
    tables: &mut Vec<RSparseBitSet>,
    trail: &mut Trail,
    decisions: &mut Vec<(usize, bool)>,
    out: &mut Vec<Cube>,
    cutoff_state: &mut CutoffState,
    sigma_all_parent: usize,
    parent_gamma: f64,
    depth: usize,
) {
    // Backstop: once the sample budget is reached, stop descending.
    if ctx.max_nodes.is_some_and(|cap| out.len() >= cap) {
        return;
    }

    // Heartbeat so long cubing runs are observable (f28 taught us: 45 blind
    // minutes is a debugging session, one stderr line is not).
    if stats.total_visited_nodes > 0 && stats.total_visited_nodes % 16384 == 0 {
        eprintln!(
            "  ...visited={} emitted={} depth={} adaptive_threshold={:.1}",
            stats.total_visited_nodes,
            out.len(),
            depth,
            cutoff_state.threshold,
        );
    }

    let sigma_dec = decisions.len();
    let sigma_all = doms.iter().filter(|d| d.is_fixed()).count();
    // Cheap candidate signals, snapshotted at this node (each is O(tensors)).
    let unfixed_vars = measure_core(ctx.cn, doms, Measure::NumUnfixedVars) as usize;
    let active_tensors = measure_core(ctx.cn, doms, Measure::NumUnfixedTensors) as usize;
    let hard_excess = measure_core(ctx.cn, doms, Measure::NumHardTensors) as usize;

    let scope: Vec<usize> = (0..doms.len()).filter(|&v| !doms[v].is_fixed()).collect();
    if scope.is_empty() {
        // Fully assigned: a SAT leaf — terminal, flagged, not conquered.
        stats.record_visit();
        out.push(Cube {
            decisions: decisions.clone(),
            sigma_dec,
            sigma_all,
            unfixed_vars,
            active_tensors,
            hard_excess,
            sigma_all_parent,
            parent_gamma,
            node_gamma: f64::NAN,
            refuted: false,
            sat: true,
            boundary: false,
        });
        return;
    }

    // findbest gives both the branching rule AND this node's γ (free — the same
    // rule we descend on), so node_gamma needs no extra probe.
    let (clauses, variables, node_gamma) = ctx.selector.findbest(
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
    let clauses = match clauses {
        // No rule (region proved locally UNSAT): terminal refuted node — a
        // refutation event, so the adaptive threshold rises to this level.
        None => {
            ctx.cutoff
                .on_refuted(cutoff_state, unfixed_vars, active_tensors, hard_excess);
            out.push(Cube {
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                unfixed_vars,
                active_tensors,
                hard_excess,
                sigma_all_parent,
                parent_gamma,
                node_gamma,
                refuted: true,
                sat: false,
                boundary: false,
            });
            return;
        }
        Some(c) => c,
    };

    // Emit THIS internal node as a difficulty sample (no cutoff): its residual is
    // conquered and correlated against the signals above. Then branch into every
    // child — a node is both sampled and expanded, so the samples span all depths.
    // At the depth cap — or when the production cutoff fires — the node is
    // flagged `boundary` and NOT expanded (findbest still ran, so its γ is
    // real), keeping the sampled prefix subtree-complete.
    let at_cap = ctx.max_depth.is_some_and(|d| depth >= d)
        || ctx.deadline.is_some_and(|d| std::time::Instant::now() >= d)
        || ctx.cutoff.fires(
            cutoff_state,
            depth,
            unfixed_vars,
            active_tensors,
            hard_excess,
        );
    out.push(Cube {
        decisions: decisions.clone(),
        sigma_dec,
        sigma_all,
        unfixed_vars,
        active_tensors,
        hard_excess,
        sigma_all_parent,
        parent_gamma,
        node_gamma,
        refuted: false,
        sat: false,
        boundary: at_cap,
    });
    if at_cap {
        return;
    }

    for cl in &clauses {
        stats.record_visit();
        trail.open();
        let m = trail.mark();
        buffer.reset_worklist();
        // Record this branch's decision literals before applying.
        let dec_base = decisions.len();
        for (i, &var) in variables.iter().enumerate() {
            if (cl.mask >> i) & 1 == 1 {
                decisions.push((var, (cl.val >> i) & 1 == 1));
            }
        }
        apply_masked_assignment(ctx.cn, doms, buffer, trail, &variables, cl.mask, cl.val);
        ct_propagate(ctx.cn, doms, masks, tables, buffer, trail);
        if doms[0] != DomainMask::NONE {
            dominate_fixpoint(ctx.cn, doms, masks, tables, buffer, trail);
            if doms[0] != DomainMask::NONE {
                let pool = occurrence_pool(ctx.cn, doms, buffer, masks, FAILED_LITERAL_POOL);
                failed_literal_fixpoint(ctx.cn, doms, masks, tables, buffer, trail, &pool);
            }
        }
        if doms[0] == DomainMask::NONE {
            // Branch closed by propagation: refuted cube (no conquer needed).
            // This child's parent is the current node, so it carries this node's
            // sigma_all and rule γ as its parent values. A refutation event:
            // the adaptive threshold rises to this child's residual level.
            let child_unfixed = measure_core(ctx.cn, doms, Measure::NumUnfixedVars) as usize;
            let child_active = measure_core(ctx.cn, doms, Measure::NumUnfixedTensors) as usize;
            let child_hard = measure_core(ctx.cn, doms, Measure::NumHardTensors) as usize;
            ctx.cutoff
                .on_refuted(cutoff_state, child_unfixed, child_active, child_hard);
            out.push(Cube {
                decisions: decisions.clone(),
                sigma_dec: decisions.len(),
                sigma_all: doms.iter().filter(|d| d.is_fixed()).count(),
                unfixed_vars: child_unfixed,
                active_tensors: child_active,
                hard_excess: child_hard,
                sigma_all_parent: sigma_all,
                parent_gamma: node_gamma,
                node_gamma: f64::NAN,
                refuted: true,
                sat: false,
                boundary: false,
            });
        } else {
            // Recurse: the child's parent is THIS node — pass this node's
            // sigma_all and rule γ down as the child's parent values.
            cube_rec(
                ctx,
                stats,
                buffer,
                doms,
                masks,
                tables,
                trail,
                decisions,
                out,
                cutoff_state,
                sigma_all,
                node_gamma,
                depth + 1,
            );
        }
        decisions.truncate(dec_base);
        trail.restore_to(m, doms, tables);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimacs::network_from_dimacs;
    use optimal_branching_core::GreedyMerge;

    /// A satisfiable instance cubed at a small theta yields cubes whose decision
    /// prefixes cover the search: every cube is a distinct decision path and at
    /// exhaustive sampling emits at least the (open) root internal node, and
    /// every open sample carries a real branching factor.
    #[test]
    fn samples_internal_nodes_of_a_small_instance() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        let (cubes, stats) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            None,
            Cutoff::None,
            None,
        );
        // At least one open internal node (the root, whose decision path may be
        // empty on a tiny instance solved in one branch). Every open sample must
        // carry the descending rule's γ ≥ 1.
        assert!(
            stats.cubes >= 1,
            "expected open samples, got {}",
            stats.cubes
        );
        let open: Vec<_> = cubes.iter().filter(|c| !c.refuted && !c.sat).collect();
        assert!(!open.is_empty(), "at least the root is an open sample");
        assert!(open
            .iter()
            .all(|c| c.node_gamma.is_finite() && c.node_gamma >= 1.0));
    }

    /// Exhaustive sampling (no cap) descends to natural leaves: the tree bottoms
    /// out in SAT/refuted terminals, and internal nodes are emitted along the way.
    #[test]
    fn exhaustive_sampling_reaches_leaves() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        let (cubes, stats) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            None,
            Cutoff::None,
            None,
        );
        assert!(stats.cubes >= 1, "internal samples emitted");
        assert!(
            cubes.iter().any(|c| c.sat),
            "descent bottoms out in a SAT leaf"
        );
    }

    /// Every emitted cube's signals are internally consistent (unfixed + fixed
    /// == n_vars, last-branch yield non-negative), and every OPEN sample carries
    /// a real node_gamma ≥ 1 (the descending rule's γ, no probe needed). The
    /// `max_nodes` cap bounds the sample count.
    #[test]
    fn cube_signals_are_consistent_and_cap_bounds_samples() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");
        let n_vars = 3usize;

        let mut p = TnProblem::from_network(cn.clone()).expect("root SAT");
        let (cubes, stats_full) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            None,
            Cutoff::None,
            None,
        );
        for c in &cubes {
            assert_eq!(
                c.unfixed_vars + c.sigma_all,
                n_vars,
                "unfixed + fixed must equal n_vars"
            );
            assert!(
                c.sigma_all >= c.sigma_all_parent,
                "sigma_all must not drop below the parent's"
            );
            if !c.refuted && !c.sat {
                assert!(
                    c.node_gamma.is_finite() && c.node_gamma >= 1.0,
                    "open sample carries a real γ ≥ 1, got {}",
                    c.node_gamma
                );
            }
        }

        // A cap of 1 emitted row must not exceed the uncapped sample count.
        let mut p2 = TnProblem::from_network(cn).expect("root SAT");
        let (capped, stats_cap) = generate_cubes(
            &mut p2,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            Some(1),
            None,
            Cutoff::None,
            None,
        );
        assert!(capped.len() <= cubes.len());
        assert!(stats_cap.cubes <= stats_full.cubes);
    }

    /// A depth cap keeps the sampled prefix subtree-complete: every open node
    /// strictly above the cap is fully expanded, nodes AT the cap are flagged
    /// `boundary` (open, with a real γ, no children), and nothing lies deeper.
    #[test]
    fn depth_cap_marks_boundary_and_stays_subtree_complete() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        let (cubes, _) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            Some(0),
            Cutoff::None,
            None,
        );
        // Cap at depth 0: the root is the single open sample, flagged boundary,
        // and no deeper node of any kind is emitted.
        assert_eq!(cubes.len(), 1);
        let root = &cubes[0];
        assert!(root.boundary && !root.refuted && !root.sat);
        assert!(root.decisions.is_empty());
        assert!(root.node_gamma.is_finite() && root.node_gamma >= 1.0);
    }

    /// The production ResidualBudget cutoff fires through the boundary path:
    /// a budget at least the root's residual emits exactly one boundary cube
    /// (the root), and a budget of 0 never fires (exhaustive descent).
    #[test]
    fn residual_budget_cutoff_emits_boundary_cubes() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");

        let mut p = TnProblem::from_network(cn.clone()).expect("root SAT");
        let (cubes, _) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            None,
            Cutoff::ResidualBudget(Measure::NumUnfixedVars, 3),
            None,
        );
        assert_eq!(cubes.len(), 1, "budget >= root residual fires at the root");
        assert!(cubes[0].boundary && cubes[0].decisions.is_empty());

        let mut p2 = TnProblem::from_network(cn).expect("root SAT");
        let (unbounded, _) = generate_cubes(
            &mut p2,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            None,
            Cutoff::ResidualBudget(Measure::NumUnfixedVars, 0),
            None,
        );
        assert!(
            unbounded.iter().all(|c| !c.boundary),
            "budget 0 never fires: exhaustive descent"
        );
        assert!(unbounded.len() > 1);
    }

    /// The march-style adaptive cutoff starts at threshold 0 and only rises on
    /// refutation events. On a refutation-free tree it must therefore behave
    /// exactly like exhaustive sampling: full descent, zero boundary cubes.
    #[test]
    fn adaptive_cutoff_without_refutations_is_exhaustive() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");

        let mut p = TnProblem::from_network(cn.clone()).expect("root SAT");
        let (adaptive, stats) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            None,
            Cutoff::march_default(Measure::NumUnfixedVars),
            None,
        );
        assert_eq!(stats.refuted, 0, "this tiny tree has no refutation events");
        assert!(
            adaptive.iter().all(|c| !c.boundary),
            "threshold never rose above 0, so nothing may fire"
        );

        let mut p2 = TnProblem::from_network(cn).expect("root SAT");
        let (exhaustive, _) = generate_cubes(
            &mut p2,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            None,
            None,
            Cutoff::None,
            None,
        );
        assert_eq!(adaptive.len(), exhaustive.len(), "identical descent");
    }
}
