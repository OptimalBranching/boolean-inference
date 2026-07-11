//! Cube generation for Cube-and-Conquer: use the region-branching rule as a
//! CUBER. Descend the branch tree, and at each node emit the current
//! decision-literal path as a cube once the node is "fixed enough" by the
//! cutoff `|sigma_dec| * |sigma_all| > theta` (the paper's condition), where
//! `sigma_dec` is the number of variables fixed by BRANCH decisions along the
//! path and `sigma_all` is the total number of fixed variables at the node.
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
//! cutoff proxy can be compared against realized conquer difficulty.

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

/// One generated cube: the decision literals as `(var, value)` pairs, plus the
/// two cutoff counts at the emitting leaf. `refuted` cubes were closed by
/// propagation before reaching the cutoff (locally UNSAT — no conquer needed).
#[derive(Clone, Debug)]
pub struct Cube {
    pub decisions: Vec<(usize, bool)>,
    pub sigma_dec: usize,
    pub sigma_all: usize,
    pub refuted: bool,
    pub sat: bool,
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
    theta: f64,
}

/// Generate a cube partition of `problem` under cutoff `theta`. The problem's
/// root propagation must already have run (as after `from_network`). Returns the
/// open cubes (to hand to a conquer solver) and generation stats. Refuted and
/// SAT leaves are included in the returned vector, flagged, so callers can audit
/// coverage; filter on `!refuted` for the conquer worklist.
pub fn generate_cubes(
    problem: &mut TnProblem,
    selector: Selector,
    measure: Measure,
    solver: &BranchSolver,
    theta: f64,
) -> (Vec<Cube>, CubeStats) {
    problem.stats.reset();
    let ctx = CubeCtx {
        cn: &problem.static_cn,
        selector,
        measure,
        solver,
        theta,
    };
    let masks = &problem.masks;
    let stats = &mut problem.stats;
    let buffer = &mut problem.buffer;
    let doms = &mut problem.doms;
    let tables = &mut problem.tables;
    let trail = &mut problem.trail;

    let mut out = Vec::new();
    let mut decisions: Vec<(usize, bool)> = Vec::new();
    let mark = trail.mark();
    // Root already propagated; if it is already solved or refuted, that is a
    // single (degenerate) cube.
    if doms[0] == DomainMask::NONE {
        out.push(Cube {
            decisions: Vec::new(),
            sigma_dec: 0,
            sigma_all: 0,
            refuted: true,
            sat: false,
        });
    } else {
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
) {
    // Cutoff test on the current (post-reduction) node: emit the decision path
    // as a cube once fixed enough. sigma_dec counts decision-fixed vars along
    // the path; sigma_all counts all fixed vars now.
    let sigma_dec = decisions.len();
    let sigma_all = doms.iter().filter(|d| d.is_fixed()).count();
    if (sigma_dec as f64) * (sigma_all as f64) > ctx.theta {
        out.push(Cube {
            decisions: decisions.clone(),
            sigma_dec,
            sigma_all,
            refuted: false,
            sat: false,
        });
        return;
    }

    let scope: Vec<usize> = (0..doms.len()).filter(|&v| !doms[v].is_fixed()).collect();
    if scope.is_empty() {
        // Fully assigned without hitting the cutoff: a SAT leaf.
        stats.record_visit();
        out.push(Cube {
            decisions: decisions.clone(),
            sigma_dec,
            sigma_all,
            refuted: false,
            sat: true,
        });
        return;
    }

    let (clauses, variables) = ctx.selector.findbest(
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
        // No rule (region proved locally UNSAT): refuted cube.
        None => {
            out.push(Cube {
                decisions: decisions.clone(),
                sigma_dec,
                sigma_all,
                refuted: true,
                sat: false,
            });
            return;
        }
        Some(c) => c,
    };

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
            out.push(Cube {
                decisions: decisions.clone(),
                sigma_dec: decisions.len(),
                sigma_all: doms.iter().filter(|d| d.is_fixed()).count(),
                refuted: true,
                sat: false,
            });
        } else {
            cube_rec(
                ctx, stats, buffer, doms, masks, tables, trail, decisions, out,
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
    /// least one is not refuted.
    #[test]
    fn cubes_partition_a_small_sat_instance() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        let (cubes, stats) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            1.0,
        );
        // Non-degenerate: at least one open cube, and open cubes have decisions.
        assert!(stats.cubes >= 1, "expected open cubes, got {}", stats.cubes);
        assert!(cubes.iter().any(|c| !c.refuted && !c.decisions.is_empty()));
    }

    /// theta = infinity means the cutoff never fires, so `generate_cubes`
    /// degenerates to a full search whose leaves are all SAT or refuted — a
    /// sanity check that the descent terminates and covers.
    #[test]
    fn infinite_theta_solves_to_leaves() {
        let cnf = "p cnf 3 4\n1 2 0\n-1 -2 0\n2 3 0\n-2 -3 0\n";
        let cn = network_from_dimacs(cnf).expect("parse");
        let mut p = TnProblem::from_network(cn).expect("root SAT");
        let (cubes, stats) = generate_cubes(
            &mut p,
            Selector::MostOccurrence { max_rows: 32 },
            Measure::NumUnfixedVars,
            &BranchSolver::Greedy(GreedyMerge),
            f64::INFINITY,
        );
        assert_eq!(stats.cubes, 0, "no open cubes at infinite theta");
        assert!(cubes.iter().any(|c| c.sat), "at least one SAT leaf");
    }
}
