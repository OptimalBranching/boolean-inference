use std::sync::Arc;

use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;
use crate::trail::Trail;

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub branching_nodes: u64,
    pub total_potential_subproblems: u64,
    pub total_visited_nodes: u64,
}

impl Stats {
    pub fn reset(&mut self) {
        *self = Stats::default();
    }
    #[inline]
    pub fn record_branch(&mut self, subproblem_count: u64) {
        self.branching_nodes += 1;
        self.total_potential_subproblems += subproblem_count;
    }
    #[inline]
    pub fn record_visit(&mut self) {
        self.total_visited_nodes += 1;
    }
}

pub struct SolverBuffer {
    pub queue: Vec<usize>,
    pub in_queue: Vec<bool>,
    pub occurrence_scores: Vec<f64>,
    /// Scratch word buffer for CT `updateTable` mask unions. Sized to the widest
    /// unique tensor so `mask_scratch[..n_words]` fits any tensor's support.
    pub mask_scratch: Vec<u64>,
    /// Per-tensor dirty-axis mask for CT delta-tracking. Bit `j` of `dirty[t]`
    /// means "axis `j` of tensor `t` has a changed variable awaiting
    /// `updateTable`". Arity <= 32, so a `u32` suffices. Invariant:
    /// `dirty[t] == 0` whenever `!in_queue[t]`.
    pub dirty: Vec<u32>,
}

impl SolverBuffer {
    /// Drop any queued work: clear the worklist and every `in_queue` flag.
    /// (`dirty` needs no reset — `dirty[t] == 0` whenever `!in_queue[t]`.)
    pub fn reset_worklist(&mut self) {
        self.queue.clear();
        for b in self.in_queue.iter_mut() {
            *b = false;
        }
    }

    pub fn new(cn: &ConstraintNetwork) -> SolverBuffer {
        let n_tensors = cn.tensors.len();
        let n_vars = cn.n_vars;
        let max_n_words = cn
            .truth_tables
            .iter()
            .map(|td| (td.support.len() + 63) / 64)
            .max()
            .unwrap_or(0);
        SolverBuffer {
            queue: Vec::with_capacity(n_tensors),
            in_queue: vec![false; n_tensors],
            occurrence_scores: vec![0.0; n_vars],
            mask_scratch: vec![0u64; max_n_words],
            dirty: vec![0u32; n_tensors],
        }
    }
}

impl Default for SolverBuffer {
    /// All-empty placeholder used only as the swap partner in the measure-scratch
    /// (never used for propagation until a real, sized buffer is swapped in).
    fn default() -> Self {
        SolverBuffer {
            queue: Vec::new(),
            in_queue: Vec::new(),
            occurrence_scores: Vec::new(),
            mask_scratch: Vec::new(),
            dirty: Vec::new(),
        }
    }
}

pub struct TnProblem {
    pub static_cn: Arc<ConstraintNetwork>,
    pub doms: Vec<DomainMask>,
    pub tables: Vec<RSparseBitSet>,
    pub masks: Arc<Vec<TableMasks>>,
    /// The one trail spanning root propagation and the whole search. Its `epoch`
    /// must stay monotonic across both: each `RSparseBitSet` stamps `saved_epoch`
    /// with the trail epoch at which it last saved a word, and a fresh trail
    /// (epoch restarting at 1) would collide with those root-propagation stamps,
    /// causing `save_word` to skip trailing and `restore_to` to leak mutations.
    pub trail: Trail,
    pub stats: Stats,
    pub buffer: SolverBuffer,
}

impl TnProblem {
    pub fn count_unfixed(&self) -> usize {
        self.doms.iter().filter(|d| !d.is_fixed()).count()
    }
    pub fn is_solved(&self) -> bool {
        self.count_unfixed() == 0
    }

    pub fn from_network(static_cn: ConstraintNetwork) -> Result<TnProblem, &'static str> {
        let n_vars = static_cn.n_vars;
        let mut doms = vec![DomainMask::BOTH; n_vars];
        let mut buffer = SolverBuffer::new(&static_cn);
        let (masks, mut tables) = crate::ct::build_tables(&static_cn);
        // seed all tensors
        for t in 0..static_cn.tensors.len() {
            buffer.queue.push(t);
            buffer.in_queue[t] = true;
        }
        // Root propagation on the SEARCH trail: the root-propagated (doms,
        // tables) become the permanent base, so the undo entries are dropped
        // (`clear`) — but the trail's monotonic `epoch` is preserved so later
        // search scopes never collide with the `saved_epoch` stamps left here.
        let mut trail = Trail::new();
        trail.open();
        crate::ct::ct_propagate(
            &static_cn,
            &mut doms,
            &masks,
            &mut tables,
            &mut buffer,
            &mut trail,
        );
        if !crate::problem::has_contradiction(&doms) {
            // Root reductions, matching every search node (solver.rs): domination
            // (pure-literal generalization) then failed-literal probing. Their
            // fixes join the permanent base alongside root propagation's.
            crate::propagate::dominate_fixpoint(
                &static_cn,
                &mut doms,
                &masks,
                &mut tables,
                &mut buffer,
                &mut trail,
            );
            if !crate::problem::has_contradiction(&doms) {
                let pool = crate::selector::occurrence_pool(
                    &static_cn,
                    &doms,
                    &mut buffer,
                    &masks,
                    crate::selector::FAILED_LITERAL_POOL,
                );
                crate::propagate::failed_literal_fixpoint(
                    &static_cn,
                    &mut doms,
                    &masks,
                    &mut tables,
                    &mut buffer,
                    &mut trail,
                    &pool,
                );
            }
        }
        if crate::problem::has_contradiction(&doms) {
            return Err("initial propagation found a contradiction");
        }
        trail.clear();
        Ok(TnProblem {
            static_cn: Arc::new(static_cn),
            doms,
            tables,
            masks: Arc::new(masks),
            trail,
            stats: Stats::default(),
            buffer,
        })
    }
}

#[inline]
pub fn has_contradiction(doms: &[DomainMask]) -> bool {
    doms.iter().any(|d| *d == DomainMask::NONE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    #[test]
    fn from_network_runs_initial_propagation() {
        // unit clause (x0): tensor over [0] with dense [F,T] forces x0 = 1.
        let cn = setup_problem(1, vec![vec![0]], vec![vec![false, true]]);
        let p = TnProblem::from_network(cn).unwrap();
        assert_eq!(p.doms[0], DomainMask::D1);
        assert!(p.is_solved());
    }
}
