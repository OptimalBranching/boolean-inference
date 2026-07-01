use std::sync::Arc;

use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;

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
    pub scratch_doms: Vec<DomainMask>,
    pub connection_scores: Vec<f64>,
    /// Scratch word buffer for CT `updateTable` mask unions. Sized to the widest
    /// unique tensor so `mask_scratch[..n_words]` fits any tensor's support.
    pub mask_scratch: Vec<u64>,
}

impl SolverBuffer {
    pub fn new(cn: &ConstraintNetwork) -> SolverBuffer {
        let n_tensors = cn.tensors.len();
        let n_vars = cn.vars.len();
        let max_n_words = cn
            .unique_tensors
            .iter()
            .map(|td| (td.support.len() + 63) / 64)
            .max()
            .unwrap_or(0);
        SolverBuffer {
            queue: Vec::with_capacity(n_tensors),
            in_queue: vec![false; n_tensors],
            scratch_doms: vec![DomainMask::BOTH; n_vars],
            connection_scores: vec![0.0; n_vars],
            mask_scratch: vec![0u64; max_n_words],
        }
    }
}

pub struct TnProblem {
    pub static_cn: Arc<ConstraintNetwork>,
    pub doms: Vec<DomainMask>,
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
        let n_vars = static_cn.vars.len();
        let mut doms = vec![DomainMask::BOTH; n_vars];
        let mut buffer = SolverBuffer::new(&static_cn);
        // seed all tensors
        for t in 0..static_cn.tensors.len() {
            buffer.queue.push(t);
            buffer.in_queue[t] = true;
        }
        crate::propagate::propagate_core(&static_cn, &mut doms, &mut buffer);
        if crate::problem::has_contradiction(&doms) {
            return Err("initial propagation found a contradiction");
        }
        Ok(TnProblem {
            static_cn: Arc::new(static_cn),
            doms,
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
