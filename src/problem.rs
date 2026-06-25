use rustc_hash::FxHashMap; // P2: fast integer-keyed map

use optimal_branching_core::Clause;

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
    pub branching_cache: FxHashMap<Clause, f64>,
}

impl SolverBuffer {
    pub fn new(cn: &ConstraintNetwork) -> SolverBuffer {
        let n_tensors = cn.tensors.len();
        let n_vars = cn.vars.len();
        SolverBuffer {
            queue: Vec::with_capacity(n_tensors),
            in_queue: vec![false; n_tensors],
            scratch_doms: vec![DomainMask::BOTH; n_vars],
            connection_scores: vec![0.0; n_vars],
            branching_cache: FxHashMap::default(),
        }
    }
}

pub struct TnProblem {
    pub static_cn: ConstraintNetwork,
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
}

#[inline]
pub fn has_contradiction(doms: &[DomainMask]) -> bool {
    doms.iter().any(|d| *d == DomainMask::NONE)
}
