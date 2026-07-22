//! Tail-aware variant of `optimal_branching_core::GreedyMerge`.
//!
//! The ordinary merge objective may trade one very weak merged child for many
//! strong children.  That trade can improve the aggregate branching factor but
//! hurt finite-worker Cube-and-Conquer makespan.  `TailGreedyMerge` keeps the
//! weakest-child quality of the unmerged table as a parameter-free guard: the
//! minimum size reduction among the initial full-row branches is frozen, and a
//! merge whose measured reduction falls below that floor is never considered.
//!
//! The implementation intentionally mirrors upstream GreedyMerge, including
//! its per-clause size-reduction memo.  The hot-path difference is one floating
//! point comparison per candidate merge; rejected weak merges also avoid later
//! queue and propagation work.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use optimal_branching_core::branching_table::gather2;
use optimal_branching_core::{
    bit_clauses, complexity_bv, BranchAndReduceProblem, BranchingRuleSolver, BranchingTable,
    Clause, Error, Measure, OptimalBranchingResult, DNF,
};

const ENERGY_EPSILON: f64 = 1e-12;
const FLOOR_EPSILON: f64 = 1e-12;

/// Greedy row merging that never creates a child weaker than the weakest
/// unmerged table row under the selected problem measure.
#[derive(Debug, Clone, Copy, Default)]
pub struct TailGreedyMerge;

impl BranchingRuleSolver for TailGreedyMerge {
    fn optimal_branching_rule<P, M>(
        &self,
        problem: &P,
        table: &BranchingTable,
        variables: &[usize],
        measure: &M,
    ) -> Result<OptimalBranchingResult, Error>
    where
        P: BranchAndReduceProblem,
        M: Measure<P>,
    {
        let candidates = bit_clauses(table);
        Ok(tail_greedymerge(&candidates, problem, variables, measure))
    }
}

/// Merge a branching table while preserving its initial minimum reduction.
pub fn tail_greedymerge<P, M>(
    clauses: &[Vec<Clause>],
    problem: &P,
    variables: &[usize],
    measure: &M,
) -> OptimalBranchingResult
where
    P: BranchAndReduceProblem,
    M: Measure<P>,
{
    greedymerge_with_floor(clauses, problem, variables, measure, true)
}

fn greedymerge_with_floor<P, M>(
    clauses: &[Vec<Clause>],
    problem: &P,
    variables: &[usize],
    measure: &M,
    guard_tail: bool,
) -> OptimalBranchingResult
where
    P: BranchAndReduceProblem,
    M: Measure<P>,
{
    let nvars = variables.len();
    let before = measure.measure(problem);
    let memo: RefCell<HashMap<(u64, u64), M::Output>> = RefCell::new(HashMap::new());
    let size_reduction = |clause: &Clause| -> M::Output {
        let key = (clause.mask, clause.val);
        if let Some(&value) = memo.borrow().get(&key) {
            return value;
        }
        let (subproblem, _) = problem.apply_branch(clause, variables);
        let value = before - measure.measure(&subproblem);
        memo.borrow_mut().insert(key, value);
        value
    };

    let reduction_merge = |left: &[Clause], right: &[Clause]| -> (Clause, f64) {
        let mut best_clause = Clause::new(0, 0);
        let mut best_reduction = 0.0_f64;
        for a in left {
            for b in right {
                let merged = gather2(nvars, a, b);
                if merged.mask == 0 {
                    continue;
                }
                let reduction: f64 = size_reduction(&merged).into();
                if reduction > best_reduction {
                    best_clause = merged;
                    best_reduction = reduction;
                }
            }
        }
        (best_clause, best_reduction)
    };

    let mut clauses: Vec<Vec<Clause>> = clauses.to_vec();
    let mut reductions: Vec<f64> = clauses
        .iter()
        .map(|group| size_reduction(&group[0]).into())
        .collect();
    let tail_floor = if guard_tail {
        reductions.iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        f64::NEG_INFINITY
    };
    let tail_safe = |reduction: f64| reduction + FLOOR_EPSILON >= tail_floor;

    loop {
        let count = clauses.len();
        let mut active = vec![true; count];
        let gamma = complexity_bv(&reductions);
        let mut weights: Vec<f64> = reductions
            .iter()
            .map(|&reduction| gamma.powf(-reduction))
            .collect();

        let mut queue = PairQueue::new();
        for i in 0..count {
            for j in (i + 1)..count {
                let (_, reduction) = reduction_merge(&clauses[i], &clauses[j]);
                if !tail_safe(reduction) {
                    continue;
                }
                let delta_energy = gamma.powf(-reduction) - weights[i] - weights[j];
                if delta_energy <= -ENERGY_EPSILON {
                    queue.upsert((i, j), delta_energy);
                }
            }
        }

        if queue.is_empty() {
            return OptimalBranchingResult {
                optimal_rule: DNF {
                    clauses: clauses.iter().map(|group| group[0]).collect(),
                },
                branching_vector: reductions,
                gamma,
            };
        }

        while let Some((i, j)) = queue.pop_min() {
            for row in [i, j] {
                active[row] = false;
                for (other, &is_active) in active.iter().enumerate() {
                    if is_active {
                        queue.remove(ordered_pair(row, other));
                    }
                }
            }

            active[i] = true;
            let (merged, reduction) = reduction_merge(&clauses[i], &clauses[j]);
            debug_assert!(tail_safe(reduction));
            reductions[i] = reduction;
            clauses[i] = vec![merged];
            weights[i] = gamma.powf(-reduction);

            for (other, &is_active) in active.iter().enumerate() {
                if i == other || !is_active {
                    continue;
                }
                let (a, b) = ordered_pair(i, other);
                let (_, candidate_reduction) = reduction_merge(&clauses[a], &clauses[b]);
                if !tail_safe(candidate_reduction) {
                    continue;
                }
                let delta_energy = gamma.powf(-candidate_reduction) - weights[a] - weights[b];
                if delta_energy <= -ENERGY_EPSILON {
                    queue.upsert((a, b), delta_energy);
                }
            }
        }

        let mut next_clauses = Vec::with_capacity(count);
        let mut next_reductions = Vec::with_capacity(count);
        for index in 0..count {
            if active[index] {
                next_clauses.push(std::mem::take(&mut clauses[index]));
                next_reductions.push(reductions[index]);
            }
        }
        clauses = next_clauses;
        reductions = next_reductions;
    }
}

#[inline]
fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

struct PairQueue {
    by_priority: BTreeSet<(OrdF64, usize, usize)>,
    priority_of: HashMap<(usize, usize), OrdF64>,
}

impl PairQueue {
    fn new() -> Self {
        Self {
            by_priority: BTreeSet::new(),
            priority_of: HashMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.priority_of.is_empty()
    }

    fn upsert(&mut self, key: (usize, usize), priority: f64) {
        let priority = OrdF64(priority);
        if let Some(old) = self.priority_of.insert(key, priority) {
            self.by_priority.remove(&(old, key.0, key.1));
        }
        self.by_priority.insert((priority, key.0, key.1));
    }

    fn remove(&mut self, key: (usize, usize)) {
        if let Some(old) = self.priority_of.remove(&key) {
            self.by_priority.remove(&(old, key.0, key.1));
        }
    }

    fn pop_min(&mut self) -> Option<(usize, usize)> {
        let &(priority, i, j) = self.by_priority.iter().next()?;
        self.by_priority.remove(&(priority, i, j));
        self.priority_of.remove(&(i, j));
        Some((i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimal_branching_core::mock::{MockProblem, NumOfVariables};
    use optimal_branching_core::{GreedyMerge, DNF};

    fn three_of_four_table() -> BranchingTable {
        BranchingTable::new(2, vec![vec![0b00], vec![0b01], vec![0b10]])
    }

    #[test]
    fn tail_guard_preserves_the_unmerged_weakest_child() {
        let problem = MockProblem {
            optimal: vec![false, false],
        };
        let table = three_of_four_table();
        let variables = [0usize, 1];

        let ordinary = GreedyMerge
            .optimal_branching_rule(&problem, &table, &variables, &NumOfVariables)
            .unwrap();
        assert!(ordinary.branching_vector.iter().any(|&value| value < 2.0));

        let guarded = TailGreedyMerge
            .optimal_branching_rule(&problem, &table, &variables, &NumOfVariables)
            .unwrap();
        assert!(guarded.branching_vector.iter().all(|&value| value >= 2.0));
        assert!(table.covered_by(&guarded.optimal_rule));
    }

    #[test]
    fn unguarded_copy_matches_upstream_greedymerge() {
        let problem = MockProblem {
            optimal: vec![false, true, false],
        };
        let table =
            BranchingTable::new(3, vec![vec![0b000], vec![0b001], vec![0b010], vec![0b100]]);
        let variables = [0usize, 1, 2];
        let upstream = GreedyMerge
            .optimal_branching_rule(&problem, &table, &variables, &NumOfVariables)
            .unwrap();
        let copied = greedymerge_with_floor(
            &bit_clauses(&table),
            &problem,
            &variables,
            &NumOfVariables,
            false,
        );
        assert_eq!(copied.optimal_rule, upstream.optimal_rule);
        assert_eq!(copied.branching_vector, upstream.branching_vector);
        assert_eq!(copied.gamma, upstream.gamma);
        assert!(table.covered_by(&DNF {
            clauses: copied.optimal_rule.clauses,
        }));
    }
}
