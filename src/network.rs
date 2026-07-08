/// Shared truth-table content for a constraint (flyweight; deduplicated).
#[derive(Clone, Debug)]
pub struct TruthTable {
    /// Satisfied configs (0-indexed, ascending), the sparse "support". This is the
    /// sole stored representation — no dense truth table is kept.
    pub support: Vec<u32>,
    /// OR over all support configs (fast feasibility scan).
    pub support_or: u32,
    /// AND over all support configs (fast feasibility scan).
    pub support_and: u32,
}

impl TruthTable {
    /// Construct from an ascending list of satisfying configs. Derives the OR/AND
    /// aggregates; `support` is stored as-is and must be strictly ascending (the
    /// `is_sat` binary search relies on it).
    pub fn from_support(support: Vec<u32>) -> TruthTable {
        debug_assert!(
            support.windows(2).all(|w| w[0] < w[1]),
            "support must be strictly ascending"
        );
        let mut support_or: u32 = 0;
        let mut support_and: u32 = 0xFFFF_FFFF;
        for &config in &support {
            support_or |= config;
            support_and &= config;
        }
        TruthTable {
            support,
            support_or,
            support_and,
        }
    }

    /// Construct from a dense truth table: derive the (ascending) support and discard
    /// the dense table — it is never stored.
    pub fn from_dense(dense: Vec<bool>) -> TruthTable {
        TruthTable::from_support(dense_to_support(&dense))
    }
}

use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Variable {
    pub deg: usize,
}

#[derive(Clone, Debug)]
pub struct Constraint {
    /// Variable ids on each axis; bit `i` of a config is `var_axes[i]`.
    pub var_axes: Vec<usize>,
    pub table_idx: usize,
}

#[derive(Clone, Debug)]
pub struct ConstraintNetwork {
    pub vars: Vec<Variable>,
    pub truth_tables: Vec<TruthTable>,
    pub tensors: Vec<Constraint>,
    /// variable -> tensor incidence (compressed var ids).
    pub v2t: Vec<Vec<usize>>,
    /// original var id -> compressed var id (None if removed).
    pub orig_to_new: Vec<Option<usize>>,
}

impl ConstraintNetwork {
    #[inline]
    pub fn table(&self, t: &Constraint) -> &TruthTable {
        &self.truth_tables[t.table_idx]
    }
    #[inline]
    pub fn support(&self, t: &Constraint) -> &[u32] {
        &self.table(t).support
    }
    #[inline]
    pub fn support_or(&self, t: &Constraint) -> u32 {
        self.table(t).support_or
    }
    #[inline]
    pub fn support_and(&self, t: &Constraint) -> u32 {
        self.table(t).support_and
    }
    /// True iff `config` (a bitmask over `t.var_axes`) satisfies the constraint.
    /// Sparse membership on the ascending support — replaces dense-table indexing.
    #[inline]
    pub fn is_sat(&self, t: &Constraint, config: u32) -> bool {
        self.support(t).binary_search(&config).is_ok()
    }
}

/// Ascending support (indices of satisfied configs) of a dense truth table.
fn dense_to_support(dense: &[bool]) -> Vec<u32> {
    dense
        .iter()
        .enumerate()
        .filter_map(|(i, &sat)| if sat { Some(i as u32) } else { None })
        .collect()
}

/// Build a `ConstraintNetwork` from raw dense tensor specs. `var_num` is the number
/// of original variables (0-based ids `0..var_num`). `tensor_data[i]` has length
/// `2^tensors_to_vars[i].len()`.
pub fn setup_problem(
    var_num: usize,
    tensors_to_vars: Vec<Vec<usize>>,
    tensor_data: Vec<Vec<bool>>,
) -> ConstraintNetwork {
    assert_eq!(tensors_to_vars.len(), tensor_data.len());
    let tensors_in: Vec<(Vec<usize>, Vec<u32>)> = tensors_to_vars
        .into_iter()
        .zip(tensor_data)
        .map(|(var_axes, dense)| {
            assert_eq!(
                dense.len(),
                1usize << var_axes.len(),
                "tensor_data size mismatch"
            );
            (var_axes, dense_to_support(&dense))
        })
        .collect();
    assemble(var_num, tensors_in)
}

/// Shared assembly from `(var_axes, support)` tensors: dedup `TruthTable`, compress
/// out unused variables, remap axes to compressed ids, build `v2t`. Dedup key is
/// `(var_axes.len(), support)` — support alone is arity-ambiguous.
pub(crate) fn assemble(
    var_num: usize,
    tensors_in: Vec<(Vec<usize>, Vec<u32>)>,
) -> ConstraintNetwork {
    let f = tensors_in.len();
    let mut tensors: Vec<Constraint> = Vec::with_capacity(f);
    let mut vars_to_tensors: Vec<Vec<usize>> = vec![Vec::new(); var_num];
    let mut unique_data: Vec<TruthTable> = Vec::new();
    let mut data_to_idx: HashMap<(usize, Vec<u32>), usize> = HashMap::new();

    for (i, (var_axes, support)) in tensors_in.into_iter().enumerate() {
        assert!(var_axes.len() <= 32, "tensor arity exceeds 32-var cap");
        debug_assert!(
            {
                let mut s = var_axes.clone();
                s.sort_unstable();
                s.dedup();
                s.len() == var_axes.len()
            },
            "CT precondition: tensor var_axes must be distinct"
        );
        let key = (var_axes.len(), support.clone());
        let table_idx = match data_to_idx.get(&key) {
            Some(&idx) => idx,
            None => {
                let idx = unique_data.len();
                data_to_idx.insert(key, idx);
                unique_data.push(TruthTable::from_support(support));
                idx
            }
        };
        for &v in &var_axes {
            vars_to_tensors[v].push(i);
        }
        tensors.push(Constraint {
            var_axes,
            table_idx,
        });
    }

    // Compress out variables that appear in no tensor.
    let mut orig_to_new: Vec<Option<usize>> = vec![None; var_num];
    let mut next_id = 0usize;
    for v in 0..var_num {
        if !vars_to_tensors[v].is_empty() {
            orig_to_new[v] = Some(next_id);
            next_id += 1;
        }
    }

    for t in tensors.iter_mut() {
        for axis in t.var_axes.iter_mut() {
            *axis = orig_to_new[*axis].expect("tensor references a compressed-out variable");
        }
    }

    let mut new_v2t: Vec<Vec<usize>> = vec![Vec::new(); next_id];
    for (tid, t) in tensors.iter().enumerate() {
        for &v in &t.var_axes {
            new_v2t[v].push(tid);
        }
    }

    let vars: Vec<Variable> = new_v2t
        .iter()
        .map(|ts| Variable { deg: ts.len() })
        .collect();

    ConstraintNetwork {
        vars,
        truth_tables: unique_data,
        tensors,
        v2t: new_v2t,
        orig_to_new,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensordata_extracts_support_and_aggregates() {
        // 2-var tensor, satisfied configs = {01, 11} (i.e. index 1 and 3)
        let dense = vec![false, true, false, true];
        let td = TruthTable::from_dense(dense);
        assert_eq!(td.support, vec![1u32, 3u32]);
        assert_eq!(td.support_or, 0b11); // 1 | 3
        assert_eq!(td.support_and, 0b01); // 1 & 3
    }

    #[test]
    fn tensordata_empty_support_aggregates() {
        let td = TruthTable::from_dense(vec![false, false]);
        assert!(td.support.is_empty());
        assert_eq!(td.support_or, 0);
        assert_eq!(td.support_and, 0xFFFF_FFFF);
    }

    #[test]
    fn setup_problem_dedups_and_builds_incidence() {
        // 3 vars (0,1,2). Two tensors with identical truth tables share TruthTable.
        // tensor 0 over vars [0,1], tensor 1 over vars [1,2], same dense table.
        let dense = vec![false, true, true, true]; // OR of 2 literals
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![dense.clone(), dense.clone()],
        );
        assert_eq!(cn.tensors.len(), 2);
        assert_eq!(cn.truth_tables.len(), 1); // deduplicated
        assert_eq!(cn.tensors[0].table_idx, cn.tensors[1].table_idx);
        // incidence: var 1 is in both tensors
        assert_eq!(cn.v2t[1], vec![0, 1]);
        assert_eq!(cn.v2t[0], vec![0]);
        assert_eq!(cn.v2t[2], vec![1]);
        assert_eq!(cn.vars.len(), 3);
        assert_eq!(cn.vars[1].deg, 2);
    }

    #[test]
    fn from_support_aggregates_match_from_dense() {
        // dense {false,true,false,true} -> support {1,3}.
        let a = TruthTable::from_dense(vec![false, true, false, true]);
        let b = TruthTable::from_support(vec![1u32, 3u32]);
        assert_eq!(a.support, b.support);
        assert_eq!(a.support_or, b.support_or);
        assert_eq!(a.support_and, b.support_and);
    }

    #[test]
    fn setup_problem_compresses_unused_vars() {
        // var 1 appears in no tensor -> compressed out; var 2 remaps to id 1.
        let dense = vec![false, true]; // 1-var tensor, satisfied when var=1
        let cn = setup_problem(3, vec![vec![0], vec![2]], vec![dense.clone(), dense]);
        assert_eq!(cn.vars.len(), 2); // var 1 dropped
        assert_eq!(cn.orig_to_new[0], Some(0));
        assert_eq!(cn.orig_to_new[1], None);
        assert_eq!(cn.orig_to_new[2], Some(1));
        // the second tensor now references compressed var id 1
        assert_eq!(cn.tensors[1].var_axes, vec![1]);
    }
}
