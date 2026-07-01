/// Shared truth-table data for a constraint tensor (flyweight; deduplicated).
#[derive(Clone, Debug)]
pub struct TensorData {
    /// dense[config] == true iff `config` satisfies the constraint.
    pub dense: Vec<bool>,
    /// Satisfied configs (0-indexed), the sparse "support".
    pub support: Vec<u32>,
    /// OR over all support configs (fast feasibility scan).
    pub support_or: u32,
    /// AND over all support configs (fast feasibility scan).
    pub support_and: u32,
}

impl TensorData {
    pub fn from_dense(dense: Vec<bool>) -> TensorData {
        let mut support = Vec::new();
        let mut support_or: u32 = 0;
        let mut support_and: u32 = 0xFFFF_FFFF;
        for (i, &sat) in dense.iter().enumerate() {
            if sat {
                let config = i as u32;
                support.push(config);
                support_or |= config;
                support_and &= config;
            }
        }
        TensorData {
            dense,
            support,
            support_or,
            support_and,
        }
    }
}

use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Variable {
    pub deg: usize,
}

#[derive(Clone, Debug)]
pub struct BoolTensor {
    /// Variable ids on each axis; bit `i` of a config is `var_axes[i]`.
    pub var_axes: Vec<usize>,
    pub data_idx: usize,
}

#[derive(Clone, Debug)]
pub struct ConstraintNetwork {
    pub vars: Vec<Variable>,
    pub unique_tensors: Vec<TensorData>,
    pub tensors: Vec<BoolTensor>,
    /// variable -> tensor incidence (compressed var ids).
    pub v2t: Vec<Vec<usize>>,
    /// original var id -> compressed var id (None if removed).
    pub orig_to_new: Vec<Option<usize>>,
}

impl ConstraintNetwork {
    #[inline]
    pub fn data(&self, t: &BoolTensor) -> &TensorData {
        &self.unique_tensors[t.data_idx]
    }
    #[inline]
    pub fn support(&self, t: &BoolTensor) -> &[u32] {
        &self.data(t).support
    }
    #[inline]
    pub fn support_or(&self, t: &BoolTensor) -> u32 {
        self.data(t).support_or
    }
    #[inline]
    pub fn support_and(&self, t: &BoolTensor) -> u32 {
        self.data(t).support_and
    }
    #[inline]
    pub fn dense<'a>(&'a self, t: &BoolTensor) -> &'a [bool] {
        &self.data(t).dense
    }
}

/// Build a `ConstraintNetwork` from raw tensor specs. `var_num` is the number of
/// original variables (0-based ids `0..var_num`). `tensor_data[i]` has length
/// `2^tensors_to_vars[i].len()`.
pub fn setup_problem(
    var_num: usize,
    tensors_to_vars: Vec<Vec<usize>>,
    tensor_data: Vec<Vec<bool>>,
) -> ConstraintNetwork {
    let f = tensors_to_vars.len();
    assert_eq!(f, tensor_data.len());

    let mut tensors: Vec<BoolTensor> = Vec::with_capacity(f);
    let mut vars_to_tensors: Vec<Vec<usize>> = vec![Vec::new(); var_num];
    let mut unique_data: Vec<TensorData> = Vec::new();
    let mut data_to_idx: HashMap<Vec<bool>, usize> = HashMap::new();

    for i in 0..f {
        let var_axes = &tensors_to_vars[i];
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
        assert_eq!(
            tensor_data[i].len(),
            1usize << var_axes.len(),
            "tensor_data size mismatch"
        );

        let data_idx = match data_to_idx.get(&tensor_data[i]) {
            Some(&idx) => idx,
            None => {
                let idx = unique_data.len();
                data_to_idx.insert(tensor_data[i].clone(), idx);
                unique_data.push(TensorData::from_dense(tensor_data[i].clone()));
                idx
            }
        };

        tensors.push(BoolTensor {
            var_axes: var_axes.clone(),
            data_idx,
        });
        for &v in var_axes {
            vars_to_tensors[v].push(i);
        }
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
        unique_tensors: unique_data,
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
        let td = TensorData::from_dense(dense);
        assert_eq!(td.support, vec![1u32, 3u32]);
        assert_eq!(td.support_or, 0b11); // 1 | 3
        assert_eq!(td.support_and, 0b01); // 1 & 3
    }

    #[test]
    fn tensordata_empty_support_aggregates() {
        let td = TensorData::from_dense(vec![false, false]);
        assert!(td.support.is_empty());
        assert_eq!(td.support_or, 0);
        assert_eq!(td.support_and, 0xFFFF_FFFF);
    }

    #[test]
    fn setup_problem_dedups_and_builds_incidence() {
        // 3 vars (0,1,2). Two tensors with identical truth tables share TensorData.
        // tensor 0 over vars [0,1], tensor 1 over vars [1,2], same dense table.
        let dense = vec![false, true, true, true]; // OR of 2 literals
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![dense.clone(), dense.clone()],
        );
        assert_eq!(cn.tensors.len(), 2);
        assert_eq!(cn.unique_tensors.len(), 1); // deduplicated
        assert_eq!(cn.tensors[0].data_idx, cn.tensors[1].data_idx);
        // incidence: var 1 is in both tensors
        assert_eq!(cn.v2t[1], vec![0, 1]);
        assert_eq!(cn.v2t[0], vec![0]);
        assert_eq!(cn.v2t[2], vec![1]);
        assert_eq!(cn.vars.len(), 3);
        assert_eq!(cn.vars[1].deg, 2);
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
