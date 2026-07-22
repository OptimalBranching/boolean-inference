//! Representation-level diagnostics for a conditioned residual problem.
//!
//! These features deliberately stop at GAC.  They describe what follows from
//! the emitted cube in a native relation network or, for a DIMACS clause
//! network, from ordinary unit propagation.  Search-only domination and
//! failed-literal choices are excluded because the cuber does not serialize
//! those choices as downstream assumptions.

use std::collections::VecDeque;

use serde::Serialize;

use crate::ct::{RSparseBitSet, TableMasks};
use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;
use crate::problem::has_contradiction;
use crate::util::is_entailed;

#[derive(Clone, Debug, Serialize)]
pub struct ResidualDiagnostics {
    pub contradiction: bool,
    pub variables: usize,
    pub fixed_variables: usize,
    pub unfixed_variables: usize,
    pub active_tensors: usize,
    pub entailed_tensors: usize,
    pub constrained_variables: usize,
    pub free_variables: usize,
    pub constrained_components: usize,
    pub components_including_free: usize,
    pub largest_component_variables: usize,
    pub largest_component_tensors: usize,
    pub component_variables_p50: Option<f64>,
    pub component_variables_p95: Option<f64>,
    pub component_tensors_p50: Option<f64>,
    pub component_tensors_p95: Option<f64>,
    pub active_incidence_edges: usize,
    pub active_degree_mean: Option<f64>,
    pub active_degree_p95: Option<f64>,
    pub active_degree_max: usize,
    pub residual_arity_mean: Option<f64>,
    pub residual_arity_p95: Option<f64>,
    pub residual_arity_max: usize,
    pub live_rows_total: usize,
    pub live_rows_mean: Option<f64>,
    pub live_rows_p95: Option<f64>,
    pub live_rows_max: usize,
    pub tensor_compression_mean_bits: Option<f64>,
    pub tensor_compression_p50_bits: Option<f64>,
}

fn quantile(values: &[usize], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((q * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index] as f64)
}

fn median_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn mean_usize(values: &[usize]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<usize>() as f64 / values.len() as f64)
}

pub fn diagnose(
    cn: &ConstraintNetwork,
    doms: &[DomainMask],
    masks: &[TableMasks],
    tables: &[RSparseBitSet],
) -> ResidualDiagnostics {
    let contradiction = has_contradiction(doms);
    let unfixed_variables = doms.iter().filter(|domain| !domain.is_fixed()).count();
    let fixed_variables = doms.len().saturating_sub(unfixed_variables);
    if contradiction {
        return ResidualDiagnostics {
            contradiction,
            variables: doms.len(),
            fixed_variables,
            unfixed_variables,
            active_tensors: 0,
            entailed_tensors: cn.tensors.len(),
            constrained_variables: 0,
            free_variables: unfixed_variables,
            constrained_components: 0,
            components_including_free: unfixed_variables,
            largest_component_variables: usize::from(unfixed_variables > 0),
            largest_component_tensors: 0,
            component_variables_p50: None,
            component_variables_p95: None,
            component_tensors_p50: None,
            component_tensors_p95: None,
            active_incidence_edges: 0,
            active_degree_mean: None,
            active_degree_p95: None,
            active_degree_max: 0,
            residual_arity_mean: None,
            residual_arity_p95: None,
            residual_arity_max: 0,
            live_rows_total: 0,
            live_rows_mean: None,
            live_rows_p95: None,
            live_rows_max: 0,
            tensor_compression_mean_bits: None,
            tensor_compression_p50_bits: None,
        };
    }

    let active: Vec<bool> = (0..cn.tensors.len())
        .map(|tensor| !is_entailed(cn, tensor, doms, masks))
        .collect();
    let active_tensors = active.iter().filter(|&&value| value).count();
    let entailed_tensors = active.len() - active_tensors;

    let mut degrees = vec![0usize; doms.len()];
    let mut residual_arities = Vec::with_capacity(active_tensors);
    let mut live_rows = Vec::with_capacity(active_tensors);
    let mut tensor_compression = Vec::with_capacity(active_tensors);
    for (tensor_id, tensor) in cn.tensors.iter().enumerate() {
        if !active[tensor_id] {
            continue;
        }
        let arity = tensor
            .var_axes
            .iter()
            .filter(|&&variable| !doms[variable].is_fixed())
            .count();
        residual_arities.push(arity);
        for &variable in &tensor.var_axes {
            if !doms[variable].is_fixed() {
                degrees[variable] += 1;
            }
        }
        let rows = tables[tensor_id]
            .words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        live_rows.push(rows);
        if rows > 0 {
            tensor_compression.push(arity as f64 - (rows as f64).log2());
        }
    }

    let active_degrees: Vec<usize> = degrees
        .iter()
        .copied()
        .filter(|&degree| degree > 0)
        .collect();
    let constrained_variables = active_degrees.len();
    let free_variables = unfixed_variables.saturating_sub(constrained_variables);

    let mut seen_variable = vec![false; doms.len()];
    let mut seen_tensor = vec![false; cn.tensors.len()];
    let mut component_variables = Vec::new();
    let mut component_tensors = Vec::new();
    for start in 0..doms.len() {
        if doms[start].is_fixed() || degrees[start] == 0 || seen_variable[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        seen_variable[start] = true;
        let mut variables = 0usize;
        let mut tensors = 0usize;
        while let Some(variable) = queue.pop_front() {
            variables += 1;
            for &tensor_id in &cn.v2t[variable] {
                if !active[tensor_id] || seen_tensor[tensor_id] {
                    continue;
                }
                seen_tensor[tensor_id] = true;
                tensors += 1;
                for &neighbor in &cn.tensors[tensor_id].var_axes {
                    if !doms[neighbor].is_fixed() && !seen_variable[neighbor] {
                        seen_variable[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        component_variables.push(variables);
        component_tensors.push(tensors);
    }

    let compression_mean = (!tensor_compression.is_empty())
        .then(|| tensor_compression.iter().sum::<f64>() / tensor_compression.len() as f64);
    ResidualDiagnostics {
        contradiction,
        variables: doms.len(),
        fixed_variables,
        unfixed_variables,
        active_tensors,
        entailed_tensors,
        constrained_variables,
        free_variables,
        constrained_components: component_variables.len(),
        components_including_free: component_variables.len() + free_variables,
        largest_component_variables: component_variables.iter().copied().max().unwrap_or(0),
        largest_component_tensors: component_tensors.iter().copied().max().unwrap_or(0),
        component_variables_p50: quantile(&component_variables, 0.50),
        component_variables_p95: quantile(&component_variables, 0.95),
        component_tensors_p50: quantile(&component_tensors, 0.50),
        component_tensors_p95: quantile(&component_tensors, 0.95),
        active_incidence_edges: residual_arities.iter().sum(),
        active_degree_mean: mean_usize(&active_degrees),
        active_degree_p95: quantile(&active_degrees, 0.95),
        active_degree_max: active_degrees.iter().copied().max().unwrap_or(0),
        residual_arity_mean: mean_usize(&residual_arities),
        residual_arity_p95: quantile(&residual_arities, 0.95),
        residual_arity_max: residual_arities.iter().copied().max().unwrap_or(0),
        live_rows_total: live_rows.iter().sum(),
        live_rows_mean: mean_usize(&live_rows),
        live_rows_p95: quantile(&live_rows, 0.95),
        live_rows_max: live_rows.iter().copied().max().unwrap_or(0),
        tensor_compression_mean_bits: compression_mean,
        tensor_compression_p50_bits: median_f64(&tensor_compression),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;
    use crate::problem::TnProblem;

    #[test]
    fn separates_constrained_components_from_entailed_free_variables() {
        let xor = vec![true, false, false, true];
        let full = vec![true, true];
        let network = setup_problem(
            5,
            vec![vec![0, 1], vec![2, 3], vec![4]],
            vec![xor.clone(), xor, full],
        );
        let problem = TnProblem::from_network_gac(network).unwrap();
        let result = diagnose(
            &problem.static_cn,
            &problem.doms,
            &problem.masks,
            &problem.tables,
        );
        assert_eq!(result.unfixed_variables, 5);
        assert_eq!(result.active_tensors, 2);
        assert_eq!(result.entailed_tensors, 1);
        assert_eq!(result.constrained_components, 2);
        assert_eq!(result.free_variables, 1);
        assert_eq!(result.components_including_free, 3);
        assert_eq!(result.largest_component_variables, 2);
        assert_eq!(result.largest_component_tensors, 1);
        assert_eq!(result.live_rows_total, 4);
        assert_eq!(result.tensor_compression_mean_bits, Some(1.0));
    }
}
