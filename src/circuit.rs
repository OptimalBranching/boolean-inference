//! Load a **structure-preserving** CircuitSAT instance — as emitted by
//! problem-reductions' `pred reduce --to CircuitSAT` — into a `ConstraintNetwork`.
//!
//! Why not DIMACS: flattening a circuit to CNF (Tseitin) destroys the gate
//! structure the region-contraction method exploits. CircuitSAT keeps every gate
//! as a `BooleanExpr` over named wires; we turn each gate into one constraint
//! tensor `out == eval(expr)` — mirroring Julia's
//! `setup_from_sat(CircuitSAT(...; use_constraints=true))`. A factoring
//! instance's `= N` pin arrives as bare `Const` gates, which become unit
//! constraints automatically (no special handling needed).
//!
//! Input JSON may be a bare CircuitSAT `data` object (`{circuit, variables}`), a
//! full problem (`{type, variant, data}`), or a `pred reduce` bundle
//! (`{source, target, path}` — the `target` is used).

use std::collections::HashMap;

use serde::Deserialize;
use thiserror::Error;

use crate::domain::DomainMask;
use crate::network::{setup_problem, ConstraintNetwork};

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("failed to parse CircuitSAT JSON: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("JSON has no CircuitSAT circuit/variables data")]
    NoData,
    #[error("gate references unknown variable {0:?}")]
    UnknownVar(String),
    #[error("gate must have exactly one output, found {0}")]
    BadOutputArity(usize),
    #[error("gate arity {0} exceeds the 32-variable tensor cap")]
    ArityCap(usize),
}

// --- problem-reductions' CircuitSAT JSON shape (serde-external-tagged enum) ---

#[derive(Debug, Deserialize)]
struct BooleanExpr {
    op: BooleanOp,
}

#[derive(Debug, Deserialize)]
enum BooleanOp {
    Var(String),
    Const(bool),
    Not(Box<BooleanExpr>),
    And(Vec<BooleanExpr>),
    Or(Vec<BooleanExpr>),
    Xor(Vec<BooleanExpr>),
}

#[derive(Debug, Deserialize)]
struct Assignment {
    expr: BooleanExpr,
    outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Circuit {
    assignments: Vec<Assignment>,
}

#[derive(Debug, Deserialize)]
struct CircuitSatData {
    circuit: Circuit,
    variables: Vec<String>,
}

/// Evaluate a gate expression given a lookup for its input wires.
fn eval(expr: &BooleanExpr, val: &impl Fn(&str) -> bool) -> bool {
    match &expr.op {
        BooleanOp::Var(n) => val(n),
        BooleanOp::Const(b) => *b,
        BooleanOp::Not(e) => !eval(e, val),
        BooleanOp::And(es) => es.iter().all(|e| eval(e, val)),
        BooleanOp::Or(es) => es.iter().any(|e| eval(e, val)),
        BooleanOp::Xor(es) => es.iter().fold(false, |acc, e| acc ^ eval(e, val)),
    }
}

/// Collect the distinct input-wire names referenced by `expr`, in first-seen order.
fn collect_vars<'a>(expr: &'a BooleanExpr, out: &mut Vec<&'a str>) {
    match &expr.op {
        BooleanOp::Var(n) => {
            if !out.contains(&n.as_str()) {
                out.push(n);
            }
        }
        BooleanOp::Const(_) => {}
        BooleanOp::Not(e) => collect_vars(e, out),
        BooleanOp::And(es) | BooleanOp::Or(es) | BooleanOp::Xor(es) => {
            for e in es {
                collect_vars(e, out);
            }
        }
    }
}

/// A CircuitSAT instance loaded into a `ConstraintNetwork`, with the wire-name
/// map needed to read named wires (e.g. factor bits) out of a solution.
pub struct CircuitProblem {
    pub network: ConstraintNetwork,
    /// Wire name -> original variable id (index into the CircuitSAT `variables`).
    pub name_to_orig: HashMap<String, usize>,
}

impl CircuitProblem {
    /// Value of a named wire in a solved assignment (`solution` over compressed
    /// vars). `None` if the wire was compressed out (appears in no gate) or is
    /// not fixed.
    pub fn wire_value(&self, solution: &[DomainMask], name: &str) -> Option<bool> {
        let orig = *self.name_to_orig.get(name)?;
        let cid = self.network.orig_to_new[orig]?;
        solution[cid].value()
    }
}

/// Pull the `{circuit, variables}` object out of a bare data object, a full
/// problem (`data` field), or a reduce bundle (`target` field).
fn extract_data(v: &serde_json::Value) -> Option<&serde_json::Value> {
    let v = v.get("target").unwrap_or(v);
    let v = v.get("data").unwrap_or(v);
    (v.get("circuit").is_some() && v.get("variables").is_some()).then_some(v)
}

/// Build a `ConstraintNetwork` from a CircuitSAT JSON string. Each gate
/// `out = expr(inputs...)` becomes one constraint tensor over `[inputs..., out]`
/// whose dense table is `true` exactly where `out == eval(expr)`.
pub fn network_from_circuit_sat(json: &str) -> Result<CircuitProblem, CircuitError> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let data_val = extract_data(&value).ok_or(CircuitError::NoData)?;
    let data: CircuitSatData = serde_json::from_value(data_val.clone())?;

    let name_to_orig: HashMap<String, usize> = data
        .variables
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let var_num = data.variables.len();

    let n_gates = data.circuit.assignments.len();
    let mut tensors_to_vars: Vec<Vec<usize>> = Vec::with_capacity(n_gates);
    let mut tensor_data: Vec<Vec<bool>> = Vec::with_capacity(n_gates);

    for a in &data.circuit.assignments {
        if a.outputs.len() != 1 {
            return Err(CircuitError::BadOutputArity(a.outputs.len()));
        }
        let out_name = a.outputs[0].as_str();

        // Input wires (distinct, first-seen order), excluding the output itself.
        let mut inputs: Vec<&str> = Vec::new();
        collect_vars(&a.expr, &mut inputs);
        inputs.retain(|&n| n != out_name);

        let arity = inputs.len() + 1;
        if arity > 32 {
            return Err(CircuitError::ArityCap(arity));
        }

        // var_axes: inputs in bits 0..n_in, output in bit n_in.
        let mut axes: Vec<usize> = Vec::with_capacity(arity);
        for &n in &inputs {
            axes.push(
                *name_to_orig
                    .get(n)
                    .ok_or_else(|| CircuitError::UnknownVar(n.to_string()))?,
            );
        }
        axes.push(
            *name_to_orig
                .get(out_name)
                .ok_or_else(|| CircuitError::UnknownVar(out_name.to_string()))?,
        );

        // Dense table: entry[config] = (out_bit == eval(expr, input bits)).
        let n_in = inputs.len();
        let table_len = 1usize << arity;
        let mut table = vec![false; table_len];
        for (config, slot) in table.iter_mut().enumerate() {
            let lookup = |name: &str| -> bool {
                match inputs.iter().position(|&x| x == name) {
                    Some(i) => (config >> i) & 1 == 1,
                    None => false, // expr only references collected inputs
                }
            };
            let out_bit = (config >> n_in) & 1 == 1;
            *slot = out_bit == eval(&a.expr, &lookup);
        }

        tensors_to_vars.push(axes);
        tensor_data.push(table);
    }

    let network = setup_problem(var_num, tensors_to_vars, tensor_data);
    Ok(CircuitProblem {
        network,
        name_to_orig,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_and_gate_builds_the_and_relation() {
        // c = a AND b, over wires [a, b, c].
        let json = r#"{
            "variables": ["a", "b", "c"],
            "circuit": { "assignments": [
                { "outputs": ["c"], "expr": { "op": { "And": [
                    { "op": { "Var": "a" } }, { "op": { "Var": "b" } }
                ] } } }
            ] }
        }"#;
        let p = network_from_circuit_sat(json).expect("load");
        assert_eq!(p.network.tensors.len(), 1);
        let t = &p.network.tensors[0];
        // a,b,c all used -> identity compression; axes = [a=0, b=1, c=2].
        assert_eq!(t.var_axes, vec![0, 1, 2]);
        // Support (config bit0=a, bit1=b, bit2=c): c == a&b -> {000,001,010,111}.
        let mut support = p.network.support(t).to_vec();
        support.sort_unstable();
        assert_eq!(support, vec![0, 1, 2, 7]);
    }

    #[test]
    fn const_gate_becomes_a_unit_constraint() {
        // s = Const(true): forces wire s = 1.
        let json = r#"{
            "variables": ["s"],
            "circuit": { "assignments": [
                { "outputs": ["s"], "expr": { "op": { "Const": true } } }
            ] }
        }"#;
        let p = network_from_circuit_sat(json).expect("load");
        let t = &p.network.tensors[0];
        assert_eq!(t.var_axes, vec![0]);
        // Over [s]: only s=1 satisfies -> support = {1}.
        assert_eq!(p.network.support(t), &[1]);
    }

    #[test]
    fn accepts_a_reduce_bundle_shape() {
        // Wrapped as a `pred reduce` bundle: {source, target:{type,variant,data}, path}.
        let json = r#"{
            "source": {},
            "path": [],
            "target": { "type": "CircuitSAT", "variant": {}, "data": {
                "variables": ["x", "y"],
                "circuit": { "assignments": [
                    { "outputs": ["y"], "expr": { "op": { "Var": "x" } } }
                ] }
            } }
        }"#;
        let p = network_from_circuit_sat(json).expect("load bundle");
        // y == x equality over [x, y] -> support {00, 11} = {0, 3}.
        let t = &p.network.tensors[0];
        let mut support = p.network.support(t).to_vec();
        support.sort_unstable();
        assert_eq!(support, vec![0, 3]);
    }
}
