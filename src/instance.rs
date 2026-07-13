//! The CANONICAL instance format: a weighted constraint network as JSON
//! (`"format": "wcn-1"`). This is the one representation the counting engine
//! officially eats; every other input (`.csp`, DIMACS `.cnf`, problem-reductions
//! CircuitSAT `.json`) is an ADAPTER that converts into it (`from_*_text`), and
//! `load_instance` dispatches on the file extension.
//!
//! Schema (serde = the spec):
//! ```json
//! {
//!   "format": "wcn-1",
//!   "n_vars": 5,
//!   "tensors": [
//!     {"vars": [0,1], "allow": [1,2,3]},
//!     {"vars": [2,3], "rows": [[0,"1"], [3,"1/2"]]}
//!   ],
//!   "var_weights": [[0, "1/2", "1/2"]],
//!   "meta": {"family": "or-chain"}
//! }
//! ```
//! - `n_vars` is the DECLARED variable count (0-based ids); vars in no tensor
//!   are unconstrained and contribute their `free_factor` to the count.
//! - Each tensor lists its scope `vars` (distinct, in bit order: bit i of a
//!   config = value of `vars[i]`) and EITHER `allow` (0/1 relation: the allowed
//!   configs, weight 1 each) OR `rows` (weighted relation: `[config, weight]`
//!   pairs) — per-row weights are the full `WRelation` expressiveness, which no
//!   text format carries (UAI-style potentials map here directly).
//! - `var_weights` are literal weights `[var, w(v=0), w(v=1)]` — sugar for a
//!   1-ary `rows` tensor. All weights are EXACT strings: integer (`3`), decimal
//!   (`0.75`), or ratio (`3/4`), parsed by `RationalWeight::parse`.
//! - `meta` is free-form and ignored by the solver (bank metadata: family,
//!   known count, seed, ...). Future M3 field: `show` (projection set).
//!
//! Counting semantics: an instance with NO `rows` tensors and NO `var_weights`
//! is plain model counting (exact `BigCount`); anything weighted counts in
//! exact rationals (`RationalWeight`). Both run the same M2 pipeline
//! (`count_with_ve`): weighted VE front-end, then region-branching search.

use std::fmt;
use std::path::Path;

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

use crate::circuit::network_from_circuit_sat;
use crate::contract::{unit_weighted_relations, WRelation};
use crate::csp::parse_csp;
use crate::dimacs::{parse_dimacs, parse_mcc_weights, MAX_CLAUSE_WIDTH};
use crate::network::{setup_problem, ConstraintNetwork};
use crate::problem::Stats;
use crate::semiring::{BigCount, RationalWeight, Weight};
use crate::solver::{count_with_ve, CountBranch};

/// The schema tag; bump on breaking changes.
pub const FORMAT_TAG: &str = "wcn-1";

/// One constraint tensor: scope + EITHER `allow` (0/1) OR `rows` (weighted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceTensor {
    pub vars: Vec<usize>,
    /// Allowed configs of a 0/1 relation (each row weight 1).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<u64>,
    /// Weighted rows `[config, weight]`; mutually exclusive with `allow`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<(u64, String)>,
}

/// A weighted constraint network — the canonical `wcn-1` instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub format: String,
    pub n_vars: usize,
    pub tensors: Vec<InstanceTensor>,
    /// Literal weights `[var, w_false, w_true]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub var_weights: Vec<(usize, String, String)>,
    /// Free-form bank metadata; the solver never reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

/// An exact count in whichever semiring the instance selected.
#[derive(Debug, Clone, PartialEq)]
pub enum CountValue {
    /// Plain model count (unweighted instance).
    Int(BigCount),
    /// Weighted count / partition function (exact rational).
    Rat(RationalWeight),
}

impl fmt::Display for CountValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CountValue::Int(c) => write!(f, "{}", c.to_decimal()),
            CountValue::Rat(w) => write!(f, "{}", w.to_ratio_string()),
        }
    }
}

impl Instance {
    /// A fresh unweighted instance over `n_vars` variables (no tensors yet).
    pub fn new(n_vars: usize) -> Instance {
        Instance {
            format: FORMAT_TAG.to_string(),
            n_vars,
            tensors: Vec::new(),
            var_weights: Vec::new(),
            meta: None,
        }
    }

    /// `true` iff counting must run in the rational (weighted) semiring.
    pub fn is_weighted(&self) -> bool {
        !self.var_weights.is_empty() || self.tensors.iter().any(|t| !t.rows.is_empty())
    }

    /// Structural validation: format tag, scope sanity (non-empty, distinct,
    /// in range, arity ≤ 63), config range, `allow`/`rows` exclusivity, weight
    /// parseability, and `var_weights` var uniqueness/range.
    pub fn validate(&self) -> Result<(), String> {
        if self.format != FORMAT_TAG {
            return Err(format!(
                "unknown format {:?} (expected {FORMAT_TAG:?})",
                self.format
            ));
        }
        for (i, t) in self.tensors.iter().enumerate() {
            if t.vars.is_empty() {
                return Err(format!("tensor {i}: empty scope"));
            }
            if t.vars.len() > 63 {
                return Err(format!(
                    "tensor {i}: arity {} exceeds 63 (u64 bitmask)",
                    t.vars.len()
                ));
            }
            let mut sorted = t.vars.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != t.vars.len() {
                return Err(format!("tensor {i}: duplicate var in scope {:?}", t.vars));
            }
            if let Some(&v) = t.vars.iter().find(|&&v| v >= self.n_vars) {
                return Err(format!("tensor {i}: var {v} >= n_vars {}", self.n_vars));
            }
            if !t.allow.is_empty() && !t.rows.is_empty() {
                return Err(format!("tensor {i}: has both `allow` and `rows`"));
            }
            let limit = 1u64 << t.vars.len();
            for &c in t.allow.iter().chain(t.rows.iter().map(|(c, _)| c)) {
                if c >= limit {
                    return Err(format!(
                        "tensor {i}: config {c} out of range for arity {}",
                        t.vars.len()
                    ));
                }
            }
            for (c, w) in &t.rows {
                RationalWeight::parse(w)
                    .map_err(|e| format!("tensor {i}: bad weight for config {c}: {e}"))?;
            }
        }
        let mut seen = vec![false; self.n_vars];
        for (v, wf, wt) in &self.var_weights {
            if *v >= self.n_vars {
                return Err(format!("var_weights: var {v} >= n_vars {}", self.n_vars));
            }
            if seen[*v] {
                return Err(format!("var_weights: duplicate entry for var {v}"));
            }
            seen[*v] = true;
            for w in [wf, wt] {
                RationalWeight::parse(w)
                    .map_err(|e| format!("var_weights: bad weight for var {v}: {e}"))?;
            }
        }
        Ok(())
    }

    // ---------------- adapters (other formats -> canonical) ----------------

    /// Adapter: native `.csp` text (tensor lines + optional `w` weight lines).
    pub fn from_csp_text(text: &str) -> Result<Instance, String> {
        let doc = parse_csp(text)?;
        let mut inst = Instance::new(doc.n_vars);
        inst.tensors = doc
            .tensors
            .into_iter()
            .map(|(vars, mut allow)| {
                allow.sort_unstable();
                allow.dedup();
                InstanceTensor {
                    vars,
                    allow,
                    rows: Vec::new(),
                }
            })
            .collect();
        inst.var_weights = doc.weights;
        inst.validate()?;
        Ok(inst)
    }

    /// Adapter: DIMACS CNF text, with MCC 2021+ `c p weight <lit> <w> 0` lines
    /// for weighted counting (the legacy Cachet `w` format and the projected
    /// `c p show` header are refused by the underlying parser). Duplicate
    /// literals in a clause are deduplicated; tautological clauses are dropped
    /// (count-safe: they constrain nothing, and their vars stay declared).
    pub fn from_dimacs_text(text: &str) -> Result<Instance, String> {
        let weights = parse_mcc_weights(text).map_err(|e| e.to_string())?;
        let (n_vars, clauses) = parse_dimacs(text).map_err(|e| e.to_string())?;
        let mut inst = Instance::new(n_vars);
        'clause: for lits in &clauses {
            let mut lits = lits.clone();
            lits.sort_unstable_by_key(|l| (l.unsigned_abs(), *l < 0));
            lits.dedup();
            for w in lits.windows(2) {
                if w[0] == -w[1] {
                    continue 'clause; // tautology: always satisfied
                }
            }
            // Dense-expansion guard: a k-literal clause materializes 2^k − 1
            // allow rows (and the engine's tensor tables are dense/u32-config,
            // hard-capped at arity 32 anyway). Real-world CNFs (MCC) carry
            // 30+-literal clauses — one such clause would allocate tens of GB
            // and take the machine down, which is exactly what happened on
            // 2026-07-13. Refuse loudly instead; long-clause CNF is outside
            // this engine's native-table representation.
            if lits.len() > MAX_CLAUSE_WIDTH {
                return Err(format!(
                    "clause length {} exceeds the dense-table limit {MAX_CLAUSE_WIDTH} \
                     (2^k row expansion; instance is outside this engine's representation)",
                    lits.len()
                ));
            }
            let vars: Vec<usize> = lits
                .iter()
                .map(|l| (l.unsigned_abs() as usize) - 1)
                .collect();
            // The single falsifying config sets every literal false.
            let forbidden: u64 = lits
                .iter()
                .enumerate()
                .filter(|(_, &l)| l < 0)
                .map(|(i, _)| 1u64 << i)
                .sum();
            let allow: Vec<u64> = (0..(1u64 << lits.len()))
                .filter(|&c| c != forbidden)
                .collect();
            inst.tensors.push(InstanceTensor {
                vars,
                allow,
                rows: Vec::new(),
            });
        }
        for (lit, w) in &weights {
            let v = (lit.unsigned_abs() as usize) - 1;
            if v >= n_vars {
                return Err(format!(
                    "weight literal {lit} exceeds declared var count {n_vars}"
                ));
            }
            let entry = match inst.var_weights.iter_mut().find(|(u, _, _)| *u == v) {
                Some(e) => e,
                None => {
                    inst.var_weights.push((v, "1".to_string(), "1".to_string()));
                    inst.var_weights.last_mut().unwrap()
                }
            };
            if *lit > 0 {
                entry.2 = w.clone();
            } else {
                entry.1 = w.clone();
            }
        }
        inst.validate()?;
        Ok(inst)
    }

    /// Adapter: problem-reductions CircuitSAT JSON (gates as native tensors,
    /// unweighted). Ids are mapped back to the ORIGINAL wire id space so the
    /// canonical instance keeps every declared wire.
    pub fn from_circuit_sat_json(json: &str) -> Result<Instance, String> {
        let cp = network_from_circuit_sat(json).map_err(|e| e.to_string())?;
        let cn = cp.network;
        let mut new_to_orig = vec![0usize; cn.n_vars];
        for (o, &c) in cn.orig_to_new.iter().enumerate() {
            if let Some(ci) = c {
                new_to_orig[ci] = o;
            }
        }
        let mut inst = Instance::new(cn.orig_to_new.len());
        inst.tensors = cn
            .tensors
            .iter()
            .map(|t| InstanceTensor {
                vars: t.var_axes.iter().map(|&c| new_to_orig[c]).collect(),
                allow: cn.support(t).iter().map(|&c| c as u64).collect(),
                rows: Vec::new(),
            })
            .collect();
        inst.validate()?;
        Ok(inst)
    }

    /// Parse canonical `wcn-1` JSON.
    pub fn from_json(json: &str) -> Result<Instance, String> {
        let inst: Instance =
            serde_json::from_str(json).map_err(|e| format!("bad wcn JSON: {e}"))?;
        inst.validate()?;
        Ok(inst)
    }

    /// Serialize to canonical JSON (pretty, stable field order via serde).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("wcn serialization cannot fail")
    }

    // ---------------- counting ----------------

    /// The plain 0/1 skeleton (weights dropped) — for the decision path and for
    /// the unweighted counting arm. Unconstrained vars are compressed out
    /// (`cn.orig_to_new` keeps the mapping).
    pub fn skeleton_network(&self) -> ConstraintNetwork {
        let mut scopes: Vec<Vec<usize>> = Vec::with_capacity(self.tensors.len());
        let mut tables: Vec<Vec<bool>> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let mut table = vec![false; 1usize << t.vars.len()];
            for &c in t.allow.iter().chain(t.rows.iter().map(|(c, _)| c)) {
                table[c as usize] = true;
            }
            scopes.push(t.vars.clone());
            tables.push(table);
        }
        setup_problem(self.n_vars, scopes, tables)
    }

    /// Exact count via the M2 pipeline (`count_with_ve`). Semiring selection:
    /// unweighted -> `BigCount` plus the `2^unconstrained` free-var multiplier;
    /// weighted -> `RationalWeight`, with EVERY declared var given a 1-ary
    /// literal-weight tensor (unit where unspecified), so no var is compressed
    /// away before its weight is folded in and no multiplier is owed.
    pub fn count(
        &self,
        budget: usize,
        max_rows: usize,
        branch: CountBranch,
    ) -> Result<(CountValue, Stats), String> {
        self.validate()?;
        if !self.is_weighted() {
            let cn = self.skeleton_network();
            let rels = unit_weighted_relations::<BigCount>(&cn);
            let (count, stats) =
                count_with_ve::<BigCount>(cn.n_vars, rels, budget, max_rows, branch);
            let unconstrained = self.n_vars - cn.n_vars;
            let multiplier = BigCount(BigUint::from(2u32).pow(unconstrained as u32));
            Ok((CountValue::Int(count.mul(&multiplier)), stats))
        } else {
            let mut rels: Vec<WRelation<RationalWeight>> =
                Vec::with_capacity(self.tensors.len() + self.n_vars);
            for (i, t) in self.tensors.iter().enumerate() {
                let mut rows: Vec<(u64, RationalWeight)> = if t.rows.is_empty() {
                    t.allow
                        .iter()
                        .map(|&c| (c, RationalWeight::int(1)))
                        .collect()
                } else {
                    t.rows
                        .iter()
                        .map(|(c, w)| Ok((*c, RationalWeight::parse(w)?)))
                        .collect::<Result<_, String>>()?
                };
                rows.sort_unstable_by_key(|&(c, _)| c);
                if rows.windows(2).any(|w| w[0].0 == w[1].0) {
                    return Err(format!("tensor {i}: duplicate config in rows"));
                }
                rels.push(WRelation {
                    vars: t.vars.clone(),
                    rows,
                });
            }
            let mut w0 = vec![RationalWeight::int(1); self.n_vars];
            let mut w1 = vec![RationalWeight::int(1); self.n_vars];
            for (v, wf, wt) in &self.var_weights {
                w0[*v] = RationalWeight::parse(wf)?;
                w1[*v] = RationalWeight::parse(wt)?;
            }
            for v in 0..self.n_vars {
                rels.push(WRelation {
                    vars: vec![v],
                    rows: vec![(0u64, w0[v].clone()), (1u64, w1[v].clone())],
                });
            }
            let (count, stats) =
                count_with_ve::<RationalWeight>(self.n_vars, rels, budget, max_rows, branch);
            Ok((CountValue::Rat(count), stats))
        }
    }
}

/// Load any supported instance file, dispatching on the extension:
/// `.csp` (native tables), `.cnf`/`.dimacs` (CNF + MCC weights), `.json`
/// (canonical `wcn-1` if the format tag matches, else CircuitSAT).
pub fn load_instance(path: &Path) -> Result<Instance, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csp" => Instance::from_csp_text(&text),
        "cnf" | "dimacs" => Instance::from_dimacs_text(&text),
        "json" => {
            let is_canonical = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v.get("format").and_then(|f| f.as_str()).map(String::from))
                .is_some_and(|f| f == FORMAT_TAG);
            if is_canonical {
                Instance::from_json(&text)
            } else {
                Instance::from_circuit_sat_json(&text)
            }
        }
        other => Err(format!(
            "unsupported extension {other:?} (expected .csp, .cnf, .dimacs, or .json)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (x0 ∨ x1) ∧ (x1 ∨ x2): 5 models over 3 vars.
    const OR_CHAIN_CSP: &str = "3\n0 1 : 1 2 3\n1 2 : 1 2 3\n";
    const OR_CHAIN_CNF: &str = "p cnf 3 2\n1 2 0\n2 3 0\n";

    fn count_int(inst: &Instance) -> String {
        let (c, _) = inst.count(16, 128, CountBranch::PerConfig).expect("count");
        c.to_string()
    }

    #[test]
    fn csp_cnf_and_canonical_json_agree() {
        let a = Instance::from_csp_text(OR_CHAIN_CSP).expect("csp");
        let b = Instance::from_dimacs_text(OR_CHAIN_CNF).expect("cnf");
        let c = Instance::from_json(&a.to_json()).expect("round-trip");
        assert_eq!(count_int(&a), "5");
        assert_eq!(count_int(&b), "5");
        assert_eq!(count_int(&c), "5");
    }

    #[test]
    fn budget_invariance_holds_through_the_canonical_path() {
        let inst = Instance::from_csp_text(OR_CHAIN_CSP).expect("csp");
        for budget in [0, 2, 16] {
            let (c, _) = inst
                .count(budget, 128, CountBranch::BlockMerge)
                .expect("count");
            assert_eq!(c.to_string(), "5");
        }
    }

    #[test]
    fn declared_but_unconstrained_vars_double_the_count() {
        // Same skeleton, 5 declared vars: 5 * 2^2 = 20.
        let inst = Instance::from_csp_text("5\n0 1 : 1 2 3\n1 2 : 1 2 3\n").expect("csp");
        assert_eq!(count_int(&inst), "20");
    }

    #[test]
    fn csp_w_lines_and_mcc_weights_agree() {
        // w(x0=0)=1/2, w(x0=1)=1/4, rest unit. Weighted count =
        // Σ_models w(x0): models with x0=0: 010,011 -> 2*(1/2)=1;
        // with x0=1: 110,101,111 -> 3*(1/4)=3/4; total 7/4.
        let csp = format!("{OR_CHAIN_CSP}w 0 1/2 1/4\n");
        let cnf = format!("{OR_CHAIN_CNF}c p weight 1 1/4 0\nc p weight -1 1/2 0\n");
        let a = Instance::from_csp_text(&csp).expect("csp");
        let b = Instance::from_dimacs_text(&cnf).expect("cnf");
        let (ca, _) = a.count(16, 128, CountBranch::PerConfig).expect("count");
        let (cb, _) = b.count(0, 128, CountBranch::BlockMerge).expect("count");
        assert_eq!(ca.to_string(), "7/4");
        assert_eq!(cb.to_string(), "7/4");
    }

    #[test]
    fn per_row_weighted_tensors_count_exactly() {
        // One tensor over {0,1}: row 0 (00) weight 1/2, row 3 (11) weight 2.
        // Partition function = 1/2 + 2 = 5/2 (var_weights default to unit).
        let json = r#"{
            "format": "wcn-1", "n_vars": 2,
            "tensors": [{"vars": [0,1], "rows": [[0,"1/2"],[3,"2"]]}]
        }"#;
        let inst = Instance::from_json(json).expect("json");
        let (c, _) = inst.count(16, 128, CountBranch::PerConfig).expect("count");
        assert_eq!(c.to_string(), "5/2");
    }

    #[test]
    fn circuit_sat_json_is_sniffed_apart_from_canonical() {
        // A canonical file must NOT be fed to the CircuitSAT adapter and vice
        // versa; sniffing keys off the `format` tag.
        let canonical = Instance::from_csp_text(OR_CHAIN_CSP)
            .expect("csp")
            .to_json();
        let v: serde_json::Value = serde_json::from_str(&canonical).unwrap();
        assert_eq!(v["format"], FORMAT_TAG);
    }

    #[test]
    fn validation_rejects_malformed_instances() {
        for bad in [
            r#"{"format":"wcn-9","n_vars":1,"tensors":[]}"#,
            r#"{"format":"wcn-1","n_vars":1,"tensors":[{"vars":[0,0],"allow":[0]}]}"#,
            r#"{"format":"wcn-1","n_vars":1,"tensors":[{"vars":[1],"allow":[0]}]}"#,
            r#"{"format":"wcn-1","n_vars":2,"tensors":[{"vars":[0,1],"allow":[4]}]}"#,
            r#"{"format":"wcn-1","n_vars":1,"tensors":[{"vars":[0],"allow":[0],"rows":[[0,"1"]]}]}"#,
            r#"{"format":"wcn-1","n_vars":1,"tensors":[],"var_weights":[[0,"x","1"]]}"#,
        ] {
            assert!(Instance::from_json(bad).is_err(), "should reject: {bad}");
        }
    }

    #[test]
    fn tautologies_and_duplicate_literals_are_normalized() {
        // (x1 ∨ ¬x1) is dropped; (x2 ∨ x2) dedups to (x2). Count over 2 declared
        // vars: x2 = true, x1 free -> 2 models.
        let inst = Instance::from_dimacs_text("p cnf 2 2\n1 -1 0\n2 2 0\n").expect("cnf");
        assert_eq!(inst.tensors.len(), 1);
        assert_eq!(count_int(&inst), "2");
    }
}
