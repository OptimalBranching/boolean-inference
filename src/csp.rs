//! Parser for the generic NATIVE constraint-network (`.csp`) format emitted by
//! `benchmarks/csp/*.py` — one explicit tensor per constraint, so region
//! contraction sees each relation whole (no CNF flattening). This module is an
//! input ADAPTER: the canonical instance representation is the `wcn-1` JSON of
//! `crate::instance`; `.csp` files convert into it via `Instance::from_csp_text`.
//!
//! Format (whitespace/newline tolerant):
//!   line 1:  <n_vars>
//!   then one tensor per line:  <v0> <v1> ... <v_{a-1}> : <cfg0> <cfg1> ...
//!     scope = the var ids before ':'; the ints after ':' are the ALLOWED
//!     configurations, each a bitmask over the scope (bit i = value of v_i).
//!   optional weight lines (weighted counting, M2.2), anywhere after line 1:
//!     w <var> <w_false> <w_true>
//!     gives variable `<var>` exact literal weights; each weight is an integer
//!     (`3`), a decimal (`0.75`), or a ratio (`3/4`), parsed downstream by
//!     `RationalWeight::parse`. The decision path ignores weight lines.
//!
//! Variable ids are 0-based; a `.csp` var `v` maps to DIMACS literal `v + 1`,
//! matching the paired `.cnf` the generators emit (they add no aux variables),
//! so a cube over a `.csp` uses the same numbering as its `.cnf`.

use crate::network::{setup_problem, ConstraintNetwork};

/// A parsed `.csp` document, before any variable compression: tensor scopes and
/// allowed configs exactly as written, plus raw `w` literal-weight lines.
#[derive(Debug, Clone)]
pub struct CspDoc {
    pub n_vars: usize,
    /// One `(scope, allowed configs)` per tensor line, in file order.
    pub tensors: Vec<(Vec<usize>, Vec<u64>)>,
    /// `(var, w_false, w_true)` per `w` line; weights kept as raw strings.
    pub weights: Vec<(usize, String, String)>,
}

/// Parse a `.csp` document. Returns a human-readable error string on malformed
/// input (missing `n_vars`, a tensor line without `:`, a non-integer field, an
/// out-of-range scope var / config bitmask / weight var, or a truncated `w` line).
pub fn parse_csp(text: &str) -> Result<CspDoc, String> {
    let mut weight_lines: Vec<&str> = Vec::new();
    let mut body = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter(|l| {
            let is_weight = l.split_whitespace().next() == Some("w");
            if is_weight {
                weight_lines.push(l);
            }
            !is_weight
        });

    let n_vars: usize = body
        .next()
        .ok_or("empty .csp: missing n_vars header")?
        .parse()
        .map_err(|e| format!("n_vars is not an integer: {e}"))?;

    let mut tensors: Vec<(Vec<usize>, Vec<u64>)> = Vec::new();
    for line in body {
        let (lhs, rhs) = line
            .split_once(':')
            .ok_or_else(|| format!("tensor line needs ':': {line:?}"))?;
        let scope: Vec<usize> = lhs
            .split_whitespace()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|e| format!("bad scope var {s:?}: {e}"))
            })
            .collect::<Result<_, _>>()?;
        if scope.is_empty() {
            return Err(format!("tensor line has empty scope: {line:?}"));
        }
        if scope.len() > 63 {
            return Err(format!(
                "tensor arity {} exceeds 63 (u64 bitmask): {line:?}",
                scope.len()
            ));
        }
        if let Some(&v) = scope.iter().find(|&&v| v >= n_vars) {
            return Err(format!("scope var {v} >= n_vars {n_vars}"));
        }
        let mut configs: Vec<u64> = Vec::new();
        for field in rhs.split_whitespace() {
            let c: u64 = field
                .parse()
                .map_err(|e| format!("bad config {field:?}: {e}"))?;
            if c >= (1u64 << scope.len()) {
                return Err(format!(
                    "config {c} out of range for arity-{} tensor",
                    scope.len()
                ));
            }
            configs.push(c);
        }
        tensors.push((scope, configs));
    }

    let mut weights: Vec<(usize, String, String)> = Vec::new();
    for line in weight_lines {
        let mut it = line.split_whitespace().skip(1);
        let var: usize = it
            .next()
            .ok_or_else(|| format!("weight line needs `w <var> <wf> <wt>`: {line:?}"))?
            .parse()
            .map_err(|e| format!("bad weight var in {line:?}: {e}"))?;
        if var >= n_vars {
            return Err(format!("weight var {var} >= n_vars {n_vars}"));
        }
        let wf = it
            .next()
            .ok_or_else(|| format!("weight line missing w_false: {line:?}"))?;
        let wt = it
            .next()
            .ok_or_else(|| format!("weight line missing w_true: {line:?}"))?;
        weights.push((var, wf.to_string(), wt.to_string()));
    }

    Ok(CspDoc {
        n_vars,
        tensors,
        weights,
    })
}

/// Parse a `.csp` document into a `ConstraintNetwork`, ignoring any `w` weight
/// lines — the decision-path entry (`solve_csp`, `gen_cubes`).
pub fn network_from_csp(text: &str) -> Result<ConstraintNetwork, String> {
    let doc = parse_csp(text)?;
    let mut scopes: Vec<Vec<usize>> = Vec::with_capacity(doc.tensors.len());
    let mut tables: Vec<Vec<bool>> = Vec::with_capacity(doc.tensors.len());
    for (scope, configs) in doc.tensors {
        let mut table = vec![false; 1usize << scope.len()];
        for c in configs {
            table[c as usize] = true;
        }
        scopes.push(scope);
        tables.push(table);
    }
    Ok(setup_problem(doc.n_vars, scopes, tables))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_small_csp() {
        // 3 vars; T0 over {0,1} allowing configs 1,2,3 (i.e. NOT 00) = OR;
        // T1 over {1,2} allowing 1,2,3 = OR. Same as the or-chain fixture.
        let text = "3\n0 1 : 1 2 3\n1 2 : 1 2 3\n";
        let cn = network_from_csp(text).expect("parse");
        assert_eq!(cn.n_vars, 3);
        assert_eq!(cn.tensors.len(), 2);
        assert_eq!(cn.tensors[0].var_axes, vec![0, 1]);
    }

    #[test]
    fn rejects_line_without_colon() {
        assert!(network_from_csp("2\n0 1 1 2 3\n").is_err());
    }

    #[test]
    fn rejects_out_of_range_scope() {
        assert!(network_from_csp("2\n0 5 : 1\n").is_err());
    }

    #[test]
    fn weight_lines_are_parsed_and_ignored_by_the_decision_path() {
        let text = "3\n0 1 : 1 2 3\nw 0 1/2 1/2\n1 2 : 1 2 3\nw 2 0.25 3\n";
        let doc = parse_csp(text).expect("parse");
        assert_eq!(doc.tensors.len(), 2);
        assert_eq!(
            doc.weights,
            vec![
                (0, "1/2".to_string(), "1/2".to_string()),
                (2, "0.25".to_string(), "3".to_string()),
            ]
        );
        // The decision entry sees the same skeleton.
        assert_eq!(network_from_csp(text).expect("parse").tensors.len(), 2);
    }

    #[test]
    fn rejects_out_of_range_or_truncated_weight_lines() {
        assert!(parse_csp("2\n0 1 : 1\nw 5 1 1\n").is_err());
        assert!(parse_csp("2\n0 1 : 1\nw 0 1\n").is_err());
    }
}
