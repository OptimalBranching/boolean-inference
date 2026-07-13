//! Parser for the generic NATIVE constraint-network (`.csp`) format emitted by
//! `benchmarks/csp/*.py` — one explicit tensor per constraint, so region
//! contraction sees each relation whole (no CNF flattening). Extracted from
//! `examples/solve_csp.rs` so both the solver and the cuber (`gen_cubes`) read
//! the same format through one function.
//!
//! Format (whitespace/newline tolerant):
//!   line 1:  <n_vars>
//!   then one tensor per line:  <v0> <v1> ... <v_{a-1}> : <cfg0> <cfg1> ...
//!     scope = the var ids before ':'; the ints after ':' are the ALLOWED
//!     configurations, each a bitmask over the scope (bit i = value of v_i).
//!
//! Variable ids are 0-based; a `.csp` var `v` maps to DIMACS literal `v + 1`,
//! matching the paired `.cnf` the generators emit (they add no aux variables),
//! so a cube over a `.csp` uses the same numbering as its `.cnf`.

use crate::network::{setup_problem, ConstraintNetwork};

/// Parse a `.csp` document into a `ConstraintNetwork`. Returns a human-readable
/// error string on malformed input (missing `n_vars`, a tensor line without
/// `:`, a non-integer field, or an out-of-range scope var / config bitmask).
pub fn network_from_csp(text: &str) -> Result<ConstraintNetwork, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let n_vars: usize = lines
        .next()
        .ok_or("empty .csp: missing n_vars header")?
        .trim()
        .parse()
        .map_err(|e| format!("n_vars is not an integer: {e}"))?;

    let mut scopes: Vec<Vec<usize>> = Vec::new();
    let mut tables: Vec<Vec<bool>> = Vec::new();
    for line in lines {
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
        let mut table = vec![false; 1usize << scope.len()];
        for field in rhs.split_whitespace() {
            let c: u64 = field
                .parse()
                .map_err(|e| format!("bad config {field:?}: {e}"))?;
            let idx = c as usize;
            if idx >= table.len() {
                return Err(format!(
                    "config {c} out of range for arity-{} tensor",
                    scope.len()
                ));
            }
            table[idx] = true;
        }
        scopes.push(scope);
        tables.push(table);
    }

    Ok(setup_problem(n_vars, scopes, tables))
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
}
