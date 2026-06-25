use crate::network::{setup_problem, ConstraintNetwork};

pub const MAX_CLAUSE_WIDTH: usize = 20;

#[derive(Debug, thiserror::Error)]
pub enum DimacsError {
    #[error("clause width {0} exceeds MAX_CLAUSE_WIDTH={1}")]
    ClauseTooWide(usize, usize),
    #[error("literal 0 used as a variable")]
    ZeroLiteral,
    #[error("malformed token: {0}")]
    BadToken(String),
}

/// Parse DIMACS CNF text into `(nvars, clauses)` with 1-based signed literals.
pub fn parse_dimacs(text: &str) -> Result<(usize, Vec<Vec<i64>>), DimacsError> {
    let mut nvars = 0usize;
    let mut clauses: Vec<Vec<i64>> = Vec::new();
    let mut current: Vec<i64> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') || line.starts_with('%') {
            continue;
        }
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                nvars = parts[3].parse().map_err(|_| DimacsError::BadToken(parts[3].into()))?;
            }
            continue;
        }
        for tok in line.split_whitespace() {
            let val: i64 = match tok.parse() {
                Ok(v) => v,
                Err(_) => continue, // tolerate stray tokens, like the Julia parser
            };
            if val == 0 {
                if !current.is_empty() {
                    clauses.push(std::mem::take(&mut current));
                }
            } else {
                nvars = nvars.max(val.unsigned_abs() as usize);
                current.push(val);
            }
        }
    }
    if !current.is_empty() {
        clauses.push(current);
    }
    Ok((nvars, clauses))
}

/// Build `(var_axes_0based, dense)` for a CNF clause. `dense[config]` is true iff
/// the clause is satisfied by the assignment encoded in `config` (bit i = var_axes[i]).
pub fn clause_to_tensor(lits: &[i64]) -> (Vec<usize>, Vec<bool>) {
    let mut vars: Vec<usize> = Vec::with_capacity(lits.len());
    let mut pos: Vec<bool> = Vec::with_capacity(lits.len());
    for &lit in lits {
        debug_assert!(lit != 0);
        vars.push((lit.unsigned_abs() as usize) - 1); // 0-based
        pos.push(lit > 0);
    }
    let k = vars.len();
    let mut dense = vec![false; 1usize << k];
    for config in 0..(1usize << k) {
        // satisfied if any literal is true under this config
        let mut sat = false;
        for i in 0..k {
            let bit = (config >> i) & 1 == 1;
            if bit == pos[i] {
                sat = true;
                break;
            }
        }
        dense[config] = sat;
    }
    (vars, dense)
}

pub fn network_from_dimacs(text: &str) -> Result<ConstraintNetwork, DimacsError> {
    let (nvars, clauses) = parse_dimacs(text)?;
    let mut tensors_to_vars: Vec<Vec<usize>> = Vec::with_capacity(clauses.len());
    let mut tensor_data: Vec<Vec<bool>> = Vec::with_capacity(clauses.len());
    for clause in &clauses {
        if clause.is_empty() {
            continue;
        }
        if clause.len() > MAX_CLAUSE_WIDTH {
            return Err(DimacsError::ClauseTooWide(clause.len(), MAX_CLAUSE_WIDTH));
        }
        let (vars, dense) = clause_to_tensor(clause);
        tensors_to_vars.push(vars);
        tensor_data.push(dense);
    }
    Ok(setup_problem(nvars, tensors_to_vars, tensor_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_dimacs() {
        let text = "c comment\np cnf 3 2\n1 -2 0\n2 3 0\n";
        let (nvars, clauses) = parse_dimacs(text).unwrap();
        assert_eq!(nvars, 3);
        assert_eq!(clauses, vec![vec![1, -2], vec![2, 3]]);
    }

    #[test]
    fn clause_to_tensor_or_semantics() {
        // clause (x1 OR NOT x2): var_axes = [0, 1] (0-based), polarity [+, -]
        // config bit0 = x1, bit1 = x2. Unsatisfied only when x1=0 AND x2=1 => config 0b10 = 2.
        let (vars, dense) = clause_to_tensor(&[1, -2]);
        assert_eq!(vars, vec![0, 1]);
        assert_eq!(dense, vec![true, true, false, true]);
    }

    #[test]
    fn network_from_dimacs_end_to_end() {
        let text = "p cnf 2 1\n1 2 0\n";
        let cn = network_from_dimacs(text).unwrap();
        assert_eq!(cn.tensors.len(), 1);
        assert_eq!(cn.vars.len(), 2);
    }
}
