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
    #[error("projected-model header `c p show` — this is a projected-counting (#∃SAT) instance; refusing to report its projected count as a total model count")]
    ProjectedInstance,
    #[error("legacy Cachet weight line `w <var> <p>` — refused; use the MCC 2021+ format `c p weight <lit> <w> 0` with BOTH polarities explicit (silent convention mixing corrupts weighted counts)")]
    CachetWeightLine,
    #[error("malformed `c p weight` line: {0}")]
    BadWeightLine(String),
}

/// Parse MCC 2021+ weighted-model-counting weight lines `c p weight <lit> <w> 0`
/// into `(literal, weight-string)` pairs (both polarities are given explicitly,
/// per the MCC format). The legacy Cachet `w <var> <p>` line format is REFUSED
/// (§7 guard b: mixing the two silently corrupts weighted counts). Non-weight
/// lines are ignored; the caller still parses the CNF via `network_from_dimacs`.
/// Weight strings are returned verbatim so the caller parses them into its exact
/// weight type (e.g. `RationalWeight`) without an intermediate `f64` rounding.
pub fn parse_mcc_weights(text: &str) -> Result<Vec<(i64, String)>, DimacsError> {
    let mut out: Vec<(i64, String)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let mut it = line.split_whitespace();
        match it.next() {
            // MCC weight line `c p weight <lit> <w> 0` (the guard consumes `p
            // weight`; a plain `c ...` comment fails the guard and is ignored).
            Some("c") if it.next() == Some("p") && it.next() == Some("weight") => {
                let lit: i64 = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| DimacsError::BadWeightLine(line.into()))?;
                let w = it
                    .next()
                    .ok_or_else(|| DimacsError::BadWeightLine(line.into()))?
                    .to_string();
                out.push((lit, w));
            }
            // A line whose first token is a bare `w` is the Cachet weight format.
            Some("w") => return Err(DimacsError::CachetWeightLine),
            _ => {}
        }
    }
    Ok(out)
}

/// Detect the MCC projected-model-counting header `c p show <vars> 0`. A total
/// model counter must REFUSE such an instance rather than silently count the
/// projected variables as if they were the whole formula (§7 guard d). Matches
/// the `c p show` prefix on a comment line (after trimming).
fn is_projected_show_line(line: &str) -> bool {
    let mut it = line.split_whitespace();
    it.next() == Some("c") && it.next() == Some("p") && it.next() == Some("show")
}

/// Parse DIMACS CNF text into `(nvars, clauses)` with 1-based signed literals.
pub fn parse_dimacs(text: &str) -> Result<(usize, Vec<Vec<i64>>), DimacsError> {
    let mut nvars = 0usize;
    let mut clauses: Vec<Vec<i64>> = Vec::new();
    let mut current: Vec<i64> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if is_projected_show_line(line) {
            return Err(DimacsError::ProjectedInstance);
        }
        if line.is_empty() || line.starts_with('c') || line.starts_with('%') {
            continue;
        }
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // `p cnf <nvars> <nclauses>` — nvars is at index 2.
            if parts.len() >= 3 {
                nvars = parts[2]
                    .parse()
                    .map_err(|_| DimacsError::BadToken(parts[2].into()))?;
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
        assert_eq!(cn.n_vars, 2);
    }

    #[test]
    fn rejects_projected_show_header() {
        // `c p show ... 0` marks a projected-counting instance — a total counter
        // must refuse it, not count the projection as a total (§7 guard d).
        let r = parse_dimacs("c p show 1 2 0\np cnf 3 1\n1 2 0\n");
        assert!(matches!(r, Err(DimacsError::ProjectedInstance)));
        // A plain comment starting with `c` but not `c p show` still parses.
        assert!(parse_dimacs("c p weight 1 0.5 0\np cnf 2 1\n1 2 0\n").is_ok());
    }

    #[test]
    fn declared_but_unused_vars_survive_as_header_count() {
        // `p cnf 5 1 / 1 2 0`: declared 5, only vars 1,2 used. The declared count
        // must be 5 (the counting front-end multiplies 2^(5−2)=8 back in for a
        // total of 24). The builder compresses vars 3,4,5 to n_vars=2.
        let (nvars, _clauses) = parse_dimacs("p cnf 5 1\n1 2 0\n").unwrap();
        assert_eq!(nvars, 5);
        let cn = network_from_dimacs("p cnf 5 1\n1 2 0\n").unwrap();
        assert_eq!(cn.n_vars, 2, "unused declared vars compressed out");
        assert_eq!(nvars - cn.n_vars, 3, "three free vars ⇒ ×2^3 multiplier");
    }

    #[test]
    fn parses_mcc_weight_lines_and_rejects_cachet() {
        // MCC 2021+ `c p weight <lit> <w> 0`, both polarities explicit.
        let mcc = "p cnf 2 1\nc p weight 1 0.75 0\nc p weight -1 0.25 0\n1 2 0\n";
        let ws = parse_mcc_weights(mcc).expect("mcc weights");
        assert_eq!(
            ws,
            vec![(1i64, "0.75".to_string()), (-1i64, "0.25".to_string())]
        );
        // Legacy Cachet `w <var> <p>` must be refused, not silently mis-parsed.
        assert!(matches!(
            parse_mcc_weights("p cnf 2 1\nw 1 0.75\n1 2 0\n"),
            Err(DimacsError::CachetWeightLine)
        ));
        // A plain CNF with no weight lines yields an empty weight list.
        assert!(parse_mcc_weights("p cnf 2 1\n1 2 0\n").unwrap().is_empty());
    }

    #[test]
    fn parse_dimacs_reads_declared_nvars_not_nclauses() {
        // `p cnf 5 2` declares 5 vars, 2 clauses. nvars must come from index 2 (=5),
        // not index 3 (=2). Literals only reach var 3, so a buggy parser returns 3.
        let (nvars, clauses) = parse_dimacs("p cnf 5 2\n1 -2 0\n2 3 0\n").unwrap();
        assert_eq!(nvars, 5);
        assert_eq!(clauses, vec![vec![1, -2], vec![2, 3]]);
    }
}
