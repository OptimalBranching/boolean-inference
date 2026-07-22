//! Parser for the repository's native extensional Boolean CSP format.
//!
//! The first non-empty line is the number of original variables. Every later
//! line is `<v0> <v1> ... : <cfg0> <cfg1> ...`, where variables are zero-based
//! and each configuration is a bitmask over the listed scope. Unlike DIMACS,
//! this preserves each local relation as one tensor.

use crate::network::{assemble, ConstraintNetwork};

#[derive(Debug, thiserror::Error)]
pub enum CspError {
    #[error("missing variable-count header")]
    MissingHeader,
    #[error("line {0}: expected '<scope> : <allowed configurations>'")]
    MissingColon(usize),
    #[error("line {line}: malformed {kind} token '{token}'")]
    BadToken {
        line: usize,
        kind: &'static str,
        token: String,
    },
    #[error("line {line}: variable {variable} is outside 0..{nvars}")]
    VariableOutOfRange {
        line: usize,
        variable: usize,
        nvars: usize,
    },
    #[error("line {0}: constraint scope is empty")]
    EmptyScope(usize),
    #[error("line {line}: constraint arity {arity} exceeds 32")]
    ArityTooLarge { line: usize, arity: usize },
    #[error("line {line}: constraint scope contains duplicate variable {variable}")]
    DuplicateVariable { line: usize, variable: usize },
    #[error("line {line}: configuration {config} exceeds {arity}-variable scope")]
    ConfigurationOutOfRange {
        line: usize,
        config: u64,
        arity: usize,
    },
}

fn data_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().filter_map(|(offset, raw)| {
        let line = raw.trim();
        (!line.is_empty() && !line.starts_with('#')).then_some((offset + 1, line))
    })
}

fn parse_usize(token: &str, line: usize, kind: &'static str) -> Result<usize, CspError> {
    token.parse().map_err(|_| CspError::BadToken {
        line,
        kind,
        token: token.to_string(),
    })
}

/// Parse a native extensional Boolean CSP into a constraint network.
pub fn network_from_csp(text: &str) -> Result<ConstraintNetwork, CspError> {
    let mut lines = data_lines(text);
    let (header_line, header) = lines.next().ok_or(CspError::MissingHeader)?;
    let nvars = parse_usize(header, header_line, "variable-count")?;
    let mut tensors = Vec::new();

    for (line_number, line) in lines {
        let (scope_text, support_text) = line
            .split_once(':')
            .ok_or(CspError::MissingColon(line_number))?;
        let scope: Vec<usize> = scope_text
            .split_whitespace()
            .map(|token| parse_usize(token, line_number, "variable"))
            .collect::<Result<_, _>>()?;
        if scope.is_empty() {
            return Err(CspError::EmptyScope(line_number));
        }
        if scope.len() > 32 {
            return Err(CspError::ArityTooLarge {
                line: line_number,
                arity: scope.len(),
            });
        }
        let mut seen = std::collections::HashSet::with_capacity(scope.len());
        for &variable in &scope {
            if variable >= nvars {
                return Err(CspError::VariableOutOfRange {
                    line: line_number,
                    variable,
                    nvars,
                });
            }
            if !seen.insert(variable) {
                return Err(CspError::DuplicateVariable {
                    line: line_number,
                    variable,
                });
            }
        }

        let limit = 1u64 << scope.len();
        let mut support: Vec<u32> = support_text
            .split_whitespace()
            .map(|token| {
                let config = token.parse::<u64>().map_err(|_| CspError::BadToken {
                    line: line_number,
                    kind: "configuration",
                    token: token.to_string(),
                })?;
                if config >= limit {
                    return Err(CspError::ConfigurationOutOfRange {
                        line: line_number,
                        config,
                        arity: scope.len(),
                    });
                }
                Ok(config as u32)
            })
            .collect::<Result<_, _>>()?;
        support.sort_unstable();
        support.dedup();
        tensors.push((scope, support));
    }

    Ok(assemble(nvars, tensors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native_relations_without_flattening() {
        let network = network_from_csp("4\n0 1 : 1 2\n1 2 3 : 0 3 5 6\n").unwrap();
        assert_eq!(network.n_vars, 4);
        assert_eq!(network.tensors.len(), 2);
        assert_eq!(network.tensors[0].var_axes, vec![0, 1]);
        assert_eq!(network.support(&network.tensors[0]), &[1, 2]);
        assert_eq!(network.tensors[1].var_axes, vec![1, 2, 3]);
        assert_eq!(network.support(&network.tensors[1]), &[0, 3, 5, 6]);
    }

    #[test]
    fn rejects_out_of_scope_configuration_and_variable() {
        assert!(matches!(
            network_from_csp("2\n0 1 : 4\n"),
            Err(CspError::ConfigurationOutOfRange { .. })
        ));
        assert!(matches!(
            network_from_csp("2\n0 2 : 0\n"),
            Err(CspError::VariableOutOfRange { .. })
        ));
    }
}
