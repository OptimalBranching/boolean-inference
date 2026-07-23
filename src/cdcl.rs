//! Persistent CDCL propagation for cube generation.
//!
//! The cuber owns one CaDiCaL instance for the whole run. Every query applies
//! the current cube decisions as assumptions and invokes CaDiCaL's standard
//! assumptions-propagation entry point. Native implications are deliberately
//! not promoted to assumptions: CaDiCaL must reconstruct their reasons so
//! conflict analysis learns clauses over the actual cube decisions. The entry
//! point performs BCP and conflict analysis, so globally valid learned clauses
//! remain in the solver for later nodes. It stops after the assumptions: the
//! cuber never calls `solve` or lets CaDiCaL make search decisions beyond the
//! cube.

use std::cell::{Cell, RefCell};
use std::io::BufRead;
use std::path::Path;
use std::rc::Rc;

use optimal_branching_core::Clause as BranchClause;
use rustsat::instances::SatInstance;
use rustsat::solvers::{FreezeVar, GetInternalStats, Learn, Propagate, Solve};
use rustsat::types::{Lit, TernaryVal, Var};
use rustsat_cadical::{CaDiCaL, Config};

use crate::domain::DomainMask;

/// Aggregate counters from the persistent CaDiCaL instance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CdclStats {
    /// Number of assumption-propagation queries issued by the cuber.
    pub propagation_calls: u64,
    /// Queries whose BCP/conflict-analysis result refuted the assumptions.
    pub propagation_conflicts: u64,
    /// Total number of literals supplied as assumptions across all queries.
    pub assumption_literals: u64,
    /// Full CDCL searches started inside the cuber. This is an architectural
    /// invariant and is always zero.
    pub full_search_calls: u64,
    pub conflicts: u64,
    pub decisions: u64,
    pub propagations: u64,
    /// Clauses reported by CaDiCaL's learner callback over the whole run.
    pub total_learned_clauses: u64,
    /// Redundant clauses currently retained in CaDiCaL's clause database.
    pub current_redundant_clauses: usize,
}

/// Cloneable handle to one persistent, single-threaded CaDiCaL instance.
///
/// Region scoring and cube generation are single-threaded. Clones therefore
/// share the solver through `Rc<RefCell<_>>`; every learned clause immediately
/// benefits later committed branches and, when enabled, later candidate probes.
#[derive(Clone)]
pub struct CdclPropagator {
    inner: Rc<CdclInner>,
}

struct CdclInner {
    solver: RefCell<CaDiCaL<'static, 'static>>,
    native_to_cnf: Vec<usize>,
    cnf_to_native: Vec<Option<usize>>,
    propagation_calls: Cell<u64>,
    propagation_conflicts: Cell<u64>,
    assumption_literals: Cell<u64>,
    total_learned_clauses: Rc<Cell<u64>>,
}

impl CdclPropagator {
    /// Load a DIMACS formula. `native_to_cnf[v]` is the zero-based DIMACS
    /// variable corresponding to compressed native variable `v`.
    pub fn from_dimacs_path(path: &Path, native_to_cnf: Vec<usize>) -> Result<Self, String> {
        let instance = SatInstance::from_dimacs_path(path)
            .map_err(|error| format!("parse CDCL CNF {}: {error}", path.display()))?;
        Self::from_instance(instance, native_to_cnf)
    }

    /// Reader form used by tests and embedders.
    pub fn from_dimacs<R: BufRead>(
        reader: &mut R,
        native_to_cnf: Vec<usize>,
    ) -> Result<Self, String> {
        let instance =
            SatInstance::from_dimacs(reader).map_err(|error| format!("parse CDCL CNF: {error}"))?;
        Self::from_instance(instance, native_to_cnf)
    }

    fn from_instance(instance: SatInstance, native_to_cnf: Vec<usize>) -> Result<Self, String> {
        let formula_vars = instance.max_var().map_or(0, |var| var.idx() + 1);
        let mapped_vars = native_to_cnf.iter().copied().max().map_or(0, |var| var + 1);
        let n_cnf_vars = formula_vars.max(mapped_vars);
        let mut seen = vec![false; n_cnf_vars];
        for &cnf_var in &native_to_cnf {
            if cnf_var > Var::MAX_IDX as usize {
                return Err(format!(
                    "flattened CNF variable {} exceeds the RustSAT limit",
                    cnf_var + 1
                ));
            }
            if std::mem::replace(&mut seen[cnf_var], true) {
                return Err(format!(
                    "two native variables map to flattened CNF variable {}",
                    cnf_var + 1
                ));
            }
        }

        let mut solver = CaDiCaL::default();
        solver
            .set_configuration(Config::Default)
            .map_err(|error| format!("configure CaDiCaL: {error}"))?;
        if n_cnf_vars > 0 {
            solver
                .reserve(Var::new((n_cnf_vars - 1) as u32))
                .map_err(|error| format!("reserve CaDiCaL variables: {error}"))?;
        }
        for clause in instance.cnf() {
            solver
                .add_clause_ref(clause)
                .map_err(|error| format!("load CaDiCaL clause: {error}"))?;
        }
        // Native variables recur as assumptions and must keep stable external
        // identities across CaDiCaL inprocessing rounds.
        for &cnf_var in &native_to_cnf {
            solver
                .freeze_var(Var::new(cnf_var as u32))
                .map_err(|error| format!("freeze CaDiCaL variable {}: {error}", cnf_var + 1))?;
        }
        let total_learned_clauses = Rc::new(Cell::new(0u64));
        let learner_count = Rc::clone(&total_learned_clauses);
        solver.attach_learner(
            move |_| learner_count.set(learner_count.get().saturating_add(1)),
            n_cnf_vars,
        );
        let mut cnf_to_native = vec![None; n_cnf_vars];
        for (native, &cnf_var) in native_to_cnf.iter().enumerate() {
            cnf_to_native[cnf_var] = Some(native);
        }

        Ok(Self {
            inner: Rc::new(CdclInner {
                solver: RefCell::new(solver),
                native_to_cnf,
                cnf_to_native,
                propagation_calls: Cell::new(0),
                propagation_conflicts: Cell::new(0),
                assumption_literals: Cell::new(0),
                total_learned_clauses,
            }),
        })
    }

    /// Propagate a hypothetical optimal-branching clause from `base`.
    pub fn propagate_clause(
        &self,
        base: &[DomainMask],
        prefix: &[(usize, bool)],
        clause: &BranchClause,
        variables: &[usize],
    ) -> Result<Vec<DomainMask>, String> {
        let mut decisions = Vec::with_capacity(prefix.len() + clause.mask.count_ones() as usize);
        decisions.extend_from_slice(prefix);
        for (index, &var) in variables.iter().enumerate() {
            if (clause.mask >> index) & 1 != 0 {
                decisions.push((var, (clause.val >> index) & 1 != 0));
            }
        }
        self.propagate_decisions(base, &decisions)
    }

    /// Propagate the explicit cube `decisions` and overlay the resulting native
    /// implications on `base`.
    ///
    /// Only `decisions` are passed to CaDiCaL as assumptions. Fixed values in
    /// `base` are a projection maintained by native propagation and are kept in
    /// the returned snapshot, but are not turned into artificial decision
    /// levels. This distinction is essential for useful first-UIP learning.
    ///
    /// A conflict is represented by the existing solver convention
    /// `snapshot[0] == DomainMask::NONE`. CaDiCaL's propagation call analyzes
    /// such a conflict before returning, retaining the learned clause globally.
    pub fn propagate_decisions(
        &self,
        base: &[DomainMask],
        decisions: &[(usize, bool)],
    ) -> Result<Vec<DomainMask>, String> {
        if base.len() != self.inner.native_to_cnf.len() {
            return Err(format!(
                "native domain length {} does not match CDCL map length {}",
                base.len(),
                self.inner.native_to_cnf.len()
            ));
        }

        let mut snapshot = base.to_vec();
        for &(var, value) in decisions {
            let requested = fixed_domain(value);
            match snapshot.get_mut(var) {
                Some(domain) if *domain == DomainMask::BOTH || *domain == requested => {
                    *domain = requested;
                }
                Some(_) => {
                    mark_conflict(&mut snapshot);
                    return Ok(snapshot);
                }
                None => return Err(format!("native branch variable {var} is out of range")),
            }
        }

        let assumptions = assumptions_from_decisions(&self.inner, decisions)?;
        self.inner
            .propagation_calls
            .set(self.inner.propagation_calls.get() + 1);
        self.inner.assumption_literals.set(
            self.inner.assumption_literals.get()
                + u64::try_from(assumptions.len()).unwrap_or(u64::MAX),
        );

        let mut solver = self.inner.solver.borrow_mut();
        let result = solver
            .propagate(&assumptions, false)
            .map_err(|error| format!("CaDiCaL branch propagation failed: {error}"))?;
        if result.conflict {
            self.inner
                .propagation_conflicts
                .set(self.inner.propagation_conflicts.get() + 1);
            // Assumption propagation analyzes conflicts outside CaDiCaL's
            // normal search loop. Run its own scheduled reduction policy at
            // this reset-to-root boundary so retained clauses stay managed.
            solver.maintain_learned_clauses();
            mark_conflict(&mut snapshot);
            return Ok(snapshot);
        }

        // `propagated` contains the assumption trail. `current_lit_val` also
        // exposes root-fixed projected variables that predate the first
        // assumption and therefore may not appear in that returned suffix.
        for (native, &cnf_var) in self.inner.native_to_cnf.iter().enumerate() {
            let literal = Var::new(cnf_var as u32).pos_lit();
            let implied = match solver.current_lit_val(literal) {
                TernaryVal::True => Some(DomainMask::D1),
                TernaryVal::False => Some(DomainMask::D0),
                TernaryVal::DontCare => None,
            };
            if let Some(implied) = implied {
                let domain = &mut snapshot[native];
                if *domain == DomainMask::BOTH || *domain == implied {
                    *domain = implied;
                } else {
                    mark_conflict(&mut snapshot);
                    return Ok(snapshot);
                }
            }
        }

        for literal in result.propagated {
            let Some(Some(native)) = self.inner.cnf_to_native.get(literal.vidx()) else {
                continue;
            };
            let implied = fixed_domain(literal.is_pos());
            let domain = &mut snapshot[*native];
            if *domain == DomainMask::BOTH || *domain == implied {
                *domain = implied;
            } else {
                mark_conflict(&mut snapshot);
                break;
            }
        }
        Ok(snapshot)
    }

    pub fn stats(&self) -> CdclStats {
        let solver = self.inner.solver.borrow();
        CdclStats {
            propagation_calls: self.inner.propagation_calls.get(),
            propagation_conflicts: self.inner.propagation_conflicts.get(),
            assumption_literals: self.inner.assumption_literals.get(),
            full_search_calls: 0,
            conflicts: solver.conflicts().try_into().unwrap_or(u64::MAX),
            decisions: solver.decisions().try_into().unwrap_or(u64::MAX),
            propagations: solver.propagations().try_into().unwrap_or(u64::MAX),
            total_learned_clauses: self.inner.total_learned_clauses.get(),
            current_redundant_clauses: solver.get_redundant().max(0) as usize,
        }
    }
}

fn assumptions_from_decisions(
    inner: &CdclInner,
    decisions: &[(usize, bool)],
) -> Result<Vec<Lit>, String> {
    let mut assumptions = Vec::with_capacity(decisions.len());
    for &(native, value) in decisions {
        if native >= inner.native_to_cnf.len() {
            return Err(format!("native decision variable {native} is out of range"));
        }
        let cnf_var: u32 = inner.native_to_cnf[native]
            .try_into()
            .map_err(|_| "flattened CNF variable exceeds u32".to_string())?;
        assumptions.push(Lit::new(cnf_var, !value));
    }
    Ok(assumptions)
}

fn mark_conflict(doms: &mut [DomainMask]) {
    if let Some(sentinel) = doms.first_mut() {
        *sentinel = DomainMask::NONE;
    }
}

#[inline]
fn fixed_domain(value: bool) -> DomainMask {
    if value {
        DomainMask::D1
    } else {
        DomainMask::D0
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn solver(cnf: &str, variables: usize) -> CdclPropagator {
        CdclPropagator::from_dimacs(&mut Cursor::new(cnf.as_bytes()), (0..variables).collect())
            .unwrap()
    }

    #[test]
    fn projects_gate_implications_to_native_domains() {
        // z <-> (a AND b):
        // (¬z∨a)(¬z∨b)(z∨¬a∨¬b)
        let cdcl = solver("p cnf 3 3\n-3 1 0\n-3 2 0\n3 -1 -2 0\n", 3);
        let base = vec![DomainMask::BOTH; 3];

        let z_true = cdcl.propagate_decisions(&base, &[(2, true)]).unwrap();
        assert_eq!(z_true, vec![DomainMask::D1, DomainMask::D1, DomainMask::D1]);

        let a_false = cdcl.propagate_decisions(&base, &[(0, false)]).unwrap();
        assert_eq!(a_false[0], DomainMask::D0);
        assert_eq!(a_false[2], DomainMask::D0);
    }

    #[test]
    fn repeated_probes_do_not_leak_assumptions() {
        let cdcl = solver("p cnf 2 1\n1 2 0\n", 2);
        let base = vec![DomainMask::BOTH; 2];
        let x0_false = cdcl.propagate_decisions(&base, &[(0, false)]).unwrap();
        assert_eq!(x0_false, vec![DomainMask::D0, DomainMask::D1]);

        let x0_true = cdcl.propagate_decisions(&base, &[(0, true)]).unwrap();
        assert_eq!(x0_true[0], DomainMask::D1);
        assert_eq!(x0_true[1], DomainMask::BOTH);
    }

    #[test]
    fn contradictory_assumptions_return_the_native_sentinel() {
        let cdcl = solver("p cnf 1 1\n1 0\n", 1);
        let result = cdcl
            .propagate_decisions(&[DomainMask::BOTH], &[(0, false)])
            .unwrap();
        assert_eq!(result, vec![DomainMask::NONE]);
    }

    #[test]
    fn auxiliary_implications_are_projected_but_auxiliaries_stay_hidden() {
        // Native variables are a,b,c (CNF 1,2,3); variable 4 is a Tseitin
        // auxiliary. a -> aux -> c, while b is unrelated.
        let cdcl = solver("p cnf 4 4\n-1 4 0\n1 -4 0\n-4 3 0\n4 -3 0\n", 3);
        let result = cdcl
            .propagate_decisions(&[DomainMask::BOTH; 3], &[(0, true)])
            .unwrap();
        assert_eq!(result[0], DomainMask::D1);
        assert_eq!(result[1], DomainMask::BOTH);
        assert_eq!(result[2], DomainMask::D1);
    }

    #[test]
    fn propagation_conflict_learns_for_later_parent_bcp() {
        // Resolving the three clauses shows that the formula entails a. Root
        // BCP initially cannot see it. Under a=0 it forces both b and c before
        // conflicting, producing the learned unit a.
        let cdcl = solver("p cnf 3 3\n1 2 0\n1 3 0\n1 -2 -3 0\n", 3);
        let parent = vec![DomainMask::BOTH; 3];
        assert_eq!(cdcl.propagate_decisions(&parent, &[]).unwrap(), parent);

        let child = cdcl.propagate_decisions(&parent, &[(0, false)]).unwrap();
        assert_eq!(child[0], DomainMask::NONE);

        let after = cdcl.propagate_decisions(&parent, &[]).unwrap();
        assert_eq!(after[0], DomainMask::D1);
        let stats = cdcl.stats();
        assert_eq!(stats.propagation_conflicts, 1);
        assert_eq!(stats.full_search_calls, 0);
        assert!(stats.conflicts >= 1);
        assert!(stats.total_learned_clauses >= 1);
    }

    #[test]
    fn open_formula_is_not_solved_by_the_cuber_cdcl() {
        // No root implication. A full CDCL solve could return SAT immediately,
        // but branch propagation must leave the formula open.
        let cdcl = solver("p cnf 3 2\n1 2 0\n-1 2 0\n", 3);
        let base = vec![DomainMask::BOTH; 3];
        assert_eq!(cdcl.propagate_decisions(&base, &[]).unwrap(), base);
        let stats = cdcl.stats();
        assert_eq!(stats.full_search_calls, 0);
        assert_eq!(stats.propagation_calls, 1);
    }

    #[test]
    fn projected_implications_are_not_reintroduced_as_assumptions() {
        let cdcl = solver("p cnf 3 2\n-1 2 0\n-2 3 0\n", 3);
        let projected = vec![DomainMask::D1, DomainMask::D1, DomainMask::D1];
        let result = cdcl.propagate_decisions(&projected, &[(0, true)]).unwrap();
        assert_eq!(result, projected);
        assert_eq!(cdcl.stats().assumption_literals, 1);
    }
}
