use boolean_inference::adapter::BranchSolver;
use boolean_inference::dimacs::{network_from_dimacs, parse_dimacs};
use boolean_inference::measure::Measure;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use boolean_inference::solver::bbsat;
use optimal_branching_core::GreedyMerge;

/// Ground-truth oracle: exhaustively enumerate all 2^nvars assignments.
fn brute_force_sat(nvars: usize, clauses: &[Vec<i64>]) -> bool {
    assert!(nvars <= 20, "brute force is only for small instances");
    for bits in 0u32..(1u32 << nvars) {
        let sat = clauses.iter().all(|c| {
            c.iter().any(|&lit| {
                let v = lit.unsigned_abs() as usize - 1;
                let b = (bits >> v) & 1 == 1;
                if lit > 0 {
                    b
                } else {
                    !b
                }
            })
        });
        if sat {
            return true;
        }
    }
    false
}

fn model_satisfies(assignment: &[bool], clauses: &[Vec<i64>]) -> bool {
    clauses.iter().all(|c| {
        c.iter().any(|&lit| {
            let v = lit.unsigned_abs() as usize - 1;
            if lit > 0 {
                assignment[v]
            } else {
                !assignment[v]
            }
        })
    })
}

/// DIMACS is used here only as a compact test generator for the solver core; it
/// is deliberately not part of boolean-inference's high-level input API.
fn solve_test_network(cnf: &str) -> Option<Vec<bool>> {
    let cn = network_from_dimacs(cnf).expect("parse test CNF");
    let orig_to_new = cn.orig_to_new.clone();
    let mut problem = TnProblem::from_network(cn).ok()?;
    let result = bbsat(
        &mut problem,
        Selector::MostOccurrence { max_rows: 128 },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    result.found.then(|| {
        orig_to_new
            .into_iter()
            .map(|cid| {
                cid.and_then(|id| result.solution[id].value())
                    .unwrap_or(false)
            })
            .collect()
    })
}

/// Solve `cnf`, compare the verdict to the oracle, and re-check any SAT witness.
fn check(cnf: &str) {
    let (nvars, clauses) = parse_dimacs(cnf).expect("parse");
    let oracle_sat = brute_force_sat(nvars, &clauses);
    match solve_test_network(cnf) {
        Some(a) => {
            assert!(
                oracle_sat,
                "solver said SAT but the oracle says UNSAT:\n{cnf}"
            );
            assert_eq!(a.len(), nvars, "model length mismatch:\n{cnf}");
            assert!(
                model_satisfies(&a, &clauses),
                "returned model is not satisfying:\n{cnf}"
            );
        }
        None => {
            assert!(
                !oracle_sat,
                "solver said UNSAT but the oracle found a model:\n{cnf}"
            );
        }
    }
}

#[test]
fn handcrafted_match_oracle() {
    let cases = [
        "p cnf 1 1\n1 0\n",                                  // unit -> SAT
        "p cnf 1 2\n1 0\n-1 0\n",                            // x ∧ ¬x -> UNSAT
        "p cnf 2 2\n1 2 0\n-1 -2 0\n",                       // XOR -> SAT
        "p cnf 3 3\n1 0\n-1 2 0\n-2 3 0\n",                  // unit chain -> SAT
        "p cnf 3 3\n-1 2 0\n-1 3 0\n-2 -3 0\n",              // failed literal on x1 -> SAT
        "p cnf 3 3\n1 2 0\n-2 3 0\n-1 3 0\n",                // pure 2-SAT -> SAT
        "p cnf 3 1\n1 2 3 0\n",                              // underconstrained -> SAT
        "p cnf 3 8\n1 2 3 0\n1 2 -3 0\n1 -2 3 0\n1 -2 -3 0\n-1 2 3 0\n-1 2 -3 0\n-1 -2 3 0\n-1 -2 -3 0\n", // all clauses -> UNSAT
    ];
    for cnf in cases {
        check(cnf);
    }
}

// Deterministic xorshift, no rng dependency (same generator as the perf probe).
fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Random 3-SAT with `n` vars and `m` clauses (distinct, non-tautological literals).
fn gen_3sat(n: usize, m: usize, mut s: u64) -> String {
    let mut out = format!("p cnf {n} {m}\n");
    for _ in 0..m {
        let mut lits: Vec<i64> = Vec::with_capacity(3);
        while lits.len() < 3 {
            let v = (xs(&mut s) as usize % n) + 1;
            if !lits.iter().any(|l| l.unsigned_abs() as usize == v) {
                let sign = if xs(&mut s) & 1 == 0 { 1i64 } else { -1 };
                lits.push(sign * v as i64);
            }
        }
        out.push_str(&format!("{} {} {} 0\n", lits[0], lits[1], lits[2]));
    }
    out
}

#[test]
fn random_3sat_matches_oracle() {
    // Vary the clause/var ratio across the phase transition (~4.26) where both
    // SAT and UNSAT instances are common. n kept small for the exhaustive oracle.
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut sat = 0usize;
    let mut unsat = 0usize;
    for &n in &[5usize, 8, 10] {
        for &ratio in &[3.0f64, 4.26, 5.0] {
            let m = (n as f64 * ratio).round() as usize;
            for _ in 0..25 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let cnf = gen_3sat(n, m, seed);
                let (nvars, clauses) = parse_dimacs(&cnf).unwrap();
                if brute_force_sat(nvars, &clauses) {
                    sat += 1;
                } else {
                    unsat += 1;
                }
                check(&cnf);
            }
        }
    }
    // Sanity: the sweep actually exercised BOTH verdicts (not all one-sided).
    assert!(
        sat > 0 && unsat > 0,
        "sweep was one-sided: {sat} SAT / {unsat} UNSAT"
    );
}
