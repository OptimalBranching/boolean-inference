// Phase 3 demonstration: from a CNF + a focus variable, watch the solver
//   (1) grow a LOCAL region around the variable,
//   (2) CONTRACT that region into the assignments that are even possible,
//   (3) turn those into an optimal BRANCHING RULE (what the search tries).
//
// Run with:  cargo run --example phase3_demo
//
// Every number printed here is computed live — nothing is hand-written. The
// point is to *see* the region → contraction → branching pipeline on tiny
// instances you can check by hand.

use boolean_inference::contract::contract_region;
use boolean_inference::dimacs::network_from_dimacs;
use boolean_inference::measure::{measure_core, Measure};
use boolean_inference::network::ConstraintNetwork;
use boolean_inference::problem::{SolverBuffer, TnProblem};
use boolean_inference::region::{k_neighboring, RegionCache};
use boolean_inference::table::compute_branching_result;
use optimal_branching_core::{Clause, IPSolver};

/// Map a compressed internal variable id back to its 1-based DIMACS number,
/// so everything we print reads in the same terms as the clauses.
fn new_to_dimacs(cn: &ConstraintNetwork) -> Vec<usize> {
    let n = cn.vars.len();
    let mut map = vec![0usize; n];
    for (orig, slot) in cn.orig_to_new.iter().enumerate() {
        if let Some(nid) = slot {
            map[*nid] = orig + 1; // DIMACS vars are 1-based
        }
    }
    map
}

fn vars_str(vars: &[usize], n2d: &[usize]) -> String {
    vars.iter()
        .map(|&v| format!("x{}", n2d[v]))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One satisfiable region configuration as `x2=0 x3=1 ...` over `vars`.
fn config_str(cfg: u64, vars: &[usize], n2d: &[usize]) -> String {
    vars.iter()
        .enumerate()
        .map(|(i, &v)| format!("x{}={}", n2d[v], (cfg >> i) & 1))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One branching clause as a conjunction of the literals it fixes.
fn clause_str(cl: &Clause, vars: &[usize], n2d: &[usize]) -> String {
    let mut lits = Vec::new();
    for (i, &v) in vars.iter().enumerate() {
        if (cl.mask >> i) & 1 == 1 {
            lits.push(format!("x{}={}", n2d[v], (cl.val >> i) & 1));
        }
    }
    if lits.is_empty() {
        "(no literal)".to_string()
    } else {
        lits.join(" ∧ ")
    }
}

fn region_demo(title: &str, clauses_human: &str, cnf: &str, focus_1based: usize) {
    println!("== {title} ==");
    println!("clauses : {clauses_human}");
    let cn = network_from_dimacs(cnf).expect("parse");
    let n2d = new_to_dimacs(&cn);
    let doms = vec![boolean_inference::domain::DomainMask::BOTH; cn.vars.len()];
    let focus = cn.orig_to_new[focus_1based - 1].expect("focus var present");
    for k in [1usize, 2] {
        let r = k_neighboring(&cn, &doms, focus, 16, k);
        let clause_ids: Vec<String> = r.tensors.iter().map(|t| format!("#{t}")).collect();
        println!(
            "focus x{focus_1based}, k={k} : region vars {{{}}}  from clauses [{}]",
            vars_str(&r.vars, &n2d),
            clause_ids.join(", ")
        );
    }
    println!();
}

fn contract_demo(title: &str, clauses_human: &str, cnf: &str, focus_1based: usize) {
    println!("== {title} ==");
    println!("clauses : {clauses_human}");
    let cn = network_from_dimacs(cnf).expect("parse");
    let n2d = new_to_dimacs(&cn);
    let doms = vec![boolean_inference::domain::DomainMask::BOTH; cn.vars.len()];
    let focus = cn.orig_to_new[focus_1based - 1].expect("focus var present");
    let region = k_neighboring(&cn, &doms, focus, 16, 2);
    let (configs, output_vars) = contract_region(&cn, &region, &doms);
    let total = 1u64 << output_vars.len();
    println!(
        "region  : vars {{{}}}  ({} possible patterns a priori)",
        vars_str(&output_vars, &n2d),
        total
    );
    println!(
        "possible: {} of {total} survive the constraints —",
        configs.len()
    );
    for c in &configs {
        println!("            {}", config_str(*c, &output_vars, &n2d));
    }
    println!();
}

fn branch_demo(title: &str, clauses_human: &str, cnf: &str, focus_1based: usize) {
    println!("== {title} ==");
    println!("clauses : {clauses_human}");
    let cn = network_from_dimacs(cnf).expect("parse");
    let n2d = new_to_dimacs(&cn);
    let p = TnProblem::from_network(cn).expect("root SAT");
    let focus = p.static_cn.orig_to_new[focus_1based - 1].expect("focus var present");
    let mut cache = RegionCache::new(&p.static_cn, &p.doms, 2, 16);
    let mut buf = SolverBuffer::new(&p.static_cn);
    let hardness = measure_core(&p.static_cn, &p.doms, Measure::NumUnfixedVars);
    let (clauses, vars) = compute_branching_result(
        &mut cache,
        &p.static_cn,
        &p.doms,
        &mut buf,
        focus,
        Measure::NumUnfixedVars,
        &IPSolver::default(),
    );
    println!(
        "focus   : x{focus_1based}   (region vars {{{}}})",
        vars_str(&vars, &n2d)
    );
    println!("measure : {hardness} unfixed vars at this node");
    match clauses {
        None => println!("rule    : none — region has no live branch here"),
        Some(cls) => {
            println!("rule    : branch {} way(s) —", cls.len());
            for (i, c) in cls.iter().enumerate() {
                println!(
                    "            branch {}: {}",
                    i + 1,
                    clause_str(c, &vars, &n2d)
                );
            }
        }
    }
    println!();
}

fn main() {
    println!("\nboolean-inference · Phase 3 live demo\n");

    // ---- Part A: region growth (local window around a variable) ----
    region_demo(
        "Region growth — a local window that widens with k",
        "(x1∨x2) ∧ (x2∨x3) ∧ (x3∨x4) ∧ (x4∨x5)",
        "p cnf 5 4\n1 2 0\n2 3 0\n3 4 0\n4 5 0\n",
        3,
    );

    // ---- Part B: contraction (which local assignments are even possible) ----
    contract_demo(
        "Contraction — 8 patterns collapse to the 3 that are consistent",
        "(¬x1∨x2) ∧ (¬x2∨x3) ∧ (¬x1∨¬x3)",
        "p cnf 3 3\n-1 2 0\n-2 3 0\n-1 -3 0\n",
        1,
    );

    // ---- Part C: branching rule (what the search actually tries) ----
    branch_demo(
        "Branching rule — XOR forces a 2-way exhaustive split",
        "(x1∨x2) ∧ (¬x1∨¬x2)   — i.e. x1 ⊕ x2",
        "p cnf 2 2\n1 2 0\n-1 -2 0\n",
        1,
    );

    branch_demo(
        "Branching rule — a 3-clause neighborhood",
        "(x1∨x2∨x3) ∧ (¬x1∨x2) ∧ (¬x2∨x3)",
        "p cnf 3 3\n1 2 3 0\n-1 2 0\n-2 3 0\n",
        2,
    );
}
