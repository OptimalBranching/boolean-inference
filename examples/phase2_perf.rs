// Quick empirical look at Phase 2 hot-path cost. NOT a rigorous benchmark —
// just enough to see how per-probe cost scales with problem size, which tells
// us whether the O(n) full-vector copy + clone in `probe_assignment` dominates.
//
//   cargo run --release --example phase2_perf

use std::time::Instant;

use boolean_inference::dimacs::network_from_dimacs;
use boolean_inference::problem::{SolverBuffer, TnProblem};
use boolean_inference::propagate::probe_assignment;

// xorshift64 — deterministic, no rng dependency.
fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn gen_cnf(n: usize, m: usize, mut s: u64) -> String {
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

fn bench(n: usize) {
    // ratio 3.0 keeps it satisfiable/underconstrained so probes do real work.
    let cnf = gen_cnf(n, n * 3, 0x9E3779B97F4A7C15 ^ n as u64);
    let cn = network_from_dimacs(&cnf).expect("parse");
    let p = match TnProblem::from_network(cn) {
        Ok(p) => p,
        Err(_) => {
            println!("n={n}: UNSAT at root, skipped");
            return;
        }
    };
    let mut buf = SolverBuffer::new(&p.static_cn);
    let nv = p.doms.len();
    let nt = p.static_cn.tensors.len();
    let k = 300_000u64;
    let mut seed = 1u64;
    let mut sink = 0u64;
    let t = Instant::now();
    for _ in 0..k {
        let v = (xs(&mut seed) as usize) % nv;
        let bit = (xs(&mut seed) >> 3) & 1;
        let res = probe_assignment(&p.static_cn, &mut buf, &p.doms, &[v], 1, bit);
        sink = sink.wrapping_add(res[v].0 as u64); // keep result live
    }
    let dt = t.elapsed();
    println!(
        "n={:>5} vars, {:>5} tensors  ->  {:>7.0} ns/probe   (sink={})",
        nv,
        nt,
        dt.as_nanos() as f64 / k as f64,
        sink
    );
}

fn main() {
    println!("\nPhase 2 probe_assignment cost vs size (single-var probe, release build)\n");
    for n in [200usize, 800, 3200, 12800] {
        bench(n);
    }
    println!("\nIf ns/probe grows ~linearly with vars despite probing only ONE var,");
    println!("the per-call full-vector copy + Vec clone is the dominant cost.\n");
}
