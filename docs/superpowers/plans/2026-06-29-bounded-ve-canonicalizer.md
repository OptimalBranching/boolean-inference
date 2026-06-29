# Bounded-Width VE Canonicalizer (Rust port) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Julia's `bounded_ve_canonicalize` — static, weighted-min-fill, width-bounded variable elimination — to the Rust solver, shrinking the branch set and producing coarser tensors before search.

**Architecture:** A self-contained `src/canonicalize.rs` runs the VE loop over working copies of the network, reusing `contract.rs`'s relational join for each bucket contraction and `network.rs::setup_problem` for the final dedup/compression; the `solve_circuit` example optionally canonicalizes (factor bits protected) before `bbsat`.

**Tech Stack:** Rust (crate `boolean-inference`), `cargo test`. Branch: `framework-apply-branch`. Reuses `optimal-branching-core`.

**Spec:** `docs/superpowers/specs/2026-06-29-bounded-ve-canonicalizer-design.md`

## Global Constraints

- **`sc = out.len() + 1`** is the elimination width (exact for our relational engine); eliminate `v` only if `out.len() <= 32` AND `sc <= budget_b`. No OMEinsum / contraction-order optimizer.
- **Weighted-min-fill order**: pop the eligible variable with the smallest `(fill, sc)` first; `fill` = neighbor pairs in `out` not already sharing an active tensor.
- **`protected` variables are never eliminated** and are given in the INPUT network's compressed-var space.
- **Correctness gate**: the set of satisfying assignments *projected onto protected variables* is identical before and after canonicalization.
- **Reuse, don't reinvent**: bucket contraction via `contract.rs` `Relation`/`join_all`; final compression + `orig_to_new` via `setup_problem`, then compose with the input's `orig_to_new`.
- `ConstraintNetwork.orig_to_new` is `Vec<Option<usize>>` (`None` = variable absent/compressed out).
- All `cargo` commands run from `/Users/xiweipan/Codes/boolean-inference`. The default `solve_dimacs` path is unchanged.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/contract.rs` | relational contraction primitives | add `dense_relation`; make `join_all` public (Task 1) |
| `src/canonicalize.rs` | the bounded-VE canonicalizer | **new** (Task 2) |
| `src/lib.rs` | module list | add `pub mod canonicalize;` (Task 2) |
| `examples/solve_circuit.rs` | CLI demo / factoring runner | optional `budget_B` arg + protect factor bits + canonicalize (Task 3) |
| `tests/factoring.rs` | end-to-end factoring | add a canonicalize→solve test (Task 3) |

---

### Task 1: `dense_relation` + public `join_all` (`src/contract.rs`)

The VE loop builds relations from raw `(var_axes, dense)` tensors that don't live in any `ConstraintNetwork`, so it can't use `tensor_relation` (which reads `cn.support`). Add `dense_relation`, and expose `join_all` for the bucket contraction.

**Files:**
- Modify: `src/contract.rs` (add `dense_relation`; change `fn join_all` → `pub fn join_all`)

**Interfaces:**
- Consumes: `Relation { pub vars: Vec<usize>, pub rows: Vec<u64> }` (existing).
- Produces: `pub fn dense_relation(var_axes: &[usize], dense: &[bool]) -> Relation`; `pub fn join_all(rels: Vec<Relation>) -> Relation`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/contract.rs`:
```rust
    #[test]
    fn dense_relation_reencodes_unsorted_axes() {
        // Tensor over var_axes = [2, 0] (UNSORTED), dense over (bit0=v2, bit1=v0).
        // dense true at config 0b10 (v2=0,v0=1) and 0b01 (v2=1,v0=0).
        // Relation must be over sorted vars [0, 2] with rows re-encoded:
        //   (v0=1,v2=0) -> bit0(v0)=1,bit1(v2)=0 -> 0b01 = 1
        //   (v0=0,v2=1) -> bit0(v0)=0,bit1(v2)=1 -> 0b10 = 2
        let dense = vec![false, true, true, false]; // idx: 00,01,10,11 over (v2,v0)
        let rel = dense_relation(&[2, 0], &dense);
        assert_eq!(rel.vars, vec![0, 2]);
        let mut rows = rel.rows.clone();
        rows.sort_unstable();
        assert_eq!(rows, vec![1u64, 2u64]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib contract::tests::dense_relation_reencodes_unsorted_axes`
Expected: FAIL — `dense_relation` not found.

- [ ] **Step 3: Implement**

Make `join_all` public — change its signature line in `src/contract.rs` from:
```rust
fn join_all(mut rels: Vec<Relation>) -> Relation {
```
to:
```rust
pub fn join_all(mut rels: Vec<Relation>) -> Relation {
```

Add `dense_relation` (place it just after `tensor_relation`):
```rust
/// The full boolean relation of a dense truth table over `var_axes` (no domain
/// slicing): rows = configs where `dense` is true, re-encoded over `var_axes` SORTED
/// ascending (canonical bit order, matching `tensor_relation`), deduplicated.
pub fn dense_relation(var_axes: &[usize], dense: &[bool]) -> Relation {
    let mut fv: Vec<(usize, usize)> = var_axes
        .iter()
        .enumerate()
        .map(|(pos, &v)| (v, pos))
        .collect();
    fv.sort_unstable_by_key(|&(v, _)| v);
    let vars: Vec<usize> = fv.iter().map(|&(v, _)| v).collect();

    let mut rows: Vec<u64> = Vec::new();
    for (config, &sat) in dense.iter().enumerate() {
        if !sat {
            continue;
        }
        let mut row = 0u64;
        for (j, &(_, pos)) in fv.iter().enumerate() {
            if (config >> pos) & 1 == 1 {
                row |= 1u64 << j;
            }
        }
        rows.push(row);
    }
    rows.sort_unstable();
    rows.dedup();
    Relation { vars, rows }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib contract`
Expected: PASS (the new test plus all existing `contract` tests).

- [ ] **Step 5: Commit**

```bash
git add src/contract.rs
git commit -m "feat(contract): dense_relation + public join_all for VE bucket contraction"
```

---

### Task 2: `bounded_ve_canonicalize` (`src/canonicalize.rs`)

The core canonicalizer + its unit tests, including the solution-preservation gate.

**Files:**
- Create: `src/canonicalize.rs`
- Modify: `src/lib.rs` (add module)

**Interfaces:**
- Consumes: `crate::contract::{dense_relation, join_all, Relation}`; `crate::network::{ConstraintNetwork, setup_problem}`.
- Produces: `pub fn bounded_ve_canonicalize(cn: &ConstraintNetwork, budget_b: usize, protected: &[usize]) -> ConstraintNetwork`.

- [ ] **Step 1: Register the module**

Add to `src/lib.rs` (alongside the other `pub mod` lines, e.g. after `pub mod api;`):
```rust
pub mod canonicalize;
```

- [ ] **Step 2: Write `src/canonicalize.rs` with the implementation and failing tests**

```rust
//! Static, width-aware constraint-network canonicalizer (bounded-width VE).
//!
//! Port of Julia `bounded_ve_canonicalize` (src/preprocessing/canonicalize.jl).
//! Eliminates a variable `v` by joining all tensors incident to `v` and projecting
//! `v` out (boolean ∃/∧), but only if the elimination width `out.len()+1` is
//! `<= budget_b` (and `out.len() <= 32`, the TensorData cap). Eligible variables are
//! removed in weighted-min-fill order. `protected` variables (read-out vars, e.g.
//! factor bits) are never eliminated and survive into the result; their values are
//! read off the result's `orig_to_new`. The elimination width `sc = out.len()+1` is
//! exact for the relational join we perform (the largest intermediate is the relation
//! over `neighbors(v) ∪ {v}`), so no contraction-order optimizer is needed.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::contract::{dense_relation, join_all};
use crate::network::{setup_problem, ConstraintNetwork};

/// A mutable tensor during elimination: axes (in cn's compressed-var space) + dense table.
struct LiveTensor {
    var_axes: Vec<usize>,
    dense: Vec<bool>,
}

/// Sorted-unique union of the incident tensors' vars, minus `v` (the produced axes).
fn out_vars(live: &[LiveTensor], tids: &[usize], v: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &t in tids {
        for &x in &live[t].var_axes {
            if x != v {
                out.push(x);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Active tensor slots incident to `v`.
fn active_incident(v2t: &[Vec<usize>], active: &[bool], v: usize) -> Vec<usize> {
    v2t[v].iter().copied().filter(|&t| active[t]).collect()
}

/// Weighted-min-fill: number of pairs in `out` that do NOT already share an active tensor.
fn fill_count(live: &[LiveTensor], v2t: &[Vec<usize>], active: &[bool], out: &[usize]) -> usize {
    let mut f = 0usize;
    for i in 0..out.len() {
        for j in (i + 1)..out.len() {
            let (a, b) = (out[i], out[j]);
            let share = v2t[a]
                .iter()
                .any(|&t| active[t] && live[t].var_axes.contains(&b));
            if !share {
                f += 1;
            }
        }
    }
    f
}

/// `Some((fill, sc))` if `v` is eligible to eliminate now, else `None`.
fn score(
    live: &[LiveTensor],
    v2t: &[Vec<usize>],
    active: &[bool],
    is_protected: &[bool],
    budget_b: usize,
    v: usize,
) -> Option<(usize, usize)> {
    if is_protected[v] {
        return None;
    }
    let tids = active_incident(v2t, active, v);
    if tids.is_empty() {
        return None;
    }
    let out = out_vars(live, &tids, v);
    let sc = out.len() + 1;
    if out.len() <= 32 && sc <= budget_b {
        Some((fill_count(live, v2t, active, &out), sc))
    } else {
        None
    }
}

/// Reshape `cn` by bucket-eliminating variables within the width `budget_b`, in
/// weighted-min-fill order. `protected` (cn compressed-var ids) are never eliminated.
/// The returned network's `orig_to_new` indexes the same original var ids as `cn`.
pub fn bounded_ve_canonicalize(
    cn: &ConstraintNetwork,
    budget_b: usize,
    protected: &[usize],
) -> ConstraintNetwork {
    let nv = cn.vars.len();

    let mut live: Vec<LiveTensor> = cn
        .tensors
        .iter()
        .map(|t| LiveTensor {
            var_axes: t.var_axes.clone(),
            dense: cn.data(t).dense.clone(),
        })
        .collect();
    let mut active: Vec<bool> = vec![true; live.len()];
    let mut v2t: Vec<Vec<usize>> = cn.v2t.clone();
    let mut is_protected = vec![false; nv];
    for &p in protected {
        if p < nv {
            is_protected[p] = true;
        }
    }

    // Min-heap on (fill, sc) via Reverse; var id is only a stable tie-break carrier.
    let mut heap: BinaryHeap<Reverse<(usize, usize, usize)>> = BinaryHeap::new();
    for v in 0..nv {
        if let Some((fill, sc)) = score(&live, &v2t, &active, &is_protected, budget_b, v) {
            heap.push(Reverse((fill, sc, v)));
        }
    }

    while let Some(Reverse((fill, sc, v))) = heap.pop() {
        // Lazy staleness: a var may have stale heap entries; only act if it is still
        // eligible with the exact (fill, sc) we popped.
        match score(&live, &v2t, &active, &is_protected, budget_b, v) {
            Some((f, s)) if f == fill && s == sc => {}
            _ => continue,
        }

        let tids = active_incident(&v2t, &active, v);
        let out = out_vars(&live, &tids, v);

        // Bucket-contract: join all incident tensors, project `v` out, densify over `out`.
        let rels: Vec<_> = tids
            .iter()
            .map(|&t| dense_relation(&live[t].var_axes, &live[t].dense))
            .collect();
        let joined = join_all(rels);
        let mut dense = vec![false; 1usize << out.len()];
        for &row in &joined.rows {
            let mut cfg = 0usize;
            for (j, &x) in out.iter().enumerate() {
                let pos = joined.vars.binary_search(&x).expect("out var present in join");
                if (row >> pos) & 1 == 1 {
                    cfg |= 1usize << j;
                }
            }
            dense[cfg] = true;
        }

        // Merge in place: reuse the first incident slot, deactivate the rest.
        let keep = tids[0];
        for &t in &tids {
            let axes = live[t].var_axes.clone();
            for x in axes {
                v2t[x].retain(|&tt| tt != t);
            }
        }
        for &t in &tids[1..] {
            active[t] = false;
        }
        live[keep] = LiveTensor {
            var_axes: out.clone(),
            dense,
        };
        for &x in &out {
            v2t[x].push(keep);
        }

        // Re-score the affected neighbors (stale entries are skipped on later pops).
        for &u in &out {
            if let Some((f, s)) = score(&live, &v2t, &active, &is_protected, budget_b, u) {
                heap.push(Reverse((f, s, u)));
            }
        }
    }

    // Finalize: hand surviving tensors to setup_problem for dedup + compression.
    let mut tv: Vec<Vec<usize>> = Vec::new();
    let mut td: Vec<Vec<bool>> = Vec::new();
    for t in 0..live.len() {
        if active[t] {
            tv.push(live[t].var_axes.clone());
            td.push(live[t].dense.clone());
        }
    }
    let new_cn = setup_problem(nv, tv, td);

    // Compose orig->cn (cn.orig_to_new) with cn->new (new_cn.orig_to_new).
    let mut orig_to_new = vec![None; cn.orig_to_new.len()];
    for (orig, &cnid) in cn.orig_to_new.iter().enumerate() {
        if let Some(c) = cnid {
            orig_to_new[orig] = new_cn.orig_to_new[c];
        }
    }
    ConstraintNetwork {
        orig_to_new,
        ..new_cn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;
    use std::collections::HashSet;

    const OR2: [bool; 4] = [false, true, true, true]; // x ∨ y

    /// All satisfying assignments of `cn` projected onto original vars `orig_vars`,
    /// as a set of bitmasks (bit j = orig_vars[j]). Brute force over compressed vars.
    fn solutions_projected(cn: &ConstraintNetwork, orig_vars: &[usize]) -> HashSet<u64> {
        let n = cn.vars.len();
        let mut out = HashSet::new();
        for cfg in 0u64..(1u64 << n) {
            let ok = cn.tensors.iter().all(|t| {
                let mut idx = 0u32;
                for (i, &v) in t.var_axes.iter().enumerate() {
                    if (cfg >> v) & 1 == 1 {
                        idx |= 1 << i;
                    }
                }
                cn.dense(t)[idx as usize]
            });
            if !ok {
                continue;
            }
            let mut key = 0u64;
            for (j, &o) in orig_vars.iter().enumerate() {
                if let Some(c) = cn.orig_to_new[o] {
                    if (cfg >> c) & 1 == 1 {
                        key |= 1u64 << j;
                    }
                }
            }
            out.insert(key);
        }
        out
    }

    #[test]
    fn eliminates_an_unprotected_chain_var() {
        // (x0∨x1)∧(x1∨x2); eliminate x1 (budget 3, nothing protected).
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![OR2.to_vec(), OR2.to_vec()]);
        let out = bounded_ve_canonicalize(&cn, 3, &[]);
        // x1 is eliminated -> only x0,x2 survive as branch vars.
        assert!(out.orig_to_new[1].is_none(), "x1 should be eliminated");
        assert!(out.orig_to_new[0].is_some() && out.orig_to_new[2].is_some());
        // Solutions over {x0,x2} preserved: (x0∨x1)∧(x1∨x2) projected to x0,x2
        // allows everything except... brute-force equality is the real check below.
        assert_eq!(
            solutions_projected(&cn, &[0, 2]),
            solutions_projected(&out, &[0, 2]),
        );
    }

    #[test]
    fn protected_var_is_never_eliminated() {
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![OR2.to_vec(), OR2.to_vec()]);
        let out = bounded_ve_canonicalize(&cn, 3, &[1]); // protect x1
        assert!(out.orig_to_new[1].is_some(), "protected x1 must survive");
    }

    #[test]
    fn budget_one_eliminates_nothing() {
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![OR2.to_vec(), OR2.to_vec()]);
        let out = bounded_ve_canonicalize(&cn, 1, &[]);
        // every elimination needs sc = out.len()+1 >= 2 > 1, so all vars survive.
        assert_eq!(out.vars.len(), cn.vars.len());
    }

    #[test]
    fn solutions_preserved_over_protected_with_elimination() {
        // 4-var chain (x0∨x1)∧(x1∨x2)∧(x2∨x3); protect {x0, x3}; budget 3.
        let cn = setup_problem(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![OR2.to_vec(), OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[0, 3]);
        assert!(out.vars.len() < cn.vars.len(), "some vars eliminated");
        assert!(out.orig_to_new[0].is_some() && out.orig_to_new[3].is_some());
        assert_eq!(
            solutions_projected(&cn, &[0, 3]),
            solutions_projected(&out, &[0, 3]),
            "solution set projected to protected vars must be preserved"
        );
    }

    #[test]
    fn produced_tensors_respect_the_budget() {
        let cn = setup_problem(
            4,
            vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            vec![OR2.to_vec(), OR2.to_vec(), OR2.to_vec()],
        );
        let out = bounded_ve_canonicalize(&cn, 3, &[]);
        for t in &out.tensors {
            assert!(t.var_axes.len() <= 3 - 1, "produced arity <= budget_b-1");
        }
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test --lib canonicalize`
Expected: PASS — all 5 canonicalize tests, including the two solution-preservation checks.

- [ ] **Step 4: Run the full suite (no regression)**

Run: `cargo test`
Expected: PASS — existing unit + integration tests unaffected (default path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/canonicalize.rs src/lib.rs
git commit -m "feat(canonicalize): bounded-width VE canonicalizer (min-fill, sc=out+1)"
```

---

### Task 3: `solve_circuit` wiring + end-to-end factoring test

Wire canonicalization into the example (protecting factor bits) and add a committed-fixture integration test.

**Files:**
- Modify: `examples/solve_circuit.rs`
- Modify: `tests/factoring.rs`

**Interfaces:**
- Consumes: `boolean_inference::canonicalize::bounded_ve_canonicalize`; `CircuitProblem { pub network, pub name_to_orig }`, `CircuitProblem::wire_value`, `network_from_circuit_sat`.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/factoring.rs`:
```rust
#[test]
fn factoring_15_canonicalized_still_solves() {
    use boolean_inference::canonicalize::bounded_ve_canonicalize;
    use boolean_inference::circuit::CircuitProblem;

    let json = include_str!("fixtures/factoring_15.circuitsat.json");
    let cp = network_from_circuit_sat(json).expect("load CircuitSAT");
    let raw_vars = cp.network.vars.len();

    // Protect the 4-bit factor wires p1..p4, q1..q4 (compressed ids).
    let mut protected = Vec::new();
    for prefix in ["p", "q"] {
        for i in 1..=4 {
            let name = format!("{prefix}{i}");
            if let Some(&orig) = cp.name_to_orig.get(&name) {
                if let Some(c) = cp.network.orig_to_new[orig] {
                    protected.push(c);
                }
            }
        }
    }
    assert!(!protected.is_empty(), "factor-bit wires must be present");

    let cn2 = bounded_ve_canonicalize(&cp.network, 10, &protected);
    assert!(cn2.vars.len() < raw_vars, "canonicalization must shrink the branch set");

    let cp2 = CircuitProblem {
        network: cn2,
        name_to_orig: cp.name_to_orig,
    };
    let mut problem = TnProblem::from_network(cp2.network.clone()).expect("root SAT");
    let solve = bbsat(
        &mut problem,
        Selector::MostOccurrence { k: 1, max_tensors: 2 },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    assert!(solve.found, "canonicalized N=15 must be SAT");
    let p = decode(&cp2, &solve.solution, "p", 4);
    let q = decode(&cp2, &solve.solution, "q", 4);
    assert_eq!(p * q, 15, "decoded factors {p} * {q} must equal 15 after VE");
}
```
(`decode` and the other imports already exist in `tests/factoring.rs`; this test reuses them.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test factoring factoring_15_canonicalized_still_solves`
Expected: FAIL — `bounded_ve_canonicalize`/`CircuitProblem` import resolves (Task 2 done) but the test is new; it should compile and run. If it fails, it indicates a real VE correctness bug to fix here.

Note: this test should actually PASS once Task 2 is in. Run it to confirm the VE preserves the factorization end-to-end.

- [ ] **Step 3: Wire canonicalization into the `solve_circuit` example**

In `examples/solve_circuit.rs`, add the import near the top:
```rust
use boolean_inference::canonicalize::bounded_ve_canonicalize;
use boolean_inference::circuit::CircuitProblem;
```

Replace the section from `let cp = network_from_circuit_sat(&json)...` through the `TnProblem::from_network(cp.network.clone())` call with:
```rust
    // Optional 4th arg: bounded-VE budget_B (0 = off).
    let budget_b: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut cp = network_from_circuit_sat(&json).expect("load CircuitSAT");
    let raw_vars = cp.network.vars.len();
    let raw_tensors = cp.network.tensors.len();

    if budget_b > 0 {
        // Protect the factor-bit wires p1..p{bits}, q1..q{bits} so they survive VE.
        let mut protected = Vec::new();
        if bits > 0 {
            for prefix in ["p", "q"] {
                for i in 1..=bits {
                    let name = format!("{prefix}{}", i);
                    if let Some(&orig) = cp.name_to_orig.get(&name) {
                        if let Some(c) = cp.network.orig_to_new[orig] {
                            protected.push(c);
                        }
                    }
                }
            }
        }
        let cn2 = bounded_ve_canonicalize(&cp.network, budget_b, &protected);
        cp = CircuitProblem {
            network: cn2,
            name_to_orig: cp.name_to_orig,
        };
    }

    let n_vars = cp.network.vars.len();
    let n_tensors = cp.network.tensors.len();
    if budget_b > 0 {
        println!(
            "canonicalized (budget_B={budget_b}): vars {raw_vars}->{n_vars}, tensors {raw_tensors}->{n_tensors}"
        );
    }

    let t0 = Instant::now();
    let mut problem = match TnProblem::from_network(cp.network.clone()) {
        Ok(p) => p,
        Err(_) => {
            println!("vars={n_vars} tensors={n_tensors} UNSAT (root contradiction)");
            return;
        }
    };
```
(The existing `n_vars`/`n_tensors` lines earlier in `main` that read `cp.network` must be removed if they now duplicate these — ensure `n_vars`/`n_tensors` are defined exactly once, after the optional canonicalize. The rest of `main` — `bbsat`, stats print, `decode` — is unchanged and now reads the possibly-canonicalized `cp`.)

- [ ] **Step 4: Build the example and run the suite**

Run: `cargo build --examples`
Expected: compiles cleanly.

Run: `cargo test`
Expected: PASS — including `factoring_15_canonicalized_still_solves` and all existing tests.

- [ ] **Step 5: Manual end-to-end check (report, do not assert)**

Run (regenerate a fixture if needed, or reuse a generated one):
`cargo run --release --example solve_circuit -- <factoring.json> <bits> difflook 12`
Expected: prints `canonicalized (budget_B=12): vars A->B, tensors C->D` with `B < A`, then `found=true` and correct `p`,`q`. Confirms the branch-set reduction + correct factoring end-to-end.

- [ ] **Step 6: Commit**

```bash
git add examples/solve_circuit.rs tests/factoring.rs
git commit -m "feat(example): canonicalize CircuitSAT before search (factor bits protected)"
```

---

## Self-Review

**1. Spec coverage:**
- `bounded_ve_canonicalize` (sc=out+1, min-fill, protected, 32-cap) → Task 2. ✓
- Reuse `join_all`/`Relation` + `dense_relation` → Task 1 + used in Task 2. ✓
- Reuse `setup_problem` + compose `orig_to_new` → Task 2 finalize. ✓
- Lazy `BinaryHeap` replacing keyed PQ → Task 2 main loop. ✓
- `solve_circuit` wiring (budget_B, protect factor bits, decode) → Task 3. ✓
- Correctness gate (solution-preservation over protected) → Task 2 tests 1 & 4; end-to-end factoring → Task 3. ✓
- Default `solve_dimacs` unchanged + no regression → Task 2 Step 4, Task 3 Step 4 (`cargo test`). ✓
- Edge cases (budget<2, protected survives, arity cap) → Task 2 tests `budget_one_eliminates_nothing`, `protected_var_is_never_eliminated`, `produced_tensors_respect_the_budget`. ✓

**2. Placeholder scan:** No TBD/"handle edge cases"/prose-only code steps; every code step has complete code. ✓

**3. Type consistency:** `bounded_ve_canonicalize(&ConstraintNetwork, usize, &[usize]) -> ConstraintNetwork`, `dense_relation(&[usize], &[bool]) -> Relation`, `join_all(Vec<Relation>) -> Relation` consistent across tasks. `orig_to_new` treated as `Vec<Option<usize>>` throughout. `CircuitProblem { network, name_to_orig }` fields match `circuit.rs`. ✓
```
