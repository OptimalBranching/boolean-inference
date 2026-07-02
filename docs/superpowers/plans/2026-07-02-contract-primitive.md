# Contract Primitive + End-to-End-Sparse Canonicalize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract one `contract(rels, keep)` primitive shared by `contract_region` and `canonicalize`'s VE step, and make `canonicalize` operate on sparse relations end to end (no `2^arity` dense allocation on its path).

**Architecture:** Add `Relation::project` and `contract = join_all + project` (binary-join internals) in `src/contract.rs`. Refactor `src/network.rs` so a shared `assemble` core builds a `ConstraintNetwork` from `(var_axes, support)` pairs, fed by both the existing dense `setup_problem` and a new sparse `setup_from_relations`. Rewrite `canonicalize` onto `contract` + `setup_from_relations`, dropping the `LiveTensor` wrapper and the dense round-trip.

**Tech Stack:** Rust, `cargo test` / `cargo check`. No new dependencies.

## Global Constraints

- **Behavior-preserving.** Node counts, factoring solutions, and every existing test must be unchanged. This is a refactor.
- **Binary-join internals only.** The `contract` kernel is `join_all(rels).project(keep)`. No generic/worst-case-optimal join (deferred). The signature must allow a generic-join kernel later without changing callers.
- **Dedup key is `(var_axes.len(), support)`** — support alone is arity-ambiguous.
- **Support is strictly ascending.** `Relation.rows` and `TensorData.support` are ascending; `is_sat` binary-search relies on it. Both construction paths must preserve this.
- `cargo check --all-targets` must stay warning-free; do not introduce new clippy warnings.

---

### Task 1: `Relation::project`

**Files:**
- Modify: `src/contract.rs` (add an `impl Relation` block after the `Relation` struct at lines 11-15; add a test in the `#[cfg(test)] mod tests` block)

**Interfaces:**
- Consumes: `Relation { pub vars: Vec<usize>, pub rows: Vec<u64> }` (existing, `src/contract.rs:11-15`).
- Produces: `Relation::project(&self, keep: &[usize]) -> Relation` — projects each row onto `keep` (a subset of `self.vars`, ascending), re-encoded over `keep` bit order, sorted + deduped.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `src/contract.rs`:

```rust
    #[test]
    fn relation_project_reencodes_and_dedups() {
        // vars [0,1,2], bit j = vars[j]: rows encode (v0,v1,v2).
        //   0b011 -> v0=1,v1=1,v2=0 ; 0b111 -> all 1 ; 0b101 -> v0=1,v1=0,v2=1
        let rel = Relation {
            vars: vec![0, 1, 2],
            rows: vec![0b011, 0b111, 0b101],
        };
        // Project onto [0,2] (new bit0=v0, bit1=v2):
        //   0b011 -> (v0=1,v2=0)=0b01 ; 0b111 -> (1,1)=0b11 ; 0b101 -> (1,1)=0b11 (dup)
        let p = rel.project(&[0, 2]);
        assert_eq!(p.vars, vec![0, 2]);
        assert_eq!(p.rows, vec![0b01, 0b11]);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib contract::tests::relation_project_reencodes_and_dedups`
Expected: FAIL — `no method named 'project' found for struct 'Relation'`.

- [ ] **Step 3: Implement `Relation::project`**

Add immediately after the `Relation` struct definition (after `src/contract.rs:15`):

```rust
impl Relation {
    /// Project each row onto `keep` (a subset of `self.vars`, ascending). Rows are
    /// re-encoded over `keep` bit order, then sorted and deduplicated. Every entry
    /// of `keep` must be present in `self.vars`.
    pub fn project(&self, keep: &[usize]) -> Relation {
        let mut rows: Vec<u64> = self
            .rows
            .iter()
            .map(|&row| {
                let mut r = 0u64;
                for (j, &v) in keep.iter().enumerate() {
                    let pos = self
                        .vars
                        .binary_search(&v)
                        .expect("projection var present in relation");
                    if (row >> pos) & 1 == 1 {
                        r |= 1u64 << j;
                    }
                }
                r
            })
            .collect();
        rows.sort_unstable();
        rows.dedup();
        Relation {
            vars: keep.to_vec(),
            rows,
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib contract::tests::relation_project_reencodes_and_dedups`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/contract.rs
git commit -m "feat(contract): add Relation::project (re-encode + dedup onto a var subset)"
```

---

### Task 2: `contract(rels, keep)` primitive

**Files:**
- Modify: `src/contract.rs` (add `contract` after `join_all`, which ends at line 228; add a test)

**Interfaces:**
- Consumes: `join_all(rels: Vec<Relation>) -> Relation` (existing, `src/contract.rs`), `Relation::project` (Task 1).
- Produces: `contract(rels: Vec<Relation>, keep: &[usize]) -> Relation` — join all `rels` on shared vars, then project onto `keep`. Precondition: `rels` non-empty.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `src/contract.rs`:

```rust
    #[test]
    fn contract_matches_join_all_then_project() {
        // (x0∨x1) over [0,1] and (x1∨x2) over [1,2]; support {1,2,3} each.
        let a = Relation { vars: vec![0, 1], rows: vec![1, 2, 3] };
        let b = Relation { vars: vec![1, 2], rows: vec![1, 2, 3] };
        let keep = vec![0, 2];
        let got = contract(vec![a.clone(), b.clone()], &keep);
        // Reference: join then hand-project (guards the extraction).
        let want = join_all(vec![a, b]).project(&keep);
        assert_eq!(got.vars, want.vars);
        assert_eq!(got.rows, want.rows);
        // Concrete: (x0∨x1)∧(x1∨x2) projected to (x0,x2) allows all four configs.
        assert_eq!(got.vars, vec![0, 2]);
        assert_eq!(got.rows, vec![0, 1, 2, 3]);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib contract::tests::contract_matches_join_all_then_project`
Expected: FAIL — `cannot find function 'contract' in this scope`.

- [ ] **Step 3: Implement `contract`**

Add after `join_all` (after `src/contract.rs:228`):

```rust
/// Join all `rels` on shared variables, then project onto `keep` (a subset of the
/// union of all rels' vars, ascending). The single contraction primitive shared by
/// `contract_region` and `canonicalize`'s VE step. Binary-join internals for now
/// (`join_all`); the signature admits a generic-join kernel later without changing
/// callers. Precondition: `rels` is non-empty (inherited from `join_all`).
pub fn contract(rels: Vec<Relation>, keep: &[usize]) -> Relation {
    join_all(rels).project(keep)
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib contract::tests::contract_matches_join_all_then_project`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/contract.rs
git commit -m "feat(contract): add contract(rels, keep) = join_all + project primitive"
```

---

### Task 3: Rewrite `contract_region` onto `contract`

**Files:**
- Modify: `src/contract.rs:248-276` (the `rels` / `join_all` / projection block inside `contract_region`)

**Interfaces:**
- Consumes: `contract` (Task 2), `tensor_relation` (existing).
- Produces: `contract_region` unchanged signature — `(cn, region, doms) -> (Vec<u64>, Vec<usize>)`.

Covered by existing brute-force tests (`two_tensor_join_matches_bruteforce`, `single_tensor_region_is_its_support`, `fixed_var_is_sliced_out_of_output`) and `region.rs`'s `region_cache_memoizes_region_and_configs`.

- [ ] **Step 1: Replace the join + projection block**

In `contract_region`, replace lines 248-276 (from `let rels: Vec<Relation> = region` through the `(configs, output_vars)` return) with:

```rust
    let rels: Vec<Relation> = region
        .tensors
        .iter()
        .map(|&tid| tensor_relation(cn, &cn.tensors[tid], doms))
        .collect();
    let contracted = contract(rels, &output_vars);
    (contracted.rows, output_vars)
```

- [ ] **Step 2: Run the covering tests to verify they pass**

Run: `cargo test --lib contract::tests region::tests`
Expected: PASS — all `contract` and `region` tests green (behavior unchanged).

- [ ] **Step 3: Confirm no warnings**

Run: `cargo check --all-targets 2>&1 | grep -c warning`
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add src/contract.rs
git commit -m "refactor(contract): contract_region uses the contract primitive"
```

---

### Task 4: Support-based assembly — `TensorData::from_support`, `assemble`, `setup_from_relations`

**Files:**
- Modify: `src/network.rs` — add `TensorData::from_support`; refactor `from_dense` to reuse it; extract `pub(crate) fn assemble`; refactor `setup_problem` to funnel through it; add a `from_support` test.
- Modify: `src/contract.rs` — add `setup_from_relations` calling `crate::network::assemble`; add a test.

**Interfaces:**
- Consumes: `Variable`, `BoolTensor`, `ConstraintNetwork`, `TensorData` (existing, `src/network.rs`); `Relation` (existing, `src/contract.rs`).
- Produces:
  - `TensorData::from_support(support: Vec<u32>) -> TensorData` (support ascending).
  - `pub(crate) fn assemble(var_num: usize, tensors_in: Vec<(Vec<usize>, Vec<u32>)>) -> ConstraintNetwork` in `src/network.rs`.
  - `setup_problem(var_num, tensors_to_vars, tensor_data) -> ConstraintNetwork` — unchanged signature/semantics.
  - `setup_from_relations(var_num: usize, rels: Vec<Relation>) -> ConstraintNetwork` in `src/contract.rs`.

- [ ] **Step 1: Write the failing `from_support` test**

Add to the `#[cfg(test)] mod tests` block in `src/network.rs`:

```rust
    #[test]
    fn from_support_aggregates_match_from_dense() {
        // dense {false,true,false,true} -> support {1,3}.
        let a = TensorData::from_dense(vec![false, true, false, true]);
        let b = TensorData::from_support(vec![1u32, 3u32]);
        assert_eq!(a.support, b.support);
        assert_eq!(a.support_or, b.support_or);
        assert_eq!(a.support_and, b.support_and);
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --lib network::tests::from_support_aggregates_match_from_dense`
Expected: FAIL — `no function or associated item named 'from_support'`.

- [ ] **Step 3: Add `from_support` and refactor `from_dense`**

In `src/network.rs`, replace the current `from_dense` (`impl TensorData { pub fn from_dense(...) {...} }`) with both methods:

```rust
    /// Construct from an ascending list of satisfying configs. Derives the OR/AND
    /// aggregates; `support` is stored as-is and must be strictly ascending (the
    /// `is_sat` binary search relies on it).
    pub fn from_support(support: Vec<u32>) -> TensorData {
        debug_assert!(
            support.windows(2).all(|w| w[0] < w[1]),
            "support must be strictly ascending"
        );
        let mut support_or: u32 = 0;
        let mut support_and: u32 = 0xFFFF_FFFF;
        for &config in &support {
            support_or |= config;
            support_and &= config;
        }
        TensorData {
            support,
            support_or,
            support_and,
        }
    }

    /// Construct from a dense truth table: derive the (ascending) support and discard
    /// the dense table — it is never stored.
    pub fn from_dense(dense: Vec<bool>) -> TensorData {
        let support: Vec<u32> = dense
            .iter()
            .enumerate()
            .filter_map(|(i, &sat)| if sat { Some(i as u32) } else { None })
            .collect();
        TensorData::from_support(support)
    }
```

- [ ] **Step 4: Run the `from_support` test + existing TensorData tests**

Run: `cargo test --lib network::tests`
Expected: PASS — `from_support_aggregates_match_from_dense`, `tensordata_extracts_support_and_aggregates`, `tensordata_empty_support_aggregates` all green.

- [ ] **Step 5: Extract `assemble` and refactor `setup_problem`**

In `src/network.rs`, replace the entire body of `setup_problem` (currently the loop building `tensors`/`unique_data`/`data_to_idx` on dense, then compression) with a thin dense→support adapter, and add the shared `assemble` core below it:

```rust
/// Build a `ConstraintNetwork` from raw dense tensor specs. `var_num` is the number
/// of original variables (0-based ids `0..var_num`). `tensor_data[i]` has length
/// `2^tensors_to_vars[i].len()`.
pub fn setup_problem(
    var_num: usize,
    tensors_to_vars: Vec<Vec<usize>>,
    tensor_data: Vec<Vec<bool>>,
) -> ConstraintNetwork {
    assert_eq!(tensors_to_vars.len(), tensor_data.len());
    let tensors_in: Vec<(Vec<usize>, Vec<u32>)> = tensors_to_vars
        .into_iter()
        .zip(tensor_data)
        .map(|(var_axes, dense)| {
            assert_eq!(
                dense.len(),
                1usize << var_axes.len(),
                "tensor_data size mismatch"
            );
            let support: Vec<u32> = dense
                .iter()
                .enumerate()
                .filter_map(|(i, &sat)| if sat { Some(i as u32) } else { None })
                .collect();
            (var_axes, support)
        })
        .collect();
    assemble(var_num, tensors_in)
}

/// Shared assembly from `(var_axes, support)` tensors: dedup `TensorData`, compress
/// out unused variables, remap axes to compressed ids, build `v2t`. Dedup key is
/// `(var_axes.len(), support)` — support alone is arity-ambiguous.
pub(crate) fn assemble(
    var_num: usize,
    tensors_in: Vec<(Vec<usize>, Vec<u32>)>,
) -> ConstraintNetwork {
    let f = tensors_in.len();
    let mut tensors: Vec<BoolTensor> = Vec::with_capacity(f);
    let mut vars_to_tensors: Vec<Vec<usize>> = vec![Vec::new(); var_num];
    let mut unique_data: Vec<TensorData> = Vec::new();
    let mut data_to_idx: HashMap<(usize, Vec<u32>), usize> = HashMap::new();

    for (i, (var_axes, support)) in tensors_in.into_iter().enumerate() {
        assert!(var_axes.len() <= 32, "tensor arity exceeds 32-var cap");
        debug_assert!(
            {
                let mut s = var_axes.clone();
                s.sort_unstable();
                s.dedup();
                s.len() == var_axes.len()
            },
            "CT precondition: tensor var_axes must be distinct"
        );
        let key = (var_axes.len(), support.clone());
        let data_idx = match data_to_idx.get(&key) {
            Some(&idx) => idx,
            None => {
                let idx = unique_data.len();
                data_to_idx.insert(key, idx);
                unique_data.push(TensorData::from_support(support));
                idx
            }
        };
        for &v in &var_axes {
            vars_to_tensors[v].push(i);
        }
        tensors.push(BoolTensor { var_axes, data_idx });
    }

    // Compress out variables that appear in no tensor.
    let mut orig_to_new: Vec<Option<usize>> = vec![None; var_num];
    let mut next_id = 0usize;
    for v in 0..var_num {
        if !vars_to_tensors[v].is_empty() {
            orig_to_new[v] = Some(next_id);
            next_id += 1;
        }
    }

    for t in tensors.iter_mut() {
        for axis in t.var_axes.iter_mut() {
            *axis = orig_to_new[*axis].expect("tensor references a compressed-out variable");
        }
    }

    let mut new_v2t: Vec<Vec<usize>> = vec![Vec::new(); next_id];
    for (tid, t) in tensors.iter().enumerate() {
        for &v in &t.var_axes {
            new_v2t[v].push(tid);
        }
    }

    let vars: Vec<Variable> = new_v2t.iter().map(|ts| Variable { deg: ts.len() }).collect();

    ConstraintNetwork {
        vars,
        unique_tensors: unique_data,
        tensors,
        v2t: new_v2t,
        orig_to_new,
    }
}
```

- [ ] **Step 6: Run existing `setup_problem` tests to verify unchanged behavior**

Run: `cargo test --lib network::tests`
Expected: PASS — `setup_problem_dedups_and_builds_incidence` and `setup_problem_compresses_unused_vars` green (dedup on `(arity, support)` is equivalent to dedup on the dense table).

- [ ] **Step 7: Write the failing `setup_from_relations` test**

Add to the `#[cfg(test)] mod tests` block in `src/contract.rs`:

```rust
    #[test]
    fn setup_from_relations_matches_dense_setup() {
        use crate::network::setup_problem;
        let or2 = vec![false, true, true, true]; // support {1,2,3}
        let dense_cn = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![or2.clone(), or2.clone()],
        );
        let rels = vec![
            Relation { vars: vec![0, 1], rows: vec![1, 2, 3] },
            Relation { vars: vec![1, 2], rows: vec![1, 2, 3] },
        ];
        let rel_cn = setup_from_relations(3, rels);
        assert_eq!(rel_cn.tensors.len(), dense_cn.tensors.len());
        assert_eq!(rel_cn.unique_tensors.len(), dense_cn.unique_tensors.len()); // both dedup to 1
        assert_eq!(rel_cn.vars.len(), dense_cn.vars.len());
        for t in 0..rel_cn.tensors.len() {
            assert_eq!(
                rel_cn.support(&rel_cn.tensors[t]),
                dense_cn.support(&dense_cn.tensors[t]),
            );
        }
    }
```

- [ ] **Step 8: Run it to verify it fails**

Run: `cargo test --lib contract::tests::setup_from_relations_matches_dense_setup`
Expected: FAIL — `cannot find function 'setup_from_relations' in this scope`.

- [ ] **Step 9: Implement `setup_from_relations`**

Add to `src/contract.rs` (after `contract`, before the `#[cfg(test)]` module). Update the top-of-file imports so `ConstraintNetwork` and `assemble` are in scope: the existing `use crate::network::{BoolTensor, ConstraintNetwork};` becomes `use crate::network::{assemble, BoolTensor, ConstraintNetwork};`.

```rust
/// Build a `ConstraintNetwork` from sparse relations — the support-based entry used
/// by `canonicalize` so it never materializes a dense table. Each `Relation`
/// contributes `(rel.vars, rel.rows as u32)`; `rel.rows` must be ascending (the
/// `Relation` invariant), which `assemble`/`from_support` require.
pub fn setup_from_relations(var_num: usize, rels: Vec<Relation>) -> ConstraintNetwork {
    let tensors_in: Vec<(Vec<usize>, Vec<u32>)> = rels
        .into_iter()
        .map(|rel| {
            let support: Vec<u32> = rel.rows.iter().map(|&r| r as u32).collect();
            (rel.vars, support)
        })
        .collect();
    assemble(var_num, tensors_in)
}
```

- [ ] **Step 10: Run the test to verify it passes**

Run: `cargo test --lib contract::tests::setup_from_relations_matches_dense_setup`
Expected: PASS.

- [ ] **Step 11: Full suite + no-warning check**

Run: `cargo test 2>&1 | grep 'test result:'` and `cargo check --all-targets 2>&1 | grep -c warning`
Expected: all suites `ok`; warning count `0`.

- [ ] **Step 12: Commit**

```bash
git add src/network.rs src/contract.rs
git commit -m "feat(network): assemble core + setup_from_relations sparse entry"
```

---

### Task 5: Rewrite `canonicalize` end-to-end sparse

**Files:**
- Modify: `src/canonicalize.rs` — drop the `LiveTensor` newtype (use `Vec<Relation>`), use `contract` in the VE step, and `setup_from_relations` at finalize (delete the dense round-trip).

**Interfaces:**
- Consumes: `contract`, `support_relation`, `setup_from_relations`, `Relation` (all from `src/contract.rs`); `ConstraintNetwork` (from `src/network.rs`).
- Produces: `bounded_ve_canonicalize(cn, budget_b, protected) -> ConstraintNetwork` — unchanged signature and behavior.

Covered by existing tests: `eliminates_an_unprotected_chain_var`, `protected_var_is_never_eliminated`, `budget_one_eliminates_nothing`, `solutions_preserved_over_protected_with_elimination`, `produced_tensors_respect_the_budget`, plus the `factoring_15_canonicalized_still_solves` integration test.

- [ ] **Step 1: Update imports and drop the `LiveTensor` wrapper**

In `src/canonicalize.rs`, change the two import lines (16-17):

```rust
use crate::contract::{contract, setup_from_relations, support_relation, Relation};
use crate::network::ConstraintNetwork;
```

Delete the `LiveTensor` struct (lines 19-23) entirely.

- [ ] **Step 2: Retype the helpers `&[LiveTensor]` -> `&[Relation]`**

In `out_vars`, `fill_count`, and `score`, change every `live: &[LiveTensor]` parameter to `live: &[Relation]`, and every `live[t].rel.vars` to `live[t].vars`. Concretely:
- `out_vars` (signature `live: &[LiveTensor]` -> `live: &[Relation]`; body `&live[t].rel.vars` -> `&live[t].vars`).
- `fill_count` (signature `live: &[LiveTensor]` -> `live: &[Relation]`; body `live[t].rel.vars.contains(&b)` -> `live[t].vars.contains(&b)`).
- `score` (signature `live: &[LiveTensor]` -> `live: &[Relation]`).

- [ ] **Step 3: Seed `live` as `Vec<Relation>`**

Replace the seeding block (currently `let mut live: Vec<LiveTensor> = cn.tensors.iter().map(|t| LiveTensor { rel: support_relation(...) }).collect();`) with:

```rust
    let mut live: Vec<Relation> = cn
        .tensors
        .iter()
        .map(|t| support_relation(&t.var_axes, cn.support(t)))
        .collect();
```

- [ ] **Step 4: Replace the VE contract + merge block**

Replace the whole span from the "Bucket-contract" comment (`let rels: Vec<Relation> = ...; let joined = join_all(rels); ... rows.sort/dedup;`) through the end of the merge block — i.e. up to and including the final `for &x in &out { v2t[x].push(keep); }`. Keep the preceding `let tids = ...;` / `let out = out_vars(...);` lines and the following "Re-score the affected neighbors" block untouched. Replacement:

```rust
        // Bucket-contract via the shared primitive: join incident tensors, project v out.
        let incident: Vec<Relation> = tids.iter().map(|&t| live[t].clone()).collect();
        let merged = contract(incident, &out);

        // Merge in place: reuse the first incident slot, deactivate the rest.
        let keep = tids[0];
        for &t in &tids {
            let axes = live[t].vars.clone();
            for x in axes {
                v2t[x].retain(|&tt| tt != t);
            }
        }
        for &t in &tids[1..] {
            active[t] = false;
        }
        live[keep] = merged;
        for &x in &out {
            v2t[x].push(keep);
        }
```

- [ ] **Step 5: Replace the finalize densify with `setup_from_relations`**

Replace the finalize block (currently the `let mut tv...; let mut td...;` loop that builds `vec![false; 1usize << rel.vars.len()]` and calls `setup_problem(nv, tv, td)`) with:

```rust
    // Finalize: hand surviving relations to setup_from_relations for dedup +
    // compression — no dense table is ever materialized.
    let surviving: Vec<Relation> = (0..live.len())
        .filter(|&t| active[t])
        .map(|t| live[t].clone())
        .collect();
    let new_cn = setup_from_relations(nv, surviving);
```

- [ ] **Step 6: Run the canonicalize tests to verify unchanged behavior**

Run: `cargo test --lib canonicalize::tests`
Expected: PASS — all five canonicalize tests green.

- [ ] **Step 7: Run the full suite + integration + no-warning check**

Run: `cargo test 2>&1 | grep 'test result:'` and `cargo check --all-targets 2>&1 | grep -c warning`
Expected: every suite `ok` (including `factoring_15_canonicalized_still_solves`); warning count `0`.

- [ ] **Step 8: Commit**

```bash
git add src/canonicalize.rs
git commit -m "refactor(canonicalize): end-to-end sparse via contract + setup_from_relations"
```

---

## Notes for the implementer

- **`support_relation` already exists** in `src/contract.rs` (added in a prior commit) and is imported by `canonicalize`; do not re-create it.
- After Task 4, `canonicalize` no longer calls `setup_problem`; Task 5 removes that import. `setup_problem` remains used by `dimacs`, `circuit`, and tests — do not remove it.
- The one behavioral subtlety already validated by tests: surviving tensors carry **ascending** axes (`Relation.vars` is sorted). This is solution-equivalent and pinned by the brute-force canonicalize tests.
