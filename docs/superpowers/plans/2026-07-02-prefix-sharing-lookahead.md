# Prefix-Sharing Look-Ahead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-config region-feasibility probe loop in
`table.rs::compute_branching_result` with a trie DFS that propagates each shared
prefix once, sharing propagated state across sibling probes via the CT trail —
provably node-identical, aiming for a multiplicative speedup on structured
(factoring) instances.

**Architecture:** A new `feasible_configs` primitive in `src/propagate.rs`
performs a depth-first traversal of a trie built from the enumerated region
configurations, ordered by the region's unfixed variables. At each trie edge it
opens a fresh trail epoch, fixes one variable, runs `ct_propagate` incrementally
from the parent's already-propagated state, prunes the subtree on contradiction,
and restores on backtrack. The existing atomic `probe` becomes the differential
test oracle. Only the step-1 loop of `compute_branching_result` changes; steps
2–4 (projection, table build, optimal rule) are untouched.

**Tech Stack:** Rust; the Compact-Table substrate (`src/ct.rs`, `src/trail.rs`),
`cargo test`, `cargo build --release`, `runscribe` for perf measurement.

## Global Constraints

- **Behavior-preserving:** the golden test `tests/ct_acceptance.rs` must remain
  green — `branching_nodes == 19761`, `total_visited_nodes == 45322` on
  `factoring_22x22`. Node counts must not change.
- **Feasibility identity:** `feasible_configs(cn, doms, .., region_vars, configs)`
  returns the same *set* as `{ c in configs : probe(cn, doms, .., &region_vars,
  full_mask, c, |d| d[0] != DomainMask::NONE) }` — order does not matter (the
  caller sorts+dedups downstream).
- **Precondition:** every config in `configs` is consistent with `doms` on
  already-fixed region vars (the caller applies the `check_mask`/`check_value`
  filter first). `feasible_configs` never fixes an already-fixed region var.
- **Trail epoch discipline:** one `trail.open()` (new epoch) per trie edge before
  fixing+propagating; `restore_to(mark)` on backtrack. Never take a nested
  restore mark inside a single epoch (CT save-once-per-epoch requires it).
- **Restore invariant:** on return, `doms`, every `tables[i]` (`words` + `limit`),
  and `buffer` (`queue` empty, all `in_queue`/`dirty` clear) are exactly as on
  entry.
- **Monotonicity:** rely on CT being a monotonic+inflationary propagator; encode
  as a `debug_assert`, not a runtime guard.
- v1 trie order is the static region-var order (ascending var id) over unfixed
  positions. MINCE/fail-first/Gray-code ordering and the project-and-intersect
  alternative are out of scope (tracked in the spec's Future section).

---

### Task 1: `feasible_configs` engine + unit tests

**Files:**
- Modify: `src/propagate.rs` (add `feasible_configs`, plus private `key_of` and
  `descend`; add unit tests to the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes (all already in `propagate.rs` scope or imported there):
  `ct::{ct_propagate, enqueue_var_change, RSparseBitSet, TableMasks}`,
  `domain::DomainMask`, `network::ConstraintNetwork`, `problem::SolverBuffer`,
  `trail::Trail`. `ct_propagate(cn, &mut doms, masks, &mut tables, &mut buffer,
  &mut trail)` sets the sentinel `doms[0] = DomainMask::NONE` on contradiction
  (trailed) and drains `buffer` clean on both paths.
- Produces: `pub fn feasible_configs(cn: &ConstraintNetwork, doms: &mut
  [DomainMask], masks: &[TableMasks], tables: &mut [RSparseBitSet], buffer: &mut
  SolverBuffer, trail: &mut Trail, region_vars: &[usize], configs: &[u64]) ->
  Vec<u64>`.

- [ ] **Step 1: Write the failing test (known feasible set on an OR-chain)**

Add to `src/propagate.rs` `mod tests`:

```rust
#[test]
fn feasible_configs_matches_known_set_on_or_chain() {
    use crate::ct::build_tables;
    use crate::trail::Trail;
    // (x0 OR x1) AND (x1 OR x2): feasible assignments over [0,1,2] are
    // {010,011,101,110,111} = {2,3,5,6,7} (bit i = value of var i).
    let or2 = vec![false, true, true, true];
    let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![or2.clone(), or2]);
    let (masks, mut tables) = build_tables(&cn);
    let mut doms = vec![DomainMask::BOTH; 3];
    let mut buf = SolverBuffer::new(&cn);
    let mut trail = Trail::new();
    let region_vars = vec![0usize, 1, 2];
    let all: Vec<u64> = (0u64..8).collect();
    let mut got = feasible_configs(
        &cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail, &region_vars, &all,
    );
    got.sort_unstable();
    assert_eq!(got, vec![2, 3, 5, 6, 7]);
    // base state fully restored
    assert_eq!(doms, vec![DomainMask::BOTH; 3]);
    assert!(buf.queue.is_empty());
    assert!(buf.in_queue.iter().all(|&q| !q));
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test --lib feasible_configs_matches_known_set_on_or_chain`
Expected: FAIL — `cannot find function feasible_configs`.

- [ ] **Step 3: Implement `key_of`, `descend`, and `feasible_configs`**

Add to `src/propagate.rs` (top-level, near `probe`):

```rust
/// MSB-first bit key reading `c`'s bits at the positions in `order`
/// (order[0] = most significant). Used to sort configs so trie subtrees are
/// contiguous ranges. `order.len()` <= 64 (region size fits in a u64 config).
#[inline]
fn key_of(c: u64, order: &[usize]) -> u64 {
    let mut k = 0u64;
    for &pos in order {
        k = (k << 1) | ((c >> pos) & 1);
    }
    k
}

/// DFS one trie level. `range` is the contiguous slice of the sorted config
/// list that agrees with the current path on `order[..level]`. Precondition:
/// `buffer` is drained clean and `doms`/`tables` reflect the parent prefix.
#[allow(clippy::too_many_arguments)]
fn descend(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    region_vars: &[usize],
    order: &[usize],
    range: &[u64],
    level: usize,
    out: &mut Vec<u64>,
) {
    if level == order.len() {
        // Leaf: the whole prefix is fixed and no ancestor contradicted, so every
        // config in `range` is feasible. (range is a single config in practice.)
        out.extend_from_slice(range);
        return;
    }
    let pos = order[level];
    let var = region_vars[pos];
    // `range` is sorted MSB-first by `order`, so at this level the configs with
    // bit `pos` == 0 form a contiguous run followed by those with bit `pos` == 1.
    let mut i = 0usize;
    while i < range.len() {
        let value = ((range[i] >> pos) & 1) as u8;
        let mut j = i;
        while j < range.len() && (((range[j] >> pos) & 1) as u8) == value {
            j += 1;
        }
        let sub = &range[i..j];
        i = j;

        trail.open(); // fresh epoch per trie edge — required for nested restore
        let m = trail.mark();
        debug_assert!(buffer.queue.is_empty(), "worklist must be drained per sibling");
        let nd = if value == 1 { DomainMask::D1 } else { DomainMask::D0 };
        // `var` is unfixed here (order holds only unfixed positions) => real change.
        trail.record_dom(var, doms[var]);
        doms[var] = nd;
        enqueue_var_change(cn, buffer, var);
        ct_propagate(cn, doms, masks, tables, buffer, trail);
        if doms[0] != DomainMask::NONE {
            descend(
                cn, doms, masks, tables, buffer, trail, region_vars, order, sub,
                level + 1, out,
            );
        }
        trail.restore_to(m, doms, tables);
    }
}

/// Return the subset of `configs` that are GAC-feasible from the current
/// `(doms, tables)`, sharing the propagation of common prefixes via a trie DFS
/// over the region's UNFIXED variables. Set-identical to probing each config
/// independently with `probe(.., |d| d[0] != DomainMask::NONE)`.
///
/// Precondition: each config agrees with `doms` on already-fixed region vars
/// (caller applies the consistency filter). On return `(doms, tables)` and
/// `buffer` are exactly as on entry.
#[allow(clippy::too_many_arguments)]
pub fn feasible_configs(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    region_vars: &[usize],
    configs: &[u64],
) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    if configs.is_empty() {
        return out;
    }
    // Trie levels = unfixed region-var positions, in region_vars (ascending) order.
    let order: Vec<usize> = (0..region_vars.len())
        .filter(|&pos| !doms[region_vars[pos]].is_fixed())
        .collect();
    if order.is_empty() {
        // All region vars fixed: each config equals the current assignment, so
        // feasibility is the (live) base state. Matches probing each as a no-op.
        if doms[0] != DomainMask::NONE {
            out.extend_from_slice(configs);
        }
        return out;
    }
    let mut sorted: Vec<u64> = configs.to_vec();
    sorted.sort_by_key(|&c| key_of(c, &order));

    // Clean the worklist once, as `probe` does.
    buffer.queue.clear();
    for b in buffer.in_queue.iter_mut() {
        *b = false;
    }
    descend(
        cn, doms, masks, tables, buffer, trail, region_vars, &order, &sorted, 0,
        &mut out,
    );
    out
}
```

- [ ] **Step 4: Run the known-set test to verify it passes**

Run: `cargo test --lib feasible_configs_matches_known_set_on_or_chain`
Expected: PASS.

- [ ] **Step 5: Add edge-case unit tests**

Add to `src/propagate.rs` `mod tests`:

```rust
#[test]
fn feasible_configs_empty_input_is_empty() {
    use crate::ct::build_tables;
    use crate::trail::Trail;
    let or2 = vec![false, true, true, true];
    let cn = setup_problem(2, vec![vec![0, 1]], vec![or2]);
    let (masks, mut tables) = build_tables(&cn);
    let mut doms = vec![DomainMask::BOTH; 2];
    let mut buf = SolverBuffer::new(&cn);
    let mut trail = Trail::new();
    let got = feasible_configs(&cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail, &[0, 1], &[]);
    assert!(got.is_empty());
}

#[test]
fn feasible_configs_prunes_infeasible_prefix() {
    use crate::ct::build_tables;
    use crate::trail::Trail;
    // (x0 OR x1): assignment 00 (=0) is the only infeasible one.
    let or2 = vec![false, true, true, true];
    let cn = setup_problem(2, vec![vec![0, 1]], vec![or2]);
    let (masks, mut tables) = build_tables(&cn);
    let mut doms = vec![DomainMask::BOTH; 2];
    let mut buf = SolverBuffer::new(&cn);
    let mut trail = Trail::new();
    let mut got = feasible_configs(
        &cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail, &[0, 1], &[0, 1, 2, 3],
    );
    got.sort_unstable();
    assert_eq!(got, vec![1, 2, 3]); // 00 pruned
    assert_eq!(doms, vec![DomainMask::BOTH; 2]);
}

#[test]
fn feasible_configs_all_region_vars_fixed_returns_all() {
    use crate::ct::build_tables;
    use crate::trail::Trail;
    let or2 = vec![false, true, true, true];
    let cn = setup_problem(2, vec![vec![0, 1]], vec![or2]);
    let (masks, mut tables) = build_tables(&cn);
    // Fix both vars consistently (x0=1,x1=0 => config bit0=1 => value 1).
    let mut doms = vec![DomainMask::D1, DomainMask::D0];
    let mut buf = SolverBuffer::new(&cn);
    let mut trail = Trail::new();
    let got = feasible_configs(
        &cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail, &[0, 1], &[1],
    );
    assert_eq!(got, vec![1]);
    assert_eq!(doms, vec![DomainMask::D1, DomainMask::D0]);
}
```

- [ ] **Step 6: Run all new unit tests**

Run: `cargo test --lib feasible_configs`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/propagate.rs
git commit -m "feat(propagate): feasible_configs trie DFS engine + unit tests"
```

---

### Task 2: Differential oracle + restore-integrity test

**Files:**
- Modify: `src/propagate.rs` (add a randomized differential test to `mod tests`)

**Interfaces:**
- Consumes: `feasible_configs` (Task 1) and `probe` (existing) as the oracle;
  `ct::build_tables`, `network::setup_problem`, `trail::Trail`.
- Produces: nothing new — hardens Task 1 against nested-restore bugs.

**Global constraints for this task (from the plan header):** feasibility
identity vs `probe`; restore invariant (byte-equal `doms` and each table's
`words`+`limit` before/after). The oracle is the atomic `probe`, one call per
config, fixing all region vars via `full_mask`.

- [ ] **Step 1: Write the differential test (multi-seed random networks)**

Add to `src/propagate.rs` `mod tests`:

```rust
#[test]
fn feasible_configs_matches_probe_oracle_randomized() {
    use crate::ct::build_tables;
    use crate::trail::Trail;

    // Tiny deterministic PRNG (xorshift64) — no external dep.
    fn next(s: &mut u64) -> u64 {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        x
    }

    for seed in 1u64..=300 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let n_vars = 3 + (next(&mut s) % 3) as usize; // 3..=5 vars
        let n_tensors = 2 + (next(&mut s) % 3) as usize; // 2..=4 tensors

        // Build tensors over random distinct-var scopes with random non-empty support.
        let mut scopes: Vec<Vec<usize>> = Vec::new();
        let mut tables_dense: Vec<Vec<bool>> = Vec::new();
        for _ in 0..n_tensors {
            let arity = 2 + (next(&mut s) % 2) as usize; // 2 or 3
            let mut vs: Vec<usize> = Vec::new();
            while vs.len() < arity {
                let v = (next(&mut s) % n_vars as u64) as usize;
                if !vs.contains(&v) {
                    vs.push(v);
                }
            }
            let rows = 1usize << arity;
            let mut support = vec![false; rows];
            let mut any = false;
            for r in support.iter_mut() {
                if next(&mut s) % 100 < 60 {
                    *r = true;
                    any = true;
                }
            }
            if !any {
                support[(next(&mut s) as usize) % rows] = true; // ensure non-empty
            }
            scopes.push(vs);
            tables_dense.push(support);
        }
        let cn = setup_problem(n_vars, scopes, tables_dense);
        let (masks, mut tables) = build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; n_vars];
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();

        // Establish a base fixpoint: randomly fix ~1/3 of vars, propagate from base.
        buf.queue.clear();
        for b in buf.in_queue.iter_mut() { *b = false; }
        for v in 0..n_vars {
            if next(&mut s) % 3 == 0 {
                let nd = if next(&mut s) & 1 == 1 { DomainMask::D1 } else { DomainMask::D0 };
                trail.record_dom(v, doms[v]);
                doms[v] = nd;
                crate::ct::enqueue_var_change(&cn, &mut buf, v);
            }
        }
        ct_propagate(&cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail);
        if doms[0] == DomainMask::NONE {
            continue; // base already contradictory — skip this seed
        }

        // Region = all vars. Candidate configs = all 2^n, filtered to those
        // consistent with the fixed vars (the caller's contract).
        let region_vars: Vec<usize> = (0..n_vars).collect();
        let (check_mask, check_value) = mask_value_bits(&doms, &region_vars);
        let full_mask: u64 = if n_vars >= 64 { u64::MAX } else { (1u64 << n_vars) - 1 };
        let all: Vec<u64> = (0u64..(1u64 << n_vars))
            .filter(|c| (c & check_mask) == check_value)
            .collect();

        // Snapshot base for restore-integrity check.
        let doms_before = doms.clone();
        let words_before: Vec<Vec<u64>> = tables.iter().map(|t| t.words.clone()).collect();
        let limit_before: Vec<u32> = tables.iter().map(|t| t.limit).collect();

        // Oracle: probe each config independently.
        let mut want: Vec<u64> = Vec::new();
        for &c in &all {
            let feas = probe(
                &cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail,
                &region_vars, full_mask, c, |d| d[0] != DomainMask::NONE,
            );
            if feas { want.push(c); }
        }
        want.sort_unstable();

        // System under test.
        let mut got = feasible_configs(
            &cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail, &region_vars, &all,
        );
        got.sort_unstable();

        assert_eq!(got, want, "seed {seed}: feasible set mismatch vs probe oracle");

        // Restore integrity.
        assert_eq!(doms, doms_before, "seed {seed}: doms not restored");
        for (i, t) in tables.iter().enumerate() {
            assert_eq!(t.words, words_before[i], "seed {seed}: table {i} words not restored");
            assert_eq!(t.limit, limit_before[i], "seed {seed}: table {i} limit not restored");
        }
        assert!(buf.queue.is_empty(), "seed {seed}: worklist leaked");
        assert!(buf.in_queue.iter().all(|&q| !q), "seed {seed}: in_queue leaked");
    }
}

/// (mask, value): bit i set in `mask` iff region_vars[i] is fixed; the same bit
/// in `value` is its fixed value. Mirrors the call-site consistency filter.
fn mask_value_bits(doms: &[DomainMask], region_vars: &[usize]) -> (u64, u64) {
    let mut mask = 0u64;
    let mut value = 0u64;
    for (i, &v) in region_vars.iter().enumerate() {
        match doms[v] {
            DomainMask::D0 => { mask |= 1 << i; }
            DomainMask::D1 => { mask |= 1 << i; value |= 1 << i; }
            _ => {}
        }
    }
    (mask, value)
}
```

- [ ] **Step 2: Run the differential test**

Run: `cargo test --lib feasible_configs_matches_probe_oracle_randomized`
Expected: PASS (300 seeds; any mismatch prints the seed for reproduction).

- [ ] **Step 3: Confirm `RSparseBitSet.words`/`limit` are readable from tests**

If Step 2 fails to compile because `words`/`limit` are private, confirm in
`src/ct.rs` that `RSparseBitSet` exposes `pub words: Vec<u64>` and
`pub limit: u32`. They are already `pub` (used by `src/trail.rs::restore_to`).
No change expected; do not widen visibility further than already present.

Run: `cargo test --lib feasible_configs`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 4: Commit**

```bash
git add src/propagate.rs
git commit -m "test(propagate): 300-seed differential oracle vs probe + restore integrity"
```

---

### Task 3: Wire `feasible_configs` into `compute_branching_result`

**Files:**
- Modify: `src/table.rs:36-71` (replace the step-1 `for &config in &cached_configs`
  loop with a filter + one `feasible_configs` call)
- Test: `tests/ct_acceptance.rs` (existing golden test — must stay green)

**Interfaces:**
- Consumes: `feasible_configs` (Task 1). `compute_branching_result` already holds
  `cn`, `doms`, `masks`, `tables`, `buffer`, `trail`, `region_vars`,
  `cached_configs`, and the computed `(check_mask, check_value)`.
- Produces: unchanged public signature of `compute_branching_result`; `feasible`
  is now built by `feasible_configs` instead of the loop. Steps 2–4 unchanged.

- [ ] **Step 1: Add the import**

In `src/table.rs`, change the propagate import line
(`use crate::propagate::probe;`) to:

```rust
use crate::propagate::feasible_configs;
```

- [ ] **Step 2: Replace the step-1 loop**

In `src/table.rs::compute_branching_result`, replace this block:

```rust
    let mut feasible: Vec<u64> = Vec::new();
    for &config in &cached_configs {
        if (config & check_mask) != check_value {
            continue;
        }
        let feasible_here = probe(
            cn,
            doms,
            masks,
            tables,
            buffer,
            trail,
            &region_vars,
            full_mask,
            config,
            |d| d[0] != DomainMask::NONE,
        );
        if feasible_here {
            feasible.push(config);
        }
    }
    if feasible.is_empty() {
        return (None, region_vars);
    }
```

with:

```rust
    // Configs consistent with the currently-fixed region vars; feasibility is
    // decided by a single prefix-sharing trie DFS instead of one probe per config.
    let filtered: Vec<u64> = cached_configs
        .iter()
        .copied()
        .filter(|&config| (config & check_mask) == check_value)
        .collect();
    let feasible = feasible_configs(cn, doms, masks, tables, buffer, trail, &region_vars, &filtered);
    if feasible.is_empty() {
        return (None, region_vars);
    }
```

Note: `full_mask` is now unused in step 1. If the compiler warns that `full_mask`
(and the `n`/`n >= 64` block computing it) is dead, delete that computation. If
`full_mask` is still used elsewhere in the function, leave it. Check with Step 4.

- [ ] **Step 3: Update the file-local unit tests that referenced `probe`**

`src/table.rs` `mod tests` calls `compute_branching_result` (not `probe`
directly), so its two tests (`branching_result_covers_the_table`,
`no_feasible_config_returns_none`) need no change. Confirm they still compile —
they use `build_tables`, `Trail`, `compute_branching_result` only.

Run: `cargo test --lib --features "" table::tests`  (or `cargo test --lib branching_result no_feasible_config`)
Expected: PASS (both).

- [ ] **Step 4: Build and clear warnings**

Run: `cargo build --release 2>&1 | tail -20`
Expected: builds; no `unused variable: full_mask` / `unused import: probe`
warnings. If either appears, remove the dead `full_mask` computation and ensure
`probe` is still imported only where used (it remains used by `mod tests` in
`propagate.rs`, so its definition stays; only `table.rs`'s import changed).

- [ ] **Step 5: Run the golden node-identity test**

Run: `cargo test --release --test ct_acceptance`
Expected: PASS — `branching_nodes == 19761`, `total_visited_nodes == 45322`.

- [ ] **Step 6: Run the full test suite**

Run: `cargo test`
Expected: PASS (all lib + integration tests).

- [ ] **Step 7: Commit**

```bash
git add src/table.rs
git commit -m "feat(table): route region feasibility through feasible_configs (prefix-sharing)"
```

---

### Task 4: Measure the speedup (runscribe)

**Files:**
- No source changes. Uses `runscribe` per the repo's experiment discipline and
  the existing solve example/binary used for the earlier CT measurements.

**Interfaces:**
- Consumes: the release binary/example that solves `factoring_22x22` (the same
  entry point used to produce the baseline `19761 / 45322 / ~9s` numbers — see
  `.superpowers/sdd/progress.md` and the CT-plan perf notes for the exact command).

- [ ] **Step 1: Declare the goal and open a hypothesis**

Run:
```bash
runscribe goal new prefix-sharing-lookahead -m "does region-feasibility prefix-sharing cut wall-clock on factoring_22x22 with node counts unchanged"
runscribe hyp new A --from A -m "trie DFS beats per-config probing on VE10 factoring_22x22"
```
Expected: allocates goal `A` and hypothesis `A1`.

- [ ] **Step 2: Measure VE-on (VE10) — the realistic config**

Run (wrap the exact solve command used for the CT baseline; substitute the real
binary/args):
```bash
runscribe run --hyp A1 --tag ve10 -- <the release solve command for factoring_22x22, VE10>
```
Expected: prints `[runscribe] recorded → <run_dir>`; solver reports
`branching_nodes=19761 visited=45322` (identical) and a wall-clock time.

- [ ] **Step 3: Measure no-VE baseline config**

Run:
```bash
runscribe run --hyp A1 --tag no-ve -- <the release solve command for factoring_22x22, no VE>
```
Expected: `[runscribe] recorded → <run_dir>`; node counts identical to the
pre-change no-VE run.

- [ ] **Step 4: Record the observation (do not promote to a finding)**

Fill the hypothesis `## Observation` section with the before/after wall-clock and
µs/branching-node for both configs, linking each run dir. Compare against the
pre-change numbers (VE10 ≈ 9.03s; no-VE ≈ 37.36s from the CT work). State the
delta; leave `## Your judgment` for the human. If node counts differ from
19761/45322, STOP — that is a behavior-preservation regression, not a perf result.

Run: `runscribe index`
Expected: tables rebuilt.

- [ ] **Step 5: Commit any ledger/notes (source tree unchanged)**

```bash
git add -A && git commit -m "chore(perf): runscribe measurement of prefix-sharing lookahead" || echo "nothing to commit"
```

---

## Self-Review

**1. Spec coverage:**
- Algorithm (trie DFS, per-level scope) → Task 1. ✅
- Trail/epoch discipline (open per edge) → Task 1 `descend` + header constraint. ✅
- Buffer discipline (clean once, drain per sibling, debug_assert) → Task 1. ✅
- Node-identity (set-return, sorted downstream) → Task 3 golden test. ✅
- Feasibility identity vs probe oracle → Task 2. ✅
- Restore invariant → Task 2 byte-equal check. ✅
- Precondition (configs filtered) → Task 3 filter + Task 2 mask_value_bits. ✅
- Edge cases (empty / all-fixed / single / pruned) → Task 1 Step 5 + Task 2. ✅
- Monotonicity debug_assert → present as `debug_assert!` on worklist; the
  monotonicity property itself is a CT invariant relied upon, not separately
  assertable at this layer (documented in header). ✅
- Perf measurement (runscribe, VE-on/no-VE, nodes identical) → Task 4. ✅
- Out-of-scope items (MINCE/Gray-code/project-intersect/difflookahead) → not
  planned, matches spec Future. ✅

**2. Placeholder scan:** The only intentionally deferred literal is "the exact
release solve command for `factoring_22x22`" in Task 4 — this is an existing repo
command the implementer must reuse verbatim from the CT-plan perf notes /
progress ledger, not new code. All source code is complete and concrete.

**3. Type consistency:** `feasible_configs` signature is identical in the spec,
Task 1 (definition), Task 2 (call), and Task 3 (call). `descend`/`key_of` are
private helpers used only within `propagate.rs`. `mask_value_bits` is a
test-only helper. `DomainMask::NONE` sentinel usage matches `ct_propagate` and
`probe`. `RSparseBitSet.words`/`limit` are `pub` (relied on in Task 2).
