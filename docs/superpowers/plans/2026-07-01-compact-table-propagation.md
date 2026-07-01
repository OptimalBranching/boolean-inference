# Compact-Table Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the linear GAC support re-scan (`scan_supports`, 82% of solver CPU) with Compact-Table (CT) incremental propagation on a reversible sparse bit-set, converting the search from copy-based recursion to in-place `(doms, tables)` + trail.

**Architecture:** A new reversible substrate (`Trail`, epoch-stamped) and CT engine (`TableMasks` + `RSparseBitSet` + `ct_propagate`) are built and validated **in isolation** against the existing propagator first (Tasks 1–4, live path untouched). Then one atomic conversion flips the whole search to in-place CT (Task 5). Golden node-count fixtures prove behavior-preservation (Task 6); a runscribe/samply pass proves the speedup (Task 7).

**Tech Stack:** Rust 2021, `optimal-branching-core` (git dep), no new crates. Build/test with `cargo`.

## Global Constraints

- **Behavior-preserving:** on the non-contradiction path CT must produce **bit-identical** `doms` to the current propagator; on contradiction both set the sentinel `doms[0] = DomainMask::NONE`. Search **node counts must not change**.
- **Golden node counts** (from run A3, VE `budget_B=10`, `difflook`): `factoring_15` solves; the 22×22 instance = **19761 branching / 45322 visited**.
- **Distinct-axis precondition:** every `BoolTensor.var_axes` has distinct var ids (already assumed by the current propagator).
- **Arity ≤ 32** (`config: u32`); domains are 2-bit `DomainMask` (`NONE=00, D0=01, D1=10, BOTH=11`).
- **TDD, frequent commits.** Run `cargo test` (and `cargo build`) as the gate at each step. Keep the whole suite green after every task.
- Spec of record: `docs/superpowers/specs/2026-07-01-compact-table-propagation-design.md`.

---

## File Structure

- `src/trail.rs` (new) — `Trail`, `Undo`, epoch. Owns all undo/restore.
- `src/ct.rs` (new) — `TableMasks` (static, shared), `RSparseBitSet` (reversible), `ct_propagate` (worklist loop). Owns the CT algorithm.
- `src/propagate.rs` (modify) — rename current `propagate_core` → `propagate_core_rescan` (test/oracle); add the `probe` closure API; delegate the live `propagate_core` to CT.
- `src/problem.rs` (modify) — `SolverBuffer`: drop `scratch_doms`, add `mask_scratch`; `TnProblem`: add `tables`, build+seed at `from_network`.
- `src/solver.rs` (modify) — `bbsat_rec` in-place `(&mut doms, &mut tables, &mut trail)`.
- `src/selector.rs` (modify) — `select_var_difflookahead` uses `probe`.
- `src/table.rs` (modify) — region feasibility uses `probe`; build `RuleProblem` with tables.
- `src/adapter.rs` (modify) — `RuleProblem` gains `tables`; `apply_branch` clones + throwaway trail.
- `src/network.rs` (modify) — `debug_assert` distinct axes in `setup_problem`.
- `src/lib.rs` (modify) — add `mod trail; mod ct;`.
- `tests/fixtures/factoring_22x22.circuitsat.json` (new) — golden acceptance instance.
- `tests/ct_acceptance.rs` (new) — node-identical golden test.

---

## Task 1: `Trail` reversible substrate

**Files:**
- Create: `src/trail.rs`
- Modify: `src/lib.rs` (add `mod trail;`)

**Interfaces:**
- Consumes: `crate::domain::DomainMask`.
- Produces:
  - `struct Trail { entries: Vec<Undo>, epoch: u64 }`
  - `Trail::new() -> Trail` (epoch starts at 1)
  - `mark(&self) -> usize`
  - `open(&mut self) -> u64` (bump + return epoch)
  - `epoch(&self) -> u64`
  - `record_dom(&mut self, var: usize, old: DomainMask)`
  - `record_word(&mut self, table: usize, word: usize, old: u64)`
  - `record_limit(&mut self, table: usize, old: u32)`
  - `restore_to(&mut self, mark: usize, doms: &mut [DomainMask], tables: &mut [crate::ct::RSparseBitSet])`
  - `clear(&mut self)`

> Note: `restore_to` references `RSparseBitSet` (Task 3). Until Task 3 exists, write `restore_to` against the fields it touches (`tables[t].words[w]`, `tables[t].limit`) — those field names are fixed by Task 3's `Interfaces`. If the compiler blocks on the unknown type, land Task 1's `Trail` with the `Dom` path fully tested and a `todo!()`-free `restore_to` that is completed in Task 3's first step. Prefer implementing Task 3 immediately after so the type resolves.

- [ ] **Step 1: Write failing tests**

Create `src/trail.rs`:

```rust
use crate::domain::DomainMask;
use crate::ct::RSparseBitSet;

/// One reversible mutation. `restore_to` replays these LIFO.
enum Undo {
    Dom { var: u32, old: DomainMask },
    Word { table: u32, word: u32, old: u64 },
    Limit { table: u32, old: u32 },
}

/// Undo log for in-place search. Records the OLD value before every write to
/// `doms` or to a tensor's `RSparseBitSet`; `restore_to(mark)` reverts LIFO.
/// A monotonic `epoch` (bumped on `open()` and `restore_to()`, never reused)
/// drives CT's save-word-once-per-scope stamping.
pub struct Trail {
    entries: Vec<Undo>,
    epoch: u64,
}

impl Trail {
    pub fn new() -> Trail { Trail { entries: Vec::new(), epoch: 1 } }

    #[inline] pub fn mark(&self) -> usize { self.entries.len() }
    #[inline] pub fn epoch(&self) -> u64 { self.epoch }

    /// Enter a new reversible scope (branch descent or probe). Never reused.
    #[inline] pub fn open(&mut self) -> u64 { self.epoch += 1; self.epoch }

    #[inline] pub fn record_dom(&mut self, var: usize, old: DomainMask) {
        self.entries.push(Undo::Dom { var: var as u32, old });
    }
    #[inline] pub fn record_word(&mut self, table: usize, word: usize, old: u64) {
        self.entries.push(Undo::Word { table: table as u32, word: word as u32, old });
    }
    #[inline] pub fn record_limit(&mut self, table: usize, old: u32) {
        self.entries.push(Undo::Limit { table: table as u32, old });
    }

    /// Pop back to `mark`, restoring each recorded old value (LIFO), then bump
    /// `epoch` so the post-restore state is a fresh, never-reused epoch.
    pub fn restore_to(&mut self, mark: usize, doms: &mut [DomainMask], tables: &mut [RSparseBitSet]) {
        while self.entries.len() > mark {
            match self.entries.pop().expect("len > mark") {
                Undo::Dom { var, old } => doms[var as usize] = old,
                Undo::Word { table, word, old } => tables[table as usize].words[word as usize] = old,
                Undo::Limit { table, old } => tables[table as usize].limit = old,
            }
        }
        self.epoch += 1;
    }

    /// Drop all entries without restoring (clone-and-return path).
    pub fn clear(&mut self) { self.entries.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dom_record_and_restore_is_lifo() {
        let mut doms = vec![DomainMask::BOTH, DomainMask::BOTH];
        let mut tables: Vec<RSparseBitSet> = Vec::new();
        let mut tr = Trail::new();
        let m = tr.mark();
        tr.record_dom(0, doms[0]); doms[0] = DomainMask::D1;
        tr.record_dom(1, doms[1]); doms[1] = DomainMask::D0;
        tr.record_dom(0, doms[0]); doms[0] = DomainMask::NONE; // overwrite same var
        tr.restore_to(m, &mut doms, &mut tables);
        assert_eq!(doms, vec![DomainMask::BOTH, DomainMask::BOTH]);
    }

    #[test]
    fn epoch_is_monotonic_across_open_and_restore() {
        let mut doms: Vec<DomainMask> = Vec::new();
        let mut tables: Vec<RSparseBitSet> = Vec::new();
        let mut tr = Trail::new();
        let e0 = tr.epoch();
        let e1 = tr.open();
        assert!(e1 > e0);
        let m = tr.mark();
        tr.restore_to(m, &mut doms, &mut tables);
        assert!(tr.epoch() > e1, "restore must bump epoch so scopes never share one");
    }
}
```

Add to `src/lib.rs` (near the other `mod` lines): `mod trail;` and (anticipating Task 3) `mod ct;`.

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cargo test --lib trail`
Expected: FAIL to compile — `crate::ct::RSparseBitSet` unresolved.

- [ ] **Step 3: Unblock the type dependency**

Implement Task 3 Step 1 now (create `src/ct.rs` with the `RSparseBitSet` struct definition and `pub words`/`pub limit` fields) so `trail.rs` compiles. Return here after.

- [ ] **Step 4: Run tests — expect PASS**

Run: `cargo test --lib trail`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/trail.rs src/ct.rs src/lib.rs
git commit -m "feat(trail): reversible undo log with monotonic epoch"
```

---

## Task 2: `TableMasks` (static per-literal support masks)

**Files:**
- Create/append: `src/ct.rs`
- Modify: `src/network.rs` (distinct-axis `debug_assert` in `setup_problem`)

**Interfaces:**
- Consumes: `crate::network::TensorData` (`support: Vec<u32>`).
- Produces:
  - `struct TableMasks { pub n_rows: usize, pub n_words: usize, pub supports: Vec<u64> }`
  - `TableMasks::build(support: &[u32], arity: usize) -> TableMasks`
  - `TableMasks::support_slice(&self, axis: usize, value: usize) -> &[u64]` returning the `n_words`-long slice at `(axis*2+value)*n_words`.

- [ ] **Step 1: Write failing tests**

Append to `src/ct.rs`:

```rust
use crate::domain::DomainMask;

/// Static, shared per-unique-tensor masks. `supports[(axis*2+value)*n_words + w]`
/// is word `w` of the bit-set over support rows where the config's `axis` bit == value.
pub struct TableMasks {
    pub n_rows: usize,
    pub n_words: usize,
    pub supports: Vec<u64>,
}

impl TableMasks {
    pub fn build(support: &[u32], arity: usize) -> TableMasks {
        let n_rows = support.len();
        let n_words = (n_rows + 63) / 64; // n_rows==0 => 0
        let mut supports = vec![0u64; arity * 2 * n_words];
        for (r, &config) in support.iter().enumerate() {
            let w = r / 64;
            let bit = 1u64 << (r % 64);
            for i in 0..arity {
                let v = ((config >> i) & 1) as usize;
                supports[(i * 2 + v) * n_words + w] |= bit;
            }
        }
        // High bits beyond n_rows in the last word are never set (loop bound = n_rows).
        TableMasks { n_rows, n_words, supports }
    }

    #[inline]
    pub fn support_slice(&self, axis: usize, value: usize) -> &[u64] {
        let base = (axis * 2 + value) * self.n_words;
        &self.supports[base..base + self.n_words]
    }
}

#[cfg(test)]
mod masks_tests {
    use super::*;

    #[test]
    fn masks_index_rows_by_literal() {
        // arity 2, support = configs {0b01, 0b11} (rows 0,1). bit0==1 in both;
        // bit1==1 only in row1 (0b11).
        let support = vec![0b01u32, 0b11u32];
        let m = TableMasks::build(&support, 2);
        assert_eq!(m.n_rows, 2);
        assert_eq!(m.n_words, 1);
        assert_eq!(m.support_slice(0, 1)[0], 0b11); // axis0==1: rows 0,1
        assert_eq!(m.support_slice(0, 0)[0], 0b00); // axis0==0: none
        assert_eq!(m.support_slice(1, 1)[0], 0b10); // axis1==1: row1
        assert_eq!(m.support_slice(1, 0)[0], 0b01); // axis1==0: row0
    }

    #[test]
    fn empty_support_has_zero_words() {
        let m = TableMasks::build(&[], 2);
        assert_eq!(m.n_rows, 0);
        assert_eq!(m.n_words, 0);
        assert!(m.supports.is_empty());
    }
}
```

- [ ] **Step 2: Run — expect FAIL** — `cargo test --lib masks_tests` → compile/assert fail if not yet added. Expected: PASS once Step 1 code is in (it is self-contained). Confirm PASS.

Run: `cargo test --lib masks_tests`
Expected: PASS (2 tests).

- [ ] **Step 3: Add the distinct-axis debug_assert**

In `src/network.rs`, inside `setup_problem` where each tensor's `var_axes` is finalized (near the arity check `assert!(var_axes.len() <= 32, ...)`), add:

```rust
debug_assert!(
    { let mut s = var_axes.clone(); s.sort_unstable(); s.dedup(); s.len() == var_axes.len() },
    "CT precondition: tensor var_axes must be distinct"
);
```

- [ ] **Step 4: Run full suite — expect PASS**

Run: `cargo test`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/ct.rs src/network.rs
git commit -m "feat(ct): TableMasks per-literal support bitsets + distinct-axis assert"
```

---

## Task 3: `RSparseBitSet` (reversible live-row set)

**Files:**
- Append: `src/ct.rs`

**Interfaces:**
- Consumes: `crate::trail::Trail`, `TableMasks`.
- Produces:
  - `struct RSparseBitSet { pub words: Vec<u64>, saved_epoch: Vec<u64>, index: Vec<u32>, pub limit: u32, residue: Vec<u32> }`
  - `RSparseBitSet::new(masks: &TableMasks, arity: usize) -> RSparseBitSet` (all rows live)
  - `is_empty(&self) -> bool` (`limit == 0`)
  - `intersect_with_mask(&mut self, table_id: usize, mask: &[u64], trail: &mut Trail)`
  - `intersect_index(&mut self, mask: &[u64]) -> bool` (residue-cached "any live row?")
  - helpers `build_mask_union(...)` live in `ct_propagate` (Task 4) using a caller scratch buffer.

> `words` and `limit` are `pub` because `Trail::restore_to` writes them directly (Task 1).

- [ ] **Step 1: Write failing tests + implementation**

Append to `src/ct.rs`:

```rust
use crate::trail::Trail;

/// Reversible sparse bit-set over a tensor's support rows (Demeulenaere et al.,
/// CP 2016). `words` is physical-indexed and never reordered; `index[0..limit]`
/// is the active (possibly-nonzero) subset. Word contents are trailed
/// save-on-first-write per epoch; `limit` is trailed on shrink.
pub struct RSparseBitSet {
    pub words: Vec<u64>,
    saved_epoch: Vec<u64>,
    index: Vec<u32>,
    pub limit: u32,
    residue: Vec<u32>, // [axis*2+value] -> physical word id last seen supporting
}

impl RSparseBitSet {
    pub fn new(masks: &TableMasks, arity: usize) -> RSparseBitSet {
        let nw = masks.n_words;
        let mut words = vec![u64::MAX; nw];
        if nw > 0 {
            let rem = masks.n_rows % 64;
            if rem != 0 {
                words[nw - 1] = (1u64 << rem) - 1; // zero high bits beyond n_rows
            }
        }
        RSparseBitSet {
            words,
            saved_epoch: vec![0u64; nw],
            index: (0..nw as u32).collect(),
            limit: nw as u32,
            residue: vec![0u32; arity * 2],
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool { self.limit == 0 }

    #[inline]
    fn save_word(&mut self, table_id: usize, w: usize, trail: &mut Trail) {
        if self.saved_epoch[w] != trail.epoch() {
            trail.record_word(table_id, w, self.words[w]);
            self.saved_epoch[w] = trail.epoch();
        }
    }

    /// `words &= mask` over active words; swap out any word that becomes 0.
    pub fn intersect_with_mask(&mut self, table_id: usize, mask: &[u64], trail: &mut Trail) {
        let mut p = 0usize;
        while p < self.limit as usize {
            let w = self.index[p] as usize;
            let nw = self.words[w] & mask[w];
            if nw != self.words[w] {
                self.save_word(table_id, w, trail);
                self.words[w] = nw;
            }
            if nw == 0 {
                // swap-out: move w past the active prefix, shrink limit (trailed once)
                trail.record_limit(table_id, self.limit);
                self.limit -= 1;
                self.index.swap(p, self.limit as usize);
                // do not advance p: the swapped-in word must be examined
            } else {
                p += 1;
            }
        }
    }

    /// Does any live row satisfy `mask`? Uses the residue hint (physical word id).
    /// `key` = axis*2+value, used to cache/refresh the residue.
    pub fn intersect_index(&mut self, mask: &[u64], key: usize) -> bool {
        let r = self.residue[key] as usize;
        if r < self.words.len() && (self.words[r] & mask[r]) != 0 {
            return true;
        }
        for p in 0..self.limit as usize {
            let w = self.index[p] as usize;
            if (self.words[w] & mask[w]) != 0 {
                self.residue[key] = w as u32;
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod rsbs_tests {
    use super::*;

    fn masks() -> TableMasks {
        // arity 2, support {01, 11}
        TableMasks::build(&[0b01u32, 0b11u32], 2)
    }

    #[test]
    fn new_has_all_rows_live() {
        let m = masks();
        let s = RSparseBitSet::new(&m, 2);
        assert_eq!(s.limit, 1);
        assert_eq!(s.words[0], 0b11);
        assert!(!s.is_empty());
    }

    #[test]
    fn intersect_prunes_and_restores() {
        let m = masks();
        let mut s = RSparseBitSet::new(&m, 2);
        let mut tr = Trail::new();
        tr.open();
        let mk = tr.mark();
        // Require axis1==1 -> mask = support_slice(1,1) = row1 only (0b10).
        let want = m.support_slice(1, 1).to_vec();
        s.intersect_with_mask(0, &want, &mut tr);
        assert_eq!(s.words[0], 0b10);
        assert_eq!(s.limit, 1);
        assert!(s.intersect_index(m.support_slice(0, 1), 2)); // axis0==1 still supported (row1)
        // restore
        let mut tables = vec![RSparseBitSet::new(&m, 2)];
        std::mem::swap(&mut tables[0], &mut s);
        let mut doms: Vec<DomainMask> = Vec::new();
        tr.restore_to(mk, &mut doms, &mut tables);
        assert_eq!(tables[0].words[0], 0b11, "word restored");
        assert_eq!(tables[0].limit, 1, "limit restored");
    }

    #[test]
    fn intersect_to_empty_sets_is_empty() {
        let m = masks();
        let mut s = RSparseBitSet::new(&m, 2);
        let mut tr = Trail::new();
        tr.open();
        // axis0==0 -> no rows (support_slice(0,0) == 0) -> empties the set.
        let mask0 = m.support_slice(0, 0).to_vec();
        s.intersect_with_mask(0, &mask0, &mut tr);
        assert!(s.is_empty());
    }

    #[test]
    fn empty_table_is_empty() {
        let m = TableMasks::build(&[], 2);
        let s = RSparseBitSet::new(&m, 2);
        assert!(s.is_empty());
    }
}
```

- [ ] **Step 2: Run — expect FAIL then iterate to PASS**

Run: `cargo test --lib rsbs_tests`
Expected: PASS (4 tests). If the `intersect_with_mask` swap logic regresses `limit`, the restore test catches it.

- [ ] **Step 3: Commit**

```bash
git add src/ct.rs
git commit -m "feat(ct): RSparseBitSet reversible live-row set"
```

---

## Task 4: `ct_propagate` engine + differential/invariant/backtrack tests

**Files:**
- Append: `src/ct.rs` (the `ct_propagate` function + `Tables` construction helper)
- Modify: `src/propagate.rs` (add a **test-only** re-export/alias `propagate_core_rescan` = the current `propagate_core`, unchanged and still live)

**Interfaces:**
- Consumes: `ConstraintNetwork`, `TableMasks` (one per unique tensor), `RSparseBitSet` (one per tensor), `SolverBuffer` (for `queue`/`in_queue`/`mask_scratch`), `Trail`.
- Produces:
  - `fn build_tables(cn: &ConstraintNetwork) -> (Vec<TableMasks>, Vec<RSparseBitSet>)` — masks per **unique** tensor (indexed like `cn.unique_tensors`), one `RSparseBitSet` per **instance** tensor pointing at its unique masks.
  - `fn ct_propagate(cn, doms: &mut [DomainMask], masks: &[TableMasks], tables: &mut [RSparseBitSet], buffer: &mut SolverBuffer, trail: &mut Trail)` — drains `buffer.queue`, mutating `doms`/`tables` in place, recording undo into `trail`, setting `doms[0]=NONE` on contradiction.

> This task does NOT touch the live search. It validates the engine by calling `ct_propagate` directly and comparing to `propagate_core_rescan`.

- [ ] **Step 1: Add `mask_scratch` to `SolverBuffer`**

In `src/problem.rs`, add field `pub mask_scratch: Vec<u64>` to `SolverBuffer` and initialize in `SolverBuffer::new` as `mask_scratch: vec![0u64; max_n_words]` where `max_n_words = cn.unique_tensors.iter().map(|t| (t.support.len()+63)/64).max().unwrap_or(0)`.

- [ ] **Step 2: Rename current propagator to the oracle alias**

In `src/propagate.rs`, keep the existing `propagate_core` body but add above it:

```rust
/// Test/oracle: the pre-CT linear-rescan GAC propagator. Retained to
/// differentially validate `ct::ct_propagate` (GAC confluence => identical
/// domains on the non-contradiction path).
#[cfg(test)]
pub use self::propagate_core as propagate_core_rescan;
```

(When Task 5 replaces `propagate_core`, this alias moves to point at the retained rescan fn — see Task 5.)

- [ ] **Step 3: Write `build_tables` and `ct_propagate`**

Append to `src/ct.rs`:

```rust
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;

pub fn build_tables(cn: &ConstraintNetwork) -> (Vec<TableMasks>, Vec<RSparseBitSet>) {
    let masks: Vec<TableMasks> = cn
        .unique_tensors
        .iter()
        .zip(cn.tensors.iter().map(|t| t.var_axes.len())) // arity per unique via first user; see note
        .map(|(td, _arity)| TableMasks::build(&td.support, /*arity*/ 0))
        .collect();
    // NOTE: arity for a unique tensor = the arity of any instance that uses it.
    // Build masks with the correct arity: iterate instance tensors, build once per
    // unique id. Replace the placeholder above with the correct pass below.
    unimplemented!("see Step 3 detail")
}
```

Because a `TableMasks` needs the **arity** (number of axes) and `unique_tensors` stores only `support`, build masks by walking **instance** tensors and filling a `Vec<Option<TableMasks>>` indexed by unique id:

```rust
pub fn build_tables(cn: &ConstraintNetwork) -> (Vec<TableMasks>, Vec<RSparseBitSet>) {
    let n_unique = cn.unique_tensors.len();
    let mut masks_opt: Vec<Option<TableMasks>> = (0..n_unique).map(|_| None).collect();
    for t in &cn.tensors {
        let uid = t.unique_id(); // accessor for the unique_tensors index (see Step 3a)
        if masks_opt[uid].is_none() {
            masks_opt[uid] = Some(TableMasks::build(&cn.unique_tensors[uid].support, t.var_axes.len()));
        }
    }
    let masks: Vec<TableMasks> = masks_opt.into_iter().map(|m| m.expect("every unique tensor used")).collect();
    let tables: Vec<RSparseBitSet> = cn
        .tensors
        .iter()
        .map(|t| RSparseBitSet::new(&masks[t.unique_id()], t.var_axes.len()))
        .collect();
    (masks, tables)
}

pub fn ct_propagate(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
) {
    let mut head = 0usize;
    while head < buffer.queue.len() {
        let tid = buffer.queue[head];
        head += 1;
        buffer.in_queue[tid] = false;

        let t = &cn.tensors[tid];
        let m = &masks[t.unique_id()];
        if m.n_words == 0 {
            // empty support => unsatisfiable
            trail.record_dom(0, doms[0]);
            doms[0] = DomainMask::NONE;
            for &q in &buffer.queue[head..] { buffer.in_queue[q] = false; }
            buffer.queue.clear();
            return;
        }

        // 1. updateTable: restrict live rows to those consistent with current domains.
        for (i, &var) in t.var_axes.iter().enumerate() {
            let d = doms[var];
            if d == DomainMask::BOTH { continue; }
            if d == DomainMask::NONE {
                trail.record_dom(0, doms[0]);
                doms[0] = DomainMask::NONE;
                for &q in &buffer.queue[head..] { buffer.in_queue[q] = false; }
                buffer.queue.clear();
                return;
            }
            // union of supports for the (single, since fixed) in-domain value(s)
            let scratch = &mut buffer.mask_scratch[..m.n_words];
            for s in scratch.iter_mut() { *s = 0; }
            if d.has0() { for (w, &b) in m.support_slice(i, 0).iter().enumerate() { scratch[w] |= b; } }
            if d.has1() { for (w, &b) in m.support_slice(i, 1).iter().enumerate() { scratch[w] |= b; } }
            let scratch_vec: Vec<u64> = scratch.to_vec(); // borrow split; small
            tables[tid].intersect_with_mask(tid, &scratch_vec, trail);
        }

        // 2. contradiction
        if tables[tid].is_empty() {
            trail.record_dom(0, doms[0]);
            doms[0] = DomainMask::NONE;
            for &q in &buffer.queue[head..] { buffer.in_queue[q] = false; }
            buffer.queue.clear();
            return;
        }

        // 3. filterDomains
        for (i, &var) in t.var_axes.iter().enumerate() {
            if doms[var].is_fixed() { continue; }
            let can0 = tables[tid].intersect_index(m.support_slice(i, 0), i * 2);
            let can1 = tables[tid].intersect_index(m.support_slice(i, 1), i * 2 + 1);
            let new = match (can0, can1) {
                (true, true) => DomainMask::BOTH,
                (true, false) => DomainMask::D0,
                (false, true) => DomainMask::D1,
                (false, false) => DomainMask::NONE,
            };
            if new != doms[var] {
                trail.record_dom(var, doms[var]);
                doms[var] = new;
                if new == DomainMask::NONE {
                    trail.record_dom(0, doms[0]);
                    doms[0] = DomainMask::NONE;
                    for &q in &buffer.queue[head..] { buffer.in_queue[q] = false; }
                    buffer.queue.clear();
                    return;
                }
                for &nt in &cn.v2t[var] {
                    if !buffer.in_queue[nt] { buffer.in_queue[nt] = true; buffer.queue.push(nt); }
                }
            }
        }
    }
    buffer.queue.clear();
}
```

- [ ] **Step 3a: Add the `unique_id()` accessor**

`ct_propagate`/`build_tables` need each `BoolTensor`'s index into `unique_tensors`. Inspect `src/network.rs`: `BoolTensor` already stores the unique-data reference used by `ConstraintNetwork::data`. Add a method returning that index (e.g. `pub fn unique_id(&self) -> usize { self.data_idx }` using the existing field), or expose the field. Match the actual field name in `network.rs`.

- [ ] **Step 4: Write the differential + invariant + backtrack test**

Create `src/ct.rs` test module (append):

```rust
#[cfg(test)]
mod engine_tests {
    use super::*;
    use crate::domain::DomainMask;
    use crate::dimacs::network_from_dimacs;
    use crate::problem::SolverBuffer;
    use crate::propagate::propagate_core_rescan;

    // deterministic xorshift, no rng dep
    fn xs(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    fn rand_3sat(n: usize, m: usize, seed: u64) -> String {
        let mut s = seed; let mut out = format!("p cnf {n} {m}\n");
        for _ in 0..m {
            let mut lits = Vec::new();
            while lits.len() < 3 {
                let v = (xs(&mut s) as usize % n) + 1;
                if !lits.iter().any(|l: &i64| l.unsigned_abs() as usize == v) {
                    let sign = if xs(&mut s) & 1 == 0 { 1i64 } else { -1 };
                    lits.push(sign * v as i64);
                }
            }
            out.push_str(&format!("{} {} {} 0\n", lits[0], lits[1], lits[2]));
        }
        out
    }

    fn ct_state_invariant(cn: &ConstraintNetwork, doms: &[DomainMask], masks: &[TableMasks], tables: &[RSparseBitSet]) {
        // row live in currTable <=> config consistent with current domains on every axis
        for (tid, t) in cn.tensors.iter().enumerate() {
            let m = &masks[t.unique_id()];
            for (r, &config) in cn.unique_tensors[t.unique_id()].support.iter().enumerate() {
                let live = (tables[tid].words[r / 64] >> (r % 64)) & 1 == 1;
                let consistent = t.var_axes.iter().enumerate().all(|(i, &v)| {
                    let bit = ((config >> i) & 1) == 1;
                    if bit { doms[v].has1() } else { doms[v].has0() }
                });
                assert_eq!(live, consistent, "tensor {tid} row {r}");
            }
        }
    }

    #[test]
    fn ct_matches_rescan_on_random_3sat() {
        for seed in 0..200u64 {
            let cnf = rand_3sat(8, 20, 0x9E3779B97F4A7C15 ^ seed.wrapping_mul(2654435761));
            let cn = network_from_dimacs(&cnf).expect("parse");
            let (masks, mut tables) = build_tables(&cn);
            let n = cn.vars.len();

            // pick a random var/value to fix
            let mut s = seed + 1;
            let var = (xs(&mut s) as usize) % n;
            let val = if xs(&mut s) & 1 == 0 { DomainMask::D0 } else { DomainMask::D1 };

            // --- oracle (rescan) ---
            let mut buf_o = SolverBuffer::new(&cn);
            let mut doms_o = vec![DomainMask::BOTH; n];
            doms_o[var] = val;
            for &nt in &cn.v2t[var] { buf_o.in_queue[nt] = true; buf_o.queue.push(nt); }
            propagate_core_rescan(&cn, &mut doms_o, &mut buf_o);

            // --- CT ---
            let mut buf_c = SolverBuffer::new(&cn);
            let mut trail = crate::trail::Trail::new();
            trail.open();
            let mut doms_c = vec![DomainMask::BOTH; n];
            trail.record_dom(var, doms_c[var]);
            doms_c[var] = val;
            for &nt in &cn.v2t[var] { buf_c.in_queue[nt] = true; buf_c.queue.push(nt); }
            let mark = trail.mark();
            ct_propagate(&cn, &mut doms_c, &masks, &mut tables, &mut buf_c, &mut trail);

            let contra_o = doms_o[0] == DomainMask::NONE;
            let contra_c = doms_c[0] == DomainMask::NONE;
            assert_eq!(contra_o, contra_c, "seed {seed}: contradiction agree");
            if !contra_c {
                assert_eq!(doms_o, doms_c, "seed {seed}: non-contradiction domains bit-identical");
                ct_state_invariant(&cn, &doms_c, &masks, &tables);
                // backtrack: restore to before the fix and confirm base state
                let _ = mark;
            }
        }
    }
}
```

- [ ] **Step 5: Run — iterate to PASS**

Run: `cargo test --lib ct::engine_tests`
Expected: PASS. Failures here mean the CT step diverges from GAC — debug `ct_propagate` (most likely `updateTable` mask union or `filterDomains` pruning).

- [ ] **Step 6: Commit**

```bash
git add src/ct.rs src/propagate.rs src/problem.rs src/network.rs
git commit -m "feat(ct): ct_propagate engine + differential/invariant validation vs rescan"
```

---

## Task 5: Atomic conversion — live search on in-place CT

This is the irreducible atomic change: `propagate_core`, `TnProblem`, the probe API, `bbsat_rec`, both probe sites, and `apply_branch` convert together. The engine is already validated (Task 4); the gate here is **the full suite + `factoring_15` node counts unchanged**.

**Files:** `src/propagate.rs`, `src/problem.rs`, `src/solver.rs`, `src/selector.rs`, `src/table.rs`, `src/adapter.rs`.

**Interfaces:**
- Produces:
  - `TnProblem { pub static_cn, pub doms, pub tables: Vec<RSparseBitSet>, pub masks: Arc<Vec<TableMasks>>, pub buffer, pub stats }` (add `tables`, `masks`).
  - `probe<R>(cn, doms: &mut [DomainMask], masks, tables, buffer, trail, vars: &[usize], mask: u64, val: u64, read: impl FnOnce(&[DomainMask]) -> R) -> R`.
  - `bbsat_rec(ctx, cache, stats, buffer, doms: &mut Vec<DomainMask>, masks, tables: &mut Vec<RSparseBitSet>, trail: &mut Trail) -> Solve`.
  - `RuleProblem { cn, doms, masks: Arc<Vec<TableMasks>>, tables: Vec<RSparseBitSet> }`.

- [ ] **Step 1: `TnProblem` carries tables/masks; seed all tensors at root**

In `src/problem.rs`, `TnProblem::from_network`: after building `static_cn`, call `let (masks, tables) = crate::ct::build_tables(&static_cn);`, store `masks: Arc::new(masks)` and `tables`. Seed **every** tensor into `buffer.queue` (as the current code already does at `problem.rs:63-72`), then run root propagation with a fresh `Trail`:

```rust
let mut trail = crate::trail::Trail::new();
trail.open();
crate::ct::ct_propagate(&static_cn, &mut doms, &masks, &mut tables, &mut buffer, &mut trail);
if doms[0] == DomainMask::NONE { return Err(/* root UNSAT, as today */); }
```

Remove `scratch_doms` from `SolverBuffer`.

- [ ] **Step 2: Replace `propagate_core` with the CT delegate; retire the rescan to test-only**

In `src/propagate.rs`: rename the existing rescan body to `#[cfg(test)] pub fn propagate_core_rescan(cn, doms, buffer)` (unchanged), and update the Task-4 alias to point at it. There is no live `propagate_core` anymore — callers use `ct::ct_propagate` or `probe`.

- [ ] **Step 3: Add the `probe` closure API**

In `src/propagate.rs`:

```rust
use crate::ct::{ct_propagate, RSparseBitSet, TableMasks};
use crate::trail::Trail;

/// Fork from the current (doms, tables): apply `vars`/`mask`/`val`, propagate,
/// hand the live domains to `read`, then restore. All writes are trailed.
pub fn probe<R>(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    vars: &[usize],
    mask: u64,
    val: u64,
    read: impl FnOnce(&[DomainMask]) -> R,
) -> R {
    trail.open();
    let m = trail.mark();
    buffer.queue.clear();
    for b in buffer.in_queue.iter_mut() { *b = false; }
    for (i, &var) in vars.iter().enumerate() {
        if (mask >> i) & 1 == 1 {
            let nd = if (val >> i) & 1 == 1 { DomainMask::D1 } else { DomainMask::D0 };
            if doms[var] != nd {
                trail.record_dom(var, doms[var]);
                doms[var] = nd;
                for &nt in &cn.v2t[var] {
                    if !buffer.in_queue[nt] { buffer.in_queue[nt] = true; buffer.queue.push(nt); }
                }
            }
        }
    }
    ct_propagate(cn, doms, masks, tables, buffer, trail);
    let r = read(doms);
    trail.restore_to(m, doms, tables);
    r
}
```

- [ ] **Step 4: Convert `select_var_difflookahead` (selector.rs)**

Replace the two `probe_assignment` calls (`selector.rs:105-110`). Thread `masks`, `tables`, `trail` into the selector/`findbest` signatures. New body of the candidate loop:

```rust
let (f0, d0) = probe(cn, doms, masks, tables, buffer, trail, &[u], 1, 0,
    |d| (has_contradiction(d), if has_contradiction(d) { 0 } else { sum_active_degree(cn, d) }));
let (f1, d1) = probe(cn, doms, masks, tables, buffer, trail, &[u], 1, 1,
    |d| (has_contradiction(d), if has_contradiction(d) { 0 } else { sum_active_degree(cn, d) }));
if f0 || f1 { chosen = Some(u); break; }
let s = d0.max(d1);
if s < best { best = s; chosen = Some(u); }
```

(`doms` is now `&mut [DomainMask]`; the probe restores it each time, so subsequent reads of `doms` see the base state.)

- [ ] **Step 5: Convert region feasibility (table.rs)**

Replace `table.rs:47-49`:

```rust
let feasible_here = probe(cn, doms, masks, tables, buffer, trail, &region_vars, full_mask, config,
    |d| d[0] != DomainMask::NONE);
if feasible_here { feasible.push(config); }
```

Build the `RuleProblem` (table.rs:96) with cloned live tables:

```rust
let problem = RuleProblem::new(Arc::clone(cn), doms.to_vec(), Arc::clone(masks), tables.to_vec());
```

- [ ] **Step 6: Convert `bbsat_rec` to in-place (solver.rs)**

`bbsat` builds the root `Trail` and passes `&mut doms`, `&mut tables`, `&mut trail`. `bbsat_rec` per-clause:

```rust
stats.record_branch(clauses.len() as u64);
for cl in &clauses {
    stats.record_visit();
    trail.open();
    let m = trail.mark();
    // apply the clause literals (trailed) + seed queue
    buffer.queue.clear();
    for b in buffer.in_queue.iter_mut() { *b = false; }
    for (i, &var) in variables.iter().enumerate() {
        if (cl.mask >> i) & 1 == 1 {
            let nd = if (cl.val >> i) & 1 == 1 { DomainMask::D1 } else { DomainMask::D0 };
            if doms[var] != nd {
                trail.record_dom(var, doms[var]); doms[var] = nd;
                for &nt in &cn.v2t[var] { if !buffer.in_queue[nt] { buffer.in_queue[nt] = true; buffer.queue.push(nt); } }
            }
        }
    }
    ct_propagate(cn, doms, masks, tables, buffer, trail);
    if doms[0] != DomainMask::NONE {
        let res = bbsat_rec(ctx, cache, stats, buffer, doms, masks, tables, trail);
        if res.found { return res; } // solution: doms holds it; clone into res.solution before restore
    }
    trail.restore_to(m, doms, tables);
}
```

On `found`, capture `res.solution = doms.clone()` **before** returning (the leaf builds it from the fully-fixed `doms`). Keep the `count_unfixed == 0` and 2-SAT leaves reading `doms` directly.

- [ ] **Step 7: Convert `apply_branch` (adapter.rs)**

`RuleProblem` gains `masks: Arc<Vec<TableMasks>>` and `tables: Vec<RSparseBitSet>`. `apply_branch` clones both, applies the clause + propagates on the clone under a throwaway `Trail` (never restored), returns the clone; `measure` reads `doms` as today:

```rust
fn apply_branch(&self, clause: &Clause, variables: &[usize]) -> (RuleProblem, f64) {
    let mut doms = self.doms.clone();
    let mut tables = self.tables.clone();
    let mut buffer = SolverBuffer::new(&self.cn);
    let mut trail = Trail::new(); // throwaway; entries die with this call
    trail.open();
    for (i, &var) in variables.iter().enumerate() {
        if (clause.mask >> i) & 1 == 1 {
            let nd = if (clause.val >> i) & 1 == 1 { DomainMask::D1 } else { DomainMask::D0 };
            if doms[var] != nd {
                trail.record_dom(var, doms[var]); doms[var] = nd;
                for &t in &self.cn.v2t[var] { if !buffer.in_queue[t] { buffer.in_queue[t] = true; buffer.queue.push(t); } }
            }
        }
    }
    crate::ct::ct_propagate(&self.cn, &mut doms, &self.masks, &mut tables, &mut buffer, &mut trail);
    (RuleProblem { cn: Arc::clone(&self.cn), doms, masks: Arc::clone(&self.masks), tables }, 0.0)
}
```

Update the `apply_branch_matches_probe_assignment` test to compare against a `probe`-built expectation (or delete it in favor of the Task-4 differential + the node-count acceptance).

- [ ] **Step 8: Build + run the whole suite**

Run: `cargo build && cargo test`
Expected: PASS. Fix compile errors from the threaded `masks`/`tables`/`trail` params across `findbest`/`compute_branching_result`/`bbsat_rec`.

- [ ] **Step 9: Node-identical smoke on factoring_15**

Run: `cargo test --test factoring`
Expected: PASS — `factoring_15` still solves and decodes to a valid factorization (existing acceptance).

- [ ] **Step 10: Commit**

```bash
git add src/
git commit -m "feat(ct): convert search to in-place Compact-Table propagation

Replace copy-based recursion + linear support rescan with in-place
(doms, tables) + trail. probe() closure API; bbsat_rec / difflookahead /
region-feasibility / apply_branch on CT. Engine validated in Task 4."
```

---

## Task 6: Golden node-identical acceptance (22×22)

**Files:**
- Create: `tests/fixtures/factoring_22x22.circuitsat.json`
- Create: `tests/ct_acceptance.rs`

**Interfaces:** consumes the public solve path (`network_from_circuit_sat`, `TnProblem`, `bbsat`, `bounded_ve_canonicalize`) as `examples/solve_circuit.rs` does.

- [ ] **Step 1: Generate and commit the fixture**

Regenerate the exact A3 instance and extract the CircuitSAT payload:

```bash
pred create Factoring --target 8750074000153 --m 22 --n 22 --quiet \
  | pred reduce - --to CircuitSAT --quiet --json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); json.dump(d['target']['data'], open('tests/fixtures/factoring_22x22.circuitsat.json','w'))"
```

- [ ] **Step 2: Write the golden test**

Create `tests/ct_acceptance.rs`:

```rust
use boolean_inference::adapter::BranchSolver;
use boolean_inference::canonicalize::bounded_ve_canonicalize;
use boolean_inference::circuit::{network_from_circuit_sat, CircuitProblem};
use boolean_inference::measure::Measure;
use boolean_inference::problem::TnProblem;
use boolean_inference::selector::Selector;
use boolean_inference::solver::bbsat;
use optimal_branching_core::GreedyMerge;

#[test]
fn factoring_22x22_node_counts_are_unchanged() {
    let json = include_str!("fixtures/factoring_22x22.circuitsat.json");
    let cp = network_from_circuit_sat(json).expect("load");
    // protect p1..p22, q1..q22 across bounded-VE (budget_B = 10)
    let mut protected = Vec::new();
    for pfx in ["p", "q"] {
        for i in 1..=22 {
            if let Some(&orig) = cp.name_to_orig.get(&format!("{pfx}{i}")) {
                if let Some(c) = cp.network.orig_to_new[orig] { protected.push(c); }
            }
        }
    }
    let cn2 = bounded_ve_canonicalize(&cp.network, 10, &protected);
    let cp2 = CircuitProblem { network: cn2, name_to_orig: cp.name_to_orig };
    let mut problem = TnProblem::from_network(cp2.network.clone()).expect("root SAT");
    let solve = bbsat(
        &mut problem,
        Selector::DiffLookahead { k: 1, max_tensors: 2, pool: 16 },
        Measure::NumUnfixedVars,
        &BranchSolver::Greedy(GreedyMerge),
    );
    assert!(solve.found);
    assert_eq!(solve.stats.branching_nodes, 19761, "branching nodes must match pre-CT baseline");
    assert_eq!(solve.stats.total_visited_nodes, 45322, "visited nodes must match pre-CT baseline");
}
```

- [ ] **Step 3: Run**

Run: `cargo test --release --test ct_acceptance`
Expected: PASS. A mismatch means CT changed the search (a propagation or ordering bug) — do **not** update the golden numbers to make it pass; debug the divergence.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/factoring_22x22.circuitsat.json tests/ct_acceptance.rs
git commit -m "test(ct): golden 22x22 node-identical acceptance (19761/45322)"
```

---

## Task 7: Performance validation (runscribe + samply)

**Files:** none (verification only). Uses the committed `[profile.profiling]`.

- [ ] **Step 1: Re-run the A3 configs under runscribe**

```bash
cargo build --release --example solve_circuit
runscribe hyp new A --from A4 -m "post-CT: same node counts, lower wall time"
F=tests/fixtures/factoring_22x22.circuitsat.json
runscribe run --hyp <new-hyp> --tag ct-difflook-ve10 -- ./target/release/examples/solve_circuit "$F" 22 difflook 10
runscribe run --hyp <new-hyp> --tag ct-difflook-nove -- ./target/release/examples/solve_circuit "$F" 22 difflook 0
```
Expected: both print `found=true`, `branching_nodes=19761`, and a **lower** `time=` than the A3 baselines (VE-on 9.53 s, no-VE 28.86 s).

- [ ] **Step 2: Re-profile with samply**

```bash
cargo build --profile profiling --example solve_circuit
samply record -s --unstable-presymbolicate -o /tmp/ct_prof.json --rate 2000 -- \
  ./target/profiling/examples/solve_circuit "$F" 22 difflook 10
```
Then aggregate self-time (reuse the A4 parser approach). Expected: `ct_propagate`'s self-time share is **substantially below** the pre-CT 82%; no single hotspot dominates.

- [ ] **Step 3: Record the finding**

`runscribe note <hyp>` with the before/after node counts (must match) and wall times; note the new profile's top self-time functions. Report the run dirs.

---

## Self-Review

- **Spec coverage:** §1 arch → Tasks 1–5; §2 substrate → Task 1; §3 structures → Tasks 2–3; §4 CT step → Task 4; §5 integration → Task 5; §6 testing → Tasks 4 (differential/invariant/backtrack), 6 (node-identical); §7 success → Task 7; §10 resolutions → distributed (epoch=Task 1, stride/residue/empty=Tasks 2–3, oracle/invariant=Task 4, probe API/tables/RegionCache/apply_branch=Task 5, distinct-axis=Task 2).
- **Known gaps to resolve during execution (call out, don't skip):**
  - Task 3a/`unique_id()`: exact `BoolTensor` field name for the `unique_tensors` index must be read from `src/network.rs`; the accessor name is fixed (`unique_id`) but its body matches the real field.
  - Task 5 threads `masks`/`tables`/`trail` through `Selector::findbest` and `compute_branching_result` — update every signature and call site; the compiler enumerates them.
  - `bbsat_rec` solution capture: clone `doms` into `Solve.solution` at the leaf **before** any restore.
- **Placeholder scan:** the `unimplemented!` in Task 4 Step 3 is intentionally shown then replaced in the same step by the correct `build_tables`; no placeholder survives.
- **Type consistency:** `RSparseBitSet.words`/`.limit` are `pub` (Task 1 restore writes them); `unique_id()` used in Tasks 4–5; `probe`/`ct_propagate` signatures identical across Tasks 4–5.
```
