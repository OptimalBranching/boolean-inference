# Compact-Table Propagation — Design

**Status:** draft (brainstorming), pending user review → implementation plan
**Date:** 2026-07-01
**Branch:** `framework-apply-branch` (Rust crate `boolean-inference`)
**Supersedes:** [`2026-06-29-trail-mechanism-design.md`](2026-06-29-trail-mechanism-design.md)
(the domain-only trail is absorbed into CT's reversible substrate; see §8).
**Informed by:** profiling (samply run A4: `propagate_core` = 82% CPU) and
[the incremental-table-propagation research](../../research/2026-06-30-incremental-table-propagation.md).

## Goal

Replace the linear GAC support re-scan (`scan_supports`, `src/propagate.rs:24-50`)
— **82% of solver CPU** — with **Compact-Table (CT)** (Demeulenaere et al., CP
2016): per-constraint GAC maintained incrementally via a **reversible sparse
bit-set** of live support rows, updated with word-parallel bit operations on
domain changes and rolled back cheaply on backtrack.

This is **behavior-preserving**: GAC is a confluent fixpoint, so CT must reach the
**bit-identical** domain state as the current propagator on every call. The
observable search is therefore unchanged — same branch decisions, same node
counts — only faster. That equivalence is the primary correctness oracle (§6).

## Non-goals (deferred)

- Delta-based "modified variable" tracking. First version **recomputes each active
  tensor's AND-mask from its variables' current domains** (arity is small); a
  changed-var delta set is a later refinement.
- Negative/compressed/short tables, MDD propagators, tuple hashing.
- STR3 path-optimality (CT chosen per research; §1 of the research doc).
- Any change to the branching rule or variable-selection *logic* — only the
  *mechanism* by which the selector probes (copy → mark/restore) changes.
- Parallel search, GPU region contraction, SIMD micro-opt (CT replaces the scan
  wholesale, so the `wide::u32x8` stopgap is dropped).

## Background: what CT replaces

Today each tensor activation in `propagate_core` (`src/propagate.rs:144`) calls
`scan_supports`, which **linearly re-examines every row of the tensor's support
table** (`TensorData.support: Vec<u32>`, `src/network.rs:3-12`) against the current
domain masks — from scratch, every time, even though typically one variable
changed. In tensor-network terms this recomputes a Boolean-semiring contraction
of the constraint tensor against the (rank-1) domain vectors on each activation.
CT caches *which rows are still live* in a reversible bit-set and updates it
incrementally, so no live-row set is ever rescanned.

---

## 1. Architecture & module layout

New / changed modules under `src/`:

| Module | Role |
|---|---|
| `src/trail.rs` (new) | Reversible substrate: level-indexed undo log for domain writes **and** bit-set-word writes. |
| `src/ct.rs` (new) | `RSparseBitSet` (reversible sparse bit-set) + `TableConstraint` static masks/residues + the CT per-tensor propagation step. |
| `src/propagate.rs` (rewrite) | `propagate_core` becomes the CT worklist loop. `scan_supports` / `apply_updates` / `compute_query_masks` deleted from the live path; the old rescan kept **test-only** as `propagate_core_rescan` (differential oracle). |
| `src/problem.rs` | `SolverBuffer` loses `scratch_doms`; search state gains `tables: Vec<RSparseBitSet>` threaded with `doms`. |
| `src/solver.rs` | `bbsat_rec` switches copy → `mark → apply+propagate → recurse → restore_to`. |
| `src/selector.rs`, `src/table.rs` | lookahead probe path switches copy → `mark/restore` (§5). |
| `src/adapter.rs` | ob-core `apply_branch` clone path deep-copies `tables`; uses a throwaway trail. |

**Data-flow (uniform for committed branches and lookahead probes):**
```
let m = trail.mark();  level += 1;
apply assignment (write doms, trailed);  ct_propagate(...);   // in place
... recurse / read marginal ...
trail.restore_to(m, &mut doms, &mut tables);  level -= 1;
```

## 2. Reversible substrate (`src/trail.rs`)

One level-indexed undo log, two tagged entry kinds, one `mark()` covering both:

```rust
enum Undo {
    Dom  { var: u32,   old: DomainMask },
    Word { table: u32, word: u32, old: u64 },
}
pub struct Trail { entries: Vec<Undo>, level: u32 }
```

- `mark() -> usize` — restore point (= `entries.len()`).
- `record_dom(var, old)` / `record_word(table, word, old)` — pushed **before** each
  write, by CT (not by callers).
- `restore_to(mark, &mut doms, &mut tables)` — pop LIFO, replay each `old`. Domain
  and word undos interleave correctly under LIFO.
- `clear()` — ob-core clone-and-return path (never restores).
- `enter()/leave()` bump/drop `level`.

**Save-word-on-first-write-per-level** (the one subtlety): CT ANDs many bits into a
`currTable` word within one step, but the word's `old` value must be trailed **only
once per level**, before its first mutation at that level. Each word carries a `u32`
`saved_level` stamp; `record_word` fires only when `stamp != trail.level`, then sets
`stamp = trail.level`. `level` = search depth. This keeps the trail O(distinct words
touched per level), not O(bit ops).

## 3. CT data structures (`src/ct.rs`)

**Static, precomputed once per _unique_ tensor** (shared across all `BoolTensor`s
that dedup to the same `unique_tensors` entry — memory reuse the dedup already
gives us):

```rust
struct TableMasks {
    n_rows: usize,
    n_words: usize,                 // ceil(n_rows / 64)
    // supports[axis*2 + value] = bitset over rows where config bit `axis` == value
    supports: Vec<u64>,             // arity*2 * n_words words
}
```
Built from `TensorData.support` at load: for each row index `r` with config `c`,
for each axis `i`, set bit `r` in `supports[i*2 + ((c>>i)&1)]`.

**Dynamic, reversible, one per `BoolTensor` instance:**

```rust
struct RSparseBitSet {
    words: Vec<u64>,          // live-row set; trailed via save-on-first-write
    saved_level: Vec<u32>,    // per-word level stamp (§2)
    index: Vec<u32>,          // permutation of word positions
    limit: usize,             // reversible: index[0..limit] are the maybe-nonzero words
    // `mask` scratch lives in SolverBuffer (reused, not trailed)
    residue: Vec<u32>,        // [axis*2+value] -> last word idx that had support (hint, not trailed)
}
```
`limit` is trailed as a plain reversible int (record old `limit` before shrinking).
Initial state: all `n_rows` bits set, `limit = n_words`, `index = 0..n_words`.

Memory: `n_words` words per tensor instance for `words`+`saved_level`+`index`;
`TableMasks` shared. For this workload `n_words ≤ 16` (VE `budget_B=10` ⇒ ≤1024
rows) and `=1` for small gates — negligible.

## 4. CT propagation step (per activated tensor)

`propagate_core` keeps the existing worklist (`SolverBuffer.queue` / `in_queue`,
seeded from changed vars' `v2t` tensors). For each popped tensor:

1. **updateTable** — restrict the live set to rows still consistent with current
   domains. For each axis `i` whose var domain is **not** `BOTH` (something was
   removed):
   - build AND-mask into scratch: `mask = ⋃_{v ∈ dom(x_i)} supports[i][v]` over
     active words only;
   - `currTable.intersect_with_mask(mask)` — `words[j] &= mask[j]` for active words;
     any word that becomes 0 is swapped out (shrink `limit`, trail `limit`), and
     each mutated word is trailed once/level.
   This is CT's *reset*-style recompute per constrained var; it touches only the
   currently-live words, never the full row list.
2. **contradiction** — if `currTable.is_empty()` → set `doms[0] = NONE` sentinel
   (record_dom first), clean the worklist exactly as today (`src/propagate.rs:157-165`),
   return.
3. **filterDomains** — for each **unfixed** var axis `i` and each value `v ∈ dom(x_i)`:
   check whether any live row supports `x_i = v` via
   `currTable.intersect_index(supports[i][v], &mut residue[i*2+v])` (residue gives an
   O(1) hit in the common case; else scan active words). If none → prune `v`:
   `record_dom` + narrow `doms[x_i]` + `enqueue_neighbors(v2t[x_i])`.

`RSparseBitSet` ops (`clear_mask` / `add_to_mask` / `intersect_with_mask` /
`intersect_index` / `is_empty`) are the CP-2016 primitives, iterating only
`index[0..limit]`.

**Equivalence to today:** step 3 prunes exactly the values with no support under
current domains = GAC = what `apply_updates` computes from `valid_or/valid_and`
today. `support_or`/`support_and` fast-paths are subsumed (a full-`BOTH` tensor
contributes no `updateTable` work).

## 5. Search & probe integration

- **`SolverBuffer`** (`src/problem.rs:28`): remove `scratch_doms`; add reusable
  `mask_scratch: Vec<u64>` (sized to max `n_words`). `queue`/`in_queue` unchanged.
- **Search state**: `doms: Vec<DomainMask>` and `tables: Vec<RSparseBitSet>` are the
  two reversible objects, threaded `&mut` through `bbsat_rec` (`src/solver.rs`) with
  a single shared `Trail`. Root init (`TnProblem::from_network`) builds `tables` and
  runs one root propagation under a throwaway mark.
- **Committed branch** (`bbsat_rec`): `mark → trail.enter()`; write branch literal(s)
  (trailed) + seed queue; `ct_propagate`; recurse; `trail.restore_to(mark)` +
  `trail.leave()`.
- **Lookahead probe** (`src/selector.rs` `DiffLookahead`, `src/table.rs`
  `compute_branching_result`): `probe_assignment` (`src/propagate.rs:100`) stops
  copying `base_doms`; a probe is `mark → apply+propagate → read resulting domains →
  restore_to(mark)`. This is what moves the 58%-inclusive probe path onto CT.
- **ob-core adapter** (`src/adapter.rs` `RuleProblem::apply_branch`): keeps
  clone-and-return semantics at the ob-core boundary, now deep-copying `tables`
  alongside `doms`; its internal propagate uses a **throwaway** `Trail` (`clear()`),
  never restored — sound because it clones and returns (same rationale the
  superseded trail spec used, §7 there).

## 6. Correctness & testing

- **Differential oracle (primary):** keep the current propagator as
  `propagate_core_rescan` (test-only). Property test over random networks × random
  partial assignments asserts CT's resulting `doms` are **bit-identical**, including
  the contradiction sentinel. GAC confluence guarantees this must hold.
- **Node-identical invariant:** an acceptance test runs `bbsat` with CT on fixtures
  and asserts `stats.branching_nodes` / `total_visited_nodes` equal recorded golden
  values — `factoring_15`, and the 22×22 instance added as a fixture
  (**19761 branching / 45322 visited**, from run A3). Any drift = a CT bug.
- **Backtrack correctness:** after `restore_to(mark)`, `doms` and every
  `RSparseBitSet` must equal their pre-`mark` snapshot, bit-identical (unit test with
  explicit snapshots; exercises save-on-first-write level stamping — the classic bug
  source).
- **`RSparseBitSet` unit tests:** each primitive in isolation.
- Existing acceptance harness (brute-force oracle) and all current tests stay green.

Built **test-driven** (per repo convention): write the differential/backtrack tests
first, then make them pass.

## 7. Success criteria

1. **Correctness:** all existing tests pass; differential oracle passes; node counts
   bit-identical to baseline (19761/45322 on the 22-bit instance).
2. **Performance:** re-run the A3 configs under runscribe (VE-on baseline 9.53 s,
   no-VE 28.86 s). Gate = **same node counts, lower wall time**. No fixed multiplier
   is promised (research caveat: absolute speedup is workload-specific), but a
   re-profile must show `propagate_core`'s self-time share fall substantially from
   82%.
3. **Profile:** samply re-run; the former hotspot is no longer dominant.

## 8. Relationship to the superseded trail spec

`2026-06-29-trail-mechanism-design.md` proposed a **domain-only** trail as a
behavior-preserving foundation for later incremental propagation. The research
confirmed (a) that trail alone yields ~0 speedup (copying <1% of runtime) and (b)
that CT needs to roll back **bit-set** state the domain-only trail can't express.
Building the domain-only trail first would therefore be throwaway scaffolding. This
design folds both needs into one substrate (§2): the same level-indexed log undoes
domain writes and `currTable` words. The superseded spec's sound sub-decisions are
retained here — in-place `doms` threaded through recursion, the contradiction
sentinel `doms[0] = NONE`, the ob-core clone-path throwaway trail.

## 9. Risks

- **Save-word-on-first-write level stamping** is the classic CT bug; §6 backtrack
  tests target it directly.
- **ob-core clone path** must deep-copy `tables` or state corrupts across the
  boundary; covered by the existing factoring acceptance test through `apply_branch`.
- **Memory**: `tables` is per-tensor-instance; bounded by `Σ n_words` ≤ a few ×
  `n_tensors` words — negligible here, but noted for very wide VE budgets.
