# Compact-Table Propagation — Design

**Status:** draft (brainstorming), revised after Codex xhigh review → pending user review → implementation plan
**Date:** 2026-07-01
**Branch:** `framework-apply-branch` (Rust crate `boolean-inference`)
**Supersedes:** [`2026-06-29-trail-mechanism-design.md`](2026-06-29-trail-mechanism-design.md)
(the domain-only trail is absorbed into CT's reversible substrate; see §8).
**Informed by:** profiling (samply run A4: `propagate_core` = 82% CPU), the
[incremental-table-propagation research](../../research/2026-06-30-incremental-table-propagation.md),
and a Codex xhigh critical review (18 findings; resolutions in §10).

## Goal

Replace the linear GAC support re-scan (`scan_supports`, `src/propagate.rs:24-50`)
— **82% of solver CPU** — with **Compact-Table (CT)** (Demeulenaere et al., CP
2016): per-constraint GAC maintained incrementally via a **reversible sparse
bit-set** of live support rows, updated with word-parallel bit operations on
domain changes and rolled back cheaply on backtrack.

**This requires converting the search from copy-based recursion to in-place +
backtrack.** Today `bbsat_rec` takes `doms: Vec<DomainMask>` **by value** and each
child gets a fresh `sub = scratch.to_vec()` — there is no trail and no restore
(`src/solver.rs:54-113`). CT's incrementality only pays off if the per-tensor live
sets **persist across an assignment and are undone on backtrack**, so the search
must thread a single mutable `(doms, tables)` under a `Trail`. That conversion —
not just adding a propagator — is the core of this work.

**Behavior-preserving claim (scoped precisely, per review §10/#7):** GAC to a
fixpoint is confluent, so on any **non-contradiction** propagation CT reaches the
**bit-identical** domain state as the current propagator. On a **contradiction**
CT sets the same sentinel (`doms[0] = NONE`) but leaves other domains unspecified
(both old and new propagators early-exit; §4). The search consumes only
(a) the contradiction *bit* from failed probes and (b) full domains from
*successful* probes (`sum_active_degree`, `measure`), plus (c) the root-built
`RegionCache` (§5). All three are preserved ⇒ **identical branch decisions and
node counts** — the oracle (§6).

## Non-goals (deferred)

- Delta-based "modified variable" tracking. First version **recomputes each active
  tensor's AND-mask from its variables' current domains** (arity is small); a
  changed-var delta set is a later refinement.
- Negative/compressed/short tables, MDD propagators, tuple hashing.
- STR3 path-optimality (CT chosen per research).
- Any change to the branching rule or variable-selection *logic* — only the
  *mechanism* by which the search holds state (copy → in-place+trail) changes.
- Parallel search, GPU region contraction, SIMD micro-opt.

**Precondition (documented invariant, review §10/#17):** every `BoolTensor.var_axes`
has **distinct** variable ids. The *current* propagator already assumes this
(`compute_query_masks`/`apply_updates` treat each axis independently), so CT does
not regress it; we make it explicit and add a `debug_assert` in `setup_problem`
(`src/network.rs`). Equality-aware tables are out of scope.

## Background: what CT replaces

Each tensor activation in `propagate_core` (`src/propagate.rs:144`) calls
`scan_supports`, which **linearly re-examines every row of the tensor's support
table** (`TensorData.support: Vec<u32>`) against current domains — from scratch,
every activation. In tensor-network terms this recomputes a Boolean-semiring
contraction of the constraint tensor against the (rank-1) domain vectors each time.
CT caches *which rows are still live* in a reversible bit-set and updates it
incrementally, so no live-row set is ever rescanned.

---

## 1. Architecture & the copy → in-place conversion

New / changed modules under `src/`:

| Module | Role |
|---|---|
| `src/trail.rs` (new) | Reversible substrate: an undo log with three entry kinds — domain writes, bit-set-word writes, and `limit` changes — plus a monotonic epoch counter. |
| `src/ct.rs` (new) | `RSparseBitSet` (reversible sparse bit-set) + `TableMasks` (shared static per-literal masks) + the CT per-tensor propagation step. |
| `src/propagate.rs` (rewrite) | `propagate_core` becomes the CT worklist loop over `(doms, tables, trail)`. `scan_supports`/`apply_updates`/`compute_query_masks` are kept **test-only** as `propagate_core_rescan` (differential oracle). |
| `src/problem.rs` | `SolverBuffer` loses `scratch_doms`, gains reusable `mask_scratch: Vec<u64>`. Search state = `(doms, tables, trail)`. `TnProblem::from_network` builds `tables` and seeds **all** tensors for root propagation (§4). |
| `src/solver.rs` | `bbsat_rec` switches owned-`doms` recursion → in-place `(&mut doms, &mut tables, &mut trail)` with `mark → apply+propagate → recurse → restore_to` per clause. `RegionCache` stays root-built and threaded unchanged (§5). |
| `src/selector.rs`, `src/table.rs` | the two lookahead-probe sites switch copy → `mark/apply/read/restore` via the redesigned probe API (§5). |
| `src/adapter.rs` | `RuleProblem` gains a `tables` field; `apply_branch` deep-copies `(doms, tables)`, applies on the clone under a **throwaway** trail, returns it (never restored). |

**Call sites that must convert (all four are "from base: apply literals →
propagate → use result"):**

| Site | File | Today | Under CT |
|---|---|---|---|
| committed branch | `solver.rs:99-107` | probe base → owned `sub` → recurse | `mark → apply+propagate in place → recurse → restore` |
| difflookahead probe | `selector.rs:105,108` | `probe_assignment` → read slice | `mark → apply+propagate → read → restore` (§5 API) |
| region feasibility probe | `table.rs:47` | `probe_assignment` → read `scratch[0]` | same `mark/read/restore` |
| ob-core `apply_branch` | `adapter.rs:63-95` | clone `doms`, propagate, return | clone `(doms, tables)`, propagate on clone (throwaway trail), return |

## 2. Reversible substrate (`src/trail.rs`)

```rust
enum Undo {
    Dom   { var: u32,   old: DomainMask },
    Word  { table: u32, word: u32, old: u64 },
    Limit { table: u32, old: u32 },
}
pub struct Trail { entries: Vec<Undo>, epoch: u64 }
```

- `mark() -> usize` — restore point (`entries.len()`).
- `open() -> u64` — bump and return `epoch` (called when entering any reversible
  scope: a branch descent **or** a probe). **Never reused.**
- `record_dom` / `record_word` / `record_limit` — push the OLD value **before** each
  write; called by CT, not by callers.
- `restore_to(mark, &mut doms, &mut tables)` — pop LIFO, replaying each `old` into
  `doms`, into `tables[t].words[w]`, or into `tables[t].limit`. Then bump `epoch`
  (so post-restore is a fresh, never-reused epoch). Domain, word, and limit undos
  interleave correctly under LIFO.
- `clear()` — drop all entries without restoring (ob-core clone path, §5).

**Epoch, not numeric level (fixes review §10/#5,#6,#12).** Save-word-once-per-scope
cannot key on search depth: siblings and repeated probes reuse the same depth, so a
word saved in one sibling would be skipped in the next and restore would corrupt the
parent. Instead a **monotonic `epoch: u64`** is bumped on every `open()` and every
`restore_to()` and never repeats. Each `RSparseBitSet` word carries `saved_epoch`;
`record_word` fires iff `saved_epoch != trail.epoch`, then stamps it. Because epochs
never recur, a word touched in a prior scope is always re-saved in a new scope.
`saved_epoch` initializes to `0` and `epoch` starts at `1`, so the first write in
any scope always saves (fixes #6). Nested scopes may re-save a word redundantly;
that is still correct under LIFO (extra entries only), just not maximally tight.

**Why `index` is not trailed (fixes review §10/#4).** Removing a dead word swaps its
slot in `index` with `index[limit-1]` and decrements `limit` (§3). Restoring `limit`
alone puts that word back inside `index[0..limit]`; the *set* of active words is
identical (only their order within `index` may differ, which no operation depends
on). So only `limit` needs a trail entry (`Limit`), not the whole permutation. Word
*contents* are restored via `Word`.

## 3. CT data structures (`src/ct.rs`)

**Static, precomputed once per _unique_ tensor** (shared across all `BoolTensor`s
that dedup to the same `unique_tensors` entry):

```rust
struct TableMasks {
    n_rows:  usize,
    n_words: usize,                 // ceil(n_rows / 64); n_rows==0 => n_words==0 (see below)
    supports: Vec<u64>,             // length (arity*2) * n_words
}
// EXACT layout (fixes review §10/#1): the bitset for "axis i takes value v" is
//   supports[(i*2 + v) * n_words .. (i*2 + v + 1) * n_words]
// bit r of that slice is 1 iff row r's config has bit i == v.
```
Built from `TensorData.support` at load. **Last-word high bits are zeroed**: only
bits `0..n_rows` are meaningful; bits `n_rows..64*n_words` stay 0 in every mask and
in `currTable` so word ops never see phantom rows (fixes review §10/#3).

**Empty support (`n_rows == 0`)** is a valid source state (an unsatisfiable tensor,
`src/network.rs`). Then `n_words == 0`; the tensor's `RSparseBitSet` starts empty and
`is_empty()` is true, so its first activation reports a contradiction — matching the
current propagator's immediate-contradiction behavior. No `index`/`residue`/word
read is performed when `n_words == 0` (fixes review §10/#3).

**Dynamic, reversible, one per `BoolTensor` instance:**

```rust
struct RSparseBitSet {
    words:       Vec<u64>,   // physical-indexed live-row set; trailed (save-on-first-write)
    saved_epoch: Vec<u64>,   // per-word epoch stamp (§2); init 0
    index:       Vec<u32>,   // permutation of physical word ids; index[0..limit] = active
    limit:       u32,        // reversible: count of possibly-nonzero words
    residue:     Vec<u32>,   // [i*2+v] -> physical word id last seen supporting (i=v); hint only
}
```
- `words` is indexed by **physical word id** and never reordered; `index` holds the
  active subset, `limit` its length. Swap-out: `swap(index[p], index[limit-1]); limit-=1`.
- Initial state: all `n_rows` bits set (high bits of last word zero), `limit=n_words`,
  `index=0..n_words`, `saved_epoch=[0; n_words]`, `residue=[0; arity*2]`.
- **`residue` semantics (fixes review §10/#2):** stores a **physical** word id. On
  `intersect_index(mask)` first test `words[residue] & mask[residue] != 0`; if so it
  is a valid support (a physical word that is 0 gives 0, so a swapped-out word simply
  misses and we fall through). Else scan `index[0..limit]`; on a hit, update `residue`.
  Residue is a hint, never trailed — a stale hint can only cost one failed word test,
  never a wrong answer.

Memory: `n_words` words each for `words`/`saved_epoch`/`index` per tensor instance,
`TableMasks` shared. For this workload `n_words ≤ 16` (VE `budget_B=10` ⇒ ≤1024
rows), `=1` for small gates — negligible.

## 4. CT propagation step (per activated tensor)

`propagate_core` keeps the existing worklist (`SolverBuffer.queue`/`in_queue`,
seeded from changed vars' `v2t` tensors). Tensors are queued on **any** neighbor
change regardless of fixedness, and a fully-fixed tensor is still processed as a
final consistency check — CT never drops a tensor's table (fixes review §10/#16).

**Precondition:** the incoming state is contradiction-free (`doms[0] != NONE`); the
search never calls propagate on an already-failed state, and the sentinel is the
only `NONE` (fixes review §10/#18). If any processed var's domain is `NONE`, that is
an immediate contradiction.

For each popped tensor `t` (skip entirely if `n_words == 0` after emitting the
empty-support contradiction):

1. **updateTable** — for each axis `i` whose var domain is **not** `BOTH`
   (i.e. `D0`/`D1`, something removed): build `mask = ⋃_{v ∈ dom(x_i)} supports[i][v]`
   into `SolverBuffer.mask_scratch` over active words only, then
   `currTable.intersect_with_mask(mask)`: for each active word `w`, `record_word` (if
   new epoch) then `words[w] &= mask[w]`; if it becomes 0, `record_limit` (once) and
   swap it out. Touches only live words, never the full row list.
2. **contradiction** — if `currTable.is_empty()` → `record_dom(0, doms[0])`, set
   `doms[0] = NONE`, clean the worklist exactly as today
   (`src/propagate.rs:157-165`), return.
3. **filterDomains** — for each **unfixed** axis `i` and each value `v ∈ dom(x_i)`:
   `currTable.intersect_index(supports[i][v], &mut residue[i*2+v])`; if no live row
   supports `x_i = v` → `record_dom` + narrow `doms[x_i]` + `enqueue_neighbors(v2t[x_i])`.

**Equivalence to today:** step 3 prunes exactly the values with no support under the
current domains = GAC = what `apply_updates` computes from `valid_or/valid_and`. A
full-`BOTH` tensor does no `updateTable` work but is still filtered — which is why
**root propagation must seed every tensor** (units and root pruning depend on it,
`src/problem.rs:63-72`; fixes review §10/#15).

## 5. Search & probe integration

**State.** `doms: Vec<DomainMask>` and `tables: Vec<RSparseBitSet>` are the two
reversible objects, threaded `&mut` through `bbsat_rec` with one shared `Trail`.
`TnProblem::from_network` builds `tables`, seeds all tensors, and runs root
propagation under an initial scope.

**`RegionCache` invariant (fixes review §10/#9).** The cache is built **once** from
root domains in `bbsat` (`src/solver.rs:39`) and threaded `&mut` unchanged; its
cached configs are encoded over root-unfixed region vars (`src/table.rs:32` warns
rebuilding at non-root doms is unsound). CT changes neither the cache's construction
nor the *values* of `doms` it depends on, so branching tables — and thus node counts
— are unaffected. This design must **not** rebuild or mutate the cache.

**Committed branch** (`bbsat_rec`): the recursion signature changes to
`(&mut doms, &mut tables, &mut trail, ...)`. Per clause:
```
let m = trail.mark(); trail.open();
apply_literals(clause, variables, doms, tables, trail);   // each write trailed
ct_propagate(cn, doms, tables, buffer, trail);
if doms[0] != NONE { recurse in place; if found { return } }
trail.restore_to(m, doms, tables);
```
The leaf/2-SAT paths still read `doms` directly (no copy needed).

**Probe API redesign (fixes review §10/#10,#11).** `probe_assignment`'s
return-a-slice contract is incompatible with restore-before-return. Replace it with
a **read-under-scope** helper that applies, propagates, hands the live state to a
caller closure, then restores:
```rust
pub fn probe<R>(cn, doms, tables, buffer, trail,
                vars, mask, val, read: impl FnOnce(&[DomainMask]) -> R) -> R
// open scope; apply literals (trailed); ct_propagate; let r = read(doms);
// restore_to(mark); r
```
- **difflookahead** (`selector.rs`): `let (f0,d0) = probe(.., |d| (has_contradiction(d), sum_active_degree(cn,d)));`
  then the same for value 1. Each probe forks from the same base and restores it, so
  the two are independent (as today), now with tables incremental instead of copied.
- **region feasibility** (`table.rs:47`): `let feas = probe(.., |d| d[0] != NONE);`.
  All probe writes (including the seeded literals) are trailed, so `restore_to` fully
  undoes them.

**ob-core `apply_branch` (fixes review §10/#13,#14).** `RuleProblem` gains
`tables: Vec<RSparseBitSet>` (and shares `TableMasks` via the `Arc<ConstraintNetwork>`).
`compute_branching_result` builds it from the node's live `(doms, tables)` (clones
both). `apply_branch` deep-copies `(doms, tables)`, applies the clause and propagates
on the **clone** under a **throwaway** `Trail` whose entries are dropped with the
clone — nothing is ever restored, so the epoch/stamp state dies with the clone and
cannot contaminate the search trail. `clear()` is documented as "for the clone path:
the trailed state is discarded together with its tables." ob-core only reads
`measure(returned)`; it never re-propagates the returned problem with a restoring
trail.

## 6. Correctness & testing (TDD)

- **Differential oracle (fixes review §10/#7,#8).** Keep the current propagator as
  `propagate_core_rescan` (test-only). Property test over random networks × random
  partial assignments asserts: on the **non-contradiction** path, CT's `doms` are
  **bit-identical**; on the **contradiction** path, both set the sentinel
  (`has_contradiction` agrees) — domains off the fixpoint are not compared.
- **CT-state invariant (fixes review §10/#8).** After propagation to a fixpoint,
  assert for every tensor: row `r ∈ currTable` ⇔ row `r`'s config is consistent with
  the current domains of all its axes. This pins CT's own state independently of the
  domain oracle, so later incremental calls start from a correct live set.
- **Backtrack correctness (fixes review §10/#4,#5,#6).** After `restore_to(m)`,
  `doms` **and** every `RSparseBitSet` (`words`, `limit`, and the active set
  `index[0..limit]`) must equal their pre-`m` snapshot. A dedicated stress test
  exercises sibling branches and repeated same-level probes touching overlapping
  words — the epoch mechanism's failure mode.
- **Node-identical acceptance.** `bbsat` under CT on fixtures asserts
  `stats.branching_nodes` / `total_visited_nodes` equal recorded golden values:
  `factoring_15`, and the 22×22 instance (**19761 branching / 45322 visited**, run
  A3). Any drift = a CT bug.
- **`RSparseBitSet` unit tests** for each primitive, including the `n_words == 0`,
  last-word-high-bit, and stale-residue cases.
- Existing acceptance harness (brute-force oracle) and all current tests stay green.

Built test-first: write the differential, invariant, and backtrack tests, then make
them pass.

## 7. Success criteria

1. **Correctness:** all existing tests pass; differential oracle, CT-state invariant,
   and backtrack tests pass; node counts bit-identical to baseline (19761/45322).
2. **Performance:** re-run the A3 configs under runscribe (VE-on 9.53 s, no-VE
   28.86 s). Gate = **same node counts, lower wall time**. No fixed multiplier is
   promised (research caveat: absolute speedup is workload-specific), but a re-profile
   must show `propagate_core`'s self-time share fall substantially from 82%.
3. **Profile:** samply re-run; the former hotspot is no longer dominant.

## 8. Relationship to the superseded trail spec

`2026-06-29-trail-mechanism-design.md` proposed a **domain-only** trail. The research
confirmed (a) a domain-only trail yields ~0 speedup (copying <1% of runtime) and (b)
CT must roll back **bit-set** state a domain-only trail cannot express. Building it
first would be throwaway scaffolding. This design folds both needs into one substrate
(§2): the same log undoes domain writes, `currTable` words, and `limit`. Retained
sound sub-decisions from the superseded spec: in-place `doms` threaded through
recursion, the contradiction sentinel `doms[0] = NONE`, the ob-core clone-path
throwaway trail.

## 9. Risks

- **Epoch save-on-first-write** is the classic CT bug surface; §6 backtrack/sibling
  stress tests target it directly, and the epoch (not level) scheme removes the
  reuse hazard.
- **In-place conversion scope.** Converting `bbsat_rec` and both probe sites off
  copies is the invasive part; the differential oracle + node-identical gate bound
  the risk. The implementation plan should stage it: (1) `RSparseBitSet` + `TableMasks`
  + CT step validated against `propagate_core_rescan` via the *existing* copy-based
  probe first; (2) then the in-place `(doms, tables)` + trail conversion; (3) then
  `apply_branch`.
- **ob-core clone cost.** `apply_branch` now clones `tables` too; bounded by
  `Σ n_words` words (small here). A pooled/COW clone is a follow-up if it shows up.
- **`RegionCache`** must remain root-built and untouched or node counts drift; called
  out as an explicit invariant (§5).

## 10. Review resolutions (Codex xhigh, 18 findings)

| # | Severity | Resolution |
|---|---|---|
| 1 | AMBIGUITY | Exact `supports` stride specified: `(i*2+v)*n_words + word` (§3). |
| 2 | AMBIGUITY | `residue` = physical word id; validate via `words[residue] & mask[residue]`, else scan + refresh; never trailed (§3). |
| 3 | MISSING | Last-word high bits zeroed; `n_rows==0 ⇒ n_words==0`, empty set ⇒ immediate contradiction, no index/residue reads (§3,§4). |
| 4 | BUG | Added `Limit` undo entry; `limit` trailed. `index` need not be trailed — restoring `limit` restores the active *set* (§2). |
| 5 | BUG | Replaced numeric-level test with a never-reused monotonic **epoch** bumped on `open()` and `restore_to()` (§2). |
| 6 | MISSING | `saved_epoch` inits to 0, `epoch` starts at 1 ⇒ first write per scope always saves; epochs never recur ⇒ no stale-stamp (§2). |
| 7 | DESIGN_RISK | Oracle scoped: bit-identical domains only on the non-contradiction path; contradiction compared by sentinel/`has_contradiction` (§ intro, §6). |
| 8 | DESIGN_RISK | Added an explicit CT-state invariant (currTable == rows consistent with domains) as its own test (§6). |
| 9 | MISSING | `RegionCache` stays root-built and threaded unchanged; stated as a hard invariant CT must not violate (§5). |
| 10 | BUG | Probe API redesigned to read-under-scope closure (`probe(.., read)`) that restores after reading (§5). |
| 11 | BUG | Seeded probe literals are trailed via `apply_literals`, so `restore_to` undoes them (§4,§5). |
| 12 | BUG | Subsumed by the epoch fix (#5): repeated same-level probes each `open()` a fresh epoch (§2,§5). |
| 13 | MISSING | `RuleProblem` gains a `tables` field, built from the node's live tables (§5). |
| 14 | AMBIGUITY | Clone path uses a throwaway trail discarded with the clone; `clear()` contract documented; no cross-contamination (§5). |
| 15 | MISSING | Root init seeds **all** tensors so full-`BOTH` tensors still get `filterDomains` (units/root pruning) (§4). |
| 16 | MISSING | Tensors queued regardless of fixedness; fully-fixed tensors processed as final consistency checks; tables never dropped (§4). |
| 17 | MISSING | Reframed: distinct-axis is a **pre-existing** assumption of the current propagator, not a CT regression; documented as a precondition + `debug_assert` (Non-goals). |
| 18 | MISSING | Propagate precondition: input contradiction-free, sentinel is the only `NONE`; a `NONE` on a processed var = immediate contradiction (§4). |
