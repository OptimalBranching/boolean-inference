# Prefix-Sharing Look-Ahead (Region-Feasibility) — Design Spec

**Date:** 2026-07-02
**Status:** design, ready for planning
**Depends on / supersedes:** builds directly on the Compact-Table substrate
(`src/ct.rs`, `src/trail.rs`, `src/propagate.rs::probe`). Research backing:
[`docs/research/2026-07-02-prefix-sharing-lookahead-verdict.md`](../../research/2026-07-02-prefix-sharing-lookahead-verdict.md).

## Goal

Eliminate the redundant re-propagation in the region-feasibility probe loop
(`table.rs::compute_branching_result` step 1) by structuring the enumerated
region configurations as a **trie over region variables** and propagating each
shared prefix **once**, sharing the propagated state across sibling probes via
the existing trail `mark`/`restore_to`. Behavior must be **node-identical** to
today (golden test 19761 branching / 45322 visited nodes on `factoring_22x22`).

## Background — the redundancy

Today, step 1 of `compute_branching_result` is:

```rust
let mut feasible: Vec<u64> = Vec::new();
for &config in &cached_configs {
    if (config & check_mask) != check_value { continue; }      // consistency filter
    let feasible_here = probe(
        cn, doms, masks, tables, buffer, trail,
        &region_vars, full_mask, config,
        |d| d[0] != DomainMask::NONE,                          // feasibility read
    );
    if feasible_here { feasible.push(config); }
}
```

Each `probe` call independently: `open()`s a scope, fixes **all** region vars
to `config`, runs `ct_propagate` **from base**, reads feasibility, and
`restore_to`s. Configurations that share a prefix of fixed region-var values
re-propagate that shared prefix from scratch, once per config. Profiling puts
this loop (`findbest`) at ~95% of runtime.

## Correctness foundation (settled)

The research verdict (24/25 claims confirmed; correctness core all 3-0):
incremental prefix-wise propagation reaches the **identical no-wipeout verdict**
per full configuration as the one-shot probe, because CT/GAC propagators are
**monotonic + inflationary**, so by Apt chaotic-iteration / Schulte–Stuckey–Tack
confluence the fixpoint is **schedule-independent**. Consequences we rely on:

1. **Feasibility identity.** A config is GAC-feasible (one-shot `d[0] != NONE`)
   **iff** the trie path to its leaf reaches the leaf with no node (including the
   leaf's own post-propagation state) contradicting.
2. **Early-prune soundness.** A contradiction at an internal prefix node ⇒
   wipeout for every extension (wipeout monotone under further restriction), so
   pruning the whole subtree can never drop a config a full probe would call
   feasible.
3. **Idempotence not required** — CT's incremental-vs-reset internal choice is
   fine.
4. **Only failure mode:** a non-monotonic region propagator. Standard CT is
   monotonic; we add a debug assertion rather than a runtime guard.

## Node-identity argument (why the branching rule is unchanged)

The step-1 output `feasible` is consumed **order-independently**: step 2 maps
each config through `project`, then `projected.sort_unstable(); projected.dedup()`
before building the `BranchingTable`. Therefore the *set* of feasible configs is
all that matters, not the order. The trie DFS returns exactly the same **set**
(by feasibility identity above), so `projected` — hence the branching table,
hence the optimal rule, hence every branch taken — is bit-for-bit unchanged.
This is what the golden 19761/45322 test verifies.

## Algorithm — trie DFS with per-level reversible scope

New primitive in `src/propagate.rs`, called once per branching node in place of
the step-1 loop:

```rust
/// Return the subset of `configs` that are GAC-feasible from the current
/// `(doms, tables)`, sharing the propagation of common prefixes. Set-identical
/// to probing each config independently with `probe(.. |d| d[0] != NONE)`.
/// On return, `(doms, tables)` and `buffer` are exactly as on entry.
#[allow(clippy::too_many_arguments)]
pub fn feasible_configs(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    region_vars: &[usize],   // region var ids, ascending (as today)
    configs: &[u64],         // already filtered by the consistency mask
) -> Vec<u64>;
```

**Trie order (v1):** the *unfixed* region-var positions in `region_vars` order
(ascending var id). Already-fixed positions carry no delta (all filtered configs
agree on them, per the consistency filter) and are **skipped** in the descent —
matching `probe`'s `if doms[var] != nd` no-op skip. (MINCE / fail-first / Gray-code
orderings are out of scope for v1 — see Future.)

**Enumeration:** sort `configs` once by the bit-key that reads the unfixed
positions in trie order (level 0 = most significant). Sorted contiguous ranges
then correspond to trie subtrees; descent partitions a range by the next unfixed
position's bit.

**Descent (recursive):**

```
descend(level, range):                       # range = contiguous sorted slice of configs
    if level == n_unfixed:                    # leaf: whole assignment fixed w/o contradiction
        push every config in range to result  # (range is a single config)
        return
    pos  = unfixed_order[level]
    var  = region_vars[pos]
    for value in {0,1} present at bit `pos` within range:   # in sorted order: 0s then 1s
        sub = subrange of `range` with bit pos == value
        if sub empty: continue
        trail.open()                          # NEW EPOCH — required for nested restore (see below)
        let m = trail.mark()
        record_dom + set doms[var] = value; enqueue_var_change(cn, buffer, var)
        ct_propagate(cn, doms, masks, tables, buffer, trail)
        if doms[0] != DomainMask::NONE:        # no contradiction at this prefix
            descend(level + 1, sub)
        trail.restore_to(m, doms, tables)      # revert this level; bumps epoch
```

## Trail / epoch discipline (the crux)

CT's reversible bit-set saves each word **once per epoch** (`saved_epoch`),
and `Trail::restore_to` reverts entries down to a `mark` **and bumps the epoch**.
This save-once stamping is only correct for restores at **epoch granularity**.
Nested restores within a single epoch would silently skip re-saving a word
already written earlier in that epoch, so `restore_to(inner_mark)` would fail to
revert the inner write.

**Rule:** call `trail.open()` (bump epoch) **before fixing+propagating at each
trie level**, and pair it with a `mark()`; `restore_to(mark)` when backtracking
that level. One epoch per trie edge. Then every word written at a level is saved
fresh in that level's epoch (its `saved_epoch` differs from any ancestor level's
epoch), so `restore_to` reverts it correctly. This is the same discipline the
main search already uses per branch in `solver.rs`.

## Buffer discipline

- Clear `buffer.queue` / `buffer.in_queue` **once** at entry to `feasible_configs`
  (as `probe` does).
- `ct_propagate` drains the worklist to empty on both the success and the
  contradiction path (it must leave `queue` empty and all `in_queue`/`dirty`
  cleared — verify against the contradiction cleanup already present for the
  rescan propagator). Rely on that so siblings never inherit pending events;
  add `debug_assert!(buffer.queue.is_empty())` before each level's
  `enqueue_var_change` to catch a leak.

## Call-site change (`table.rs`)

Replace the step-1 loop with:

```rust
let feasible = feasible_configs(
    cn, doms, masks, tables, buffer, trail, &region_vars, &filtered_configs,
);
if feasible.is_empty() { return (None, region_vars); }
```

where `filtered_configs` is `cached_configs` after the existing
`(config & check_mask) == check_value` filter. Steps 2–4 (projection, table
build, `optimal_rule`) are **unchanged**.

`probe` stays in `propagate.rs` — it is no longer used by `table.rs` but becomes
the **differential oracle** for testing `feasible_configs` (probe each config
independently and compare the feasible set).

## Testing

1. **Golden node-identity (primary guard):** the existing `tests/ct_acceptance.rs`
   must still assert `branching_nodes == 19761`, `total_visited_nodes == 45322`
   on `factoring_22x22`. This is the behavior-preservation gate.
2. **Differential vs the probe oracle:** for randomized `(cn, doms, region_vars,
   configs)` fixtures (multi-seed, mirroring the CT `engine_tests` discipline),
   assert `feasible_configs(...)` returns the **same set** as
   `{ c in configs : probe(.., &region_vars, full_mask, c, |d| d[0] != NONE) }`,
   and that `(doms, tables)` are restored to entry state afterward (byte-equal).
3. **Backtrack/restore integrity:** after `feasible_configs`, re-run a plain
   `ct_propagate` from base and confirm no residual domain narrowing or bit-set
   drift (the trail fully restored).
4. **Edge cases:** empty `configs`; all region vars already fixed (0 unfixed
   levels → each surviving config is a leaf); single config; a config set where
   an internal prefix contradicts (subtree pruned) — all must match the oracle.

## Performance expectation (honest)

march_eq's tree-based look-ahead measured 72–78% time reduction on structured
multiplier instances; our factoring/CircuitSAT is the favorable regime. But the
substrate differs (GAC-over-config-set vs binary BCP), so the gain **must be
measured**, not assumed. Measure with `runscribe` on `factoring_22x22` (VE-on and
no-VE), reporting wall-clock and µs/branching-node; node counts must be identical.

## Out of scope / future (explicitly deferred)

- **Variable ordering beyond static region order:** MINCE-style
  connectivity/min-cut ordering, fail-first refinement, Gray-code/minimal-change
  enumeration to amortize restore. v1 uses the natural region-var order.
- **Project-and-intersect alternative:** computing the GAC-feasible config set
  directly by one region tensor contraction / table-MDD intersection, instead of
  a trie of probes. Research flags this as a plausible bigger win but an **open,
  unbenchmarked** experiment. Track as a follow-up A/B against `feasible_configs`.
- **difflookahead prefix-sharing** (`selector.rs`): sharing is weaker there
  (independent single-var fixes); deferred to a separate spec.

## Risks

- **Nested-restore correctness** (the epoch discipline) is the one place a bug
  would be silent-but-wrong; the differential oracle + restore-integrity test are
  aimed squarely at it.
- **No win / a loss** if regions are small or configs share little prefix
  (overhead > sharing). Mitigation: measure; the change is isolated and can be
  reverted at the single call site. Consider a threshold (fall back to the flat
  probe loop when `configs.len()` or region size is below a cutoff) only if
  measurement shows a regression — not in v1 by default.
