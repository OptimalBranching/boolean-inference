# Tree-Based Look-Ahead — Implementation Reference

**Date:** 2026-07-01
**Purpose:** Concrete implementation notes for tree-based look-ahead (the ~3.7–4.5×
lever identified in [the probe-count research](2026-07-01-lookahead-probe-count-optimization.md)),
sourced from Heule et al.'s march_eq (SAT 2004), its source (`tree.c`, `lookahead.c`,
`common.h`), and the Handbook of Satisfiability ch. 5.

## The core idea (march_eq)

Look-ahead probes many literals, each `assign → propagate → measure → undo`. Many
probes share propagation: if literal A implies B (via binary clauses) and both are
probed, A's propagation work needn't be redone under B. march organizes the look-ahead
literals into a **spanning forest of the binary-implication graph (BIG)**: parent implies
its children, so the parent's propagation is done once and each child probe runs *on top
of* the parent's already-propagated state.

**The elegant trick — nested timestamps, no per-literal undo.** A single integer
`currentTimeStamp` (CTS) is the "visibility floor": literal `a` is fixed iff
`timeAssignments[a] >= CTS`. Entering a tree node adds its `gap` to CTS; exiting subtracts
it. Assignments are stamped with the current CTS. Because a parent probes at a *higher*
CTS than its children, the parent's stamps stay visible to children; sibling stamps (at
disjoint CTS ranges) don't bleed across. "Backtracking" is just `CTS -= gap` — no undo
list. (Forced/backbone literals get `CTS = LOOK_MAX` so they're always visible.)

Traversal is a flat pre-order walk of a `treeArray` of `(literal, gap)`:
```
for (literal, gap) in treeArray:      # DFS pre-order
    CTS += gap                         # open scope
    treelookvar(literal)               # BCP at this CTS; children inherit it
    CTS -= gap                         # close scope
```
`treelookvar` skips the probe entirely if the literal is already fixed by an ancestor
(`IS_FIXED`), turns a UNSAT probe into a forced complement (failed literal), and detects
necessary assignments (both polarities imply the same literal).

**Tree construction** (per node, `tree.c`): (1) DFS post-order over the BIG restricted to
candidates; (2) Tarjan SCC contraction (a cycle = equivalent literals; `x` and `¬x` in one
SCC ⇒ UNSAT); (3) greedy spanning forest (attach each literal under the highest-rank
parent that implies it) serialized to `treeArray` with nested `gap`s.

**Measured speedup** (march_eq Table 3, tree size ~unchanged):
`longmult8/10/12` = 3.66× / 4.48× / 4.11×; `pyhala` factoring ≈ 1.6×; circuit `hwb` ≈
1.25×; **random k-SAT is 15–20% SLOWER** (sparse BIG ⇒ tree overhead > sharing). The win
scales with how dense the implication structure is — i.e. structured/multiplier/factoring.

Source: march_eq paper http://www.cs.cmu.edu/~mheule/publications/35420345.pdf ·
Handbook ch.5 https://www.cs.cmu.edu/~mheule/publications/p01c05_lah.pdf ·
source https://github.com/marijnheule/march-SAT-solver (`tree.c`, `lookahead.c`, `common.h`)

## How this maps onto OUR solver (the important part)

Our solver does **general GAC / table propagation**, not binary BCP, so march's *exact*
BIG+timestamp machinery is not a 1:1 port. But the *principle* — share the propagation of
a common prefix across sibling probes that fork from the same base — maps directly, and we
already have the substrate for it.

**What we have that makes this cheap:** the reversible bit-set + trail + **delta-tracking**
built for Compact-Table. A child scope that inherits the parent's fixed variables only
needs to propagate the *new* variable's delta — it does NOT re-propagate the parent's
work. That is exactly the sharing march gets from the timestamp trick, expressed through
our trail instead. (This is why the CT work was not wasted — it's the enabling substrate.)

**The two probe sites and how to share them:**

1. **Region-feasibility probes** (`table.rs`): today, for each enumerated region
   configuration, a *fresh* probe fixes the whole region and propagates from base. Many
   configurations **share a prefix of fixed background variables**. Restructure as a tree:
   fix the shared prefix once (one propagation), then branch per configuration on top,
   reusing the prefix's propagated state via the trail. Configurations that agree on the
   first k region vars share those k propagations.

2. **DiffLookahead probes** (`selector.rs`): probes both polarities of ~16 candidates, each
   from base. Sharing here is weaker (candidates are independent single-var fixes), but if
   fixing candidate `u` forces candidate `v` (a derived implication), then `v`'s probe can
   run under `u`'s state — the march tree idea. Cheapest first version: skip a candidate's
   probe if an earlier probe already fixed it (the `IS_FIXED`/`treelookvar` short-circuit),
   which our delta state already knows.

**Trail protocol (our analogue of the timestamp trick):** don't `restore_to` between
prefix-sharing siblings. Keep the parent's trail segment on the stack while probing each
child; `mark` at the parent, descend into a child with a nested `mark`, `restore_to` the
child's mark to move to the next sibling, and `restore_to` the parent's mark only when the
whole subtree is done. This is precisely the `CTS += gap` / `CTS -= gap` nesting, done with
`Trail::mark`/`restore_to` (which already undoes both domains and bit-set state).

**Minimal first version (highest value, least risk):** the region-feasibility loop is the
clearest win because its probes provably share background variables. Sort/group the
enumerated configurations by common prefix, propagate shared prefixes once, and vary on
top. Guard with the golden node-identity test (feasibility results must be unchanged).

## Honest caveats (from the research)

- The 3.7–4.5× is march_eq on multiplier instances with *binary* BCP sharing; our GAC
  probes differ, so **the gain here must be measured**, not assumed.
- For general GAC the per-probe saving can be smaller than for BCP, because GAC may
  re-examine constraint tables even for already-fixed values — though our delta-tracking is
  designed to skip exactly that (process only changed axes), which is the whole reason it
  should transfer.
- Random/sparse structure gets *no* benefit (or a slowdown). Our factoring circuits are
  dense/structured — the favorable regime — but other workloads may not be.
- Building the sharing structure has overhead (march does SCC + spanning per node); keep it
  cheap or it eats the win.

## Sources

- march_eq: http://www.cs.cmu.edu/~mheule/publications/35420345.pdf
- Handbook of Satisfiability, ch.5 (look-ahead): https://www.cs.cmu.edu/~mheule/publications/p01c05_lah.pdf
- march_dl (double look-ahead): https://www.cs.cmu.edu/~mheule/publications/JSAT2_3_Heule.pdf
- march source: https://github.com/marijnheule/march-SAT-solver (`tree.c`, `lookahead.c`, `common.h`, `preselect.c`)
