# Incremental Table-Constraint Propagation — Research

**Date:** 2026-06-30
**Why:** Profiling (samply, 43-bit factoring) showed `propagate::propagate_core` =
**82% of CPU**, dominated by the GAC support re-scan `scan_supports`
(`src/propagate.rs:39-49`), which linearly re-examines **every row of a tensor's
support table on every activation**. This report surveys the algorithms that fix
exactly this, how comparable solvers implement them, and how they map onto this
crate. Informs the [trail-mechanism spec](../superpowers/specs/2026-06-29-trail-mechanism-design.md).

**Method:** deep-research harness — 6 search angles, 21 sources fetched, 88 claims
extracted, top 25 adversarially verified (3-vote, need 2/3 to refute). All 7
findings below survived **unanimous 3-0** verification against primary sources.

---

## Bottom line

The literature points to **one dominant answer: Compact-Table (CT)**
(Demeulenaere et al., CP 2016). It enforces GAC by maintaining a **reversible
sparse bit-set of currently-valid tuples**, invalidated incrementally with bitwise
word operations on value removals and restored cheaply on backtrack. It beats all
prior SOTA (STR2, STR3, GAC4R, MDD4R, AC5-TC) and is the **default table
propagator in OR-Tools CP-SAT and OscaR**. Our `TensorData.support: Vec<u32>` is
already exactly the tuple table CT operates on, so this is a **port, not a
research problem** — with one real integration gap (below).

The trail (the planned [trail-mechanism](../superpowers/plans/2026-06-29-trail-mechanism.md))
is **necessary substrate but not itself the speedup** — the literature is explicit
and even *refuted* the contrary. This matches our profile: copying/alloc is <1%,
so the trail alone moves nothing; the win is the incremental bit-set on top.

---

## 1. Algorithm SOTA

### Compact-Table (CT) — the recommendation
*Demeulenaere, Hartert, Lecoutre, Perez, Perron, Régin, Schaus, CP 2016.*
[arxiv.org/abs/1604.06641](https://arxiv.org/abs/1604.06641)

- Maintains `currTable`, a **reversible sparse bit-set** over the rows of the
  support table; tuples are invalidated *incrementally* on value removals via
  bit-set AND operations, plus **residues** (cache last-supporting word) and a
  **reset** operation.
- Iterates **only non-zero words**; the bit-set is **trailed** (restored on
  backtrack), never recomputed.
- **Complexity / memory:** ~`#tuples/64` words per constraint for the bit-set,
  plus a sparse index of non-zero word positions, plus one precomputed
  `supports[var][value]` bit-mask per variable-value. Word-level parallelism
  amortizes table size by 64×.
- **When it wins:** moderate arity/domain, dense-to-medium tables. Reported
  fastest on 94.47% of instances, ~3.77× avg over best competitor. *(Caveat:
  those are the authors' own OscaR experiments; independent adoption by OR-Tools/
  OscaR/Choco mitigates self-report bias but absolute speedups on our
  tensor-network solver will differ.)*

### STR / STR2 / STR3 — the ancestors
- **STR** (Ullmann ~2007): dynamically maintains the table so only currently-valid
  supports remain, deleting tuples as values are removed.
- **STR2** (Lecoutre, *Constraints* 2011,
  [PDF](http://cse.unl.edu/~choueiry/Documents/STR2-Lecoutre-Long.pdf)): limits
  validity-checking and support-search to changed vars (`Sval`) and unbound vars
  still needing support (`Ssup`); **potentially up to `r`× faster** (r = arity),
  so the win grows with arity. *Measured ~2× in practice; the r-factor is
  best-case.* Not fully incremental across calls — prunes per-call work but still
  rebuilds.
- **STR3** (Lecoutre/Likitvivatanavong/Yap, *AIJ* 2015,
  [link](https://www.sciencedirect.com/science/article/pii/S000437021400143X)):
  **path-optimal** — completely avoids re-traversing live rows along any branch of
  the search tree. **Outperforms STR2 precisely when tables stay large during
  search** (most rows remain valid). When tables shrink drastically, STR2's
  simpler reduction is competitive.
- **Cheap backtrack without a per-tuple undo log** (STR family, finding [3]): a
  `position[]` permutation + `currentLimit[]` + per-level `levelLimits[c][p]`
  restore the table boundary in O(1) (two pointers) — a **level-indexed
  reversible/sparse-set substrate**, not a trail of individual tuple writes. This
  is the design pattern CT's reversible sparse bit-set generalizes.

### Watched literals — the complementary framework
*Gent, Jefferson, Miguel, Minion, CP 2006
([PDF](https://sites.cs.st-andrews.ac.uk/people/ipg1/papers/GentJeffersonMiguelCP06.pdf));
Gent et al., JAIR 2013.*

- Frames support search as **backtrack-stable list scanning** and quantifies our
  exact problem: **stateless reset-and-rescan is Θ(N²) `acceptable()` calls per
  leaf** for a length-N list — *that is the `scan_supports` pattern*. Storing a
  last-support pointer and restoring it on backtrack (state-restoration / trail)
  drops it to **≤ N per branch**; the optimal **circular** scheme is
  amortized-optimal with worst-case constant factor **2**.
- Watched literals trade worst-case optimality for **zero backtrack-restoration
  cost** (watches stay valid as domains enlarge on backtrack → nothing in
  backtrackable memory, no copying on undo), and convert a coarse constraint-level
  propagator into a free fine-grained value-targeted one. Downside: the watched
  GAC table propagator is **not** worst-case time-optimal (a tuple may be checked
  up to twice per node vs once per branch in trailed GAC-2001).

---

## 2. Comparable solver implementations

- **OR-Tools CP-SAT** and **OscaR**: CT is the **default** table propagator
  (verified via finding [0] / CT adoption).
- **Choco**: adopts CT (named alongside OR-Tools/OscaR in the corroborating
  sources).
- **MiniCP** (canonical trail-based teaching solver, finding [6]): a `Trail`
  records old values (push at a node, `pop().restore()` to undo) so reversible
  objects (`StateInt`, `StateSparseSet`, domains) roll back automatically;
  constraint implementors "**should only focus on incremental aspects down in the
  search tree**." Its incremental table-style propagator (`Element2D`) maintains
  **per-value support counters** + advancing boundary pointers, decrementing on
  removal and pruning when a count hits zero — relying on the trail to restore
  counters/pointers on backtrack.
- **Rust ecosystem:** the evidence base surfaced **no mature, usable Rust CP/SAT
  crate implementing CT / STR / reversible sparse bit-sets**. → **Plan for a port,
  not a dependency.**

**Coverage gap (honest):** the research did *not* find implementation-level docs on
whether Choco/Gecode/CP-SAT use trailing vs recomputation internally for table
propagation — only that CT is "the default." Confirming the exact reversible data
structures needs a source-code pass. The Rust "no prior art" is an *absence of
evidence*; a direct crates.io/GitHub survey would harden it.

---

## 3. Trail vs incremental speedup — settled

The literature treats the **trail/undo-log as necessary substrate, not the source
of speedup** (finding [6]). The contrary claim — that watched/non-trailing wins
*because trail maintenance is the bottleneck* — was **REFUTED** in verification.
Translation for us: the trail mechanism is a prerequisite that, by itself, gives
~0 speedup (consistent with our profile: copying <1%). The real win is the
incremental bit-set algorithm that sits on it.

---

## 4. Feasibility in this codebase (our own analysis)

Our data structures are a near-perfect fit for CT:

| CT concept | This crate |
|---|---|
| tuple table | `TensorData.support: Vec<u32>` — each `u32` is one satisfying config over `var_axes` (arity ≤ 32) |
| `supports[var][value]` masks | precompute per `(axis, value)`; **boolean ⇒ 2 masks/axis**; **shareable across tensors that dedup to the same `unique_tensors` entry** (memory win — dedup already exists) |
| reversible `currTable` | **per `BoolTensor`**, a bit-set over its support rows |
| value removal → AND | replace `scan_supports` with `currTable &= supports[i][val]`, then prune any unfixed var/value whose `currTable & supports[i][v]` is empty |

**Sizes (this workload):** arity ≤ 32; VE `budget_B=10` ⇒ support ≤ 2¹⁰ = 1024
rows ⇒ `currTable` ≤ **16 u64 words**; small gates (arity 2–3) ⇒ **1 word**. So
the word-parallel AND is tiny and CT's 64× amortization applies directly.

**The one real integration gap:** the planned `Trail` only undoes **domains**
(`(var, old DomainMask)`). CT also needs to undo **bit-set state** (`currTable`).
Options:
1. Generalize the trail substrate to a **reversible sparse bit-set** (CT's own
   sparse-set + trailed limit + save-word-on-first-write-per-level), or
2. Keep the domain trail and add a **second** reversible structure for `currTable`.

This is the key design decision before implementation — and it's exactly what the
trail-mechanism spec must account for if the trail is to be the substrate for the
55→82% win.

---

## 5. Recommended path

1. **Land the trail first** (behavior-preserving substrate) — but understand it
   buys ~0 speedup alone; its purpose is to enable step 2.
2. **Implement CT** on top: per-tensor reversible `currTable`, precomputed shared
   `supports[axis][value]` masks, residues, reset. This is the 82% win.
3. **Stopgap (optional, low-risk, no architecture change):** SIMD-vectorize
   `scan_supports` with `wide::u32x8` (crate already depends on `wide`) for an
   immediate ~2–4× on the scan while CT is built.

**Decide CT vs STR3 with one cheap experiment** (open question from the research):
profile whether `scan_supports` re-examines **mostly-live rows** (→ CT/STR3 win
big) or **mostly-shrunk tables** (→ STR2's simpler scheme suffices). Our A3
cross-check (fewer-but-wider VE tensors beat many-small no-VE tensors) hints the
hotspot is *churn over live rows* — favoring CT/STR3 — but measuring the live-row
fraction would settle it.

---

## Sources (primary)

- Compact-Table — CP 2016: https://arxiv.org/abs/1604.06641 ·
  https://link.springer.com/chapter/10.1007/978-3-319-44953-1_14
- STR2 — Constraints 2011: http://cse.unl.edu/~choueiry/Documents/STR2-Lecoutre-Long.pdf ·
  https://link.springer.com/article/10.1007/s10601-011-9107-6
- STR3 — AIJ 2015: https://www.sciencedirect.com/science/article/pii/S000437021400143X
- Watched literals — CP 2006: https://sites.cs.st-andrews.ac.uk/people/ipg1/papers/GentJeffersonMiguelCP06.pdf ·
  JAIR 2013: https://www.jair.org/index.php/jair/article/download/10839/25868/20212
- MiniCP — A4CP 2017 slides: https://school.a4cp.org/summer2017/slidedecks/MiniCP.pdf
