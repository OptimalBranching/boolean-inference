# Structured-CSP Decision Benchmarks — Where This Solver Can Argue an Edge

**Date:** 2026-07-02
**Method:** deep-research (5 angles, 23 verified claims, 2 refuted).
**Question:** Which structured CSP *decision* families should we add to argue an
advantage for the boolean optimal-branching / region-contraction solver against
CSP SOTA (OR-Tools CP-SAT, ACE, Choco, Gecode) — and how to compare fairly.

## Bottom line — read this first (the honest caveat)

**The "beats CSP SOTA" thesis is an EXTRAPOLATION, not yet demonstrated.** All
primary optimal-branching evidence (arXiv:2412.07685, O(1.0441^n) average on
3-regular graphs; OptimalBranching.jl) is for **Maximum Independent Set
OPTIMIZATION**, not CSP decision, and contains **no head-to-head vs any
CP/SAT solver**. The path forward is to *measure* on the families below, and to
make **defensible** claims (node count / branching factor; "solves X in time T
that CP-SAT cannot") rather than "faster than CP-SAT everywhere."

Also: the classic "SAT-encoding beats native CP" result (Sugar #1 in all global
categories, CSC'2009) is **pre-2010** — modern OR-Tools CP-SAT has advanced
enormously since, so that result does **not** transfer as current-SOTA evidence.

## Ranked shortlist

### Tier 1 — realistic wins

**1. Quasigroup / Latin-square completion (QCP / QWH)** — top pick.
- **Why it fits region-contraction:** dense, regular, local structure — the
  constraint network is 2N interconnected size-N cliques (each row/column
  all-different = a clique) with small-world topology (max path length 2). This
  is exactly the clique-dense local structure region-contraction rewards.
- **Decision problem:** yes — equivalent to precoloring-extension / list-coloring
  of the rook's graph (Gomes & Shmoys, "Completing Quasigroups or Latin Squares:
  A Structured Graph Coloring Problem").
- **Hardness frontier (tunable):** cost peaks ~42% pre-assignment; order-20 hard
  band ~60–65% filling (below 50% propagation deduces nothing; above 70%
  propagation alone solves it). Use **QWH** ("quasigroup with holes") for
  guaranteed-satisfiable, balanced, harder-in-transition instances.
- **Boolean encoding:** all-different is best via a **dual** representation —
  order-encoding for integers + channeled **direct/one-hot** for all-different
  (BEE). On QCP (25×25, 264 holes) dual is ~38× faster / ~20× fewer clauses than
  order-encoding alone. Open: does that encoding preserve the region reward for
  *our* solver specifically? — must verify.
- **Baseline to beat:** OR-Tools CP-SAT / ACE on the native XCSP3 model.

**2. Structured graph coloring** — strong second.
- **Why:** clique/neighborhood-local structure; sharp colorability frontier
  tunable by allowed-colour count.
- **Encoding:** clean direct/one-hot (one var per (vertex,colour), per-edge
  conflict clauses, exactly-one-per-vertex). The **partial-ordering (POP) SAT
  encoding** solved the most DIMACS instances (91/134) — beat all evaluated ILP
  formulations (Faber–Jabrayilov–Mutzel, SAT 2024). (Per-family "beats SOTA on
  wap0-/queen-" was **refuted** — cite only the aggregate.)
- **Instances:** SATLIB DIMACS GCP — g125.18 / **g125.17** / g250.15 / g250.29
  (up to 7250 vars / 454k clauses). g125.17 (17 colours) ≫ harder than g125.18
  (18) on the same graph — pick colour counts near the chromatic number.
- **Also:** register-allocation graphs, Mycielski (triangle-free, high χ).

### Tier 2 — plausible, need assessment
- All-different-dominated combinatorial problems generally (Sudoku-family,
  quasigroup variants).
- Combinatorial designs (Steiner triple systems, covering/packing arrays, BIBD),
  Costas arrays, Langford, social golfer, pentomino/exact-cover — **none were
  covered by the verified claims**; booleanization-preserves-locality is unknown.

### Long shots — avoid as headline claims
- Random 3-SAT / random extensional CSP (sparse, no region reward; CDCL crushes).
- Global-arithmetic-dominated families.

## Encoding is load-bearing (theory-backed)

The **order encoding** provably maps tractable CSP classes (max-closed,
connected-row-convex) to tractable SAT classes; **sparse/direct and log
encodings do not** (Petke & Jeavons, SAT 2011). But for **all-different**
specifically, direct/one-hot is superior (hence BEE's dual scheme). Takeaway:
choose the encoding to **preserve exploitable local structure**, and always
report which encoding was used.

## Comparison protocol (defensible)

- **Metrics:** report **branching factor + search-tree node count** (our proven
  edge) alongside wall-clock and PAR2. Strongest defensible claims: "fewer nodes
  / smaller branching factor" and "solves instance X within time T that CP-SAT
  cannot" — NOT "faster than CP-SAT everywhere."
- **Avoid the model-vs-solver confound:** run CP-SAT / ACE / Choco on the
  **native XCSP3 / MiniZinc model** (their intended representation), not on our
  booleanization; report our encoding explicitly.
- **Discipline:** full established series (no cherry-picking), fixed time limit,
  hardware/tuning parity.

## Benchmark sources

- **XCSP3 instances repository** (xcsp.org) — >23,000 instances, XCSP3 XML,
  native CSP decision models (also the source for the fair CP-SAT baseline).
- **SATLIB DIMACS GCP** and DIMACS graph-coloring — SAT-encoded coloring, CNF.
- **QCP / QWH generators** (Barták 2004; Gomes & Shmoys 2002) — parameterized by
  order N and filling/hole ratio near the ~42%-pre-assignment frontier.

## Open questions (the experiments to actually run)

1. Does boolean-inference beat/compete with **modern** OR-Tools CP-SAT on any
   concrete QCP/QWH or structured-coloring instance? Unestablished — measure it.
2. Which boolean encoding best preserves QCP's clique-dense local structure for a
   region-contraction solver (dual order+direct vs one-hot vs log-support)?
3. Exact winnable-instance parameters vs **modern** CP-SAT (the phase-transition
   data is solver-agnostic and old).
4. Do Tier-2 families (Steiner, covering arrays, Costas, Langford, social golfer,
   exact-cover) have locality-preserving booleanizations? Unassessed.

## Sources (primary)

- Optimal branching: https://arxiv.org/abs/2412.07685 · OptimalBranching.jl
- Gomes & Shmoys, QCP as structured graph coloring.
- Barták, quasigroup phase-transition (ITI 2004/223).
- Faber, Jabrayilov, Mutzel, POP encoding for graph coloring (SAT 2024, LIPIcs 305).
- Petke & Jeavons, order encoding & tractability (SAT 2011).
- Metodi & Codish, BEE / dual all-different (arXiv:1206.3883).
- Tamura et al., Sugar / order encoding (CSC'2009 — pre-2010, contextual only).
- Instances: https://www.xcsp.org · SATLIB DIMACS GCP.
