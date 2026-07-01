# Why CT Gave Only 5% — and Where the Large Win Is

**Date:** 2026-07-01
**Trigger:** Replacing the 82%-CPU linear GAC support-scan with a validated incremental
Compact-Table propagator (reversible sparse bit-set + delta-tracking) yielded only ~5%
wall-clock on the realistic config — not the large speedup expected from eliminating an
"82% hotspot". This looked wrong. It was.
**Method:** deep-research (5 angles, all 8 findings 3-0 adversarially verified) +
a local A/B diagnostic on the 22×22 factoring instance.

## Bottom line

**CT was the wrong axis.** Per-propagation speed is a *linear* lever on a fraction of
runtime; for a heavy-lookahead branch-and-reduce solver the dominant cost is the **NUMBER
of propagation probes**, not their unit cost. The solver runs
`lookahead × candidates × region-configs × nodes` propagations — CT made one class ~30%
cheaper per call and left the count (and the 28% measurement path) untouched. Amdahl did
the rest.

**The large, behavior-preserving win is tree-based look-ahead**: the sibling probes all
fork from the same base state and share implications, so propagate each shared implication
**once** instead of re-propagating per sibling. march_eq measured **3.7×–4.5×** on exactly
the multiplier/factoring CircuitSAT family (`longmult8/10/12`), with **search-tree size
unchanged** — the same behavior-preserving, multiplicative character CT failed to deliver,
but on the correct axis.

**And CT was not wasted:** the reversible sparse bit-set + trail + delta-tracking we built
is precisely the substrate needed to reuse propagation incrementally across sibling probes.
It's the foundation for the real win, not a dead end.

## Local diagnostic (this instance)

| selector | branching nodes | time | µs/node |
|---|---|---|---|
| difflook (heavy lookahead) | 19,761 | 8.82s | ~446 |
| most (no lookahead) | 134,298 (6.8×) | 13.82s | ~103 |

Heavy lookahead is **4.3× costlier per node but 6.8× fewer nodes → net faster**. So the
lookahead earns its keep; it is not waste. But `findbest` (the per-node branching decision)
is **95% of runtime**, dominated by many same-base probes — exactly the count term.

## Findings (all 3-0 verified)

1. **The lever is probe COUNT, not per-call cost.** Look-ahead solvers pre-select only a
   subset of variables to probe because probing all is "very costly"; the Handbook names
   unit propagation inside lookahead "the most costly aspect," and its three standard
   remedies all reduce *redundant propagations* (count), not per-call speed. This is why a
   faster CT propagator barely moved wall-clock.
   *(Heule/Franco/van Maaren — march_dl, Handbook of Satisfiability ch. lookahead.)*

2. **Tree-based look-ahead = highest-leverage, directly applicable, behavior-preserving.**
   march_eq Table 3: `longmult8/10/12` = **3.7×/4.5×/4.1×**, `pyhala` factoring ~1.6×, tree
   size essentially unchanged (longmult8 7918→8149). Mechanism: "sharing trees" propagate
   each shared implication once across the sibling probes. These multiplier/factoring
   instances get **by far** the largest speedups (random/crafted only get 3–26% or worse).
   Maps directly onto our difflookahead (both polarities × ~16 candidates) and the
   region-feasibility probes, which all fork from a shared base.
   *(march_eq, Heule et al.)*

3. **Trigger the most expensive (double-)lookahead lazily.** Second-level lookahead
   "should not be called after every look-ahead"; an adaptive trigger caused a **~13×**
   runtime swing on `ezfact` factoring instances by call frequency alone.
   *(march_dl.)*

4. **Cheaper surrogate branch-measure for the ~28% measurement path.** The costly
   clause-weighted measure (`evalcls`) can be replaced by `evalvar` (just count variables
   assigned during lookahead) — cheaper, needs no eager structures, and the cube-and-conquer
   authors found it **more** effective on industrial/structured instances. A better target
   than making the exact measure's propagation faster.
   *(Heule/Kullmann/Wieringa/Biere — Cube and Conquer.)*

5. **Cube-and-Conquer architecture.** Pay heavy lookahead only at the *top* to partition
   into cubes, then hand reduced subproblems to CDCL; the hybrid beats both pure lookahead
   and pure CDCL. How much to cube is chosen by a runtime-estimate heuristic.
   *(Cube and Conquer; arXiv 2212.02405.)*

6. **Pure heavy lookahead is architecturally mismatched to structured CircuitSAT.**
   Lookahead is strong mainly on random k-SAT/UNSAT; CDCL (learning + watched literals +
   backjumping + restarts) has dominated *structured* instances for two decades.
   *(Handbook lookahead ch.; arXiv 2008.02215.)*

7. **Factoring's true bottleneck is search-space SIZE, not per-step cost.** SAT-based
   factoring scales ~O(2^y) and doesn't match the Number Field Sieve; **no constant-factor
   propagation speedup closes that gap.** Aim for constant-factor wins (like the 3.7–4.5×)
   and structural pruning — not an asymptotic miracle.
   *(arXiv 1910.09592, 1902.01448.)*

8. **Region/candidate probe count is the primary cost driver in this framework.** Branch
   quality comes from a per-region optimization enumerating configurations of a sub-graph of
   tens of vertices — so the number of region/candidate probes is exactly what to attack.
   *(arXiv 2412.07685, optimal-branching; single-source → medium confidence.)*

## Recommended plan (ranked by leverage × directness)

1. **Tree-based / shared look-ahead (the big one).** The difflookahead probes both
   polarities of ~16 candidates and the region-feasibility step probes every enumerated
   config — all forking from the same node base. Instead of `mark → apply → full-propagate →
   read → restore` independently per probe, **share the common propagation**: build the
   sibling probes as a tree off the base so implications entailed by the shared prefix
   propagate once. Our reversible bit-set + trail already supports the incremental
   apply/undo this needs. Expected: multiplicative, behavior-preserving (guard with the
   19761/45322 golden test). **This is the win the 5% CT result was missing.**

2. **Cheaper surrogate measure (`evalvar`) on the 28% `apply_branch` path.** Replace the
   apply-and-fully-propagate-then-clause-measure per candidate with a variable-count
   surrogate; cut or reuse the propagation the measurement does. May change node counts
   (different branch choice) — measure both time and nodes.

3. **Lazy/adaptive triggering** of the most expensive probes (region-feasibility over all
   configs; any double-lookahead) via a cheap predictor.

4. **(Architectural, later)** Cube-and-conquer or lightweight clause learning to cut the
   node count itself — the only lever that changes the *asymptotics*, bounded by the fact
   that factoring is exponential for all known SAT methods.

## Honest caveats

- The 3.7–4.5× is march_eq (a different, binary-implication-tree solver, pre-2010). It is
  strong evidence the *mechanism* helps this benchmark family, but the gain **here must be
  measured** — our probe structure (difflookahead + region-feasibility) differs.
- `evalvar` was better on *structured* instances specifically; it's a good bet for factoring
  but not universal, and it may shift node counts.
- Cube-and-conquer/CDCL superiority is "architecture-relative on structured instances" —
  factoring is empirically hard for CDCL too; it loses to NFS regardless.

## What this means for the CT work already done

CT is **behavior-correct, validated, and the right substrate** — keep it. It is marginally
faster on the realistic config and, more importantly, provides the incremental
apply/propagate/undo machinery that **tree-based look-ahead is built on**. The mistake was
treating per-propagation speed as the goal; the goal is fewer propagations, and CT is the
tool that makes cross-probe sharing cheap.

## Sources (primary)

- march_dl (double lookahead): https://www.cs.cmu.edu/~mheule/publications/march_dl.pdf
- Handbook of Satisfiability, look-ahead chapter: https://www.cs.cmu.edu/~mheule/publications/p01c05_lah.pdf
- march_eq: https://www.researchgate.net/publication/228956352_March_eq_Implementing_efficiency_and_additional_reasoning_into_a_lookahead_sat-solver
- Cube and Conquer: https://www.cs.utexas.edu/~marijn/publications/cube.pdf
- Optimal branching framework: https://arxiv.org/pdf/2412.07685
- SAT factoring complexity: https://arxiv.org/pdf/1910.09592 · https://arxiv.org/pdf/1902.01448
