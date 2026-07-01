# Prefix-Sharing Look-Ahead — Correctness Verdict & Implementation Scheme

**Date:** 2026-07-02
**Method:** deep-research (5 angles, 24/25 claims confirmed; correctness core all 3-0).
**Question:** Is trie-structured prefix-sharing of the region-feasibility probes
(fix one region var, propagate, descend, share the propagated prefix across
sibling probes via trail mark/restore) behavior-identical to today's one-shot
per-config probe — and is it the right lever?

## Bottom line

**Yes — provably behavior-identical, and it is the right lever for structured
instances. Build it.**

## (1) Correctness — confident YES

Fixing region vars one at a time and propagating deltas down a trie reaches the
**identical no-wipeout feasibility verdict** per full configuration as the
one-shot probe.

- **Why:** CT/GAC propagators are **monotonic + inflationary** contracting
  domain-reduction operators. By Apt's chaotic-iteration theorem and
  Schulte–Stuckey–Tack propagation-solver confluence, a finite set of such
  operators over a finite domain lattice has a **unique common fixpoint that
  every fair schedule reaches**. Interleaving variable-fixings with propagators
  is just one fair schedule → same fixpoint → same wipeout verdict. Propagation
  order is "a pure efficiency knob, not a correctness one."
- **Idempotence is NOT required.** Non-idempotent propagators (watched-literal
  style, or CT's incremental-vs-reset internal choice) still converge to the
  same unique fixpoint. Idempotence only enables a re-queue optimization.
- **Early pruning is sound.** A contradiction at an internal prefix node ⇒
  wipeout for **every** extension (wipeout is monotone under further
  restriction). No leaf-feasible config can be mislabeled by subtree pruning.
- **Only failure mode:** a **non-monotonic** propagator makes propagation
  non-confluent (reached fixpoint becomes order-dependent). Standard CT is
  monotonic, so this doesn't arise — but assert it.

Sources: Apt *The Essence of Constraint Propagation* (arXiv:cs/9811024,
cs/9909009, cs/0012010); Tack 2009 dissertation (Gecode theory); Schulte–Stuckey
TOPLAS 2008; Schulte–Tack CP 2009 (weak monotonicity).

## (2) Mechanism — an established technique

march_eq's **tree-based look-ahead** is this exact share-and-restore scheme:
propagate shared prefix once → propagate child literal → backtrack child →
propagate sibling → unwind the shared parent last; recurse. DFS with per-level
trail marks. Handbook of Satisfiability Ch.5 §5.5.3; march_eq §8, Fig 5.8.

## (3) Trail / restore mechanics

- Per-level trail marks; propagate the shared prefix once at the parent,
  descend by fixing one more var + propagating its delta from the parent's
  already-propagated state, `restore_to` the parent mark before the next
  sibling, unwind the parent only when all siblings are done.
- CT's save-on-first-write reversible bit-set gives O(delta) restore.
- **Clear/re-seed the propagation worklist per sibling** so a sibling does not
  inherit the previous sibling's pending events (the one concrete leak to guard).
- Watch: restore cost dominating (mitigate via minimal-change enumeration),
  residue/lastSize invalidation.

## (4) Magnitude — honest

march_eq measured **72–78% time reduction on longmult multipliers**
(longmult8/10/12) but **−17–20% on uniform random**. Our factoring/CircuitSAT is
on the favorable (structured) side. Numbers are indicative, not transferable
(binary BCP substrate, not GAC-over-config-set).

## (5) Ordering (weakest-verified, 2-1)

- Static structural order (MINCE-style connectivity-clustered / recursive
  min-cut bisection) to maximize shared-prefix reuse — favors a cheap
  fixed-order trie (sort configs once) over dynamic per-node order.
- Fail-first refinement for early contradiction.
- Gray-code / minimal-change enumeration so consecutive configs differ in one
  var → amortize restore cost.

## The one open alternative — flagged for a follow-up experiment

For a **known** enumerated config set, computing the GAC-feasible configs
**directly** — project the constraint network onto the region and intersect as a
table/MDD/ZDD, or (since we are a tensor network) **contract the region once** —
is a plausible alternative to N probes. A fetched source notes SAT feasibility =
a single tensor-network contraction (squared 2-norm = #satisfying-assignments).
No survived source benchmarks project-and-intersect head-to-head against
trie-sharing, so it is a **genuinely open experiment**, not a proven win. Keep
trie-sharing as the safe primary; measure project-and-intersect as a follow-up.

## What this means for the design

Green-light trie prefix-sharing of the `table.rs` step-1 feasibility loop.
Fold in: static MINCE-ish region-var order, per-sibling worklist reset,
monotonicity assert, minimal-change enumeration, and the 19761/45322 golden
node-identity test as the behavior-preservation guard.

## Sources (primary)

- Apt, *The Essence of Constraint Propagation*: https://arxiv.org/pdf/cs/9811024
- Tack, *Constraint Propagation — Models, Techniques, Implementation* (2009): https://www.ps.uni-saarland.de/Publications/documents/Tack_2009_ConstraintPropagation.pdf
- Schulte & Stuckey, TOPLAS 2008: https://people.eng.unimelb.edu.au/pstuckey/papers/toplas08.pdf
- Schulte & Tack, *Weakly Monotonic Propagators*, CP 2009: https://chschulte.github.io/papers/schultetack-cp-2009.html
- Heule & van Maaren, look-ahead chapter: https://www.cs.cmu.edu/~mheule/publications/p01c05_lah.pdf
- march_eq: http://www.cs.cmu.edu/~mheule/publications/35420345.pdf
- Compact-Table (CP 2016): https://arxiv.org/abs/1604.06641
- MINCE: https://www.researchgate.net/publication/220349198_MINCE_A_static_global_variable-ordering_heuristic_for_SAT_search_and_BDD_manipulation
