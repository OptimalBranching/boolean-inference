# Sparse Boolean Tensor Contraction: Beyond the Naive Binary Join

**Date:** 2026-07-02
**Context:** We are extracting a single `contract` primitive shared by `canonicalize`
(bounded-width variable elimination) and `contract_region` (per-node region
contraction). Both are currently implemented as pairwise binary hash joins
(`contract.rs::join` / `join_all`). This note surveys how to implement the
underlying sparse-relation contraction properly, and — critically — decides
*which* techniques actually pay off for **our** case: boolean relations, small
arity, small (binary) domains, stored as sparse support (lists of satisfying
tuples).

---

## 1. What "contract" is, and why three code paths are the same operation

Eliminating a variable `v` means: **join all relations mentioning `v`, then
project `v` out.** This is one step of tensor-network contraction — equivalently
Dechter bucket elimination, equivalently the inner loop of FAQ's *InsideOut*.
Doing it for a sequence of variables contracts a portion of the network.

Three places in the codebase are the *same* join-then-project primitive, deployed
differently:

| Call site | `rels` (inputs) | `keep` (projection target) | Granularity | Effect |
|---|---|---|---|---|
| `canonicalize` VE step | tensors incident to `v` | neighbours(`v`) \ {`v`} | global, once | rewrites the network |
| `contract_region` | region tensors, sliced by `doms` | unfixed region vars | per branching node | builds a branching table |
| `feasible_configs` | region configs (already contracted) | unfixed vars | per branching node | drives propagation probes |

`feasible_configs` is already a **hand-rolled variable-at-a-time descent** (a
prefix-sharing trie DFS). The other two use binary hash joins. The goal is a
single primitive:

```
contract(rels, keep) = join all rels on shared vars, project onto keep
```

so that the code reflects the fact that these are one operation.

In FAQ / PANDA terms: `canonicalize` = *contract/eliminate the low-width tractable
part*; `contract_region` + optimal branching = *branch on the hard core, using
local contraction to build branching tables*. Same primitive, two roles in the
"eliminate vs branch" strategy.

---

## 2. Algorithm landscape (beyond naive)

1. **Binary hash join (current).** Fold relations pairwise. Intermediate results
   can grow to `2^{width of the intermediate}` even when the final output is tiny.
   Blows up on cyclic joins.
2. **NPRR (Ngo–Porat–Ré–Rudra, PODS 2012).** First worst-case optimal join
   (WCOJ); recursive; achieves the AGM bound `O(N^{ρ*})`, where `ρ*` is the
   fractional edge-cover number. Foundational; rarely implemented directly now.
3. **Generic Join (Ngo–Ré–Rudra, "Skew Strikes Back", 2014).** The clean WCOJ:
   fix a global variable order; for each variable `x`, **intersect** the
   `x`-projections of all relations containing `x`; for each surviving value,
   recurse. Also achieves the AGM bound. This is the form worth implementing.
4. **Leapfrog Triejoin (Veldhuizen, ICDT 2014).** Generic join + sorted tries +
   *leapfrog / galloping* seek to intersect sorted unary extensions. The gallop is
   an optimization for **large domains with cardinality skew**.
5. **EmptyHeaded / LevelHeaded (Aberger et al.).** Engineered WCOJ. Each trie level
   picks between a **sorted-array** and a **bitset** layout by density, and does
   set intersection with SIMD. Their headline finding: *unoptimized set
   intersection is ~95% of WCOJ runtime* — the intersection implementation is the
   performance core, not the algorithm skeleton.
6. **Free Join (Wang–Willsey–Suciu, SIGMOD 2023).** Unifies binary join and WCOJ
   into one plan and one data structure, because **pure WCOJ loses to binary join
   on acyclic queries** found in practice. The modern "best of both" answer.

---

## 3. The decisive analysis: what WCOJ actually buys for **boolean** relations

The classic WCOJ premise: relations of size `N` over a **large** domain; binary
plans can materialize `N^{integer-cover}` intermediates while the output is only
`N^{ρ*}` (fractional cover); WCOJ hits `N^{ρ*}`.

For **boolean** relations this premise partly dissolves, and being honest about it
is what keeps us from cargo-culting LFTJ:

- A relation of arity `k` has at most `2^k` rows, so `N ≤ 2^{max arity}` is a
  *constant*, not a growing parameter.
- A binary-join intermediate over `w` variables has at most `2^w` rows, so the
  blow-up is bounded by **`2^{width}`** — the tensor-contraction / treewidth cost
  model. This is exactly the quantity `budget_b` bounds in `canonicalize`.

So for boolean, **WCOJ's benefit is not a smaller exponent in `N`; it is
output-sensitivity.** Generic join extends only partial tuples consistent with
*every* relation, so intermediates never exceed the AGM bound of the sub-query's
output — never `2^{width}`. WCOJ wins precisely when:

> **the region is cyclic AND the constraints are tight (feasible set sparse),**

because then `2^{width}` ≫ (actual feasible completions), and generic join hugs
the output while the binary join (and dense contraction) pays `2^{width}`.

| Workload | Use | Why |
|---|---|---|
| **factoring** (arity 2–3, near-tree, small regions) | binary ≈ WCOJ | width small, no blow-up to avoid → **wash** (matches all prior measurements) |
| **QWH / all-different** (Latin-square = tight cyclic cliques) | **WCOJ wins clearly** | intermediate `2^{width}` huge, feasible set sparse → output-sensitivity bites |

**Conclusion:** generic join is a latent, AGM-backed lever *for the structured-CSP
direction*, and a wash for factoring. This is consistent with the whole
optimization arc — the factoring hot path is at its floor; the win is reserved for
the cyclic, tight regions that structured CSP (QWH) produces.

---

## 4. Which variant to implement: generic join (radix trie), **not** full LFTJ

- **Skip leapfrog / galloping.** The boolean domain is `{0,1}`: each variable has
  at most two values, so per-variable "intersection" is trivially "is `0` allowed
  by all relations? is `1`?". Galloping — an optimization for large skewed domains
  — buys nothing here.
- **Implement generic join as a radix-trie descent on the packed bitmask.** Rows
  are `u64` bitmasks. Sort each relation's rows by the relevant bits in the global
  variable order; at variable `x`, partition each relation's live row-slice by the
  `x` bit (a binary split of a sorted sub-range), intersect the allowed values,
  recurse. This is the **Worst-Case Optimal Radix Triejoin** idea (arXiv:1912.12747),
  and it is *structurally identical to the existing `feasible_configs::descend`*.
- **Carry over the one EmptyHeaded lesson that transfers:** representation by
  density and a well-written intersection. For us relations are small → sorted
  `u64` arrays; a level that happens to be dense degenerates to a 2-bit mask. The
  intersection is cheap either way; there is no SIMD problem at our sizes.
- **Free Join is likely overkill for now.** Its value is unifying binary and WCOJ
  across a whole acyclic/cyclic query plan. Our contractions are small; a generic
  join with a reasonable variable order already degenerates to binary-join cost on
  the acyclic/small cases. Keep it in mind if region contraction ever becomes a
  measured bottleneck on mixed workloads.

---

## 5. The payoff: one primitive absorbs three code paths

`feasible_configs` is *already* a variable-at-a-time generic join. So implementing
`contract` as a radix generic join does more than de-duplicate `canonicalize` and
`contract_region` — it reveals all three are the same primitive:

```
contract(rels, keep) = fix a variable order; radix-trie descent;
                       per-variable intersect allowed values; project leaf onto keep
  ├─ canonicalize VE step:  rels = incident tensors,          keep = neighbours(v) \ {v}
  ├─ contract_region:       rels = region tensors (doms-sliced), keep = unfixed region vars
  └─ feasible_configs:      the same descent, emitting propagation probes as it goes
```

That is the end state of the "make the code admit these are one operation"
cleanup.

---

## 6. Implementation sketch

```rust
// contract.rs — the single contraction primitive (boolean radix generic join).
// Each Relation.rows is sorted by the relevant bits of the global variable order;
// `keep` is the ascending set of output variables.
pub fn contract(rels: &[Relation], keep: &[usize]) -> Relation {
    // Global order = ascending union of all rels' vars (ascending is fine).
    // DFS(prefix): for each rel containing the current variable x, keep the
    //   slice of its rows consistent with `prefix` (a sorted sub-range).
    //   allowed_x = ∩_{rel ∋ x} { v ∈ {0,1} : some row in rel's slice has x = v }
    //   for v in allowed_x: narrow every containing rel's slice to rows with x = v,
    //     recurse.
    //   Leaf: project the assigned prefix onto `keep`, emit one row.
    // Key: only descend into allowed_x branches → dead branches pruned at once →
    //   intermediates bounded by AGM/output, never by 2^{width}.
}
```

"Narrow a sorted slice to `x = v`" is exactly the move in the current
`feasible_configs::descend`; extract and reuse it.

Notes:
- Variable order affects constants (not worst-case optimality). Sorted-ascending is
  a fine default; a min-degree-style order can help but is not required.
- `canonicalize` keeps its own VE scheduling (min-fill heap, `budget_b`, protected
  vars) — that is elimination-order logic, separate from the contraction kernel.
  Only the kernel is shared.
- This also removes the sparse→dense→sparse round-trip currently in
  `canonicalize`'s finalize: with a support-based `setup_problem` entry, surviving
  relations pass through without ever materializing a `2^{arity}` dense table.

---

## 7. Recommendation

1. Extract `contract(rels, keep)` as a boolean radix generic join in `contract.rs`,
   and a `Relation::project` / slice-narrow helper factored out of
   `feasible_configs::descend`.
2. Rewrite `contract_region` and `canonicalize`'s VE step on top of it; add a
   support-based `setup_problem` entry so `canonicalize` is end-to-end sparse
   (drops the dense round-trip; removes the vestigial `LiveTensor` wrapper).
3. Treat `feasible_configs` as the third caller (fuse when convenient).
4. **Measure, don't assume:** expect a wash on factoring (arity 2–3), a real win
   only on cyclic/tight structured-CSP regions. Gate the actual QWH-facing work on
   a measured blow-up ratio (max intermediate rows / final output rows) once such
   instances exist.

---

## References

- Atserias, Grohe, Marx. *Size Bounds and Query Plans for Relational Joins* (AGM bound), FOCS 2008.
- Ngo, Porat, Ré, Rudra. *Worst-case Optimal Join Algorithms* (NPRR), PODS 2012.
- Ngo, Ré, Rudra. *Skew Strikes Back: New Developments in the Theory of Join Algorithms* (Generic Join), 2014. arXiv:1310.3314.
- Veldhuizen. *Leapfrog Triejoin: A Simple, Worst-Case Optimal Join Algorithm*, ICDT 2014.
- Aberger et al. *EmptyHeaded: A Relational Engine for Graph Processing*, SIGMOD 2016. arXiv:1503.02368.
- Aberger et al. *LevelHeaded: Making Worst-Case Optimal Joins Work in the Common Case*. arXiv:1708.07859.
- *Worst-Case Optimal Radix Triejoin*. arXiv:1912.12747.
- Wang, Willsey, Suciu. *Free Join: Unifying Worst-Case Optimal and Traditional Joins*, SIGMOD 2023. arXiv:2301.10841.
- Abo Khamis, Ngo, Rudra. *FAQ: Questions Asked Frequently* (InsideOut / variable elimination), PODS 2016. arXiv:1504.04044.
