#!/usr/bin/env python3
"""Ising partition-function instances emitted NATIVELY as wcn-1 JSON — the
partition-function-on-native-structure family (survey §3.1 / §9 P4, replicating
the TensorOrder protocol of 2D lattices + random regular graphs).

Model: spins s_i in {-1,+1} on a graph; couplings J_ij in {+1,-1}
(--disorder spinglass) or all +1 (--disorder ferro). Boolean var x_i = (s_i+1)/2.
Each edge carries the factor
    phi_ij = w        if s_i == s_j and J_ij = +1   (agree, ferro)
           = 1/w      otherwise-mismatched sign of J_ij * s_i * s_j
i.e. phi_ij = w^(J_ij * s_i * s_j) with a RATIONAL base w (default 2). This is
the standard exact-rational surrogate for e^(beta*J): Z is computed in the exact
rational semiring with no transcendental rounding, and any real-temperature Z
corresponds to w = e^beta by substitution. Edge tensors are per-row weighted
`rows` — exactly the WRelation expressiveness wcn-1 exists for.

Ground truth: for n_spins <= --brute-max (default 20) the generator brute-forces
Z with exact Fractions and stores it in meta.known_Z; bank_check-style tools (or
a one-line diff against `count`) verify the engine bit-exactly. Larger sizes
carry no embedded truth — the planar Pfaffian/Kac-Ward reference is noted as
future work (it would inflate scope here).

Reproducibility: rng = Random(seed * 1000003 + n_spins), so an instance is fully
determined by (topology, size, disorder, w, seed).

Usage:
  gen_ising.py --topology grid2d --L 4 --seed 1 --out ising_g4s1.json
  gen_ising.py --topology random3reg --v 12 --disorder spinglass --seed 2 --out i.json
"""
import argparse
import json
import random
import sys
from fractions import Fraction


def grid2d_edges(L):
    """Open-boundary LxL square lattice; vertex id = r*L + c."""
    edges = []
    for r in range(L):
        for c in range(L):
            v = r * L + c
            if c + 1 < L:
                edges.append((v, v + 1))
            if r + 1 < L:
                edges.append((v, v + L))
    return L * L, edges


def random_regular_edges(v, d, rng):
    """Random d-regular simple graph via the configuration model (retry until
    simple). v*d must be even."""
    if (v * d) % 2:
        raise SystemExit(f"v*d must be even (v={v}, d={d})")
    while True:
        stubs = [i for i in range(v) for _ in range(d)]
        rng.shuffle(stubs)
        pairs = [(stubs[i], stubs[i + 1]) for i in range(0, len(stubs), 2)]
        seen = set()
        ok = True
        for a, b in pairs:
            if a == b or (min(a, b), max(a, b)) in seen:
                ok = False
                break
            seen.add((min(a, b), max(a, b)))
        if ok:
            return v, sorted(seen)


def parse_rational(s):
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    return Fraction(s)


def frac_str(f):
    return str(f.numerator) if f.denominator == 1 else \
        f"{f.numerator}/{f.denominator}"


def brute_force_z(n, edges, couplings, w):
    """Exact Z = sum over 2^n spin configs of prod_e w^(J_e * s_i * s_j)."""
    winv = 1 / w
    z = Fraction(0)
    for cfg in range(1 << n):
        p = Fraction(1)
        for (a, b), j in zip(edges, couplings):
            agree = ((cfg >> a) & 1) == ((cfg >> b) & 1)
            # J*s_a*s_b = +J on agreement, -J on disagreement.
            p *= (w if j > 0 else winv) if agree else (winv if j > 0 else w)
        z += p
    return z


def build_instance(n, edges, couplings, w, meta):
    tensors = []
    for (a, b), j in zip(edges, couplings):
        agree = w if j > 0 else 1 / w
        disagree = 1 / agree
        # config bit0 = a, bit1 = b: 0b00/0b11 agree; 0b01/0b10 disagree.
        rows = [[0, frac_str(agree)], [1, frac_str(disagree)],
                [2, frac_str(disagree)], [3, frac_str(agree)]]
        tensors.append({"vars": [a, b], "rows": rows})
    return {"format": "wcn-1", "n_vars": n, "tensors": tensors, "meta": meta}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", choices=["grid2d", "random3reg"],
                    required=True)
    ap.add_argument("--L", type=int, help="grid side (grid2d)")
    ap.add_argument("--v", type=int, help="vertices (random3reg; v*3 even)")
    ap.add_argument("--disorder", choices=["ferro", "spinglass"],
                    default="spinglass")
    ap.add_argument("--w", default="2",
                    help="rational agree-weight base (default 2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--brute-max", type=int, default=20,
                    help="embed exact brute-force Z when n_spins <= this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = parse_rational(args.w)
    if w <= 0:
        raise SystemExit("w must be a positive rational")

    if args.topology == "grid2d":
        if not args.L:
            raise SystemExit("--L required for grid2d")
        n, edges = grid2d_edges(args.L)
        size_tag = f"L{args.L}"
        rng = random.Random(args.seed * 1000003 + n)
    else:
        if not args.v:
            raise SystemExit("--v required for random3reg")
        rng = random.Random(args.seed * 1000003 + args.v)
        n, edges = random_regular_edges(args.v, 3, rng)
        size_tag = f"v{args.v}"

    if args.disorder == "ferro":
        couplings = [1] * len(edges)
    else:
        couplings = [rng.choice((1, -1)) for _ in edges]

    meta = {
        "family": "ising",
        "topology": args.topology,
        "size": size_tag,
        "disorder": args.disorder,
        "w": frac_str(w),
        "seed": args.seed,
        "n_spins": n,
        "n_edges": len(edges),
    }
    if n <= args.brute_max:
        meta["known_Z"] = frac_str(brute_force_z(n, edges, couplings, w))

    inst = build_instance(n, edges, couplings, w, meta)
    with open(args.out, "w") as f:
        json.dump(inst, f, separators=(",", ":"))
        f.write("\n")
    known = meta.get("known_Z", "-")
    print(f"ising {args.topology} {size_tag} {args.disorder} seed={args.seed}: "
          f"n={n} edges={len(edges)} known_Z={'yes' if known != '-' else 'no'} "
          f"-> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
