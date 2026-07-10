#!/usr/bin/env python3
"""Bounded-treewidth extensional (table) CSP — the theoretical home of
structural constraint solving. n boolean vars laid on a line; each constraint's
scope is a random arity-`a` subset drawn from a sliding window of width `w`, so
the constraint graph has treewidth <= w (a path/band decomposition). Each
constraint is an explicit RANDOM table: each of the 2^a tuples is allowed
independently with prob (1 - tightness). Random tables make unit propagation
WEAK (no clause is short), while the bounded width keeps exact inference
(our region contraction / variable elimination) cheap — the regime where a
structural solver should beat CDCL, which sees only the flat encoding.

Native .csp: each constraint = one arity-a tensor (allowed tuples). CNF: each
FORBIDDEN tuple = one clause (the direct/support encoding CDCL must use).

Usage: gen_extcsp.py --n 60 --w 8 --a 3 --m 90 --tight 0.5 --seed 1 \
         --csp o.csp --cnf o.cnf
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--w", type=int, default=8, help="window width (~treewidth bound)")
    ap.add_argument("--a", type=int, default=3, help="constraint arity")
    ap.add_argument("--m", type=int, default=0, help="constraints (default 1.5n)")
    ap.add_argument("--tight", type=float, default=0.5, help="fraction of forbidden tuples")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csp", required=True)
    ap.add_argument("--cnf", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed * 100003 + args.n)
    n, w, a = args.n, args.w, args.a
    m = args.m or round(1.5 * n)

    cons = []  # (scope, allowed set)
    for _ in range(m):
        start = rng.randrange(max(1, n - w + 1))
        window = list(range(start, min(start + w, n)))
        if len(window) < a:
            window = list(range(n))
        scope = sorted(rng.sample(window, a))
        allowed = [c for c in range(1 << a) if rng.random() > args.tight]
        if not allowed:                       # never emit an empty (trivially-UNSAT) table
            allowed = [rng.randrange(1 << a)]
        cons.append((scope, allowed))

    with open(args.csp, "w") as f:
        f.write(f"{n}\n")
        for scope, allowed in cons:
            f.write(" ".join(map(str, scope)) + " : " + " ".join(map(str, allowed)) + "\n")

    clauses = []
    for scope, allowed in cons:
        aset = set(allowed)
        for c in range(1 << a):
            if c not in aset:                 # forbid tuple c: one clause
                lits = []
                for i, v in enumerate(scope):
                    bit = (c >> i) & 1
                    lits.append(-(v + 1) if bit else (v + 1))
                clauses.append(lits)
    with open(args.cnf, "w") as f:
        f.write(f"c extcsp n={n} w={w} a={a} m={m} tight={args.tight} seed={args.seed}\n")
        f.write(f"p cnf {n} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")


if __name__ == "__main__":
    main()
