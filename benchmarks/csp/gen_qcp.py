#!/usr/bin/env python3
"""Quasigroup completion (Latin square with holes) — the classic CP-beats-SAT
family. Fill an m x m grid so each row and column is a permutation of 0..m-1,
given some prefilled cells. Generated from a full Latin square with holes
punched, so always SAT; hardest around ~60% holes.

Native .csp uses the standard DECOMPOSED all-different (exactly-one per cell,
per row-value, per col-value) — the SAME decomposition SAT uses, because the
global all-different does not compress into a small boolean tensor. So this
tests whether our GAC on wide exactly-one tensors beats CDCL even without the
true all-different global propagation.

Usage: gen_qcp.py --m 8 --holes 0.6 --seed 1 --csp o.csp --cnf o.cnf
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--holes", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csp", required=True)
    ap.add_argument("--cnf", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed * 100003 + args.m)
    m = args.m
    # Full Latin square via cyclic construction + random row/col/symbol permutations.
    base = [[(i + j) % m for j in range(m)] for i in range(m)]
    rp = list(range(m)); rng.shuffle(rp)
    cp = list(range(m)); rng.shuffle(cp)
    sp = list(range(m)); rng.shuffle(sp)
    L = [[sp[base[rp[i]][cp[j]]] for j in range(m)] for i in range(m)]
    # Punch holes.
    given = {}
    for i in range(m):
        for j in range(m):
            if rng.random() > args.holes:
                given[(i, j)] = L[i][j]

    def var(r, c, v):
        return (r * m + c) * m + v  # 0-based

    n_vars = m * m * m
    scopes, tables = [], []
    full = list(range(1 << m))
    one_hot = [c for c in full if bin(c).count("1") == 1]

    def exactly_one(ids):
        scopes.append(list(ids))
        tables.append(one_hot)

    for r in range(m):
        for c in range(m):
            exactly_one([var(r, c, v) for v in range(m)])       # one value per cell
    for r in range(m):
        for v in range(m):
            exactly_one([var(r, c, v) for c in range(m)])       # value once per row
    for c in range(m):
        for v in range(m):
            exactly_one([var(r, c, v) for r in range(m)])       # value once per col

    # CNF (1-based): exactly-one = at-least-one + at-most-one pairs; givens as units.
    clauses = []
    for sc in scopes:
        clauses.append([x + 1 for x in sc])
        for a in range(len(sc)):
            for b in range(a + 1, len(sc)):
                clauses.append([-(sc[a] + 1), -(sc[b] + 1)])
    given_units = [var(r, c, v) for (r, c), v in given.items()]
    for u in given_units:
        clauses.append([u + 1])

    with open(args.csp, "w") as f:
        f.write(f"{n_vars}\n")
        for sc, tb in zip(scopes, tables):
            f.write(" ".join(map(str, sc)) + " : " + " ".join(map(str, tb)) + "\n")
        for u in given_units:              # given: unit tensor fixing var to 1
            f.write(f"{u} : 1\n")

    with open(args.cnf, "w") as f:
        f.write(f"c qcp m={m} holes={args.holes} given={len(given)} seed={args.seed}\n")
        f.write(f"p cnf {n_vars} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")


if __name__ == "__main__":
    main()
