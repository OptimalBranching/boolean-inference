#!/usr/bin/env python3
"""Tseitin formulas — the canonical resolution-hard family. Each graph EDGE is a
boolean var; each VERTEX gets a parity constraint (XOR of its incident edge vars
= its charge). If the total charge is odd the formula is UNSAT, and on an
EXPANDER graph every resolution/CDCL refutation is exponentially long — kissat's
worst case. On a bounded-treewidth graph (grid/path) variable elimination is
poly. This probes whether our region contraction (local elimination) beats CDCL
where CDCL is provably weak.

Families:
  regular  — random d-regular graph on v vertices (expander w.h.p. for d>=3)
  grid     — r x c grid (treewidth ~ min(r,c); bounded if one side is small)

Emits native .csp (one parity tensor per vertex) and DIMACS .cnf (each parity
expands to 2^(deg-1) clauses), same system both ways.

Usage:
  gen_tseitin.py --family regular --v 40 --d 3 --seed 1 --csp o.csp --cnf o.cnf
  gen_tseitin.py --family grid --rows 4 --cols 20 --seed 1 --csp o.csp --cnf o.cnf
"""
import argparse
import random


def parity(c):
    return bin(c).count("1") & 1


def regular_graph(v, d, rng):
    """Random d-regular graph via the pairing model (retry on failure)."""
    for _ in range(200):
        stubs = [x for x in range(v) for _ in range(d)]
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            return sorted(edges)
    raise RuntimeError("failed to build regular graph")


def grid_graph(r, c):
    def vid(i, j):
        return i * c + j
    edges = []
    for i in range(r):
        for j in range(c):
            if j + 1 < c:
                edges.append((vid(i, j), vid(i, j + 1)))
            if i + 1 < r:
                edges.append((vid(i, j), vid(i + 1, j)))
    return sorted(edges), r * c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["regular", "grid"], required=True)
    ap.add_argument("--v", type=int, default=40)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csp", required=True)
    ap.add_argument("--cnf", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed * 100003 + args.v + args.cols)
    if args.family == "regular":
        edges = regular_graph(args.v, args.d, rng)
        nv = args.v
    else:
        edges, nv = grid_graph(args.rows, args.cols)

    # edge var id = index in `edges`. Incident edges per vertex.
    inc = {v: [] for v in range(nv)}
    for ei, (a, b) in enumerate(edges):
        inc[a].append(ei)
        inc[b].append(ei)

    # charges: all 0 except one vertex flipped to 1 -> total charge odd -> UNSAT.
    charge = {v: 0 for v in range(nv)}
    charge[0] = 1

    n_edges = len(edges)
    # native: one parity tensor per vertex over its incident edge vars.
    csp_lines = [str(n_edges)]
    cnf_clauses = []
    for v in range(nv):
        scope = inc[v]
        if not scope:
            continue
        b = charge[v]
        k = len(scope)
        allowed = [c for c in range(1 << k) if parity(c) == b]
        csp_lines.append(" ".join(map(str, scope)) + " : " + " ".join(map(str, allowed)))
        for c in range(1 << k):
            if parity(c) != b:
                lits = []
                for i, e in enumerate(scope):
                    bit = (c >> i) & 1
                    lits.append(-(e + 1) if bit else (e + 1))
                cnf_clauses.append(lits)

    with open(args.csp, "w") as f:
        f.write("\n".join(csp_lines) + "\n")
    with open(args.cnf, "w") as f:
        f.write(f"c tseitin {args.family} edges={n_edges} verts={nv} seed={args.seed}\n")
        f.write(f"p cnf {n_edges} {len(cnf_clauses)}\n")
        for cl in cnf_clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")


if __name__ == "__main__":
    main()
