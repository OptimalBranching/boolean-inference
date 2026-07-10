#!/usr/bin/env python3
"""Generate k-coloring DIMACS CNF instances, with a graph-structure knob.

Goal (runscribe goal D): find a decision family where our region-contraction
solver's LOCALITY exploitation beats CDCL. Graph coloring on LOW-TREEWIDTH
graphs is the mechanism bet — a region grown around a vertex stays a small
local patch, so the branching tree tracks the tree-decomposition width, while
CDCL treats the CNF as flat and cannot exploit the bounded width structurally.

Two graph families, same n and edge budget so hardness is comparable:
  banded  — vertices on a line; edges only within bandwidth `w`. Treewidth <= w
            (a path/band decomposition), i.e. bounded, LOCAL structure.
  random  — Erdos-Renyi G(n, m): the SAME number of edges placed uniformly at
            random. Treewidth ~ n; no local structure. CDCL's home turf.

Encoding (standard direct k-coloring): x[v,c] = "vertex v has color c".
  - at-least-one:  (x[v,0] v ... v x[v,k-1])           per vertex
  - at-most-one:   (~x[v,c] v ~x[v,c'])                 per vertex, c<c'
  - edge-differ:   (~x[u,c] v ~x[v,c])                  per edge, per color
Same encoding for both families and for our solver + kissat (one CNF, two
solvers) — no reduction routing, maximum reuse.

Usage:
  gen_coloring.py --family banded --n 40 --k 4 --w 4 --seed 1 > out.cnf
  gen_coloring.py --family random --n 40 --k 4 --edges 148 --seed 1 > out.cnf
"""
import argparse
import random
import sys


def banded_edges(n, w, rng):
    """Edges within bandwidth w on a line 0..n-1. Includes the path (i,i+1) so
    the graph is connected, then fills in-band chords. Treewidth <= w."""
    edges = set()
    for i in range(n):
        for j in range(i + 1, min(i + w + 1, n)):
            edges.add((i, j))
    return edges


def random_edges(n, m, rng):
    """m distinct random edges (Erdos-Renyi G(n,m))."""
    edges = set()
    m = min(m, n * (n - 1) // 2)
    while len(edges) < m:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["banded", "random"], required=True)
    ap.add_argument("--n", type=int, required=True, help="vertices")
    ap.add_argument("--k", type=int, required=True, help="colors")
    ap.add_argument("--w", type=int, default=4, help="bandwidth (banded only)")
    ap.add_argument("--edges", type=int, default=0, help="edge count (random; default=match banded w)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dump-graph", default="", help="also write '<n> <k>' + edge list here")
    args = ap.parse_args()

    rng = random.Random(args.seed * 100003 + args.n)
    if args.family == "banded":
        edges = banded_edges(args.n, args.w, rng)
    else:
        m = args.edges or (args.n * args.w - args.w * (args.w + 1) // 2)  # ~banded count
        edges = random_edges(args.n, m, rng)

    n, k = args.n, args.k

    if args.dump_graph:
        with open(args.dump_graph, "w") as f:
            f.write(f"{n} {k}\n")
            for (u, v) in sorted(edges):
                f.write(f"{u} {v}\n")

    def var(v, c):  # 1-based DIMACS var id
        return v * k + c + 1

    clauses = []
    for v in range(n):
        clauses.append([var(v, c) for c in range(k)])            # at-least-one
        for c in range(k):
            for c2 in range(c + 1, k):
                clauses.append([-var(v, c), -var(v, c2)])         # at-most-one
    for (u, v) in sorted(edges):
        for c in range(k):
            clauses.append([-var(u, c), -var(v, c)])              # edge differ

    out = [f"c k-coloring family={args.family} n={n} k={k} "
           f"w={args.w} edges={len(edges)} seed={args.seed}",
           f"p cnf {n * k} {len(clauses)}"]
    for cl in clauses:
        out.append(" ".join(map(str, cl)) + " 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
