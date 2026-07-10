#!/usr/bin/env python3
"""Exact Cover by 3-Sets (X3C) decision — the family the project's novelty (the
exact-cover gamma branching rules, issue #14 / #8-#13) is built for. Universe of
3q elements; a collection of 3-element subsets; is there a subcollection that
partitions the universe (covers each element EXACTLY once)?

A planted exact cover guarantees SAT; `--extra` random 3-subsets are distractors
that make search hard. Constraint per element: exactly one selected subset
covers it (arity = #subsets containing it). Emits native .csp (one exactly-one
tensor per element) and DIMACS .cnf (at-least-one + at-most-one pairs).

Usage: gen_x3c.py --q 20 --extra 40 --seed 1 --csp o.csp --cnf o.cnf
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, required=True, help="universe = 3q elements")
    ap.add_argument("--extra", type=int, default=0, help="distractor 3-subsets")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csp", required=True)
    ap.add_argument("--cnf", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed * 100003 + args.q)
    n = 3 * args.q
    elems = list(range(n))
    rng.shuffle(elems)
    subsets = [tuple(sorted(elems[3 * i:3 * i + 3])) for i in range(args.q)]  # planted cover
    seen = set(subsets)
    while len(subsets) < args.q + args.extra:
        s = tuple(sorted(rng.sample(range(n), 3)))
        if s not in seen:
            seen.add(s)
            subsets.append(s)
    rng.shuffle(subsets)

    m = len(subsets)                      # subset vars, 0-based
    covers = {e: [] for e in range(n)}    # element -> subset ids containing it
    for si, s in enumerate(subsets):
        for e in s:
            covers[e].append(si)

    def one_hot(k):
        return [c for c in range(1 << k) if bin(c).count("1") == 1]

    with open(args.csp, "w") as f:
        f.write(f"{m}\n")
        for e in range(n):
            sc = covers[e]
            f.write(" ".join(map(str, sc)) + " : " + " ".join(map(str, one_hot(len(sc)))) + "\n")

    clauses = []
    for e in range(n):
        sc = covers[e]
        clauses.append([s + 1 for s in sc])                       # at-least-one
        for a in range(len(sc)):
            for b in range(a + 1, len(sc)):
                clauses.append([-(sc[a] + 1), -(sc[b] + 1)])      # at-most-one
    with open(args.cnf, "w") as f:
        f.write(f"c x3c q={args.q} subsets={m} extra={args.extra} seed={args.seed}\n")
        f.write(f"p cnf {m} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")


if __name__ == "__main__":
    main()
