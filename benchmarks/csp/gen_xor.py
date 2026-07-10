#!/usr/bin/env python3
"""Random k-XOR-SAT: m parity equations over n boolean vars, each
x_{i0} XOR ... XOR x_{i(k-1)} = b. Emits the SAME system two ways:

  native .csp  — one parity TENSOR per equation (arity k; allowed configs are
                 those with the right parity). Our solver sees the relation whole.
  DIMACS .cnf  — each equation expands to 2^(k-1) clauses. This is what a CDCL
                 solver (kissat) gets.

Why this family (runscribe goal D): XOR/parity is the textbook case where
CDCL/resolution is EXPONENTIALLY weak (kissat has no Gaussian elimination),
yet the system is linear-algebra-easy. If our region contraction acts like
local elimination over the parity tensors, this is a home-turf win CDCL can't
match.

Usage:
  gen_xor.py --n 60 --ratio 0.9 --k 3 --seed 1 --csp out.csp --cnf out.cnf
"""
import argparse
import random


def parity(c):
    return bin(c).count("1") & 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=0.9, help="m/n")
    ap.add_argument("--m", type=int, default=0, help="equations (overrides ratio)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csp", required=True)
    ap.add_argument("--cnf", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed * 100003 + args.n)
    n, k = args.n, args.k
    m = args.m or max(1, round(args.ratio * n))

    eqs = []  # (vars0based, b)
    for _ in range(m):
        vs = rng.sample(range(n), k)
        b = rng.randrange(2)
        eqs.append((sorted(vs), b))

    # native: allowed configs = those whose parity == b
    with open(args.csp, "w") as f:
        f.write(f"{n}\n")
        for (vs, b) in eqs:
            allowed = [c for c in range(1 << k) if parity(c) == b]
            f.write(" ".join(map(str, vs)) + " : " + " ".join(map(str, allowed)) + "\n")

    # cnf: forbid each config with parity != b (one clause each). DIMACS vars 1-based.
    clauses = []
    for (vs, b) in eqs:
        for c in range(1 << k):
            if parity(c) != b:
                lits = []
                for i, v in enumerate(vs):
                    bit = (c >> i) & 1
                    lits.append(-(v + 1) if bit else (v + 1))
                clauses.append(lits)
    with open(args.cnf, "w") as f:
        f.write(f"c random {k}-XOR n={n} m={m} seed={args.seed}\n")
        f.write(f"p cnf {n} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")


if __name__ == "__main__":
    main()
