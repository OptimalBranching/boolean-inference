#!/usr/bin/env python3
"""Generate the factoring benchmark set as JSONL, one CircuitSAT instance per line.

For each size n it writes `data/factoring_<n>x<n>.jsonl` with 10 balanced
semiprime instances: N = p*q where p, q are independent n-bit primes (each in
[2^(n-1), 2^n), so both factors genuinely use n bits — the hard case). The first
line reuses a pinned N (the historical single-fixture N, or this project's
22x22 N = 8750074000153) so earlier results stay reproducible; the rest are
random.

Each line is the exact `pred create Factoring --target N --m n --n n --json |
pred reduce - --to CircuitSAT --json` bundle (compact) — the same schema the
solver already parses. `source.data` in each line carries m/n/N, so no side
metadata is needed; the factors are recovered by solving.

Reproducibility: the RNG is seeded PER SIZE (`SEED_BASE + n`), so any subset of
sizes, in any order or invocation grouping, regenerates byte-identically.

Usage (from anywhere, with `pred` on PATH):
    python3 benchmarks/gen_factoring.py 12 14 16 18 20 22
"""
import json
import os
import random
import subprocess
import sys

SEED_BASE = 20260709

# Pinned first instances (line 0) — preserved for reproducibility of prior runs.
# 12/16/18/20 are the historical single-fixture Ns; 22 is the pinned session N
# (the legacy 22x22 fixture had no source field to read it from). Sizes not
# listed here (e.g. 14) are fully random.
PINNED = {
    12: 8780899,
    16: 2022838889,
    18: 31166617913,
    20: 323303587291,
    22: 8750074000153,
}


def is_prime(n):
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def rand_prime(rng, bits):
    while True:
        c = rng.randrange(2 ** (bits - 1), 2 ** bits) | 1
        if is_prime(c):
            return c


def make(N, m, n):
    p1 = subprocess.run(
        ["pred", "create", "Factoring", "--target", str(N), "--m", str(m), "--n", str(n), "--json"],
        capture_output=True, text=True,
    )
    p2 = subprocess.run(
        ["pred", "reduce", "-", "--to", "CircuitSAT", "--json"],
        input=p1.stdout, capture_output=True, text=True,
    )
    return json.loads(p2.stdout)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
for n in [int(x) for x in sys.argv[1:]]:
    rng = random.Random(SEED_BASE + n)  # per-size seed → invocation-independent
    Ns = []
    if n in PINNED:
        Ns.append(PINNED[n])
    while len(Ns) < 10:
        N = rand_prime(rng, n) * rand_prime(rng, n)
        if N not in Ns:
            Ns.append(N)
    path = os.path.join(OUT, f"factoring_{n}x{n}.jsonl")
    with open(path, "w") as f:
        for N in Ns:
            f.write(json.dumps(make(N, n, n), separators=(",", ":")) + "\n")
    print(f"n={n}: {len(Ns)} instances -> {path} ({os.path.getsize(path) // 1024} KB)")
