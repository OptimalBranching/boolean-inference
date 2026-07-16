#!/usr/bin/env python3
"""Generate deterministic balanced-semiprime targets independently of circuits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small:
        if value % prime == 0:
            return value == prime
    odd = value - 1
    power = 0
    while odd % 2 == 0:
        odd //= 2
        power += 1
    # Deterministic for unsigned 64-bit integers; still a strong fixed-base test above it.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def random_prime(rng: random.Random, bits: int) -> int:
    if bits < 2:
        raise ValueError("factor width must be at least 2 bits")
    while True:
        candidate = rng.randrange(1 << (bits - 1), 1 << bits) | 1
        if is_prime(candidate):
            return candidate


def records(widths: list[int], count: int, seed_base: int):
    for width in widths:
        rng = random.Random(seed_base + width)
        seen: set[int] = set()
        index = 0
        while index < count:
            left = random_prime(rng, width)
            right = random_prime(rng, width)
            target = left * right
            if target in seen:
                continue
            seen.add(target)
            public = {
                "id": f"fact-{width}-{index:04d}",
                "generator": "balanced-semiprime-v1",
                "factor_bits": width,
                "target": target,
                "seed": seed_base + width,
                "sequence_index": index,
            }
            oracle = {**public, "left_factor": left, "right_factor": right}
            yield public, oracle
            index += 1


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, action="append", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed-base", type=int, default=20260709)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--oracle-out",
        type=Path,
        help="optional private witness file; never pass this file to a solver",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    generated = list(records(args.width, args.count, args.seed_base))
    write_jsonl(args.out, [public for public, _ in generated])
    if args.oracle_out:
        write_jsonl(args.oracle_out, [oracle for _, oracle in generated])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
