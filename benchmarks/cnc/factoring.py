#!/usr/bin/env python3
"""Generate n×n SAT/UNSAT factoring instances as CircuitSAT and DIMACS CNF."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from benchmarks.pipeline.circuit import (
    pin_port_values,
    sha256_file,
    write_json,
    write_jsonl,
)
from benchmarks.pipeline.cnf import encode_validated_circuit
from benchmarks.pipeline.multipliers import (
    DETERMINISTIC_MILLER_RABIN_LIMIT,
    generate_multiplier,
    is_prime,
    records as factoring_records,
)

Outcome = Literal["sat", "unsat"]
SAT_SEED_BASE = 20260709
UNSAT_MATCH_CANDIDATES = 64


@dataclass(frozen=True, slots=True)
class FactoringTarget:
    instance_id: str
    factor_bits: int
    expected_outcome: Outcome
    target: int
    target_bits: int
    seed: int
    sequence_index: int
    paired_sat_id: str | None


def sat_targets(
    width: int,
    count: int,
    *,
    seed_base: int = SAT_SEED_BASE,
    instance_prefix: str = "factoring",
) -> tuple[list[FactoringTarget], list[dict]]:
    if not 2 <= width <= 39 or count < 1:
        raise ValueError("factor width must be between 2 and 39; count must be positive")
    if width <= 12:
        prime_count = sum(
            is_prime(candidate)
            for candidate in range(1 << (width - 1), 1 << width)
        )
        distinct_products = prime_count * (prime_count + 1) // 2
        if count > distinct_products:
            raise ValueError(
                f"width {width} has only {distinct_products} distinct "
                "balanced semiprime products"
            )
    targets = []
    oracles = []
    for public, oracle in factoring_records([width], count, seed_base):
        target = public["target"]
        index = public["sequence_index"]
        instance_id = f"{instance_prefix}-n{width}-sat-{index:02d}"
        targets.append(
            FactoringTarget(
                instance_id=instance_id,
                factor_bits=width,
                expected_outcome="sat",
                target=target,
                target_bits=target.bit_length(),
                seed=public["seed"],
                sequence_index=index,
                paired_sat_id=None,
            )
        )
        oracles.append(
            {
                "instance_id": instance_id,
                "expected_outcome": "sat",
                "left_factor": oracle["left_factor"],
                "right_factor": oracle["right_factor"],
            }
        )
    return targets, oracles


def unsat_targets(
    width: int,
    sat: list[FactoringTarget],
    *,
    instance_prefix: str = "factoring",
) -> tuple[list[FactoringTarget], list[dict]]:
    factor_max = (1 << width) - 1
    reachable_max = factor_max * factor_max
    targets = []
    oracles = []
    seen: set[int] = set()
    for index, paired in enumerate(sat):
        candidates = []
        delta = 2
        lower = 1 << (paired.target_bits - 1)
        upper = min(1 << paired.target_bits, reachable_max + 1)
        max_delta = max(paired.target - lower, upper - 1 - paired.target)
        while len(candidates) < UNSAT_MATCH_CANDIDATES and delta <= max_delta:
            for target in (paired.target - delta, paired.target + delta):
                if (
                    lower <= target < upper
                    and target > factor_max
                    and target not in seen
                    and target < DETERMINISTIC_MILLER_RABIN_LIMIT
                    and is_prime(target)
                ):
                    candidates.append(target)
                    if len(candidates) == UNSAT_MATCH_CANDIDATES:
                        break
            delta += 2
        if not candidates:
            raise ValueError(
                f"{paired.instance_id}: no distinct prime UNSAT target in range"
            )
        target = min(
            candidates,
            key=lambda value: (
                (value ^ paired.target).bit_count(),
                abs(value - paired.target),
                value,
            ),
        )
        seen.add(target)
        instance_id = f"{instance_prefix}-n{width}-unsat-{index:02d}"
        targets.append(
            FactoringTarget(
                instance_id=instance_id,
                factor_bits=width,
                expected_outcome="unsat",
                target=target,
                target_bits=target.bit_length(),
                seed=paired.seed,
                sequence_index=index,
                paired_sat_id=paired.instance_id,
            )
        )
        oracles.append(
            {
                "instance_id": instance_id,
                "expected_outcome": "unsat",
                "target_is_prime": True,
                "target_exceeds_max_factor": True,
                "target_within_multiplier_range": True,
                "paired_sat_id": paired.instance_id,
            }
        )
    return targets, oracles


def validate_targets(targets: list[FactoringTarget], oracles: list[dict]) -> None:
    by_id = {target.instance_id: target for target in targets}
    oracle_ids = [oracle.get("instance_id") for oracle in oracles]
    if len(by_id) != len(targets) or len(oracle_ids) != len(set(oracle_ids)):
        raise ValueError("duplicate factoring target or oracle id")
    if set(oracle_ids) != set(by_id):
        raise ValueError("factoring targets and oracles are not one-to-one")
    for oracle in oracles:
        target = by_id[oracle["instance_id"]]
        if target.target_bits != target.target.bit_length():
            raise ValueError(f"{target.instance_id}: stale target bit length")
        if target.expected_outcome == "sat":
            left = int(oracle["left_factor"])
            right = int(oracle["right_factor"])
            if (
                target.paired_sat_id is not None
                or left.bit_length() != target.factor_bits
                or right.bit_length() != target.factor_bits
                or left * right != target.target
            ):
                raise ValueError(f"{target.instance_id}: invalid SAT oracle")
        elif target.expected_outcome == "unsat":
            factor_max = (1 << target.factor_bits) - 1
            if (
                target.paired_sat_id not in by_id
                or target.target <= factor_max
                or target.target > factor_max * factor_max
                or target.target >= DETERMINISTIC_MILLER_RABIN_LIMIT
                or not is_prime(target.target)
            ):
                raise ValueError(f"{target.instance_id}: invalid UNSAT oracle")
        else:
            raise ValueError(f"{target.instance_id}: invalid expected outcome")


def materialize(
    widths: list[int],
    count: int,
    out_dir: Path,
    *,
    seed_base: int = SAT_SEED_BASE,
    instance_prefix: str = "factoring",
) -> list[dict]:
    if not widths or len(widths) != len(set(widths)):
        raise ValueError("factor widths must be non-empty and unique")
    all_targets: list[FactoringTarget] = []
    all_oracles: list[dict] = []
    for width in widths:
        sat, sat_oracles = sat_targets(
            width,
            count,
            seed_base=seed_base,
            instance_prefix=instance_prefix,
        )
        unsat, unsat_oracles = unsat_targets(
            width,
            sat,
            instance_prefix=instance_prefix,
        )
        all_targets.extend((*sat, *unsat))
        all_oracles.extend((*sat_oracles, *unsat_oracles))
    validate_targets(all_targets, all_oracles)

    raw_by_width = {
        width: generate_multiplier(width, "array-ripple") for width in widths
    }
    manifest = []
    for target in all_targets:
        instance_dir = (
            out_dir
            / f"n{target.factor_bits}"
            / target.expected_outcome
            / f"{target.sequence_index:02d}"
        )
        circuit_path = instance_dir / "instance.circuitsat.json"
        cnf_path = instance_dir / "instance.cnf"
        metadata_path = instance_dir / "instance.meta.json"
        circuit = pin_port_values(
            raw_by_width[target.factor_bits], {"product": target.target}
        )
        circuit.setdefault("metadata", {})["factoring"] = {
            **asdict(target),
        }
        write_json(circuit_path, circuit)
        encode_validated_circuit(circuit).write_dimacs(cnf_path)
        metadata = {
            **asdict(target),
            "circuit": str(circuit_path.relative_to(out_dir)),
            "circuit_sha256": sha256_file(circuit_path),
            "cnf": str(cnf_path.relative_to(out_dir)),
            "cnf_sha256": sha256_file(cnf_path),
        }
        write_json(metadata_path, metadata)
        manifest.append(
            {
                **metadata,
                "metadata": str(metadata_path.relative_to(out_dir)),
            }
        )
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    write_jsonl(out_dir / "oracles.jsonl", all_oracles)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, action="append", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=SAT_SEED_BASE)
    parser.add_argument("--instance-prefix", default="factoring")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    widths = sorted(args.width)
    try:
        manifest = materialize(
            widths,
            args.count,
            args.out_dir,
            seed_base=args.seed_base,
            instance_prefix=args.instance_prefix,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"wrote {len(manifest)} CircuitSAT/CNF instance pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
