#!/usr/bin/env python3
"""Generate and audit the hard-UNSAT factoring study declared by issue #51."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sympy
import yaml

from benchmarks.pipeline.circuit import (
    canonical_bytes,
    pin_port_values,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from benchmarks.pipeline.cnf import encode_validated_circuit
from benchmarks.pipeline.multipliers import generate_multiplier


MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
EXPECTED_WIDTHS = ((64, 32), (72, 36), (80, 40))
EXPECTED_METHOD_BUDGETS = {
    "monolithic-kissat": ("none",),
    "march-cu-dynamic": ("dynamic-default",),
    "region-cc": ("low", "medium", "high"),
    "structure-blind-cc": ("low", "medium", "high"),
}
EXPECTED_BANDS = {"low": 1 << 14, "medium": 1 << 16, "high": 1 << 18}
EXPECTED_TOOL_REVISIONS = {
    "kissat": "8af8e56f174b778aef3aa45af9f739b2a5f492c2",
    "march_cu": "705b60c6491ef2b61988b3ce6ac674be1b90571d",
}


class HardRegimeError(ValueError):
    """The study contract or one of its artifacts violates the frozen protocol."""


def mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HardRegimeError(f"{label} must be an object")
    return value


def positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise HardRegimeError(f"{label} must be a positive integer")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardRegimeError(f"cannot read contract {path}: {exc}") from exc
    contract = mapping(value, "contract")
    validate_contract(contract)
    return contract


def contract_sha256(contract: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(contract))


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != 1:
        raise HardRegimeError("contract schema_version must be 1")
    if contract.get("study_id") != "array-ripple-hard-unsat-cnc-v1":
        raise HardRegimeError("unexpected study_id")
    if contract.get("issue") != 51:
        raise HardRegimeError("contract must identify issue 51")
    if contract.get("architecture") != "array-ripple":
        raise HardRegimeError("hard-regime primary table must use array-ripple")
    if contract.get("expected_outcome") != "unsat":
        raise HardRegimeError("hard-regime targets must declare UNSAT")

    widths = contract.get("widths")
    if not isinstance(widths, list):
        raise HardRegimeError("widths must be a list")
    actual_widths = []
    seeds = []
    for index, raw in enumerate(widths):
        width = mapping(raw, f"widths[{index}]")
        product = positive_int(width.get("product_width"), "product_width")
        factor = positive_int(width.get("factor_input_width"), "factor_input_width")
        if product != 2 * factor:
            raise HardRegimeError(
                f"product width {product} must be twice factor-input width {factor}"
            )
        actual_widths.append((product, factor))
        for split, expected_count in (("calibration", 3), ("held_out", 10)):
            spec = mapping(width.get(split), f"width {product} {split}")
            count = positive_int(spec.get("count"), f"width {product} {split}.count")
            seed = positive_int(spec.get("seed"), f"width {product} {split}.seed")
            if count != expected_count:
                raise HardRegimeError(
                    f"width {product} {split} count must be {expected_count}"
                )
            seeds.append(seed)
    if tuple(actual_widths) != EXPECTED_WIDTHS:
        raise HardRegimeError(
            "width ladder must be product/factor pairs 64/32, 72/36, and 80/40"
        )
    duplicates = sorted(seed for seed, count in Counter(seeds).items() if count > 1)
    if duplicates:
        raise HardRegimeError(f"calibration and held-out seeds overlap: {duplicates}")

    bands = mapping(contract.get("frontier_bands"), "frontier_bands")
    actual_bands = {}
    for name, raw_spec in bands.items():
        spec = mapping(raw_spec, f"frontier band {name}")
        actual_bands[name] = positive_int(
            spec.get("center_cubes"), f"frontier band {name}.center_cubes"
        )
        ratio = spec.get("accepted_ratio")
        if ratio != [0.75, 1.25]:
            raise HardRegimeError(
                f"frontier band {name}.accepted_ratio must be [0.75, 1.25]"
            )
    if actual_bands != EXPECTED_BANDS:
        raise HardRegimeError("frontier bands must be centered at 2^14, 2^16, and 2^18")

    methods = mapping(contract.get("methods"), "methods")
    if set(methods) != set(EXPECTED_METHOD_BUDGETS):
        raise HardRegimeError("method set does not match the frozen issue #51 matrix")
    for name, expected in EXPECTED_METHOD_BUDGETS.items():
        budgets = mapping(methods[name], f"method {name}").get("budgets")
        if not isinstance(budgets, list) or tuple(budgets) != expected:
            raise HardRegimeError(f"method {name} budgets must be {list(expected)}")
    if methods["march-cu-dynamic"].get("cutoff_policy") != "upstream-default-dynamic":
        raise HardRegimeError("march_cu must retain its upstream default dynamic cutoff")
    for name in ("region-cc", "structure-blind-cc"):
        if methods[name].get("max_rows") != 512:
            raise HardRegimeError(f"method {name} max_rows must be 512")

    limits = mapping(contract.get("limits_seconds"), "limits_seconds")
    expected_limits = {"monolithic": 600, "cubing": 7200, "per_cube_conquer": 1800}
    if limits != expected_limits:
        raise HardRegimeError(f"limits_seconds must be {expected_limits}")
    scheduling = mapping(contract.get("scheduling"), "scheduling")
    if scheduling.get("measured_workers") != 32:
        raise HardRegimeError("measured worker count must be 32")
    if scheduling.get("lpt_replay_workers") != [32, 128, 512]:
        raise HardRegimeError("LPT replay worker counts must be 32, 128, and 512")
    statistics = mapping(contract.get("statistics"), "statistics")
    if statistics.get("unit") != "held-out-instance":
        raise HardRegimeError("statistical unit must be the held-out instance")
    adjustment = mapping(statistics.get("budget_adjustment"), "budget_adjustment")
    if adjustment.get("common_grid") != "overlap-low-geometric-mid-overlap-high":
        raise HardRegimeError("budget adjustment common grid is not preregistered")
    calibration = mapping(contract.get("calibration"), "calibration")
    if calibration.get("held_out_recalibration") != "forbidden":
        raise HardRegimeError("held-out threshold tuning must be forbidden")
    tool_sources = mapping(contract.get("tool_sources"), "tool_sources")
    if set(tool_sources) != set(EXPECTED_TOOL_REVISIONS):
        raise HardRegimeError("tool_sources must pin Kissat and march_cu")
    for name, revision in EXPECTED_TOOL_REVISIONS.items():
        source = mapping(tool_sources[name], f"tool source {name}")
        if source.get("revision") != revision:
            raise HardRegimeError(f"tool source {name} revision is not frozen")


def strong_miller_rabin(value: int, bases: Iterable[int] = MILLER_RABIN_BASES) -> bool:
    """Return whether ``value`` passes the declared independent strong-MR checks."""

    if value < 2:
        return False
    small = tuple(bases)
    for prime in small:
        if value % prime == 0:
            return value == prime
    odd = value - 1
    power = 0
    while odd % 2 == 0:
        odd //= 2
        power += 1
    for base in small:
        if base >= value:
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


def _targets_for_split(
    product_width: int,
    factor_width: int,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    lower = 1 << (product_width - 1)
    factor_max = (1 << factor_width) - 1
    reachable_max = factor_max * factor_max
    records = []
    seen: set[int] = set()
    while len(records) < count:
        candidate = rng.randrange(lower, reachable_max + 1) | 1
        target = int(sympy.nextprime(candidate - 1))
        if target > reachable_max or target in seen:
            continue
        seen.add(target)
        index = len(records)
        split_label = "cal" if split == "calibration" else "test"
        range_checks = {
            "full_product_width": target.bit_length() == product_width,
            "above_factor_input_range": target > factor_max,
            "within_reachable_product_range": target <= reachable_max,
        }
        mr_passed = strong_miller_rabin(target)
        sympy_passed = bool(sympy.isprime(target))
        if not all(range_checks.values()) or not mr_passed or not sympy_passed:
            raise HardRegimeError("deterministic target generation produced an invalid prime")
        records.append(
            {
                "schema_version": 1,
                "id": f"prime-p{product_width}-{split_label}-{index:02d}",
                "generator": "deterministic-prime-hard-unsat-v1",
                "architecture": "array-ripple",
                "semantic_task": "unsigned-factoring",
                "expected_outcome": "unsat",
                "product_width": product_width,
                "factor_input_width": factor_width,
                "split": split,
                "split_index": index,
                "seed": seed,
                "target": target,
                "primality_checks": {
                    "sympy_isprime": {
                        "result": sympy_passed,
                        "version": sympy.__version__,
                    },
                    "strong_miller_rabin": {
                        "bases": list(MILLER_RABIN_BASES),
                        "result": mr_passed,
                    },
                },
                "range_checks": range_checks,
                "unsat_argument": "prime-target-above-unsigned-factor-input-range",
            }
        )
    return records


def target_records(contract: dict[str, Any]) -> list[dict[str, Any]]:
    validate_contract(contract)
    records = []
    for width in contract["widths"]:
        for split in ("calibration", "held_out"):
            spec = width[split]
            records.extend(
                _targets_for_split(
                    width["product_width"],
                    width["factor_input_width"],
                    split,
                    spec["count"],
                    spec["seed"],
                )
            )
    return records


def verify_target_records(
    contract: dict[str, Any], records: list[dict[str, Any]]
) -> list[str]:
    validate_contract(contract)
    expected = target_records(contract)
    if records != expected:
        raise HardRegimeError(
            "target records do not byte-semantically match deterministic regeneration"
        )
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise HardRegimeError("target records contain duplicate instance IDs")
    split_targets: dict[str, set[int]] = {"calibration": set(), "held_out": set()}
    for record in records:
        product_width = positive_int(record.get("product_width"), "record product_width")
        factor_width = positive_int(
            record.get("factor_input_width"), "record factor_input_width"
        )
        target = positive_int(record.get("target"), "record target")
        if product_width != 2 * factor_width:
            raise HardRegimeError(f"{record.get('id')}: factor/product width confusion")
        factor_max = (1 << factor_width) - 1
        if target.bit_length() != product_width:
            raise HardRegimeError(f"{record.get('id')}: target does not use product width")
        if target <= factor_max or target > factor_max * factor_max:
            raise HardRegimeError(f"{record.get('id')}: target violates factor/product range")
        if not sympy.isprime(target) or not strong_miller_rabin(target):
            raise HardRegimeError(f"{record.get('id')}: target fails primality verification")
        split = record.get("split")
        if split not in split_targets:
            raise HardRegimeError(f"{record.get('id')}: invalid split")
        split_targets[split].add(target)
    overlap = split_targets["calibration"] & split_targets["held_out"]
    if overlap:
        raise HardRegimeError(f"calibration and held-out targets overlap: {sorted(overlap)}")
    return [
        f"PASS contract: {contract_sha256(contract)}",
        f"PASS targets: {len(records)} deterministic prime UNSAT instances",
        "PASS widths: product/factor pairs are 64/32, 72/36, and 80/40",
        "PASS split: calibration and held-out seeds and targets are disjoint",
        "PASS primality: SymPy and independent strong Miller-Rabin checks agree",
    ]


def materialize_instances(
    contract: dict[str, Any], records: list[dict[str, Any]], out_dir: Path
) -> list[dict[str, Any]]:
    verify_target_records(contract, records)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "targets.jsonl", records)
    contract_digest = contract_sha256(contract)
    circuits: dict[int, tuple[dict[str, Any], Path]] = {}
    manifest = []
    for record in records:
        factor_width = record["factor_input_width"]
        if factor_width not in circuits:
            raw = generate_multiplier(factor_width, "array-ripple")
            raw_path = out_dir / "raw" / f"array-ripple-f{factor_width}.json"
            write_json(raw_path, raw)
            circuits[factor_width] = (raw, raw_path)
        raw, raw_path = circuits[factor_width]
        pinned = pin_port_values(raw, {"product": record["target"]})
        benchmark = pinned.setdefault("metadata", {}).setdefault("benchmark", {})
        benchmark.update(
            {
                **record,
                "contract_sha256": contract_digest,
                "family": "hard-unsat-prime-factoring",
            }
        )
        instance_dir = out_dir / "instances" / f"p{record['product_width']}" / record["id"]
        circuitsat = instance_dir / f"{record['id']}.circuitsat.json"
        cnf = instance_dir / f"{record['id']}.cnf"
        metadata = instance_dir / f"{record['id']}.meta.json"
        write_json(circuitsat, pinned)
        encoding_started = time.perf_counter()
        encoding_cpu_started = time.process_time()
        encode_validated_circuit(pinned).write_dimacs(cnf)
        encoding_wall_s = time.perf_counter() - encoding_started
        encoding_cpu_s = time.process_time() - encoding_cpu_started
        meta = {
            **record,
            "contract_sha256": contract_digest,
            "raw_circuit": str(raw_path.relative_to(out_dir)),
            "raw_circuit_sha256": sha256_file(raw_path),
            "circuitsat": str(circuitsat.relative_to(out_dir)),
            "circuitsat_sha256": sha256_file(circuitsat),
            "cnf": str(cnf.relative_to(out_dir)),
            "cnf_sha256": sha256_file(cnf),
            "encoding_wall_s": encoding_wall_s,
            "encoding_cpu_s": encoding_cpu_s,
        }
        write_json(metadata, meta)
        manifest.append({**meta, "metadata": str(metadata.relative_to(out_dir))})
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    return manifest


def command_validate_contract(args: argparse.Namespace) -> None:
    contract = load_contract(args.contract)
    print(f"PASS contract: {contract_sha256(contract)}")


def command_generate_targets(args: argparse.Namespace) -> None:
    contract = load_contract(args.contract)
    records = target_records(contract)
    write_jsonl(args.out, records)
    for line in verify_target_records(contract, records):
        print(line)


def command_verify_targets(args: argparse.Namespace) -> None:
    contract = load_contract(args.contract)
    for line in verify_target_records(contract, read_jsonl(args.targets)):
        print(line)


def command_materialize(args: argparse.Namespace) -> None:
    contract = load_contract(args.contract)
    records = target_records(contract)
    manifest = materialize_instances(contract, records, args.out_dir)
    print(f"PASS materialization: {len(manifest)} CircuitSAT/CNF pairs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("validate-contract")
    command.add_argument("contract", type=Path)
    command.set_defaults(handler=command_validate_contract)
    command = commands.add_parser("generate-targets")
    command.add_argument("contract", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(handler=command_generate_targets)
    command = commands.add_parser("verify-targets")
    command.add_argument("contract", type=Path)
    command.add_argument("targets", type=Path)
    command.set_defaults(handler=command_verify_targets)
    command = commands.add_parser("materialize")
    command.add_argument("contract", type=Path)
    command.add_argument("--out-dir", type=Path, required=True)
    command.set_defaults(handler=command_materialize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (HardRegimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FAIL: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
