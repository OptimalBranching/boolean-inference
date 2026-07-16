#!/usr/bin/env python3
"""Record structural and artifact-size measurements for a multiplier corpus."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .circuit import CircuitError, load_json, write_json
except ImportError:  # direct script execution
    from circuit import CircuitError, load_json, write_json  # type: ignore


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CircuitError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def circuit_shape(path: Path) -> tuple[int, int]:
    value = load_json(path)
    value = value.get("target", value)
    if isinstance(value, dict):
        value = value.get("data", value)
    if not isinstance(value, dict):
        raise CircuitError(f"{path}: missing CircuitSAT data")
    variables = value.get("variables")
    assignments = value.get("circuit", {}).get("assignments")
    if not isinstance(variables, list) or not isinstance(assignments, list):
        raise CircuitError(f"{path}: malformed CircuitSAT data")
    return len(variables), len(assignments)


def cnf_shape(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("p cnf "):
                fields = line.split()
                if len(fields) == 4:
                    return int(fields[2]), int(fields[3])
                break
    raise CircuitError(f"{path}: missing DIMACS header")


def summarize(manifest_path: Path, root: Path, raw_dir: Path) -> dict:
    groups: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for record in read_jsonl(manifest_path):
        try:
            key = (
                int(record["factor_bits"]),
                str(record["architecture"]),
                str(record["raw_circuit"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CircuitError(f"malformed manifest record: {record!r}") from exc
        groups[key].append(record)
    if not groups:
        raise CircuitError(f"{manifest_path}: no instances found")

    measurements = []
    for (width, architecture, raw_name), records in sorted(groups.items()):
        raw_path = raw_dir / raw_name
        variables, assignments = circuit_shape(raw_path)
        cnf_shapes = {
            cnf_shape(root / str(record["cnf"])) for record in records
        }
        if len(cnf_shapes) != 1:
            raise CircuitError(
                f"{architecture}-{width}: inconsistent CNF dimensions"
            )
        cnf_variables, cnf_clauses = cnf_shapes.pop()
        measurements.append(
            {
                "architecture": architecture,
                "factor_bits": width,
                "instances": len(records),
                "raw_variables": variables,
                "raw_assignments": assignments,
                "raw_bytes": raw_path.stat().st_size,
                "cnf_variables": cnf_variables,
                "cnf_clauses": cnf_clauses,
                "circuitsat_bytes": sum(
                    (root / str(record["circuitsat"])).stat().st_size
                    for record in records
                ),
                "cnf_bytes": sum(
                    (root / str(record["cnf"])).stat().st_size
                    for record in records
                ),
            }
        )
    return {
        "instance_count": sum(len(records) for records in groups.values()),
        "measurements": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize(args.manifest, args.root, args.raw_dir)
    except (CircuitError, OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
