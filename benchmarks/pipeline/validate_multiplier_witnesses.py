#!/usr/bin/env python3
"""Simulate multiplier circuits with private factors and verify their products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .circuit import (
        CircuitError,
        CircuitSimulator,
        bits_to_int,
        circuit_data,
        int_to_bit_values,
        load_json,
        port_bits,
        ports,
    )
except ImportError:  # direct script execution
    from circuit import (  # type: ignore
        CircuitError,
        CircuitSimulator,
        bits_to_int,
        circuit_data,
        int_to_bit_values,
        load_json,
        port_bits,
        ports,
    )


def read_jsonl(path: Path) -> list[dict]:
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CircuitError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def validate(manifest_path: Path, oracle_path: Path, raw_dir: Path) -> int:
    oracle: dict[str, dict] = {}
    for record in read_jsonl(oracle_path):
        target_id = record.get("id")
        if not isinstance(target_id, str) or not target_id:
            raise CircuitError(
                f"{oracle_path}: every oracle record needs a non-empty id"
            )
        if target_id in oracle:
            raise CircuitError(f"{oracle_path}: duplicate target id {target_id!r}")
        oracle[target_id] = record
    if not oracle:
        raise CircuitError(f"{oracle_path}: no private factors found")

    prepared: dict[
        Path, tuple[CircuitSimulator, list[str], list[str], list[str]]
    ] = {}
    checked = 0
    for record in read_jsonl(manifest_path):
        target_id = record.get("target_id")
        witness = oracle.get(target_id)
        if witness is None:
            raise CircuitError(f"missing private factors for target {target_id!r}")
        try:
            expected = int(record["target"])
            left = int(witness["left_factor"])
            right = int(witness["right_factor"])
            raw_path = raw_dir / record["raw_circuit"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CircuitError(
                f"malformed manifest or oracle record for {target_id!r}"
            ) from exc
        if left * right != expected:
            raise CircuitError(
                f"private factors do not multiply to target {target_id!r}"
            )

        if raw_path not in prepared:
            data = circuit_data(load_json(raw_path))
            input_ports = ports(data, "input")
            output_ports = ports(data, "output")
            if len(input_ports) != 2 or len(output_ports) != 1:
                raise CircuitError(
                    f"{raw_path}: expected exactly two input ports and one output port"
                )
            inputs = [port_bits(data, name, "input") for name in sorted(input_ports)]
            outputs = port_bits(data, next(iter(output_ports)), "output")
            prepared[raw_path] = (
                CircuitSimulator(data),
                inputs[0],
                inputs[1],
                outputs,
            )

        simulator, left_bits, right_bits, output_bits = prepared[raw_path]
        values = int_to_bit_values(left_bits, left)
        values.update(int_to_bit_values(right_bits, right))
        result = simulator.simulate(values)
        actual = bits_to_int(output_bits, result)
        if actual != expected:
            raise CircuitError(
                f"{raw_path}: witness for {target_id!r} produced {actual}, expected {expected}"
            )
        checked += 1
    if checked == 0:
        raise CircuitError(f"{manifest_path}: no instances found")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        checked = validate(args.manifest, args.oracle, args.raw_dir)
    except (CircuitError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"PASS multiplication witnesses: {checked} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
