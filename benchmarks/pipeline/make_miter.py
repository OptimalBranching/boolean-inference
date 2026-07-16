#!/usr/bin/env python3
"""Build a global or bit-level equivalence miter from two CircuitSAT circuits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .circuit import (
        CircuitError,
        _decode,
        assignment,
        circuit_data,
        const,
        load_json,
        nary,
        port_bits,
        ports,
        unary,
        validate_circuit,
        var,
        write_json,
    )
except ImportError:  # direct script execution
    from circuit import (  # type: ignore
        CircuitError,
        _decode,
        assignment,
        circuit_data,
        const,
        load_json,
        nary,
        port_bits,
        ports,
        unary,
        validate_circuit,
        var,
        write_json,
    )


def rename_expression(expr: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    op, arg = _decode(expr)
    if op == "Var":
        return var(names[arg])
    if op == "Const":
        return const(arg)
    if op == "Not":
        return unary("Not", rename_expression(arg, names))
    return nary(op, [rename_expression(child, names) for child in arg])


def shared_input_names(
    left: dict, right: dict, input_map: dict[str, str] | None
) -> tuple[dict[str, str], dict[str, str], dict]:
    left_ports = ports(left, "input")
    right_ports = ports(right, "input")
    mapping = input_map or {name: name for name in left_ports}
    if set(mapping) != set(left_ports) or set(mapping.values()) != set(right_ports):
        raise CircuitError(
            "input mapping must pair every left and right input port exactly once"
        )
    left_names: dict[str, str] = {}
    right_names: dict[str, str] = {}
    metadata = {}
    for port_name in sorted(left_ports):
        right_port_name = mapping[port_name]
        left_bits = port_bits(left, port_name, "input")
        right_bits = port_bits(right, right_port_name, "input")
        if len(left_bits) != len(right_bits):
            raise CircuitError(
                f"miter input ports {port_name!r} and {right_port_name!r} have different widths"
            )
        shared = [f"input:{port_name}[{index}]" for index in range(len(left_bits))]
        left_names.update(zip(left_bits, shared, strict=True))
        right_names.update(zip(right_bits, shared, strict=True))
        metadata[port_name] = {"direction": "input", "bits": shared, "lsb_first": True}
    return left_names, right_names, metadata


def namespace(
    data: dict, side: str, shared: dict[str, str]
) -> tuple[list[str], list[dict], dict[str, str]]:
    names = {name: shared.get(name, f"{side}:{name}") for name in data["variables"]}
    variables = list(dict.fromkeys(names.values()))
    assignments = [
        assignment(
            names[item["outputs"][0]],
            rename_expression(item["expr"], names),
        )
        for item in data["circuit"]["assignments"]
    ]
    return variables, assignments, names


def build_miter(
    left: dict,
    right: dict,
    left_output: str,
    right_output: str,
    bit: int | None,
    input_map: dict[str, str] | None = None,
) -> dict:
    validate_circuit(left)
    validate_circuit(right)
    left_shared, right_shared, input_ports = shared_input_names(left, right, input_map)
    left_variables, left_assignments, left_names = namespace(left, "left", left_shared)
    right_variables, right_assignments, right_names = namespace(
        right, "right", right_shared
    )
    left_bits = port_bits(left, left_output, "output")
    right_bits = port_bits(right, right_output, "output")
    if len(left_bits) != len(right_bits):
        raise CircuitError("miter output ports have different widths")
    indices = list(range(len(left_bits))) if bit is None else [bit]
    if any(index < 0 or index >= len(left_bits) for index in indices):
        raise CircuitError(f"miter bit must be in [0, {len(left_bits)})")

    diff_names = [f"miter:diff[{index}]" for index in indices]
    diff_assignments = [
        assignment(
            diff_name,
            nary(
                "Xor",
                [
                    var(left_names[left_bits[index]]),
                    var(right_names[right_bits[index]]),
                ],
            ),
        )
        for index, diff_name in zip(indices, diff_names, strict=True)
    ]
    bad = "miter:bad"
    reduction_variables = []
    reduction_assignments = []
    level = list(diff_names)
    depth = 0
    while len(level) > 1:
        next_level = []
        for pair_index in range(0, len(level), 2):
            pair = level[pair_index : pair_index + 2]
            if len(pair) == 1:
                next_level.append(pair[0])
                continue
            output = f"miter:or[{depth}][{pair_index // 2}]"
            reduction_variables.append(output)
            reduction_assignments.append(
                assignment(output, nary("Or", [var(name) for name in pair]))
            )
            next_level.append(output)
        level = next_level
        depth += 1
    reduction_assignments.append(assignment(bad, var(level[0])))
    result = {
        "variables": list(
            dict.fromkeys(
                [
                    *left_variables,
                    *right_variables,
                    *diff_names,
                    *reduction_variables,
                    bad,
                ]
            )
        ),
        "circuit": {
            "assignments": [
                *left_assignments,
                *right_assignments,
                *diff_assignments,
                *reduction_assignments,
                assignment(bad, const(True)),
            ]
        },
        "metadata": {
            "format": "circuitsat-benchmark-v1",
            "semantic_task": "bit-level-equivalence"
            if bit is not None
            else "global-equivalence",
            "expected_outcome": "unsat",
            "ports": {
                **input_ports,
                "miter_bad": {"direction": "output", "bits": [bad], "lsb_first": True},
            },
            "left_output": left_output,
            "right_output": right_output,
            "bit": bit,
        },
    }
    validate_circuit(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-output", required=True)
    parser.add_argument("--right-output", required=True)
    parser.add_argument(
        "--input-map",
        action="append",
        default=[],
        metavar="LEFT=RIGHT",
        help="map differently named input ports; repeat for every pair",
    )
    parser.add_argument("--bit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        input_map = {}
        for item in args.input_map:
            if "=" not in item:
                raise CircuitError("--input-map must be LEFT=RIGHT")
            left_name, right_name = item.split("=", 1)
            if not left_name or not right_name or left_name in input_map:
                raise CircuitError(
                    "--input-map must contain unique non-empty port names"
                )
            input_map[left_name] = right_name
        write_json(
            args.out,
            build_miter(
                circuit_data(load_json(args.left)),
                circuit_data(load_json(args.right)),
                args.left_output,
                args.right_output,
                args.bit,
                input_map or None,
            ),
        )
    except CircuitError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
