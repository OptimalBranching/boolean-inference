"""Construct matched factoring instances, preimages, and equivalence miters."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .circuit import (
    CircuitError,
    CircuitSimulator,
    assignment,
    circuit_data,
    const,
    decode_expression,
    load_json,
    nary,
    pin_port_values,
    port_bits,
    port_values,
    ports,
    sha256_file,
    unary,
    validate_circuit,
    var,
    write_json,
)
from .cnf import encode_validated_circuit

REQUIRED_ARCHITECTURES = {
    "array-ripple",
    "wallace-ripple",
    "dadda-ripple",
    "booth-r4-wallace",
    "booth-r4-dadda",
    "karatsuba",
}


@dataclass(frozen=True, slots=True)
class PreparedCircuit:
    data: dict
    path: Path
    product_port: str
    product_width: int
    digest: str
    provenance: dict


def normalize_targets(targets: list[dict]) -> list[tuple[str, int, int]]:
    normalized = []
    ids: set[str] = set()
    for record in targets:
        target_id = record.get("id")
        if not isinstance(target_id, str) or not target_id:
            raise CircuitError(f"malformed target id: {record!r}")
        if target_id in ids:
            raise CircuitError(f"duplicate target id {target_id!r}")
        ids.add(target_id)
        try:
            width = int(record["factor_bits"])
            target = int(record["target"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CircuitError(f"malformed target record: {record!r}") from exc
        if width < 1 or target < 0:
            raise CircuitError(f"malformed target record: {record!r}")
        normalized.append((target_id, width, target))
    if not normalized:
        raise CircuitError("target list must not be empty")
    return normalized


def prepare_circuit(
    architecture: str,
    width: int,
    path_template: str,
    product_port: str,
) -> PreparedCircuit:
    raw_path = Path(path_template.format(bits=width))
    raw = circuit_data(load_json(raw_path))
    validate_circuit(raw)
    raw_metadata = raw.get("metadata", {})
    declared_architecture = raw_metadata.get("architecture")
    if declared_architecture is not None and declared_architecture != architecture:
        raise CircuitError(
            f"{raw_path}: declares architecture {declared_architecture!r}, expected {architecture!r}"
        )
    declared_width = raw_metadata.get("factor_bits")
    if declared_width is not None and declared_width != width:
        raise CircuitError(
            f"{raw_path}: declares factor width {declared_width}, expected {width}"
        )
    input_widths = sorted(
        len(port_bits(raw, name, "input")) for name in ports(raw, "input")
    )
    if input_widths != [width, width]:
        raise CircuitError(
            f"{raw_path}: factoring multiplier must have exactly two {width}-bit inputs"
        )
    product_width = len(port_bits(raw, product_port, "output"))
    if product_width != width * 2:
        raise CircuitError(
            f"{raw_path}: product port must be exactly {width * 2} bits, got {product_width}"
        )
    provenance = {
        key: raw_metadata[key]
        for key in (
            "source_id",
            "source_revision",
            "source_verilog_sha256",
            "yosys_version",
            "normalization",
            "generator",
            "generator_sha256",
        )
        if raw_metadata.get(key) is not None
    }
    return PreparedCircuit(
        data=raw,
        path=raw_path,
        product_port=product_port,
        product_width=product_width,
        digest=sha256_file(raw_path),
        provenance=provenance,
    )


def generate_multiplier_instances(
    targets: list[dict],
    netlists: dict[str, str],
    product_ports: dict[str, str],
    default_product_port: str,
    out_dir: Path,
) -> list[dict]:
    normalized_targets = normalize_targets(targets)
    widths = {width for _, width, _ in normalized_targets}
    cache = {
        (architecture, width): prepare_circuit(
            architecture,
            width,
            path_template,
            product_ports.get(architecture, default_product_port),
        )
        for architecture, path_template in sorted(netlists.items())
        for width in sorted(widths)
    }
    records = []
    for target_id, width, target in normalized_targets:
        for architecture in sorted(netlists):
            prepared = cache[(architecture, width)]
            if target >= 1 << prepared.product_width:
                raise CircuitError(
                    f"target {target_id} does not fit {architecture} output {prepared.product_port!r}"
                )
            instance_id = f"{target_id}-{architecture}"
            instance_dir = out_dir / f"w{width}" / target_id / architecture
            circuitsat_path = instance_dir / f"{instance_id}.circuitsat.json"
            cnf_path = instance_dir / f"{instance_id}.cnf"
            meta_path = instance_dir / f"{instance_id}.meta.json"
            pinned = pin_port_values(prepared.data, {prepared.product_port: target})
            benchmark_metadata = pinned.setdefault("metadata", {}).setdefault(
                "benchmark", {}
            )
            benchmark_metadata.update(
                {
                    "id": instance_id,
                    "family": "balanced-semiprime-factoring",
                    "architecture": architecture,
                    "target_id": target_id,
                    "target": target,
                    "factor_bits": width,
                    "semantic_task": "circuit-preimage",
                    "expected_outcome": "sat",
                }
            )
            write_json(circuitsat_path, pinned)
            encode_validated_circuit(pinned).write_dimacs(cnf_path)
            metadata = {
                **benchmark_metadata,
                "raw_circuit": prepared.path.name,
                "raw_circuit_sha256": prepared.digest,
                "source_provenance": prepared.provenance,
                "circuitsat": str(circuitsat_path.relative_to(out_dir)),
                "circuitsat_sha256": sha256_file(circuitsat_path),
                "cnf": str(cnf_path.relative_to(out_dir)),
                "cnf_sha256": sha256_file(cnf_path),
            }
            write_json(meta_path, metadata)
            records.append(metadata)
    return records


def rename_expression(expr: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    op, arg = decode_expression(expr)
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
    if not left_bits:
        raise CircuitError("miter output ports must not be empty")
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


def sample_inputs(data: dict, rng: random.Random) -> dict[str, bool]:
    bits = [
        bit
        for port_name in sorted(ports(data, "input"))
        for bit in port_bits(data, port_name, "input")
    ]
    return {bit: bool(rng.getrandbits(1)) for bit in bits}


def generate_preimages(
    data: dict, count: int, seed: int, out_dir: Path, family: str
) -> list[dict]:
    if count < 1:
        raise CircuitError("count must be positive")
    if not ports(data, "input") or not ports(data, "output"):
        raise CircuitError(
            "preimage generation requires input and output port metadata"
        )
    rng = random.Random(seed)
    simulator = CircuitSimulator(data)
    records = []
    seen_outputs: set[tuple[tuple[str, int], ...]] = set()
    draw = 0
    while len(records) < count:
        draw += 1
        if draw > count * 1000:
            raise CircuitError(
                "could not sample enough non-degenerate input/output pairs"
            )
        inputs = sample_inputs(data, rng)
        if not any(inputs.values()):
            continue
        values = simulator.simulate(inputs)
        outputs = port_values(data, values, "output")
        output_bits = [
            values[bit]
            for port_name in sorted(ports(data, "output"))
            for bit in port_bits(data, port_name, "output")
        ]
        if not any(output_bits) or (len(output_bits) > 1 and all(output_bits)):
            continue
        output_key = tuple(sorted(outputs.items()))
        if output_key in seen_outputs:
            continue
        seen_outputs.add(output_key)

        index = len(records)
        instance_id = f"{family}-{index:04d}"
        instance_dir = out_dir / instance_id
        circuitsat_path = instance_dir / f"{instance_id}.circuitsat.json"
        cnf_path = instance_dir / f"{instance_id}.cnf"
        metadata_path = instance_dir / f"{instance_id}.meta.json"
        pinned = pin_port_values(data, outputs)
        write_json(circuitsat_path, pinned)
        encode_validated_circuit(pinned).write_dimacs(cnf_path)
        metadata = {
            "id": instance_id,
            "family": family,
            "semantic_task": "circuit-preimage",
            "expected_outcome": "sat",
            "seed": seed,
            "draw": draw,
            "pinned_outputs": outputs,
            "source_provenance": {
                key: data.get("metadata", {}).get(key)
                for key in (
                    "source_id",
                    "source_revision",
                    "source_verilog_sha256",
                    "yosys_version",
                    "normalization",
                    "generator",
                    "generator_sha256",
                )
                if data.get("metadata", {}).get(key) is not None
            },
            "circuitsat": str(circuitsat_path.relative_to(out_dir)),
            "circuitsat_sha256": sha256_file(circuitsat_path),
            "cnf": str(cnf_path.relative_to(out_dir)),
            "cnf_sha256": sha256_file(cnf_path),
        }
        write_json(metadata_path, metadata)
        records.append(metadata)
    return records
