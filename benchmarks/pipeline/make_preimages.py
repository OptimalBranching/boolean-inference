#!/usr/bin/env python3
"""Generate deterministic SAT preimages from an unpinned combinational circuit."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    from .circuit import (
        CircuitSimulator,
        CircuitError,
        circuit_data,
        load_json,
        pin_port_values,
        port_bits,
        port_values,
        ports,
        sha256_file,
        write_json,
    )
    from .cnf import encode_validated_circuit
except ImportError:  # direct script execution
    from circuit import (  # type: ignore
        CircuitSimulator,
        CircuitError,
        circuit_data,
        load_json,
        pin_port_values,
        port_bits,
        port_values,
        ports,
        sha256_file,
        write_json,
    )
    from cnf import encode_validated_circuit  # type: ignore


def sample_inputs(data: dict, rng: random.Random) -> dict[str, bool]:
    bits = [
        bit
        for port_name in sorted(ports(data, "input"))
        for bit in port_bits(data, port_name, "input")
    ]
    return {bit: bool(rng.getrandbits(1)) for bit in bits}


def generate(
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        generate(
            circuit_data(load_json(args.input)),
            args.count,
            args.seed,
            args.out_dir,
            args.family,
        )
    except CircuitError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
