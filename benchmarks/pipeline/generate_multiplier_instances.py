#!/usr/bin/env python3
"""Pin matched factoring targets across a set of raw multiplier circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .circuit import (
        CircuitError,
        circuit_data,
        load_json,
        pin_port_values,
        port_bits,
        ports,
        sha256_file,
        validate_circuit,
        write_json,
    )
    from .cnf import encode_circuit
except ImportError:  # direct script execution
    from circuit import (  # type: ignore
        CircuitError,
        circuit_data,
        load_json,
        pin_port_values,
        port_bits,
        ports,
        sha256_file,
        validate_circuit,
        write_json,
    )
    from cnf import encode_circuit  # type: ignore


REQUIRED_ARCHITECTURES = {
    "array-ripple",
    "wallace-ripple",
    "dadda-ripple",
    "booth-r4-wallace",
    "booth-r4-dadda",
    "karatsuba",
}


def key_value(text: str) -> tuple[str, str]:
    try:
        key, value = text.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=VALUE") from exc
    if not key or not value:
        raise argparse.ArgumentTypeError("expected non-empty NAME=VALUE")
    return key, value


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


def generate(
    targets: list[dict],
    netlists: dict[str, str],
    product_ports: dict[str, str],
    default_product_port: str,
    out_dir: Path,
) -> list[dict]:
    records = []
    cache: dict[tuple[str, int], tuple[dict, Path]] = {}
    for target_record in targets:
        try:
            target_id = str(target_record["id"])
            width = int(target_record["factor_bits"])
            target = int(target_record["target"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CircuitError(f"malformed target record: {target_record!r}") from exc
        for architecture in sorted(netlists):
            cache_key = (architecture, width)
            if cache_key not in cache:
                raw_path = Path(netlists[architecture].format(bits=width))
                raw = circuit_data(load_json(raw_path))
                validate_circuit(raw)
                cache[cache_key] = (raw, raw_path)
            raw, raw_path = cache[cache_key]
            raw_metadata = raw.get("metadata", {})
            declared_architecture = raw_metadata.get("architecture")
            if (
                declared_architecture is not None
                and declared_architecture != architecture
            ):
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
            product_port = product_ports.get(architecture, default_product_port)
            product_width = len(port_bits(raw, product_port, "output"))
            if product_width != width * 2:
                raise CircuitError(
                    f"{raw_path}: product port must be exactly {width * 2} bits, got {product_width}"
                )
            if target >= 1 << product_width:
                raise CircuitError(
                    f"target {target_id} does not fit {architecture} output {product_port!r}"
                )
            instance_id = f"{target_id}-{architecture}"
            instance_dir = out_dir / f"w{width}" / target_id / architecture
            circuitsat_path = instance_dir / f"{instance_id}.circuitsat.json"
            cnf_path = instance_dir / f"{instance_id}.cnf"
            meta_path = instance_dir / f"{instance_id}.meta.json"
            pinned = pin_port_values(raw, {product_port: target})
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
            cnf_path.parent.mkdir(parents=True, exist_ok=True)
            cnf_path.write_text(encode_circuit(pinned).dimacs(), encoding="utf-8")
            metadata = {
                **benchmark_metadata,
                "raw_circuit": raw_path.name,
                "raw_circuit_sha256": sha256_file(raw_path),
                "source_provenance": {
                    key: raw.get("metadata", {}).get(key)
                    for key in (
                        "source_id",
                        "source_revision",
                        "source_verilog_sha256",
                        "generator",
                        "generator_sha256",
                    )
                    if raw.get("metadata", {}).get(key) is not None
                },
                "circuitsat": str(circuitsat_path.relative_to(out_dir)),
                "circuitsat_sha256": sha256_file(circuitsat_path),
                "cnf": str(cnf_path.relative_to(out_dir)),
                "cnf_sha256": sha256_file(cnf_path),
            }
            write_json(meta_path, metadata)
            records.append(metadata)
    return records


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in sorted(records, key=lambda item: item["id"])
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--netlist",
        type=key_value,
        action="append",
        required=True,
        metavar="ARCH=PATH_TEMPLATE",
        help="PATH_TEMPLATE may contain {bits}",
    )
    parser.add_argument("--product-port", type=key_value, action="append", default=[])
    parser.add_argument("--default-product-port", default="product")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    netlists = dict(args.netlist)
    if len(netlists) != len(args.netlist):
        parser.error("an architecture was declared more than once")
    if args.require_all:
        missing = REQUIRED_ARCHITECTURES - set(netlists)
        if missing:
            parser.error(
                "missing required architectures: " + ", ".join(sorted(missing))
            )
    try:
        records = generate(
            read_jsonl(args.targets),
            netlists,
            dict(args.product_port),
            args.default_product_port,
            args.out_dir,
        )
        write_manifest(args.out_dir / "manifest.jsonl", records)
    except (CircuitError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
