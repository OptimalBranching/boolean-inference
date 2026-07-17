#!/usr/bin/env python3
"""Unified command line for benchmark generation and validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .artifacts import (
    collect,
    fetch_public,
    validate_dimacs,
    validate_multiplier_witnesses,
)
from .circuit import (
    CircuitError,
    circuit_data,
    load_json,
    read_jsonl,
    validate_circuit,
    write_json,
    write_jsonl,
)
from .cnf import encode_circuit
from .instances import (
    REQUIRED_ARCHITECTURES,
    build_miter,
    generate_multiplier_instances,
    generate_preimages,
)
from .multipliers import generate_multiplier, records
from .verilog import import_verilog


def key_value(text: str) -> tuple[str, str]:
    try:
        key, value = text.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=VALUE") from exc
    if not key or not value:
        raise argparse.ArgumentTypeError("expected non-empty NAME=VALUE")
    return key, value


def command_targets(args: argparse.Namespace) -> None:
    generated = list(records(args.width, args.count, args.seed_base))
    write_jsonl(args.out, [public for public, _ in generated])
    if args.oracle_out:
        write_jsonl(args.oracle_out, [oracle for _, oracle in generated])


def command_multiplier(args: argparse.Namespace) -> None:
    write_json(
        args.out, generate_multiplier(args.bits, args.architecture, args.base_case)
    )


def command_factor(args: argparse.Namespace) -> None:
    netlists = dict(args.netlist)
    if len(netlists) != len(args.netlist):
        raise CircuitError("an architecture was declared more than once")
    if args.require_all:
        missing = REQUIRED_ARCHITECTURES - set(netlists)
        if missing:
            raise CircuitError(
                "missing required architectures: " + ", ".join(sorted(missing))
            )
    generated = generate_multiplier_instances(
        read_jsonl(args.targets),
        netlists,
        dict(args.product_port),
        args.default_product_port,
        args.out_dir,
    )
    write_jsonl(
        args.out_dir / "manifest.jsonl",
        sorted(generated, key=lambda item: item["id"]),
    )


def command_preimages(args: argparse.Namespace) -> None:
    generate_preimages(
        circuit_data(load_json(args.input)),
        args.count,
        args.seed,
        args.out_dir,
        args.family,
    )


def command_miter(args: argparse.Namespace) -> None:
    input_map = dict(args.input_map)
    if len(input_map) != len(args.input_map):
        raise CircuitError("an input port was mapped more than once")
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


def command_cnf(args: argparse.Namespace) -> None:
    encode_circuit(circuit_data(load_json(args.input))).write_dimacs(args.out)


def command_import_verilog(args: argparse.Namespace) -> None:
    write_json(
        args.out,
        import_verilog(
            args.input,
            args.top,
            args.yosys,
            args.keep_yosys_json,
            args.source_id,
            args.source_revision,
            args.architecture,
        ),
    )


def command_fetch(args: argparse.Namespace) -> None:
    print(fetch_public(args.url, args.out, args.sha256))


def command_manifest(args: argparse.Namespace) -> None:
    generated = [
        record
        for root in args.root
        for record in collect(root, args.base or args.out.parent)
    ]
    if len({record["id"] for record in generated}) != len(generated):
        raise CircuitError("duplicate instance id across manifest roots")
    write_jsonl(args.out, sorted(generated, key=lambda item: item["id"]))


def command_validate(args: argparse.Namespace) -> None:
    if args.circuitsat:
        data = circuit_data(load_json(args.circuitsat))
        validate_circuit(data)
        print(
            f"PASS CircuitSAT: {len(data['variables'])} variables, "
            f"{len(data['circuit']['assignments'])} assignments"
        )
    if args.cnf:
        variables, clauses, width = validate_dimacs(
            args.cnf, args.max_clause_width or None
        )
        print(f"PASS DIMACS: {variables} variables, {clauses} clauses, max width {width}")
    if args.solver:
        result = subprocess.run([str(args.solver), str(args.cnf)], check=False)
        expected = 10 if args.expect == "sat" else 20 if args.expect == "unsat" else None
        if expected is not None and result.returncode != expected:
            raise CircuitError(
                f"solver returned {result.returncode}, expected {expected} ({args.expect})"
            )
        print(f"PASS solver: exit {result.returncode}")


def command_witnesses(args: argparse.Namespace) -> None:
    checked = validate_multiplier_witnesses(args.manifest, args.oracle, args.raw_dir)
    print(f"PASS multiplication witnesses: {checked} instances")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("targets", help="generate deterministic semiprimes")
    command.add_argument("--width", type=int, action="append", required=True)
    command.add_argument("--count", type=int, required=True)
    command.add_argument("--seed-base", type=int, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--oracle-out", type=Path)
    command.set_defaults(handler=command_targets)

    command = commands.add_parser("multiplier", help="generate Array or Karatsuba")
    command.add_argument("--bits", type=int, required=True)
    command.add_argument("--architecture", choices=("array-ripple", "karatsuba"), required=True)
    command.add_argument("--base-case", type=int, default=4)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(handler=command_multiplier)

    command = commands.add_parser("factor", help="pin targets across multipliers")
    command.add_argument("--targets", type=Path, required=True)
    command.add_argument("--netlist", type=key_value, action="append", required=True)
    command.add_argument("--product-port", type=key_value, action="append", default=[])
    command.add_argument("--default-product-port", default="product")
    command.add_argument("--require-all", action="store_true")
    command.add_argument("--out-dir", type=Path, required=True)
    command.set_defaults(handler=command_factor)

    command = commands.add_parser("preimages", help="generate circuit preimages")
    command.add_argument("input", type=Path)
    command.add_argument("--family", required=True)
    command.add_argument("--count", type=int, required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--out-dir", type=Path, required=True)
    command.set_defaults(handler=command_preimages)

    command = commands.add_parser("miter", help="build an equivalence miter")
    command.add_argument("left", type=Path)
    command.add_argument("right", type=Path)
    command.add_argument("--left-output", required=True)
    command.add_argument("--right-output", required=True)
    command.add_argument("--input-map", type=key_value, action="append", default=[])
    command.add_argument("--bit", type=int)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(handler=command_miter)

    command = commands.add_parser("cnf", help="encode CircuitSAT as DIMACS")
    command.add_argument("input", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(handler=command_cnf)

    command = commands.add_parser("import-verilog", help="normalize Verilog with Yosys")
    command.add_argument("input", type=Path)
    command.add_argument("--top", required=True)
    command.add_argument("--yosys", default="yosys")
    command.add_argument("--source-id")
    command.add_argument("--source-revision")
    command.add_argument("--architecture")
    command.add_argument("--keep-yosys-json", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(handler=command_import_verilog)

    command = commands.add_parser("fetch", help="download and hash a public artifact")
    command.add_argument("url")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--sha256")
    command.set_defaults(handler=command_fetch)

    command = commands.add_parser("manifest", help="collect artifact metadata")
    command.add_argument("root", type=Path, nargs="+")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--base", type=Path)
    command.set_defaults(handler=command_manifest)

    command = commands.add_parser("validate", help="validate generated artifacts")
    command.add_argument("--circuitsat", type=Path)
    command.add_argument("--cnf", type=Path)
    command.add_argument("--solver", type=Path)
    command.add_argument("--expect", choices=("sat", "unsat"))
    command.add_argument("--max-clause-width", type=int, default=20)
    command.set_defaults(handler=command_validate)

    command = commands.add_parser("witnesses", help="validate private factors")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--oracle", type=Path, required=True)
    command.add_argument("--raw-dir", type=Path, required=True)
    command.set_defaults(handler=command_witnesses)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        if not args.circuitsat and not args.cnf:
            parser.error("validate needs --circuitsat and/or --cnf")
        if args.solver and not args.cnf:
            parser.error("--solver requires --cnf")
        if args.expect and not args.solver:
            parser.error("--expect requires --solver")
    try:
        args.handler(args)
    except (CircuitError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
