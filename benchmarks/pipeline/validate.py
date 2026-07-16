#!/usr/bin/env python3
"""Validate generated CircuitSAT/DIMACS artifacts and optionally check a verdict."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from .circuit import CircuitError, circuit_data, load_json, validate_circuit
except ImportError:  # direct script execution
    from circuit import CircuitError, circuit_data, load_json, validate_circuit  # type: ignore


def validate_dimacs(
    path: Path, maximum_clause_width: int | None = None
) -> tuple[int, int, int]:
    declared_vars = declared_clauses = None
    clauses = 0
    max_width = 0
    max_variable = 0
    current = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p "):
                parts = line.split()
                if declared_vars is not None or len(parts) != 4 or parts[1] != "cnf":
                    raise CircuitError(f"{path}: malformed DIMACS header")
                declared_vars, declared_clauses = int(parts[2]), int(parts[3])
                if declared_vars < 0 or declared_clauses < 0:
                    raise CircuitError(f"{path}: negative DIMACS header count")
                continue
            if declared_vars is None:
                raise CircuitError(f"{path}: clause appears before DIMACS header")
            for token in line.split():
                literal = int(token)
                if literal == 0:
                    clauses += 1
                    max_width = max(max_width, len(current))
                    current = []
                else:
                    current.append(literal)
                    max_variable = max(max_variable, abs(literal))
    if current:
        raise CircuitError(f"{path}: final clause is missing terminator 0")
    if declared_vars is None or declared_clauses is None:
        raise CircuitError(f"{path}: missing DIMACS header")
    if clauses != declared_clauses:
        raise CircuitError(
            f"{path}: header declares {declared_clauses} clauses, found {clauses}"
        )
    if max_variable > declared_vars:
        raise CircuitError(
            f"{path}: literal references variable {max_variable} beyond declared {declared_vars}"
        )
    if maximum_clause_width is not None and max_width > maximum_clause_width:
        raise CircuitError(
            f"{path}: clause width {max_width} exceeds supported limit {maximum_clause_width}"
        )
    return declared_vars, clauses, max_width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuitsat", type=Path)
    parser.add_argument("--cnf", type=Path)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--expect", choices=("sat", "unsat"))
    parser.add_argument(
        "--max-clause-width",
        type=int,
        default=20,
        help="current boolean-inference DIMACS limit; use 0 to disable",
    )
    args = parser.parse_args()
    if not args.circuitsat and not args.cnf:
        parser.error("pass --circuitsat and/or --cnf")
    try:
        if args.circuitsat:
            data = circuit_data(load_json(args.circuitsat))
            validate_circuit(data)
            print(
                f"PASS CircuitSAT: {len(data['variables'])} variables, "
                f"{len(data['circuit']['assignments'])} assignments"
            )
        if args.cnf:
            variables, clauses, max_width = validate_dimacs(
                args.cnf, args.max_clause_width or None
            )
            print(
                f"PASS DIMACS: {variables} variables, {clauses} clauses, max width {max_width}"
            )
        if args.expect and not args.solver:
            parser.error("--expect requires --solver")
        if args.solver:
            if not args.cnf:
                parser.error("--solver requires --cnf")
            result = subprocess.run([str(args.solver), str(args.cnf)], check=False)
            expected_code = (
                10 if args.expect == "sat" else 20 if args.expect == "unsat" else None
            )
            if expected_code is not None and result.returncode != expected_code:
                raise CircuitError(
                    f"solver returned {result.returncode}, expected {expected_code} ({args.expect})"
                )
            print(f"PASS solver: exit {result.returncode}")
    except (CircuitError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
