#!/usr/bin/env python3
"""Encode a CircuitSAT JSON instance as deterministic Tseitin DIMACS."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .circuit import CircuitError, circuit_data, load_json
    from .cnf import encode_circuit
except ImportError:  # direct script execution
    from circuit import CircuitError, circuit_data, load_json  # type: ignore
    from cnf import encode_circuit  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        encoded = encode_circuit(circuit_data(load_json(args.input)))
    except CircuitError as exc:
        parser.error(str(exc))
    encoded.write_dimacs(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
