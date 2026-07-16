#!/usr/bin/env python3
"""Pin one or more integer-valued CircuitSAT output ports."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .circuit import (
        CircuitError,
        circuit_data,
        load_json,
        pin_port_values,
        write_json,
    )
except ImportError:  # direct script execution
    from circuit import (  # type: ignore
        CircuitError,
        circuit_data,
        load_json,
        pin_port_values,
        write_json,
    )


def parse_pin(text: str) -> tuple[str, int]:
    try:
        name, raw = text.split("=", 1)
        return name, int(raw, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pin must be PORT=INTEGER") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--pin", type=parse_pin, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        pins = dict(args.pin)
        if len(pins) != len(args.pin):
            raise CircuitError("an output port was pinned more than once")
        write_json(
            args.out,
            pin_port_values(circuit_data(load_json(args.input)), pins),
        )
    except CircuitError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
