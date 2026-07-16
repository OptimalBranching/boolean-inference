#!/usr/bin/env python3
"""Normalize Verilog with Yosys and convert it to structure-preserving CircuitSAT."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from .circuit import CircuitError, load_json, sha256_file, write_json
    from .yosys_json_to_circuitsat import convert
except ImportError:  # direct script execution
    from circuit import CircuitError, load_json, sha256_file, write_json  # type: ignore
    from yosys_json_to_circuitsat import convert  # type: ignore


IDENTIFIER = re.compile(r"^[A-Za-z_\\$][A-Za-z0-9_.$\\]*$")


def yosys_quote(path: Path) -> str:
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def import_verilog(
    source: Path,
    top: str,
    yosys: str,
    keep_yosys_json: Path | None = None,
    source_id: str | None = None,
    source_revision: str | None = None,
    architecture: str | None = None,
) -> dict:
    if not IDENTIFIER.fullmatch(top):
        raise CircuitError(f"unsupported top-module identifier {top!r}")
    if bool(source_id) != bool(source_revision):
        raise CircuitError("source_id and source_revision must be provided together")
    executable = shutil.which(yosys)
    if executable is None:
        raise CircuitError(f"cannot find Yosys executable {yosys!r}")
    if not source.is_file():
        raise CircuitError(f"Verilog source does not exist: {source}")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        yosys_json = temporary / "netlist.json"
        script = temporary / "normalize.ys"
        script.write_text(
            "\n".join(
                [
                    f"read_verilog {yosys_quote(source)}",
                    f"hierarchy -check -top {top}",
                    "proc",
                    "flatten",
                    "techmap",
                    "simplemap",
                    "opt_clean",
                    f"write_json {yosys_quote(yosys_json)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [executable, "-q", "-s", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise CircuitError(
                f"Yosys failed with exit {result.returncode}:\n{result.stderr.strip()}"
            )
        if keep_yosys_json:
            keep_yosys_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(yosys_json, keep_yosys_json)
        data = convert(load_json(yosys_json), top)
    data["metadata"]["source_verilog"] = source.name
    data["metadata"]["source_verilog_sha256"] = sha256_file(source)
    if source_id:
        data["metadata"]["source_id"] = source_id
    if source_revision:
        data["metadata"]["source_revision"] = source_revision
    if architecture:
        data["metadata"]["architecture"] = architecture
    data["metadata"]["normalization"] = [
        "proc",
        "flatten",
        "techmap",
        "simplemap",
        "opt_clean",
    ]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--top", required=True)
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--source-id")
    parser.add_argument("--source-revision")
    parser.add_argument("--architecture")
    parser.add_argument("--keep-yosys-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
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
    except CircuitError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
