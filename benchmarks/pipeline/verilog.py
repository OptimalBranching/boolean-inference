"""Convert simplemapped Yosys netlists and import Verilog as CircuitSAT."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .circuit import (
    CircuitError,
    assignment,
    const,
    load_json,
    nary,
    sha256_file,
    unary,
    validate_circuit,
    var,
)

Bit = int | str


def choose_module(
    document: dict[str, Any], requested: str | None
) -> tuple[str, dict[str, Any]]:
    modules = document.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise CircuitError("Yosys JSON contains no modules")
    if requested is not None:
        if requested not in modules:
            raise CircuitError(f"Yosys JSON contains no module {requested!r}")
        return requested, modules[requested]
    if len(modules) != 1:
        raise CircuitError("Yosys JSON has multiple modules; pass --module")
    return next(iter(modules.items()))


def all_integer_bits(module: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for port in module.get("ports", {}).values():
        result.update(bit for bit in port.get("bits", []) if isinstance(bit, int))
    for cell in module.get("cells", {}).values():
        for connection in cell.get("connections", {}).values():
            result.update(bit for bit in connection if isinstance(bit, int))
    return result


def bit_names(module: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    for port_name, port in sorted(module.get("ports", {}).items()):
        for index, bit in enumerate(port.get("bits", [])):
            if isinstance(bit, int):
                names.setdefault(bit, f"{port_name}[{index}]")
    for net_name, net in sorted(module.get("netnames", {}).items()):
        if net.get("hide_name"):
            continue
        for index, bit in enumerate(net.get("bits", [])):
            if isinstance(bit, int):
                candidate = (
                    net_name
                    if len(net.get("bits", [])) == 1
                    else f"{net_name}[{index}]"
                )
                names.setdefault(bit, candidate)
    for bit in sorted(all_integer_bits(module)):
        names.setdefault(bit, f"wire${bit}")
    if len(set(names.values())) != len(names):
        raise CircuitError("Yosys bit naming produced duplicate variable names")
    return names


def expression_for_bit(bit: Bit, names: dict[int, str]) -> dict[str, Any]:
    if isinstance(bit, int):
        return var(names[bit])
    if bit == "0":
        return const(False)
    if bit == "1":
        return const(True)
    raise CircuitError(f"unsupported Yosys constant bit {bit!r}")


def one_connection(cell_name: str, cell: dict[str, Any], port: str) -> Bit:
    bits = cell.get("connections", {}).get(port)
    if not isinstance(bits, list) or len(bits) != 1:
        raise CircuitError(
            f"cell {cell_name!r} port {port!r} is not single-bit; run techmap; simplemap first"
        )
    return bits[0]


def cell_expression(
    cell_name: str, cell: dict[str, Any], names: dict[int, str]
) -> tuple[int, dict[str, Any]]:
    directions = cell.get("port_directions", {})
    output_ports = [
        name for name, direction in directions.items() if direction == "output"
    ]
    if len(output_ports) != 1:
        raise CircuitError(f"cell {cell_name!r} must have exactly one output port")
    output_bit = one_connection(cell_name, cell, output_ports[0])
    if not isinstance(output_bit, int):
        raise CircuitError(f"cell {cell_name!r} drives a constant")

    def value(port: str) -> dict[str, Any]:
        return expression_for_bit(one_connection(cell_name, cell, port), names)

    kind = cell.get("type")
    if kind in {"$_BUF_", "$buf"}:
        expr = value("A")
    elif kind in {"$_NOT_", "$not", "$logic_not"}:
        expr = unary("Not", value("A"))
    elif kind in {"$_AND_", "$and", "$logic_and"}:
        expr = nary("And", [value("A"), value("B")])
    elif kind in {"$_OR_", "$or", "$logic_or"}:
        expr = nary("Or", [value("A"), value("B")])
    elif kind in {"$_XOR_", "$xor"}:
        expr = nary("Xor", [value("A"), value("B")])
    elif kind in {"$_XNOR_", "$xnor"}:
        expr = unary("Not", nary("Xor", [value("A"), value("B")]))
    elif kind == "$_NAND_":
        expr = unary("Not", nary("And", [value("A"), value("B")]))
    elif kind == "$_NOR_":
        expr = unary("Not", nary("Or", [value("A"), value("B")]))
    elif kind == "$_ANDNOT_":
        expr = nary("And", [value("A"), unary("Not", value("B"))])
    elif kind == "$_ORNOT_":
        expr = nary("Or", [value("A"), unary("Not", value("B"))])
    elif kind in {"$_MUX_", "$mux"}:
        select = value("S")
        expr = nary(
            "Or",
            [
                nary("And", [unary("Not", select), value("A")]),
                nary("And", [select, value("B")]),
            ],
        )
    else:
        raise CircuitError(
            f"unsupported Yosys cell type {kind!r} in {cell_name!r}; normalize with techmap; simplemap"
        )
    return output_bit, expr


def convert(document: dict[str, Any], module_name: str | None = None) -> dict[str, Any]:
    selected_name, module = choose_module(document, module_name)
    names = bit_names(module)
    assignments = []
    driven: set[int] = set()
    for cell_name, cell in sorted(module.get("cells", {}).items()):
        if cell.get("type") == "$scopeinfo":
            continue
        output_bit, expr = cell_expression(cell_name, cell, names)
        if output_bit in driven:
            raise CircuitError(f"multiple cells drive {names[output_bit]!r}")
        driven.add(output_bit)
        assignments.append(assignment(names[output_bit], expr))

    port_metadata: dict[str, Any] = {}
    for port_name, port in sorted(module.get("ports", {}).items()):
        bits = port.get("bits", [])
        if any(not isinstance(bit, int) for bit in bits):
            raise CircuitError(f"top-level port {port_name!r} contains constant bits")
        port_metadata[port_name] = {
            "direction": port.get("direction"),
            "bits": [names[bit] for bit in bits],
            "lsb_first": True,
        }

    result = {
        "circuit": {"assignments": assignments},
        "variables": [names[bit] for bit in sorted(names)],
        "metadata": {
            "format": "circuitsat-benchmark-v1",
            "source_format": "yosys-json",
            "module": selected_name,
            "ports": port_metadata,
        },
    }
    validate_circuit(result)
    return result


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
    version_result = subprocess.run(
        [executable, "-V"], capture_output=True, text=True, check=False
    )
    if version_result.returncode != 0:
        raise CircuitError(
            f"cannot query Yosys version: {version_result.stderr.strip()}"
        )
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        yosys_json = temporary / "netlist.json"
        script = temporary / "normalize.ys"
        script.write_text(
            "\n".join(
                [
                    f"read_verilog -sv {yosys_quote(source)}",
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
    data["metadata"]["yosys_version"] = version_result.stdout.strip()
    return data
