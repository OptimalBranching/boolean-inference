"""Shared CircuitSAT helpers for benchmark generation and validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


class CircuitError(ValueError):
    """A circuit document is malformed or cannot be evaluated."""


def var(name: str) -> dict[str, Any]:
    return {"op": {"Var": name}}


def const(value: bool) -> dict[str, Any]:
    return {"op": {"Const": bool(value)}}


def unary(op: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"op": {op: value}}


def nary(op: str, values: list[dict[str, Any]]) -> dict[str, Any]:
    return {"op": {op: values}}


def assignment(output: str, expr: dict[str, Any]) -> dict[str, Any]:
    return {"outputs": [output], "expr": expr}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CircuitError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CircuitError(f"{path} must contain one JSON object")
    return value


def circuit_data(document: dict[str, Any]) -> dict[str, Any]:
    """Extract and copy the bare ``{circuit, variables}`` CircuitSAT data."""
    value: Any = document.get("target", document)
    if isinstance(value, dict):
        value = value.get("data", value)
    if (
        not isinstance(value, dict)
        or "circuit" not in value
        or "variables" not in value
    ):
        raise CircuitError("JSON has no CircuitSAT circuit/variables data")
    return copy.deepcopy(value)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decode(expr: dict[str, Any]) -> tuple[str, Any]:
    try:
        tagged = expr["op"]
    except (KeyError, TypeError) as exc:
        raise CircuitError(f"malformed Boolean expression: {expr!r}") from exc
    if not isinstance(tagged, dict) or len(tagged) != 1:
        raise CircuitError(f"Boolean expression must have one tagged op: {expr!r}")
    return next(iter(tagged.items()))


def expression_variables(expr: dict[str, Any]) -> set[str]:
    op, arg = _decode(expr)
    if op == "Var":
        if not isinstance(arg, str):
            raise CircuitError("Var payload must be a string")
        return {arg}
    if op == "Const":
        if not isinstance(arg, bool):
            raise CircuitError("Const payload must be boolean")
        return set()
    if op == "Not":
        return expression_variables(arg)
    if op in {"And", "Or", "Xor"}:
        if not isinstance(arg, list):
            raise CircuitError(f"{op} payload must be a list")
        result: set[str] = set()
        for child in arg:
            result.update(expression_variables(child))
        return result
    raise CircuitError(f"unsupported Boolean operation {op!r}")


def evaluate_expression(expr: dict[str, Any], values: dict[str, bool]) -> bool:
    op, arg = _decode(expr)
    if op == "Var":
        if arg not in values:
            raise KeyError(arg)
        return values[arg]
    if op == "Const":
        return arg
    if op == "Not":
        return not evaluate_expression(arg, values)
    if op == "And":
        return all(evaluate_expression(child, values) for child in arg)
    if op == "Or":
        return any(evaluate_expression(child, values) for child in arg)
    if op == "Xor":
        parity = False
        for child in arg:
            parity ^= evaluate_expression(child, values)
        return parity
    raise CircuitError(f"unsupported Boolean operation {op!r}")


def validate_circuit(data: dict[str, Any]) -> None:
    variables = data.get("variables")
    assignments = data.get("circuit", {}).get("assignments")
    if not isinstance(variables, list) or not all(
        isinstance(item, str) for item in variables
    ):
        raise CircuitError("variables must be a list of strings")
    if len(variables) != len(set(variables)):
        raise CircuitError("variables contains duplicate names")
    if not isinstance(assignments, list):
        raise CircuitError("circuit.assignments must be a list")
    known = set(variables)
    for index, item in enumerate(assignments):
        outputs = item.get("outputs") if isinstance(item, dict) else None
        if (
            not isinstance(outputs, list)
            or len(outputs) != 1
            or not isinstance(outputs[0], str)
        ):
            raise CircuitError(
                f"assignment {index} must have exactly one string output"
            )
        if outputs[0] not in known:
            raise CircuitError(
                f"assignment {index} writes unknown variable {outputs[0]!r}"
            )
        unknown = expression_variables(item.get("expr")) - known
        if unknown:
            raise CircuitError(
                f"assignment {index} references unknown variables: {', '.join(sorted(unknown))}"
            )


def ports(
    data: dict[str, Any], direction: str | None = None
) -> dict[str, dict[str, Any]]:
    result = data.get("metadata", {}).get("ports", {})
    if not isinstance(result, dict):
        raise CircuitError("metadata.ports must be an object")
    if direction is None:
        return result
    return {
        name: spec
        for name, spec in result.items()
        if isinstance(spec, dict) and spec.get("direction") == direction
    }


def port_bits(
    data: dict[str, Any], name: str, direction: str | None = None
) -> list[str]:
    spec = ports(data).get(name)
    if not isinstance(spec, dict):
        raise CircuitError(f"unknown circuit port {name!r}")
    if direction is not None and spec.get("direction") != direction:
        raise CircuitError(f"port {name!r} is not an {direction} port")
    bits = spec.get("bits")
    if not isinstance(bits, list) or not all(isinstance(bit, str) for bit in bits):
        raise CircuitError(f"port {name!r} has malformed bit metadata")
    if spec.get("lsb_first") is not True:
        raise CircuitError(f"port {name!r} does not declare lsb_first=true")
    return bits


def bits_to_int(bits: list[str], values: dict[str, bool]) -> int:
    result = 0
    for index, name in enumerate(bits):
        if values[name]:
            result |= 1 << index
    return result


def int_to_bit_values(bits: list[str], value: int) -> dict[str, bool]:
    if value < 0 or value >= 1 << len(bits):
        raise CircuitError(f"value {value} does not fit in {len(bits)} bits")
    return {name: bool((value >> index) & 1) for index, name in enumerate(bits)}


def simulate(data: dict[str, Any], inputs: dict[str, bool]) -> dict[str, bool]:
    """Evaluate an acyclic CircuitSAT document and check duplicate constraints."""
    validate_circuit(data)
    values = dict(inputs)
    assignments = list(data["circuit"]["assignments"])
    pending = list(enumerate(assignments))
    while pending:
        progress = False
        deferred = []
        for index, item in pending:
            try:
                result = evaluate_expression(item["expr"], values)
            except KeyError:
                deferred.append((index, item))
                continue
            output = item["outputs"][0]
            if output in values and values[output] != result:
                raise CircuitError(
                    f"assignment {index} contradicts the existing value of {output!r}"
                )
            values[output] = result
            progress = True
        if not progress:
            unresolved = sorted(
                {
                    name
                    for _, item in deferred
                    for name in expression_variables(item["expr"])
                    if name not in values
                }
            )
            raise CircuitError(
                "circuit is cyclic or inputs are missing: " + ", ".join(unresolved)
            )
        pending = deferred
    return values


def pin_port_values(data: dict[str, Any], values: dict[str, int]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    pins: dict[str, bool] = {}
    for port_name, value in values.items():
        bits = port_bits(result, port_name, "output")
        pins.update(int_to_bit_values(bits, value))
    result["circuit"]["assignments"].extend(
        assignment(name, const(value)) for name, value in sorted(pins.items())
    )
    metadata = result.setdefault("metadata", {})
    metadata["pinned_outputs"] = dict(sorted(values.items()))
    validate_circuit(result)
    return result


def port_values(
    data: dict[str, Any], values: dict[str, bool], direction: str
) -> dict[str, int]:
    return {
        name: bits_to_int(port_bits(data, name, direction), values)
        for name in sorted(ports(data, direction))
    }
