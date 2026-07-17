"""Shared CircuitSAT helpers for benchmark generation and validation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterable
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def decode_expression(expr: dict[str, Any]) -> tuple[str, Any]:
    try:
        tagged = expr["op"]
    except (KeyError, TypeError) as exc:
        raise CircuitError(f"malformed Boolean expression: {expr!r}") from exc
    if not isinstance(tagged, dict) or len(tagged) != 1:
        raise CircuitError(f"Boolean expression must have one tagged op: {expr!r}")
    return next(iter(tagged.items()))


def expression_variables(expr: dict[str, Any]) -> set[str]:
    op, arg = decode_expression(expr)
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
    op, arg = decode_expression(expr)
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
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CircuitError("metadata must be an object")
    port_specs = metadata.get("ports")
    if port_specs is None:
        return
    if not isinstance(port_specs, dict):
        raise CircuitError("metadata.ports must be an object")
    for name, spec in port_specs.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise CircuitError("every port must have a non-empty name and object spec")
        if spec.get("direction") not in {"input", "output"}:
            raise CircuitError(f"port {name!r} has unsupported direction")
        bits = spec.get("bits")
        if (
            not isinstance(bits, list)
            or not bits
            or not all(isinstance(bit, str) for bit in bits)
        ):
            raise CircuitError(f"port {name!r} must contain non-empty string bits")
        if len(bits) != len(set(bits)):
            raise CircuitError(f"port {name!r} contains duplicate bits")
        unknown_bits = set(bits) - known
        if unknown_bits:
            raise CircuitError(
                f"port {name!r} references unknown variables: "
                + ", ".join(sorted(unknown_bits))
            )
        if spec.get("lsb_first") is not True:
            raise CircuitError(f"port {name!r} does not declare lsb_first=true")


def ports(
    data: dict[str, Any], direction: str | None = None
) -> dict[str, dict[str, Any]]:
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CircuitError("metadata must be an object")
    result = metadata.get("ports", {})
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


class CircuitSimulator:
    """Validate once, then evaluate a circuit in linear dependency order."""

    def __init__(self, data: dict[str, Any]):
        validate_circuit(data)
        self.assignments = data["circuit"]["assignments"]
        self.variables = set(data["variables"])
        self.dependencies = [
            expression_variables(item["expr"]) for item in self.assignments
        ]

    def simulate(self, inputs: dict[str, bool]) -> dict[str, bool]:
        unknown_inputs = set(inputs) - self.variables
        if unknown_inputs:
            raise CircuitError(
                "inputs contain unknown variables: " + ", ".join(sorted(unknown_inputs))
            )
        values = dict(inputs)
        waiting: dict[str, list[int]] = defaultdict(list)
        remaining = []
        ready: deque[int] = deque()
        for index, dependencies in enumerate(self.dependencies):
            missing = dependencies - values.keys()
            remaining.append(len(missing))
            if not missing:
                ready.append(index)
            for name in missing:
                waiting[name].append(index)

        completed = [False] * len(self.assignments)
        completed_count = 0
        while ready:
            index = ready.popleft()
            if completed[index]:
                continue
            item = self.assignments[index]
            result = evaluate_expression(item["expr"], values)
            output = item["outputs"][0]
            if output in values and values[output] != result:
                raise CircuitError(
                    f"assignment {index} contradicts the existing value of {output!r}"
                )
            is_new = output not in values
            values[output] = result
            completed[index] = True
            completed_count += 1
            if is_new:
                for dependent in waiting.pop(output, []):
                    remaining[dependent] -= 1
                    if remaining[dependent] == 0:
                        ready.append(dependent)

        if completed_count != len(self.assignments):
            unresolved = sorted(
                name
                for index, dependencies in enumerate(self.dependencies)
                if not completed[index]
                for name in dependencies
                if name not in values
            )
            raise CircuitError(
                "circuit is cyclic or inputs are missing: "
                + ", ".join(dict.fromkeys(unresolved))
            )
        return values


def simulate(data: dict[str, Any], inputs: dict[str, bool]) -> dict[str, bool]:
    return CircuitSimulator(data).simulate(inputs)


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
