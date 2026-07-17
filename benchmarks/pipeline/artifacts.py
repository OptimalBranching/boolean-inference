"""Dataset acquisition, manifests, and artifact validation."""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

from .circuit import (
    CircuitError,
    CircuitSimulator,
    bits_to_int,
    circuit_data,
    int_to_bit_values,
    load_json,
    port_bits,
    ports,
    read_jsonl,
    sha256_file,
    write_json,
)


def fetch_public(url: str, out: Path, expected_sha256: str | None = None) -> str:
    """Download an artifact atomically and record its source and digest."""
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(url) as response:  # noqa: S310 - explicit CLI URL
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise CircuitError(
                f"download digest mismatch: expected {expected_sha256}, got {actual}"
            )
        temporary_path.replace(out)
        write_json(
            out.with_suffix(out.suffix + ".source.json"),
            {"url": url, "sha256": actual, "artifact": out.name},
        )
        return actual
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def collect(root: Path, base: Path | None = None) -> list[dict]:
    records = []
    ids: set[str] = set()
    for metadata_path in sorted(root.rglob("*.meta.json")):
        record = load_json(metadata_path)
        instance_id = record.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise CircuitError(f"{metadata_path}: missing instance id")
        if instance_id in ids:
            raise CircuitError(f"duplicate instance id {instance_id!r}")
        ids.add(instance_id)
        for key in ("circuitsat", "cnf"):
            relative = record.get(key)
            expected = record.get(f"{key}_sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise CircuitError(f"{metadata_path}: missing {key} artifact or digest")
            artifact = root / relative
            actual = sha256_file(artifact)
            if actual != expected:
                raise CircuitError(f"{metadata_path}: {key} digest mismatch")
            if base is not None:
                try:
                    record[key] = str(artifact.relative_to(base))
                except ValueError as exc:
                    raise CircuitError(
                        f"{artifact}: artifact is outside manifest base {base}"
                    ) from exc
        records.append(record)
    if not records:
        raise CircuitError(f"{root}: no *.meta.json records found")
    return sorted(records, key=lambda item: item["id"])


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


def validate_multiplier_witnesses(manifest_path: Path, oracle_path: Path, raw_dir: Path) -> int:
    oracle: dict[str, dict] = {}
    for record in read_jsonl(oracle_path):
        target_id = record.get("id")
        if not isinstance(target_id, str) or not target_id:
            raise CircuitError(
                f"{oracle_path}: every oracle record needs a non-empty id"
            )
        if target_id in oracle:
            raise CircuitError(f"{oracle_path}: duplicate target id {target_id!r}")
        oracle[target_id] = record
    if not oracle:
        raise CircuitError(f"{oracle_path}: no private factors found")

    prepared: dict[
        Path, tuple[CircuitSimulator, list[str], list[str], list[str]]
    ] = {}
    checked = 0
    for record in read_jsonl(manifest_path):
        target_id = record.get("target_id")
        witness = oracle.get(target_id)
        if witness is None:
            raise CircuitError(f"missing private factors for target {target_id!r}")
        try:
            expected = int(record["target"])
            left = int(witness["left_factor"])
            right = int(witness["right_factor"])
            raw_path = raw_dir / record["raw_circuit"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CircuitError(
                f"malformed manifest or oracle record for {target_id!r}"
            ) from exc
        if left * right != expected:
            raise CircuitError(
                f"private factors do not multiply to target {target_id!r}"
            )

        if raw_path not in prepared:
            data = circuit_data(load_json(raw_path))
            input_ports = ports(data, "input")
            output_ports = ports(data, "output")
            if len(input_ports) != 2 or len(output_ports) != 1:
                raise CircuitError(
                    f"{raw_path}: expected exactly two input ports and one output port"
                )
            inputs = [port_bits(data, name, "input") for name in sorted(input_ports)]
            outputs = port_bits(data, next(iter(output_ports)), "output")
            prepared[raw_path] = (
                CircuitSimulator(data),
                inputs[0],
                inputs[1],
                outputs,
            )

        simulator, left_bits, right_bits, output_bits = prepared[raw_path]
        values = int_to_bit_values(left_bits, left)
        values.update(int_to_bit_values(right_bits, right))
        result = simulator.simulate(values)
        actual = bits_to_int(output_bits, result)
        if actual != expected:
            raise CircuitError(
                f"{raw_path}: witness for {target_id!r} produced {actual}, expected {expected}"
            )
        checked += 1
    if checked == 0:
        raise CircuitError(f"{manifest_path}: no instances found")
    return checked
