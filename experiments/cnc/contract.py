#!/usr/bin/env python3
"""Audit and fingerprint the frozen Cube-and-Conquer study contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised in a clean environment
    raise SystemExit(
        "PyYAML is required; install experiments/requirements.txt"
    ) from exc


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = EXPERIMENTS_DIR / "cnc-study.schema.json"
UNRESOLVED = re.compile(r"^(?:todo|tbd|fixme|unresolved|placeholder)$", re.I)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A contract or lock file is malformed."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(loader: UniqueKeySafeLoader, node: Any, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ContractError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = load_yaml_text(path.read_text(encoding="utf-8"), str(path))
    except (OSError, yaml.YAMLError, ContractError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    return value


def load_yaml_text(text: str, source: str) -> dict[str, Any]:
    value = yaml.load(text, Loader=UniqueKeySafeLoader)
    if not isinstance(value, dict):
        raise ContractError(f"{source} must contain one YAML mapping")
    return value


def canonical_bytes(contract: dict[str, Any]) -> bytes:
    """Return the canonical JSON-v1 representation used by the lock file."""
    try:
        rendered = json.dumps(
            contract,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"contract is not canonical-JSON compatible: {exc}") from exc
    return rendered.encode("utf-8")


def digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(contract)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ContractError(f"schema uses unsupported external reference {ref!r}")
    value: Any = root
    for component in ref[2:].split("/"):
        value = value[component.replace("~1", "/").replace("~0", "~")]
    return value


def _type_matches(value: Any, expected: str) -> bool:
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    return checks[expected](value)


def schema_errors(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str = "$",
) -> list[str]:
    """Validate the JSON Schema subset used by the checked-in contract schema.

    Keeping this small validator beside the schema lets the issue's exact audit
    command remain dependency-light. The schema itself is Draft 2020-12 and can
    additionally be checked by any full JSON Schema implementation.
    """
    if "$ref" in schema:
        return schema_errors(value, _resolve_ref(root, schema["$ref"]), root, path)

    errors: list[str] = []
    expected = schema.get("type")
    expected_types = [expected] if isinstance(expected, str) else expected
    if expected_types and not any(_type_matches(value, item) for item in expected_types):
        return [f"{path}: expected {' or '.join(expected_types)}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value.keys() - properties.keys():
                errors.append(f"{path}: unknown field {key!r}")
        for key, child in value.items():
            if key in properties:
                errors.extend(schema_errors(child, properties[key], root, f"{path}.{key}"))

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if schema.get("uniqueItems"):
            seen: set[str] = set()
            for item in value:
                marker = json.dumps(item, sort_keys=True, separators=(",", ":"))
                if marker in seen:
                    errors.append(f"{path}: duplicate array item {item!r}")
                seen.add(marker)
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, item_schema, root, f"{path}[{index}]"))

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern and re.search(pattern, value) is None:
            errors.append(f"{path}: does not match {pattern!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")

    return errors


def unresolved_values(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if value is None:
        errors.append(f"{path}: null is unresolved")
    elif isinstance(value, str) and UNRESOLVED.fullmatch(value.strip()):
        errors.append(f"{path}: unresolved placeholder {value!r}")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(unresolved_values(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(unresolved_values(child, f"{path}[{index}]"))
    return errors


def duplicate_ids(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    paths = (
        ("benchmarks", "families"),
        ("methods",),
        ("provenance", "tools"),
        ("metrics", "definitions"),
        ("comparisons",),
        ("outputs", "tables"),
        ("outputs", "figures"),
        ("outputs", "negative_controls"),
    )
    for components in paths:
        value: Any = contract
        try:
            for component in components:
                value = value[component]
        except (KeyError, TypeError):
            continue
        ids = [item.get("id") for item in value if isinstance(item, dict)]
        repeated = sorted({item for item in ids if item is not None and ids.count(item) > 1})
        if repeated:
            errors.append(f"duplicate IDs in {'.'.join(components)}: {', '.join(repeated)}")
    for family in contract.get("benchmarks", {}).get("families", []):
        instances = family.get("instances", [])
        ids = [item.get("id") for item in instances if isinstance(item, dict)]
        repeated = sorted({item for item in ids if item is not None and ids.count(item) > 1})
        if repeated:
            errors.append(
                f"duplicate instance IDs in {family.get('id', '<unnamed>')}: "
                + ", ".join(repeated)
            )
    return errors


def check_completeness(contract: dict[str, Any]) -> list[str]:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read schema: {exc}"]
    errors = schema_errors(contract, schema, schema)
    errors.extend(unresolved_values(contract))
    errors.extend(duplicate_ids(contract))
    return errors


def check_split(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for family in contract.get("benchmarks", {}).get("families", []):
        family_id = family.get("id", "<unnamed>")
        instance_ids = {item.get("id") for item in family.get("instances", [])}
        assignment: dict[str, set[str]] = {
            name: set(family.get("splits", {}).get(name, []))
            for name in ("training", "validation", "held_out")
        }
        tuning = assignment["training"] | assignment["validation"]
        overlap = tuning & assignment["held_out"]
        if overlap:
            errors.append("an instance appears in both tuning and held-out data")
        assigned = set().union(*assignment.values())
        missing = instance_ids - assigned
        unknown = assigned - instance_ids
        if missing:
            errors.append(f"{family_id}: unassigned instances: {', '.join(sorted(missing))}")
        if unknown:
            errors.append(f"{family_id}: split references unknown instances: {', '.join(sorted(unknown))}")
        for left, right in (("training", "validation"), ("training", "held_out"), ("validation", "held_out")):
            duplicate = assignment[left] & assignment[right]
            if duplicate and not (right == "held_out" and duplicate <= overlap):
                errors.append(
                    f"{family_id}: instances shared by {left} and {right}: "
                    + ", ".join(sorted(duplicate))
                )
    return list(dict.fromkeys(errors))


def check_comparisons(contract: dict[str, Any]) -> list[str]:
    methods = contract.get("methods", [])
    method_ids = {method.get("id") for method in methods}
    roles = {method.get("role") for method in methods}
    required_roles = {
        "project",
        "direct_solve_reference",
        "cheap_branching_control",
        "partitioning_baseline",
    }
    errors = []
    missing_roles = required_roles - roles
    if missing_roles:
        errors.append(f"missing method roles: {', '.join(sorted(missing_roles))}")
    for comparison in contract.get("comparisons", []):
        for side in ("left", "right"):
            if comparison.get(side) not in method_ids:
                errors.append(
                    f"comparison {comparison.get('id', '<unnamed>')} references unknown method "
                    f"{comparison.get(side)!r}"
                )
        if not comparison.get("controlled_factors"):
            errors.append(f"comparison {comparison.get('id', '<unnamed>')} has no controlled factors")
    if not contract.get("comparisons"):
        errors.append("no controlled comparisons are declared")
    return errors


def check_accounting(contract: dict[str, Any]) -> list[str]:
    definitions = contract.get("metrics", {}).get("definitions", [])
    metric_ids = {metric.get("id") for metric in definitions}
    required = {"cubing_cost", "conquer_work", "makespan", "end_to_end_cost"}
    errors = []
    missing = required - metric_ids
    if missing:
        errors.append(f"missing required metric definitions: {', '.join(sorted(missing))}")

    outputs = contract.get("outputs", {})
    reported: set[str] = set(outputs.get("raw_records", {}).get("metrics", []))
    for section in ("tables", "figures"):
        for artifact in outputs.get(section, []):
            reported.update(artifact.get("metrics", []))
    undeclared = reported - metric_ids
    if undeclared:
        errors.append(f"reported metrics lack definitions: {', '.join(sorted(undeclared))}")
    unused = metric_ids - reported
    if unused:
        errors.append(f"declared metrics are never recorded or reported: {', '.join(sorted(unused))}")
    return errors


def check_version_history(contract: dict[str, Any], contract_path: Path) -> list[str]:
    """Reject reuse of a protocol version for different committed contents."""
    resolved = contract_path.resolve()
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=resolved.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if root_result.returncode != 0:
        return []
    root = Path(root_result.stdout.strip()).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return []
    revisions = subprocess.run(
        ["git", "rev-list", "HEAD", "--", relative],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if revisions.returncode != 0:
        return []

    current_version = contract.get("protocol", {}).get("version")
    current_digest = digest(contract)
    errors = []
    for revision in revisions.stdout.splitlines():
        result = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        try:
            historical = load_yaml_text(result.stdout, f"{revision}:{relative}")
            historical_digest = digest(historical)
        except (yaml.YAMLError, ContractError):
            continue
        historical_version = historical.get("protocol", {}).get("version")
        if historical_version == current_version and historical_digest != current_digest:
            errors.append(
                f"protocol version {current_version} is already frozen to a different digest"
            )
            break
        if isinstance(historical_version, int) and isinstance(current_version, int):
            if historical_version > current_version:
                errors.append(
                    f"protocol version regressed from {historical_version} to {current_version}"
                )
                break
    return errors


def check_freeze(
    contract: dict[str, Any], contract_path: Path, lock_path: Path
) -> list[str]:
    try:
        lock = load_yaml(lock_path)
    except ContractError as exc:
        return [str(exc)]
    errors = []
    if lock.get("algorithm") != "sha256":
        errors.append("lock algorithm must be sha256")
    if lock.get("canonicalization") != "canonical-json-v1":
        errors.append("lock canonicalization must be canonical-json-v1")
    expected = lock.get("contract_digest", "")
    actual = digest(contract)
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        errors.append("lock contract_digest is not a lowercase SHA-256 digest")
    elif expected != actual:
        errors.append("canonical digest does not match the lock")
    expected_schema = lock.get("schema_digest", "")
    actual_schema = file_digest(SCHEMA_PATH)
    if expected_schema != actual_schema:
        errors.append("schema digest does not match the lock")
    if lock.get("protocol_version") != contract.get("protocol", {}).get("version"):
        errors.append("protocol version does not match the lock")
    errors.extend(check_version_history(contract, contract_path))
    return errors


def audit(contract_path: Path, lock_path: Path | None) -> int:
    try:
        contract = load_yaml(contract_path)
    except ContractError as exc:
        print(f"FAIL completeness: {exc}")
        return 1

    checks = [
        ("completeness", check_completeness(contract), "no required field is unresolved"),
        ("split", check_split(contract), "tuning and held-out instances are disjoint"),
        ("comparisons", check_comparisons(contract), "methods and controlled contrasts are explicit"),
        ("accounting", check_accounting(contract), "every reported metric has a declared definition"),
    ]
    if lock_path is not None:
        checks.append(
            (
                "freeze",
                check_freeze(contract, contract_path, lock_path),
                "canonical digest matches the lock",
            )
        )

    failed = False
    for name, errors, success in checks:
        if errors:
            failed = True
            print(f"FAIL {name}: {errors[0]}")
            for error in errors[1:]:
                print(f"  - {error}")
        else:
            print(f"PASS {name}: {success}")
    return int(failed)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="audit a study contract")
    audit_parser.add_argument("contract", type=Path)
    audit_parser.add_argument("lock", nargs="?", type=Path)
    digest_parser = subparsers.add_parser("digest", help="print a contract's canonical digest")
    digest_parser.add_argument("contract", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "audit":
        return audit(args.contract, args.lock)
    try:
        print(digest(load_yaml(args.contract)))
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
