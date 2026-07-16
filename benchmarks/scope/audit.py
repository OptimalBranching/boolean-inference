#!/usr/bin/env python3
"""Validate and fingerprint the versioned benchmark-scope document."""

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
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required; install benchmarks/scope/requirements.txt"
    ) from exc


ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "benchmark-scope.schema.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED = re.compile(r"^(?:todo|tbd|fixme|unresolved|placeholder)$", re.I)

REQUIRED_ARCHITECTURES = {
    "array-ripple",
    "wallace-ripple",
    "dadda-ripple",
    "booth-r4-wallace",
    "booth-r4-dadda",
    "karatsuba",
}
REQUIRED_EXTERNAL_SUITES = {
    "sat-2016-datapath",
    "sat-competition-2024-multiplier-equivalence",
    "sat-competition-2025-multiplier-equivalence",
}
REQUIRED_ARITHMETIC_CIRCUITS = {"divisor", "square", "square-root"}
REQUIRED_BENCHMARK_WIDTHS = [64, 96, 128]
REQUIRED_SMOKE_WIDTHS = [24, 32]
REQUIRED_EXCLUSIONS = {
    "random-k-sat",
    "broad-xcsp",
    "hwmcc-sequential",
    "counting-and-uai",
    "approximate-truncated-floating-point",
    "sequential-shift-add",
}


class ScopeError(ValueError):
    """The scope, schema, or lock is malformed."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(
    loader: UniqueKeySafeLoader, node: Any, deep: bool = False
) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ScopeError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError, ScopeError) as exc:
        raise ScopeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScopeError(f"{path} must contain one YAML mapping")
    return value


def canonical_bytes(scope: dict[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            scope,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScopeError(f"scope is not canonical-JSON compatible: {exc}") from exc
    return rendered.encode("utf-8")


def digest(scope: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(scope)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ScopeError(f"unsupported external schema reference {ref!r}")
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
        "boolean": lambda item: isinstance(item, bool),
    }
    return checks[expected](value)


def schema_errors(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str = "$",
) -> list[str]:
    """Validate the JSON Schema subset used by benchmark-scope.schema.json."""
    if "$ref" in schema:
        return schema_errors(value, _resolve_ref(root, schema["$ref"]), root, path)

    errors: list[str] = []
    expected = schema.get("type")
    if expected and not _type_matches(value, expected):
        return [f"{path}: expected {expected}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")

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
                errors.extend(
                    schema_errors(child, properties[key], root, f"{path}.{key}")
                )

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if schema.get("uniqueItems"):
            markers = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(markers) != len(set(markers)):
                errors.append(f"{path}: contains duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(
                    schema_errors(item, schema["items"], root, f"{path}[{index}]")
                )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern and re.search(pattern, value) is None:
            errors.append(f"{path}: does not match {pattern!r}")

    if isinstance(value, int) and not isinstance(value, bool):
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


def item_ids(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [
        item.get("id")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def duplicate_id_errors(scope: dict[str, Any]) -> list[str]:
    collections = {
        "architectures": scope.get("controlled_multiplier_factoring", {}).get(
            "architectures"
        ),
        "external suites": scope.get("external_multiplier_equivalence", {}).get(
            "suites"
        ),
        "arithmetic circuits": scope.get("non_multiplier_arithmetic", {}).get(
            "circuits"
        ),
        "diagnostic families": scope.get("diagnostic_controls", {}).get("families"),
        "exclusions": scope.get("exclusions"),
        "sources": scope.get("sources"),
    }
    errors = []
    for label, collection in collections.items():
        ids = item_ids(collection)
        repeated = sorted({item for item in ids if ids.count(item) > 1})
        if repeated:
            errors.append(f"duplicate IDs in {label}: {', '.join(repeated)}")
    return errors


def check_completeness(scope: dict[str, Any]) -> list[str]:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read schema: {exc}"]
    errors = schema_errors(scope, schema, schema)
    errors.extend(unresolved_values(scope))
    errors.extend(duplicate_id_errors(scope))
    return errors


def missing(required: set[str], actual: list[str], label: str) -> list[str]:
    absent = required - set(actual)
    return [f"missing required {label}: {', '.join(sorted(absent))}"] if absent else []


def check_multiplier_coverage(scope: dict[str, Any]) -> list[str]:
    group = scope.get("controlled_multiplier_factoring", {})
    architectures = group.get("architectures", [])
    errors = missing(
        REQUIRED_ARCHITECTURES, item_ids(architectures), "multiplier architectures"
    )
    if group.get("pairing", {}).get("key") != "target-integer":
        errors.append("controlled multipliers must be paired by target-integer")
    scale = group.get("scale_policy", {})
    if not isinstance(scale, dict):
        scale = {}
    benchmark_widths = scale.get("benchmark_widths", [])
    smoke_widths = scale.get("smoke_only_widths", [])
    if benchmark_widths != REQUIRED_BENCHMARK_WIDTHS:
        errors.append("formal factor-width ladder must be 64, 96, and 128 bits")
    if smoke_widths != REQUIRED_SMOKE_WIDTHS:
        errors.append("smoke-only factor widths must be 24 and 32 bits")
    if (
        isinstance(benchmark_widths, list)
        and isinstance(smoke_widths, list)
        and all(isinstance(width, int) for width in benchmark_widths + smoke_widths)
        and set(benchmark_widths) & set(smoke_widths)
    ):
        errors.append("formal and smoke-only factor widths must not overlap")
    if scale.get("instance_counts") != "deferred-to-evaluation-protocol":
        errors.append("instance counts belong to the later evaluation protocol")
    strategy = group.get("generator_strategy", {})
    required_generators = {
        "boolean-inference-structural-multipliers",
        "multgen",
        "genmul",
    }
    if not required_generators <= set(strategy.get("structural_sources", [])):
        errors.append(
            "controlled multiplier sources must cover native, Multgen, and GenMul circuits"
        )
    if strategy.get("independent_cnf_crosscheck") != "purdom-sabry-factoring":
        errors.append("Purdom-Sabry must remain the independent CNF-only cross-check")
    if strategy.get("independent_cnf_crosscheck") in strategy.get(
        "structural_sources", []
    ):
        errors.append("the CNF-only cross-check cannot be a structural circuit source")
    ppg = {
        item.get("partial_product_generation")
        for item in architectures
        if isinstance(item, dict)
    }
    ppa = {
        item.get("partial_product_accumulation")
        for item in architectures
        if isinstance(item, dict)
    }
    if len(ppg) < 3:
        errors.append(
            "multiplier scope must cover simple, encoded, and recursive partial-product generation"
        )
    if len(ppa) < 3:
        errors.append(
            "multiplier scope must cover at least three accumulation topologies"
        )
    return errors


def check_breadth(scope: dict[str, Any]) -> list[str]:
    external = scope.get("external_multiplier_equivalence", {})
    arithmetic = scope.get("non_multiplier_arithmetic", {})
    errors = missing(
        REQUIRED_EXTERNAL_SUITES, item_ids(external.get("suites")), "external suites"
    )
    errors.extend(
        missing(
            REQUIRED_ARITHMETIC_CIRCUITS,
            item_ids(arithmetic.get("circuits")),
            "non-multiplier arithmetic circuits",
        )
    )
    tasks = set(external.get("semantic_tasks", []))
    required_tasks = {"global-equivalence", "bit-level-equivalence"}
    if not required_tasks <= tasks:
        errors.append(
            "external validation must include global and bit-level equivalence"
        )
    if (
        external.get("expected_outcome") != "unsat"
        or arithmetic.get("expected_outcome") != "sat"
    ):
        errors.append(
            "scope must cover both SAT preimages and UNSAT equivalence miters"
        )

    source_ids = set(item_ids(scope.get("sources")))
    strategy = scope.get("controlled_multiplier_factoring", {}).get(
        "generator_strategy", {}
    )
    references = set(strategy.get("structural_sources", []))
    references.update(
        {strategy.get("independent_cnf_crosscheck"), arithmetic.get("source")}
    )
    references.update(
        item.get("source")
        for item in external.get("suites", [])
        if isinstance(item, dict)
    )
    unknown = {item for item in references if item and item not in source_ids}
    if unknown:
        errors.append(f"unknown source references: {', '.join(sorted(unknown))}")
    return errors


def check_boundaries(scope: dict[str, Any]) -> list[str]:
    errors = missing(
        REQUIRED_EXCLUSIONS, item_ids(scope.get("exclusions")), "exclusions"
    )
    if scope.get("diagnostic_controls", {}).get("status") != "conditional":
        errors.append("general-Boolean diagnostic families must remain conditional")
    deferred = " ".join(
        scope.get("next_stage", {}).get("deliberately_deferred", [])
    ).lower()
    if "width" in deferred:
        errors.append("formal benchmark widths must not remain deferred")
    for subject in ("instance count", "tuning", "train", "workers", "metrics"):
        if subject not in deferred:
            errors.append(f"next stage does not explicitly defer {subject}")
    return errors


def check_version_history(scope: dict[str, Any], scope_path: Path) -> list[str]:
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=scope_path.resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if root_result.returncode != 0:
        return []
    root = Path(root_result.stdout.strip()).resolve()
    try:
        relative = scope_path.resolve().relative_to(root).as_posix()
    except ValueError:
        return []
    revisions = subprocess.run(
        ["git", "rev-list", "HEAD", "--", relative],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    current_version = scope.get("scope", {}).get("version")
    current_digest = digest(scope)
    for revision in revisions.stdout.splitlines():
        historical = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if historical.returncode != 0:
            continue
        try:
            old = yaml.load(historical.stdout, Loader=UniqueKeySafeLoader)
        except (yaml.YAMLError, ScopeError):
            continue
        if (
            isinstance(old, dict)
            and old.get("scope", {}).get("version") == current_version
            and digest(old) != current_digest
        ):
            return [
                f"scope version {current_version} is already frozen to a different digest"
            ]
    return []


def check_freeze(scope: dict[str, Any], scope_path: Path, lock_path: Path) -> list[str]:
    try:
        lock = load_yaml(lock_path)
    except ScopeError as exc:
        return [str(exc)]
    errors = []
    if lock.get("algorithm") != "sha256":
        errors.append("lock algorithm must be sha256")
    if lock.get("canonicalization") != "canonical-json-v1":
        errors.append("lock canonicalization must be canonical-json-v1")
    expected = lock.get("scope_digest", "")
    if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
        errors.append("lock scope_digest is not a lowercase SHA-256 digest")
    elif expected != digest(scope):
        errors.append("canonical digest does not match the lock")
    if lock.get("schema_digest") != file_digest(SCHEMA_PATH):
        errors.append("schema digest does not match the lock")
    if lock.get("scope_version") != scope.get("scope", {}).get("version"):
        errors.append("scope version does not match the lock")
    errors.extend(check_version_history(scope, scope_path))
    return errors


def audit(scope_path: Path, lock_path: Path | None) -> int:
    try:
        scope = load_yaml(scope_path)
    except ScopeError as exc:
        print(f"FAIL completeness: {exc}")
        return 1

    checks = [
        (
            "completeness",
            check_completeness(scope),
            "scope matches the schema and has no unresolved fields",
        ),
        (
            "multiplier-coverage",
            check_multiplier_coverage(scope),
            "required multiplier structures and matched targets are explicit",
        ),
        (
            "breadth",
            check_breadth(scope),
            "external miters and non-multiplication arithmetic are included",
        ),
        (
            "boundaries",
            check_boundaries(scope),
            "conditional families and justified exclusions are explicit",
        ),
    ]
    if lock_path is not None:
        checks.append(
            (
                "freeze",
                check_freeze(scope, scope_path, lock_path),
                "canonical digest matches the scope lock",
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
    audit_parser = subparsers.add_parser("audit", help="audit a benchmark scope")
    audit_parser.add_argument("scope", type=Path)
    audit_parser.add_argument("lock", nargs="?", type=Path)
    digest_parser = subparsers.add_parser(
        "digest", help="print the canonical scope digest"
    )
    digest_parser.add_argument("scope", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "audit":
        return audit(args.scope, args.lock)
    try:
        print(digest(load_yaml(args.scope)))
    except ScopeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
