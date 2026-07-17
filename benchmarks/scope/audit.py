#!/usr/bin/env python3
"""Check the benchmark populations and boundaries declared by the scope YAML."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL = {
    "scope",
    "coverage_axes",
    "controlled_multiplier_factoring",
    "external_multiplier_equivalence",
    "non_multiplier_arithmetic",
    "diagnostic_controls",
    "exclusions",
    "sources",
    "next_stage",
}
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
REQUIRED_ARITHMETIC = {"divisor", "square", "square-root"}
REQUIRED_EXCLUSIONS = {
    "random-k-sat",
    "broad-xcsp",
    "hwmcc-sequential",
    "counting-and-uai",
    "approximate-truncated-floating-point",
    "sequential-shift-add",
}


def mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item["id"]
        for item in value
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def unresolved(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"todo", "tbd", "fixme", "placeholder"}
    if isinstance(value, dict):
        return any(unresolved(item) for item in value.values())
    if isinstance(value, list):
        return any(unresolved(item) for item in value)
    return False


def missing(required: set[str], actual: list[str], label: str) -> list[str]:
    absent = sorted(required - set(actual))
    return [f"missing required {label}: {', '.join(absent)}"] if absent else []


def check_completeness(scope: dict[str, Any]) -> list[str]:
    errors = []
    absent = REQUIRED_TOP_LEVEL - scope.keys()
    unknown = scope.keys() - REQUIRED_TOP_LEVEL
    if absent:
        errors.append(f"missing top-level fields: {', '.join(sorted(absent))}")
    if unknown:
        errors.append(f"unknown top-level fields: {', '.join(sorted(unknown))}")
    metadata = mapping(scope.get("scope"))
    if metadata.get("status") != "frozen" or not isinstance(
        metadata.get("version"), int
    ):
        errors.append("scope needs an integer version and frozen status")
    if unresolved(scope):
        errors.append("scope contains an unresolved placeholder")
    for label, values in {
        "architectures": ids(mapping(scope.get("controlled_multiplier_factoring")).get("architectures")),
        "external suites": ids(mapping(scope.get("external_multiplier_equivalence")).get("suites")),
        "arithmetic circuits": ids(mapping(scope.get("non_multiplier_arithmetic")).get("circuits")),
        "sources": ids(scope.get("sources")),
    }.items():
        duplicates = sorted(item for item, count in Counter(values).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate IDs in {label}: {', '.join(duplicates)}")
    return errors


def check_multiplier_coverage(scope: dict[str, Any]) -> list[str]:
    group = mapping(scope.get("controlled_multiplier_factoring"))
    scale = mapping(group.get("scale_policy"))
    errors = missing(
        REQUIRED_ARCHITECTURES, ids(group.get("architectures")), "architectures"
    )
    if group.get("pairing", {}).get("key") != "target-integer":
        errors.append("controlled multipliers must be paired by target-integer")
    if scale.get("benchmark_widths") != [64, 96, 128]:
        errors.append("formal factor-width ladder must be 64, 96, and 128 bits")
    if scale.get("smoke_only_widths") != [24, 32]:
        errors.append("smoke-only factor widths must be 24 and 32 bits")
    return errors


def check_breadth(scope: dict[str, Any]) -> list[str]:
    external = mapping(scope.get("external_multiplier_equivalence"))
    arithmetic = mapping(scope.get("non_multiplier_arithmetic"))
    errors = missing(
        REQUIRED_EXTERNAL_SUITES, ids(external.get("suites")), "external suites"
    )
    errors += missing(
        REQUIRED_ARITHMETIC, ids(arithmetic.get("circuits")), "arithmetic circuits"
    )
    if external.get("expected_outcome") != "unsat":
        errors.append("equivalence miters must have expected outcome UNSAT")
    if arithmetic.get("expected_outcome") != "sat":
        errors.append("arithmetic preimages must have expected outcome SAT")
    source_ids = set(ids(scope.get("sources")))
    referenced = {arithmetic.get("source")}
    referenced.update(
        item.get("source")
        for item in external.get("suites", [])
        if isinstance(item, dict)
    )
    unknown = sorted(item for item in referenced if item and item not in source_ids)
    if unknown:
        errors.append(f"unknown source references: {', '.join(unknown)}")
    return errors


def check_boundaries(scope: dict[str, Any]) -> list[str]:
    errors = missing(REQUIRED_EXCLUSIONS, ids(scope.get("exclusions")), "exclusions")
    if mapping(scope.get("diagnostic_controls")).get("status") != "conditional":
        errors.append("general-Boolean diagnostics must remain conditional")
    deferred = " ".join(
        mapping(scope.get("next_stage")).get("deliberately_deferred", [])
    ).lower()
    for subject in ("instance count", "tuning", "train", "workers", "metrics"):
        if subject not in deferred:
            errors.append(f"evaluation protocol does not defer {subject}")
    return errors


def audit(path: Path) -> int:
    try:
        scope = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        print(f"FAIL completeness: cannot read scope: {exc}")
        return 1
    if not isinstance(scope, dict):
        print("FAIL completeness: scope must be a YAML mapping")
        return 1

    checks = [
        ("completeness", check_completeness(scope), "scope is complete and versioned"),
        ("multiplier-coverage", check_multiplier_coverage(scope), "required structures and widths are explicit"),
        ("breadth", check_breadth(scope), "external miters and non-multiplier arithmetic are included"),
        ("boundaries", check_boundaries(scope), "evaluation choices remain out of scope"),
    ]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    return audit(parser.parse_args().scope)


if __name__ == "__main__":
    raise SystemExit(main())
