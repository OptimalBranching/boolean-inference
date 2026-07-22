"""Join cube residual audits to conquer outcomes and test structural mediation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from benchmarks.cnc.trace_mechanism import TraceError, _quantile, _spearman, read_jsonl


RESIDUAL_FIELDS = (
    "fixed_variables",
    "unfixed_variables",
    "active_tensors",
    "entailed_tensors",
    "constrained_variables",
    "free_variables",
    "constrained_components",
    "largest_component_variables",
    "largest_component_tensors",
    "active_incidence_edges",
    "active_degree_mean",
    "residual_arity_mean",
    "live_rows_total",
    "tensor_compression_mean_bits",
)


def distribution(values: list[float]) -> dict[str, float | None]:
    mean = statistics.fmean(values) if values else None
    return {
        "total": sum(values),
        "mean": mean,
        "median": statistics.median(values) if values else None,
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values) if values else None,
        "cv": (
            statistics.pstdev(values) / mean
            if values and mean is not None and mean != 0
            else None
        ),
    }


def cube_sha256(literals: list[int]) -> str:
    return hashlib.sha256(
        (" ".join(map(str, literals)) + " 0\n").encode()
    ).hexdigest()


def _indexed(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["cube_index"])
        if index in indexed:
            raise TraceError(f"{label}: duplicate cube_index {index}")
        indexed[index] = row
    if set(indexed) != set(range(len(rows))):
        raise TraceError(f"{label}: cube indexes are not contiguous from zero")
    return indexed


def _values(
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    getter: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], float],
) -> list[float]:
    return [float(getter(native, cnf, conquer)) for native, cnf, conquer in rows]


def _residual_distributions(
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    position: int,
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for field in RESIDUAL_FIELDS:
        values = [row[position]["residual"].get(field) for row in rows]
        if all(value is not None for value in values):
            result[field] = distribution([float(value) for value in values])
    return result


def _tail_summary(
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    fraction: float = 0.05,
) -> dict[str, Any]:
    count = max(1, math.ceil(fraction * len(rows)))
    ordered = sorted(rows, key=lambda row: int(row[2]["conflicts"]), reverse=True)
    tail = ordered[:count]
    body = ordered[count:]
    total_conflicts = sum(int(row[2]["conflicts"]) for row in ordered)

    def medians(
        selected: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    ) -> dict[str, float | None]:
        if not selected:
            return {
                "conflicts": None,
                "cube_literals": None,
                "fixed_variables": None,
                "active_tensors": None,
                "largest_component_variables": None,
                "live_rows_total": None,
            }
        return {
            "conflicts": statistics.median(
                int(row[2]["conflicts"]) for row in selected
            ),
            "cube_literals": statistics.median(
                int(row[0]["cube_literals"]) for row in selected
            ),
            "fixed_variables": statistics.median(
                int(row[0]["residual"]["fixed_variables"]) for row in selected
            ),
            "active_tensors": statistics.median(
                int(row[0]["residual"]["active_tensors"]) for row in selected
            ),
            "largest_component_variables": statistics.median(
                int(row[0]["residual"]["largest_component_variables"])
                for row in selected
            ),
            "live_rows_total": statistics.median(
                int(row[0]["residual"]["live_rows_total"]) for row in selected
            ),
        }

    return {
        "fraction": fraction,
        "cubes": count,
        "conflict_share": (
            sum(int(row[2]["conflicts"]) for row in tail) / total_conflicts
            if total_conflicts
            else None
        ),
        "tail_medians": medians(tail),
        "body_medians": medians(body),
    }


def analyze_arm(
    name: str,
    native_audit_path: Path,
    cnf_audit_path: Path,
    conquer_path: Path,
) -> dict[str, Any]:
    native = _indexed(read_jsonl(native_audit_path), f"{name} native audit")
    cnf = _indexed(read_jsonl(cnf_audit_path), f"{name} CNF audit")
    conquer = _indexed(read_jsonl(conquer_path), f"{name} conquer")
    if not (set(native) == set(cnf) == set(conquer)):
        raise TraceError(f"{name}: audits and conquer do not contain the same cubes")

    linked = []
    fixed_assignment_matches = 0
    for index in range(len(native)):
        native_row = native[index]
        cnf_row = cnf[index]
        conquer_row = conquer[index]
        if native_row.get("schema_version") != 2 or cnf_row.get("schema_version") != 2:
            raise TraceError(f"{name}: residual audit schema v2 is required")
        native_literals = [int(value) for value in native_row["literals"]]
        cnf_literals = [int(value) for value in cnf_row["literals"]]
        if native_literals != cnf_literals:
            raise TraceError(f"{name}: native/CNF literals differ at cube {index}")
        if int(conquer_row["cube_literals"]) != len(native_literals):
            raise TraceError(f"{name}: literal count differs at cube {index}")
        if conquer_row["cube_sha256"] != cube_sha256(native_literals):
            raise TraceError(f"{name}: cube SHA-256 differs at cube {index}")
        if conquer_row.get("censored", False):
            raise TraceError(f"{name}: cube {index} is censored")
        if conquer_row.get("conflicts") is None:
            raise TraceError(f"{name}: cube {index} has no conflict count")
        if (
            native_row["fixed_mask_hex"] == cnf_row["fixed_mask_hex"]
            and native_row["fixed_value_hex"] == cnf_row["fixed_value_hex"]
        ):
            fixed_assignment_matches += 1
        linked.append((native_row, cnf_row, conquer_row))

    conflicts = _values(linked, lambda _n, _c, outcome: outcome["conflicts"])
    log_conflicts = [math.log1p(value) for value in conflicts]
    feature_getters: dict[
        str, Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], float]
    ] = {
        "cube_literals": lambda native, _cnf, _outcome: native["cube_literals"],
        "gac_additional_fixed_variables": lambda native, _cnf, _outcome: native[
            "gac_additional_fixed_variables"
        ],
        "native_fixed_variables": lambda native, _cnf, _outcome: native["residual"][
            "fixed_variables"
        ],
        "native_active_tensors": lambda native, _cnf, _outcome: native["residual"][
            "active_tensors"
        ],
        "native_largest_component_variables": lambda native, _cnf, _outcome: native[
            "residual"
        ]["largest_component_variables"],
        "native_active_incidence_edges": lambda native, _cnf, _outcome: native[
            "residual"
        ]["active_incidence_edges"],
        "native_live_rows_total": lambda native, _cnf, _outcome: native["residual"][
            "live_rows_total"
        ],
        "cnf_active_clauses": lambda _native, cnf, _outcome: cnf["residual"][
            "active_tensors"
        ],
    }
    spearman = {
        feature: _spearman(_values(linked, getter), log_conflicts)
        for feature, getter in feature_getters.items()
    }
    exact_fraction = fixed_assignment_matches / len(linked)
    return {
        "name": name,
        "cubes": len(linked),
        "cube_hash_join_verified": True,
        "native_input_kind": linked[0][0]["input_kind"],
        "cnf_input_kind": linked[0][1]["input_kind"],
        "native_cnf_fixed_assignment_matches": fixed_assignment_matches,
        "native_cnf_fixed_assignment_match_fraction": exact_fraction,
        "native_cnf_propagation_equivalent_on_frontier": exact_fraction == 1.0,
        "cube_literals": distribution(
            _values(linked, lambda native, _cnf, _outcome: native["cube_literals"])
        ),
        "gac_additional_fixed_variables": distribution(
            _values(
                linked,
                lambda native, _cnf, _outcome: native[
                    "gac_additional_fixed_variables"
                ],
            )
        ),
        "native_residual": _residual_distributions(linked, 0),
        "cnf_residual": _residual_distributions(linked, 1),
        "conflicts": distribution(conflicts),
        "hardest_five_percent": _tail_summary(linked),
        "within_arm_spearman_log1p_conflicts": spearman,
        "within_arm_associations_are_exploratory": True,
    }


def compare_arms(arms: list[dict[str, Any]], reference: str) -> dict[str, Any]:
    by_name = {arm["name"]: arm for arm in arms}
    if len(by_name) != len(arms):
        raise TraceError("duplicate arm name")
    if reference not in by_name:
        raise TraceError(f"missing reference arm {reference}")
    baseline = by_name[reference]

    def ratio(value: float | int | None, base: float | int | None) -> float | None:
        if value is None or base in (None, 0):
            return None
        return float(value) / float(base)

    comparisons = {}
    for name, arm in by_name.items():
        if name == reference:
            continue
        comparisons[name] = {
            "cubes": ratio(arm["cubes"], baseline["cubes"]),
            "conflicts_total": ratio(
                arm["conflicts"]["total"], baseline["conflicts"]["total"]
            ),
            "conflicts_mean": ratio(
                arm["conflicts"]["mean"], baseline["conflicts"]["mean"]
            ),
            "conflicts_p95": ratio(
                arm["conflicts"]["p95"], baseline["conflicts"]["p95"]
            ),
            "cube_literals_mean": ratio(
                arm["cube_literals"]["mean"], baseline["cube_literals"]["mean"]
            ),
            "native_fixed_variables_mean": ratio(
                arm["native_residual"]["fixed_variables"]["mean"],
                baseline["native_residual"]["fixed_variables"]["mean"],
            ),
            "native_active_tensors_mean": ratio(
                arm["native_residual"]["active_tensors"]["mean"],
                baseline["native_residual"]["active_tensors"]["mean"],
            ),
            "hardest_five_percent_conflict_share": ratio(
                arm["hardest_five_percent"]["conflict_share"],
                baseline["hardest_five_percent"]["conflict_share"],
            ),
        }
    return {
        "schema_version": 1,
        "reference": reference,
        "all_cube_hash_joins_verified": all(
            arm["cube_hash_join_verified"] for arm in arms
        ),
        "all_native_cnf_propagation_equivalent_on_frontier": all(
            arm["native_cnf_propagation_equivalent_on_frontier"] for arm in arms
        ),
        "arms": arms,
        "ratios_to_reference": comparisons,
        "inference_scope": (
            "descriptive divergent-tree residual mediation on one instance; "
            "cube-level associations are not independent instance-level causal evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        nargs=4,
        metavar=("NAME", "NATIVE_AUDIT", "CNF_AUDIT", "CONQUER"),
        required=True,
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        arms = [
            analyze_arm(name, Path(native), Path(cnf), Path(conquer))
            for name, native, cnf, conquer in args.arm
        ]
        result = compare_arms(arms, args.reference)
    except (OSError, KeyError, TypeError, ValueError, TraceError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
