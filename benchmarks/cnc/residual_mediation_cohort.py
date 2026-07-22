"""Aggregate treatment/reference residual mediation with instances as units."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmarks.cnc.residual_mediation import analyze_arm
from benchmarks.cnc.trace_mechanism import TraceError, _spearman


RATIO_FIELDS = (
    "cube_ratio",
    "fixed_variables_ratio",
    "active_tensors_ratio",
    "decision_literals_ratio",
    "conflicts_total_ratio",
    "conflicts_mean_ratio",
    "conflicts_p95_ratio",
    "conflicts_cv_ratio",
    "hardest_five_percent_conflict_share_ratio",
    "inverse_count_scaled_p95_ratio",
)


def _ratio(value: float | int | None, reference: float | int | None) -> float:
    if value is None or reference in (None, 0):
        raise TraceError("cohort ratio has a missing or zero reference value")
    result = float(value) / float(reference)
    if result <= 0 or not math.isfinite(result):
        raise TraceError("cohort ratios must be finite and positive")
    return result


def compare_instance(
    instance_id: str,
    treatment: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    cube_ratio = _ratio(treatment["cubes"], reference["cubes"])
    p95_ratio = _ratio(treatment["conflicts"]["p95"], reference["conflicts"]["p95"])
    return {
        "instance_id": instance_id,
        "treatment_cubes": treatment["cubes"],
        "reference_cubes": reference["cubes"],
        "cube_ratio": cube_ratio,
        "fixed_variables_ratio": _ratio(
            treatment["native_residual"]["fixed_variables"]["mean"],
            reference["native_residual"]["fixed_variables"]["mean"],
        ),
        "active_tensors_ratio": _ratio(
            treatment["native_residual"]["active_tensors"]["mean"],
            reference["native_residual"]["active_tensors"]["mean"],
        ),
        "decision_literals_ratio": _ratio(
            treatment["cube_literals"]["mean"],
            reference["cube_literals"]["mean"],
        ),
        "conflicts_total_ratio": _ratio(
            treatment["conflicts"]["total"], reference["conflicts"]["total"]
        ),
        "conflicts_mean_ratio": _ratio(
            treatment["conflicts"]["mean"], reference["conflicts"]["mean"]
        ),
        "conflicts_p95_ratio": p95_ratio,
        "conflicts_cv_ratio": _ratio(
            treatment["conflicts"]["cv"], reference["conflicts"]["cv"]
        ),
        "hardest_five_percent_conflict_share_ratio": _ratio(
            treatment["hardest_five_percent"]["conflict_share"],
            reference["hardest_five_percent"]["conflict_share"],
        ),
        # Sensitivity only: if per-cube p95 scaled exactly as 1 / task count,
        # multiplying by the task-count ratio would map both frontiers to a
        # common granularity.  This is not a learned or causal adjustment.
        "inverse_count_scaled_p95_ratio": p95_ratio * cube_ratio,
        "treatment_native_cnf_propagation_equivalent": treatment[
            "native_cnf_propagation_equivalent_on_frontier"
        ],
        "reference_native_cnf_propagation_equivalent": reference[
            "native_cnf_propagation_equivalent_on_frontier"
        ],
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise TraceError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def summarize_cohort(
    instances: list[dict[str, Any]], treatment_name: str, reference_name: str
) -> dict[str, Any]:
    if not instances:
        raise TraceError("empty residual mediation cohort")
    ids = [row["instance_id"] for row in instances]
    if len(ids) != len(set(ids)):
        raise TraceError("duplicate residual mediation instance_id")

    geometric_means = {
        field: geometric_mean([float(row[field]) for row in instances])
        for field in RATIO_FIELDS
    }
    feature_fields = (
        "cube_ratio",
        "fixed_variables_ratio",
        "active_tensors_ratio",
        "decision_literals_ratio",
    )
    outcome_fields = (
        "conflicts_total_ratio",
        "conflicts_mean_ratio",
        "conflicts_p95_ratio",
        "conflicts_cv_ratio",
        "hardest_five_percent_conflict_share_ratio",
    )
    associations = {
        feature: {
            outcome: _spearman(
                [math.log(float(row[feature])) for row in instances],
                [math.log(float(row[outcome])) for row in instances],
            )
            for outcome in outcome_fields
        }
        for feature in feature_fields
    }
    return {
        "schema_version": 1,
        "treatment": treatment_name,
        "reference": reference_name,
        "instances": len(instances),
        "all_native_cnf_propagation_equivalent": all(
            row["treatment_native_cnf_propagation_equivalent"]
            and row["reference_native_cnf_propagation_equivalent"]
            for row in instances
        ),
        "geometric_mean_ratios": geometric_means,
        "feature_to_outcome_spearman": associations,
        "per_instance": instances,
        "inverse_count_scaled_p95_is_sensitivity_only": True,
        "inference_scope": (
            "instance-level descriptive cohort; inverse-count p95 scaling is a "
            "declared granularity sensitivity, not a causal correction"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        action="append",
        nargs=7,
        metavar=(
            "ID",
            "TREAT_NATIVE",
            "TREAT_CNF",
            "TREAT_CONQUER",
            "REF_NATIVE",
            "REF_CNF",
            "REF_CONQUER",
        ),
        required=True,
    )
    parser.add_argument("--treatment", default="treatment")
    parser.add_argument("--reference", default="reference")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        comparisons = []
        for (
            instance_id,
            treatment_native,
            treatment_cnf,
            treatment_conquer,
            reference_native,
            reference_cnf,
            reference_conquer,
        ) in args.instance:
            treatment = analyze_arm(
                args.treatment,
                Path(treatment_native),
                Path(treatment_cnf),
                Path(treatment_conquer),
            )
            reference = analyze_arm(
                args.reference,
                Path(reference_native),
                Path(reference_cnf),
                Path(reference_conquer),
            )
            comparisons.append(compare_instance(instance_id, treatment, reference))
        result = summarize_cohort(comparisons, args.treatment, args.reference)
    except (OSError, KeyError, TypeError, ValueError, TraceError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
