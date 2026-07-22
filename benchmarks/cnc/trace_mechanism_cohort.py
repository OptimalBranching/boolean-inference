"""Aggregate mechanism traces at the instance level for a frozen cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmarks.cnc.trace_mechanism import (
    TraceError,
    _spearman,
    link_conquer,
    read_jsonl,
    summarize,
    summarize_conquer,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    if any(value <= 0 for value in values):
        raise TraceError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def analyze_instance(
    instance_id: str,
    trace_path: Path,
    frontier_path: Path,
    cubing_result_path: Path,
    region_conquer_path: Path,
    baseline_conquer_path: Path,
) -> dict[str, Any]:
    trace = read_jsonl(trace_path)
    mechanism = summarize(trace, require_replay=True)
    region_rows = read_jsonl(region_conquer_path)
    baseline_rows = read_jsonl(baseline_conquer_path)
    region = summarize_conquer(region_rows)
    baseline = summarize_conquer(baseline_rows)
    linked = link_conquer(trace, region_rows)

    cubing_result = json.loads(cubing_result_path.read_text())
    expected_frontier_sha256 = cubing_result["frontier"]["sha256"]
    actual_frontier_sha256 = sha256_file(frontier_path)
    if actual_frontier_sha256 != expected_frontier_sha256:
        raise TraceError(f"{instance_id}: frontier SHA-256 mismatch")
    if region["rows"] != cubing_result["measurement"]["cube_count"]:
        raise TraceError(f"{instance_id}: frozen cube count mismatch")

    replay = mechanism["same_state_replay"]
    cover_nodes = int(mechanism["cover_nodes"])
    selected_branches = int(mechanism["selected"]["branches"])
    naive_branches = int(replay["naive_branches"])
    selected_decision_literals = int(mechanism["selected"]["decision_literals"])
    naive_decision_literals = int(replay["decision_literals"]["naive"])

    def ratio(metric: str, field: str) -> float:
        return float(region[metric][field]) / float(baseline[metric][field])

    return {
        "instance_id": instance_id,
        "frontier_sha256": actual_frontier_sha256,
        "frontier_verified": True,
        "cover_nodes": cover_nodes,
        "selected_branches": selected_branches,
        "naive_branches": naive_branches,
        "selected_decision_literals": selected_decision_literals,
        "naive_decision_literals": naive_decision_literals,
        "selected_over_naive_branches": (
            selected_branches / naive_branches if naive_branches else None
        ),
        "single_branch_fraction": mechanism["selected"]["single_branch_fraction"],
        "sibling_pairs": mechanism["selected"]["sibling_pairs"],
        "potentially_overlapping_sibling_pairs": mechanism["selected"][
            "potentially_overlapping_sibling_pairs"
        ],
        "syntactically_disjoint_cover_nodes": mechanism["selected"][
            "syntactically_disjoint_cover_nodes"
        ],
        "syntactically_disjoint_cover_fraction": mechanism["selected"][
            "syntactically_disjoint_cover_fraction"
        ],
        "selected_better_binary_fraction": (
            replay["selected_better_than_binary"] / cover_nodes if cover_nodes else None
        ),
        "selected_better_naive_fraction": (
            replay["selected_better_than_naive"] / cover_nodes if cover_nodes else None
        ),
        "selected_better_binary_nodes": replay["selected_better_than_binary"],
        "selected_better_naive_nodes": replay["selected_better_than_naive"],
        "median_selected_gamma": mechanism["selected"]["median_gamma_open"],
        "median_binary_gamma": replay["median_gamma_binary_open"],
        "median_naive_gamma": replay["median_gamma_naive_open"],
        "median_compression_bits": mechanism["region"]["median_compression_bits"],
        "median_boundary_ratio": mechanism["region"]["median_boundary_ratio"],
        "probe_survival_fraction": mechanism["region"]["probe_survival_fraction"],
        "rule_solver_ms": mechanism["actual_timing_ms"]["rule_solver"],
        "region_growth_ms": mechanism["actual_timing_ms"]["region_growth"],
        "feasibility_probe_ms": mechanism["actual_timing_ms"]["feasibility_probe"],
        "region_cubes": region["rows"],
        "baseline_cubes": baseline["rows"],
        "region_over_baseline": {
            "total_conflicts": ratio("conflicts", "total"),
            "conflicts_cv": ratio("conflicts", "cv"),
            "conflicts_p95": ratio("conflicts", "p95"),
            "conflicts_max": ratio("conflicts", "max"),
            "elapsed_p95": ratio("elapsed_s", "p95"),
            "elapsed_max": ratio("elapsed_s", "max"),
        },
        "within_region_spearman": linked["spearman"],
    }


def summarize_cohort(instances: list[dict[str, Any]]) -> dict[str, Any]:
    if not instances:
        raise TraceError("empty cohort")

    mechanism_features = [
        "single_branch_fraction",
        "selected_better_binary_fraction",
        "selected_better_naive_fraction",
        "median_compression_bits",
        "median_boundary_ratio",
        "probe_survival_fraction",
        "selected_over_naive_branches",
        "syntactically_disjoint_cover_fraction",
    ]
    ratio_features = [
        "total_conflicts",
        "conflicts_cv",
        "conflicts_p95",
        "conflicts_max",
        "elapsed_p95",
        "elapsed_max",
    ]

    ratio_summary = {
        feature: {
            "geometric_mean": geometric_mean(
                [float(row["region_over_baseline"][feature]) for row in instances]
            ),
            "region_wins": sum(
                float(row["region_over_baseline"][feature]) < 1.0 for row in instances
            ),
        }
        for feature in ratio_features
    }
    mechanism_summary = {
        feature: statistics.median(float(row[feature]) for row in instances)
        for feature in mechanism_features
    }

    association: dict[str, dict[str, float | None]] = {}
    for mechanism_feature in mechanism_features:
        mechanism_values = [float(row[mechanism_feature]) for row in instances]
        association[mechanism_feature] = {
            ratio_feature: _spearman(
                mechanism_values,
                [
                    math.log(float(row["region_over_baseline"][ratio_feature]))
                    for row in instances
                ],
            )
            for ratio_feature in ratio_features
        }

    timing = {
        key: sum(float(row[key]) for row in instances)
        for key in ("rule_solver_ms", "region_growth_ms", "feasibility_probe_ms")
    }
    recorded_structural_ms = sum(timing.values())
    total_cover_nodes = sum(int(row["cover_nodes"]) for row in instances)
    total_selected_branches = sum(int(row["selected_branches"]) for row in instances)
    total_naive_branches = sum(int(row["naive_branches"]) for row in instances)
    total_selected_decision_literals = sum(
        int(row["selected_decision_literals"]) for row in instances
    )
    total_naive_decision_literals = sum(
        int(row["naive_decision_literals"]) for row in instances
    )
    total_selected_better_binary = sum(
        int(row["selected_better_binary_nodes"]) for row in instances
    )
    total_selected_better_naive = sum(
        int(row["selected_better_naive_nodes"]) for row in instances
    )
    total_sibling_pairs = sum(int(row["sibling_pairs"]) for row in instances)
    total_potentially_overlapping_pairs = sum(
        int(row["potentially_overlapping_sibling_pairs"]) for row in instances
    )
    total_syntactically_disjoint_nodes = sum(
        int(row["syntactically_disjoint_cover_nodes"]) for row in instances
    )

    return {
        "schema_version": 1,
        "instances": len(instances),
        "all_frontiers_verified": all(row["frontier_verified"] for row in instances),
        "mechanism_medians": mechanism_summary,
        "region_over_baseline": ratio_summary,
        "mechanism_to_outcome_spearman": association,
        "instrumented_cubing_timing_ms": timing,
        "rule_solver_fraction_of_recorded_structural_time": (
            timing["rule_solver_ms"] / recorded_structural_ms
            if recorded_structural_ms
            else None
        ),
        "same_state_totals": {
            "cover_nodes": total_cover_nodes,
            "selected_branches": total_selected_branches,
            "naive_branches": total_naive_branches,
            "selected_over_naive_branches": (
                total_selected_branches / total_naive_branches
                if total_naive_branches
                else None
            ),
            "selected_decision_literals": total_selected_decision_literals,
            "naive_decision_literals": total_naive_decision_literals,
            "selected_over_naive_decision_literals": (
                total_selected_decision_literals / total_naive_decision_literals
                if total_naive_decision_literals
                else None
            ),
            "selected_better_binary_nodes": total_selected_better_binary,
            "selected_better_naive_nodes": total_selected_better_naive,
            "sibling_pairs": total_sibling_pairs,
            "potentially_overlapping_sibling_pairs": (
                total_potentially_overlapping_pairs
            ),
            "syntactically_disjoint_cover_nodes": (
                total_syntactically_disjoint_nodes
            ),
            "syntactically_disjoint_cover_fraction": (
                total_syntactically_disjoint_nodes / total_cover_nodes
                if total_cover_nodes
                else None
            ),
        },
        "inference_scope": (
            "exploratory one-family/one-width instance-level association; "
            "not a cross-family causal estimate"
        ),
        "per_instance": instances,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--cubing-result-dir", type=Path, required=True)
    parser.add_argument("--region-conquer-dir", type=Path, required=True)
    parser.add_argument("--baseline-conquer-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    try:
        instances = []
        for trace_path in sorted(args.trace_dir.glob("*.jsonl")):
            instance_id = trace_path.stem
            instances.append(
                analyze_instance(
                    instance_id,
                    trace_path,
                    args.frontier_dir / f"{instance_id}.icnf",
                    args.cubing_result_dir / f"{instance_id}.json",
                    args.region_conquer_dir / instance_id / "region.jsonl",
                    args.baseline_conquer_dir / instance_id / "march.jsonl",
                )
            )
        result = summarize_cohort(instances)
    except (OSError, KeyError, TypeError, TraceError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
