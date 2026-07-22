"""Compare complete full-tree cubing arms at matched frontier scale."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.cnc.conquer_parallel import read_cubes
from benchmarks.cnc.trace_mechanism import TraceError, _quantile, read_jsonl, summarize_conquer
from benchmarks.cnc.trace_mechanism_cohort import sha256_file


def _literal_distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def analyze_arm(
    name: str,
    frontier_path: Path,
    conquer_path: Path,
    *,
    expected: str = "unsat",
) -> dict[str, Any]:
    cubes = list(read_cubes(frontier_path))
    if not cubes:
        raise TraceError(f"{name}: empty frontier")
    rows = read_jsonl(conquer_path)
    by_index = {int(row["cube_index"]): row for row in rows}
    if len(by_index) != len(rows) or set(by_index) != set(range(len(cubes))):
        raise TraceError(f"{name}: conquer rows do not exactly match frontier cubes")

    for cube_index, cube in enumerate(cubes):
        row = by_index[cube_index]
        if int(row["cube_literals"]) != len(cube):
            raise TraceError(f"{name}: cube {cube_index} literal-count mismatch")
        if not row.get("censored", False) and row.get("result") != expected:
            raise TraceError(
                f"{name}: cube {cube_index} returned {row.get('result')}, expected {expected}"
            )

    censored = sum(bool(row.get("censored", False)) for row in rows)
    summary = summarize_conquer(rows)
    released = [row.get("released_monotonic_ns") for row in rows]
    collected = [row.get("collected_monotonic_ns") for row in rows]
    measured_makespan_s = None
    if all(type(value) is int for value in (*released, *collected)):
        measured_makespan_s = (max(collected) - min(released)) / 1e9
    return {
        "name": name,
        "frontier": str(frontier_path.resolve()),
        "frontier_sha256": sha256_file(frontier_path),
        "conquer": str(conquer_path.resolve()),
        "cubes": len(cubes),
        "complete": censored == 0 and summary["uncensored"] == len(cubes),
        "censored": censored,
        "decision_literals": _literal_distribution([len(cube) for cube in cubes]),
        "conflicts": summary["conflicts"],
        "elapsed_s": summary["elapsed_s"],
        "measured_makespan_s": measured_makespan_s,
    }


def add_cubing_seconds(
    arms: list[dict[str, Any]], cubing_seconds: dict[str, float]
) -> None:
    names = {arm["name"] for arm in arms}
    if not set(cubing_seconds) <= names:
        unknown = sorted(set(cubing_seconds) - names)
        raise TraceError(f"cubing time supplied for unknown arms {unknown}")
    for arm in arms:
        cubing_s = cubing_seconds.get(arm["name"])
        if cubing_s is not None and cubing_s < 0:
            raise TraceError(f"{arm['name']}: cubing seconds must be non-negative")
        arm["cubing_s"] = cubing_s
        makespan = arm.get("measured_makespan_s")
        arm["end_to_end_s"] = (
            cubing_s + float(makespan)
            if cubing_s is not None and makespan is not None
            else None
        )


def compare_arms(
    arms: list[dict[str, Any]], reference: str, *, compare_elapsed: bool = False
) -> dict[str, Any]:
    by_name = {arm["name"]: arm for arm in arms}
    if len(by_name) != len(arms):
        raise TraceError("duplicate arm name")
    if reference not in by_name:
        raise TraceError(f"missing reference arm {reference}")
    baseline = by_name[reference]

    ratios: dict[str, dict[str, float]] = {}
    for name, arm in by_name.items():
        if name == reference:
            continue
        metrics: dict[str, float] = {}
        distributions = ["conflicts"]
        if compare_elapsed:
            distributions.append("elapsed_s")
        for distribution in distributions:
            for field in ("total", "cv", "p95", "max"):
                value = arm[distribution][field]
                base = baseline[distribution][field]
                if value is not None and base not in (None, 0):
                    metrics[f"{distribution}_{field}"] = float(value) / float(base)
        metrics["cubes"] = float(arm["cubes"]) / float(baseline["cubes"])
        metrics["decision_literals_mean"] = float(
            arm["decision_literals"]["mean"]
        ) / float(baseline["decision_literals"]["mean"])
        if compare_elapsed:
            for field in ("measured_makespan_s", "end_to_end_s"):
                value = arm.get(field)
                base = baseline.get(field)
                if value is not None and base not in (None, 0):
                    metrics[field] = float(value) / float(base)
        ratios[name] = metrics

    return {
        "schema_version": 1,
        "reference": reference,
        "all_complete": all(arm["complete"] for arm in arms),
        "arms": arms,
        "ratios_to_reference": ratios,
        "elapsed_ratios_included": compare_elapsed,
        "inference_scope": (
            "full-tree single-instance attribution at approximately matched frontier size; "
            "later residual states differ by design; elapsed ratios require an explicitly "
            "shared runtime environment"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        nargs=3,
        metavar=("NAME", "FRONTIER", "CONQUER"),
        required=True,
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--expected", default="unsat")
    parser.add_argument(
        "--compare-elapsed",
        action="store_true",
        help="include elapsed ratios only when all arms share hardware and concurrency",
    )
    parser.add_argument(
        "--cubing-seconds",
        action="append",
        nargs=2,
        metavar=("NAME", "SECONDS"),
        default=[],
        help="attach measured cubing time for end-to-end makespan comparison",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        arms = [
            analyze_arm(name, Path(frontier), Path(conquer), expected=args.expected)
            for name, frontier, conquer in args.arm
        ]
        cubing_seconds = {name: float(seconds) for name, seconds in args.cubing_seconds}
        if len(cubing_seconds) != len(args.cubing_seconds):
            raise TraceError("duplicate --cubing-seconds arm")
        add_cubing_seconds(arms, cubing_seconds)
        result = compare_arms(
            arms, args.reference, compare_elapsed=args.compare_elapsed
        )
    except (OSError, KeyError, TypeError, ValueError, TraceError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
