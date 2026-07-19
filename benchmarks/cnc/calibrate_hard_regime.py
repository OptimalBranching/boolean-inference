#!/usr/bin/env python3
"""Freeze one CC threshold per product-width, selector, and frontier band."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from benchmarks.cnc.calibrate_cc_difficulty import (
    CalibrationError,
    run_cuber,
    sha256_file,
)
from benchmarks.cnc.hard_regime import (
    HardRegimeError,
    contract_sha256,
    load_contract,
)
from benchmarks.pipeline.circuit import read_jsonl, write_json


def task_counts(row: dict[str, Any]) -> list[int]:
    instances = row.get("instances")
    if not isinstance(instances, list) or not instances:
        raise CalibrationError("width-level response row has no instances")
    counts = []
    for instance in instances:
        if not isinstance(instance, dict) or not isinstance(instance.get("tasks"), int):
            raise CalibrationError("width-level response has a malformed task count")
        counts.append(instance["tasks"])
    return counts


def median_tasks(row: dict[str, Any]) -> float:
    return float(statistics.median(task_counts(row)))


def calibration_loss(row: dict[str, Any], target: int) -> tuple[float, float]:
    errors = [abs(math.log2(max(1, count) / target)) for count in task_counts(row)]
    return float(statistics.median(errors)), max(errors)


def choose_width_response(
    rows: list[dict[str, Any]], target: int, minimum: int, maximum: int
) -> dict[str, Any]:
    if not rows:
        raise CalibrationError("empty width-level cutoff response")
    inside = [row for row in rows if minimum <= median_tasks(row) <= maximum]
    if not inside:
        raise CalibrationError(
            f"no calibrated threshold reaches accepted task range [{minimum}, {maximum}]"
        )
    return min(
        inside,
        key=lambda row: (
            *calibration_loss(row, target),
            abs(median_tasks(row) - target),
            int(row["threshold"]),
        ),
    )


def calibration_inputs(
    contract: dict[str, Any], manifest_path: Path, product_width: int
) -> list[dict[str, Any]]:
    root = manifest_path.parent
    digest = contract_sha256(contract)
    selected = []
    for record in read_jsonl(manifest_path):
        if record.get("product_width") != product_width or record.get("split") != "calibration":
            continue
        if record.get("factor_input_width") * 2 != product_width:
            raise CalibrationError(f"{record.get('id')}: factor/product width confusion")
        if record.get("architecture") != "array-ripple":
            raise CalibrationError(f"{record.get('id')}: calibration must use array-ripple")
        if record.get("contract_sha256") != digest:
            raise CalibrationError(f"{record.get('id')}: contract hash mismatch")
        relative = record.get("circuitsat")
        if not isinstance(relative, str):
            raise CalibrationError(f"{record.get('id')}: missing CircuitSAT path")
        instance = (root / relative).resolve()
        if sha256_file(instance) != record.get("circuitsat_sha256"):
            raise CalibrationError(f"{record.get('id')}: CircuitSAT hash mismatch")
        selected.append(
            {
                "id": record["id"],
                "path": instance,
                "sha256": record["circuitsat_sha256"],
            }
        )
    selected.sort(key=lambda row: row["id"])
    if len(selected) != 3:
        raise CalibrationError(
            f"product width {product_width} needs exactly 3 calibration instances"
        )
    return selected


def calibrate_width(
    *,
    contract: dict[str, Any],
    instances: list[dict[str, Any]],
    product_width: int,
    selector: str,
    cuber: Path,
    out_dir: Path,
    initial_threshold: int = 1024,
    maximum_threshold: int = 1 << 120,
    probe_runner: Callable[..., dict[str, int | float]] = run_cuber,
) -> dict[str, Any]:
    if selector not in {"region", "structure-blind"}:
        raise CalibrationError(f"unsupported selector {selector!r}")
    if len(instances) != 3:
        raise CalibrationError("width-level calibration requires exactly three instances")
    if initial_threshold <= 0 or maximum_threshold < initial_threshold:
        raise CalibrationError("invalid calibration threshold range")
    method = "region-cc" if selector == "region" else "structure-blind-cc"
    max_rows = int(contract["methods"][method]["max_rows"])
    timeout_s = float(contract["limits_seconds"]["cubing"])
    out_dir.mkdir(parents=True, exist_ok=True)
    observed: dict[int, dict[str, Any]] = {}

    def probe(threshold: int) -> dict[str, Any]:
        if threshold in observed:
            return observed[threshold]
        threshold_dir = out_dir / "candidates" / f"threshold-{threshold}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for instance in instances:
            result = probe_runner(
                cuber,
                instance["path"],
                threshold,
                threshold_dir / f"{instance['id']}.icnf",
                threshold_dir / f"{instance['id']}.log",
                max_rows,
                selector=selector,
                timeout_s=timeout_s,
            )
            rows.append(
                {
                    "id": instance["id"],
                    "tasks": int(result["tasks"]),
                    "elapsed_s": float(result["elapsed_s"]),
                    "cpu_s": float(result["user_s"]) + float(result["system_s"]),
                }
            )
        observed[threshold] = {"threshold": threshold, "instances": rows}
        return observed[threshold]

    bands = contract["frontier_bands"]
    brackets: dict[str, tuple[int, int]] = {}
    for band, spec in bands.items():
        target = int(spec["center_cubes"])
        lower, upper = 0, initial_threshold
        probe(lower)
        while median_tasks(probe(upper)) < target and upper < maximum_threshold:
            lower, upper = upper, min(upper * 2, maximum_threshold)
        if median_tasks(probe(upper)) < target:
            raise CalibrationError(
                f"maximum threshold did not reach {band} target {target}"
            )
        while upper - lower > 1:
            middle = (lower + upper) // 2
            if median_tasks(probe(middle)) >= target:
                upper = middle
            else:
                lower = middle
        brackets[band] = (lower, upper)

    response = sorted(observed.values(), key=lambda row: int(row["threshold"]))
    selections = {}
    for band, spec in bands.items():
        target = int(spec["center_cubes"])
        minimum = math.ceil(target * float(spec["accepted_ratio"][0]))
        maximum = math.floor(target * float(spec["accepted_ratio"][1]))
        selected = choose_width_response(response, target, minimum, maximum)
        threshold = int(selected["threshold"])
        final_rows = []
        for instance in instances:
            final_dir = out_dir / "selected" / band / instance["id"]
            final_dir.mkdir(parents=True, exist_ok=True)
            final = probe_runner(
                cuber,
                instance["path"],
                threshold,
                final_dir / "frontier.icnf",
                final_dir / "cuber.log",
                max_rows,
                final_dir / "nodes.jsonl",
                selector,
                timeout_s,
            )
            candidate = (
                out_dir
                / "candidates"
                / f"threshold-{threshold}"
                / f"{instance['id']}.icnf"
            )
            frontier = final_dir / "frontier.icnf"
            if sha256_file(candidate) != sha256_file(frontier):
                raise CalibrationError(
                    f"{instance['id']}: traced rerun changed selected frontier"
                )
            final_rows.append(
                {
                    "id": instance["id"],
                    "tasks": int(final["tasks"]),
                    "frontier_sha256": sha256_file(frontier),
                    "trace_sha256": sha256_file(final_dir / "nodes.jsonl"),
                    "cubing_elapsed_s": float(final["elapsed_s"]),
                    "cubing_cpu_s": float(final["user_s"])
                    + float(final["system_s"]),
                }
            )
        selections[band] = {
            "target_tasks": target,
            "accepted_task_range": [minimum, maximum],
            "selected_threshold": threshold,
            "search_bracket": list(brackets[band]),
            "median_tasks": median_tasks(selected),
            "calibration_loss": calibration_loss(selected, target)[0],
            "within_target_range": minimum <= median_tasks(selected) <= maximum,
            "instances": final_rows,
        }

    record = {
        "schema_version": 1,
        "kind": "width-level-cc-calibration-lock",
        "contract_sha256": contract_sha256(contract),
        "product_width": product_width,
        "selector": selector,
        "method": method,
        "max_rows": max_rows,
        "cuber_sha256": sha256_file(cuber),
        "calibration_instances": [
            {"id": instance["id"], "sha256": instance["sha256"]}
            for instance in instances
        ],
        "bands": selections,
        "response": response,
    }
    write_json(out_dir / "calibration-lock.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--product-width", type=int, choices=(64, 72, 80), required=True)
    parser.add_argument(
        "--selector", choices=("region", "structure-blind"), required=True
    )
    parser.add_argument("--cuber", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-threshold", type=int, default=1024)
    parser.add_argument("--maximum-threshold", type=int, default=1 << 120)
    args = parser.parse_args()
    try:
        contract = load_contract(args.contract)
        instances = calibration_inputs(contract, args.manifest, args.product_width)
        result = calibrate_width(
            contract=contract,
            instances=instances,
            product_width=args.product_width,
            selector=args.selector,
            cuber=args.cuber,
            out_dir=args.out_dir,
            initial_threshold=args.initial_threshold,
            maximum_threshold=args.maximum_threshold,
        )
    except (CalibrationError, HardRegimeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
