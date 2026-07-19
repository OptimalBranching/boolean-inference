#!/usr/bin/env python3
"""Aggregate issue #51 terminal cells with held-out instances as the sample unit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from benchmarks.cnc.hard_regime import HardRegimeError, contract_sha256, load_contract
from benchmarks.cnc.verify_hard_regime import selected_cells
from benchmarks.pipeline.circuit import load_json, write_json, write_jsonl


class AggregateError(HardRegimeError):
    """Terminal records cannot produce the preregistered instance-level analysis."""


METRICS = {
    "conquer_work_cpu_s": "Conquer work",
    "conquer_span_s": "Maximum cube time",
    "end_to_end_wall_s": "End-to-end wall time",
    "measured_32_worker_makespan_s": "Measured 32-worker makespan",
    "lpt_32_s": "LPT 32-worker makespan",
    "lpt_128_s": "LPT 128-worker makespan",
    "lpt_512_s": "LPT 512-worker makespan",
}


def metric_value(terminal: dict[str, Any], metric: str) -> float | None:
    metrics = terminal.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if metric.startswith("lpt_"):
        workers = metric.split("_")[1]
        replay = metrics.get("lpt_makespan_by_workers_s")
        value = replay.get(workers) if isinstance(replay, dict) else None
    else:
        value = metrics.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return float(value)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise AggregateError("geometric mean requires positive observations")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def bootstrap_ci(
    values: list[float], samples: int, confidence: float, seed: int, key: str
) -> list[float] | None:
    if not values:
        return None
    key_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed ^ key_seed)
    replicates = [
        geometric_mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return [percentile(replicates, alpha), percentile(replicates, 1.0 - alpha)]


def interpolate_log(points: list[tuple[float, float]], target: float) -> float | None:
    usable = sorted((x, y) for x, y in points if x > 0 and y > 0)
    if not usable or target < usable[0][0] or target > usable[-1][0]:
        return None
    for x, y in usable:
        if math.isclose(x, target, rel_tol=1e-12, abs_tol=0.0):
            return y
    for (left_x, left_y), (right_x, right_y) in zip(usable, usable[1:]):
        if left_x <= target <= right_x:
            if left_x == right_x:
                return geometric_mean([left_y, right_y])
            fraction = (math.log(target) - math.log(left_x)) / (
                math.log(right_x) - math.log(left_x)
            )
            return math.exp(math.log(left_y) + fraction * (math.log(right_y) - math.log(left_y)))
    return None


def load_terminals(
    matrix: dict[str, Any], runs_root: Path, scope: str
) -> dict[str, dict[str, Any]]:
    terminals = {}
    for cell in selected_cells(matrix, scope):
        path = runs_root / "cells" / cell["cell_id"] / "terminal.json"
        if not path.is_file():
            raise AggregateError(f"missing terminal cell {cell['cell_id']}")
        terminal = load_json(path)
        if terminal.get("cell_id") != cell["cell_id"]:
            raise AggregateError(f"terminal identity mismatch for {cell['cell_id']}")
        terminals[cell["cell_id"]] = terminal
    return terminals


def paired_raw_observations(
    matrix: dict[str, Any], terminals: dict[str, dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    cells = {
        (cell["instance_id"], cell["method"], cell["budget"]): cell
        for cell in matrix["cells"]
        if cell["split"] == "held_out"
    }
    observations = []
    for instance_id in sorted({key[0] for key in cells}):
        for budget in ("low", "medium", "high"):
            region = cells.get((instance_id, "region-cc", budget))
            blind = cells.get((instance_id, "structure-blind-cc", budget))
            if region is None or blind is None:
                continue
            region_terminal = terminals.get(region["cell_id"])
            blind_terminal = terminals.get(blind["cell_id"])
            if region_terminal is None or blind_terminal is None:
                continue
            region_value = metric_value(region_terminal, metric)
            blind_value = metric_value(blind_terminal, metric)
            complete = (
                region_terminal.get("state") == "complete"
                and blind_terminal.get("state") == "complete"
                and region_value is not None
                and blind_value is not None
            )
            observations.append(
                {
                    "analysis": "raw-nominal-budget",
                    "instance_id": instance_id,
                    "product_width": region["product_width"],
                    "budget": budget,
                    "metric": metric,
                    "region_state": region_terminal.get("state"),
                    "blind_state": blind_terminal.get("state"),
                    "region_value": region_value,
                    "blind_value": blind_value,
                    "ratio": region_value / blind_value if complete else None,
                    "complete_pair": complete,
                }
            )
    return observations


def adjusted_observations(
    matrix: dict[str, Any], terminals: dict[str, dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    cells = {
        (cell["instance_id"], cell["method"], cell["budget"]): cell
        for cell in matrix["cells"]
        if cell["split"] == "held_out"
    }
    observations = []
    declared_instances = sorted(
        {
            cell["instance_id"]
            for cell in matrix["cells"]
            if cell["split"] == "held_out"
            and cell["method"] in {"region-cc", "structure-blind-cc"}
            and cell["cell_id"] in terminals
        }
    )
    for instance_id in declared_instances:
        series = {}
        product_width = next(
            cell["product_width"]
            for cell in matrix["cells"]
            if cell["instance_id"] == instance_id
        )
        for method in ("region-cc", "structure-blind-cc"):
            points = []
            states = []
            for budget in ("low", "medium", "high"):
                cell = cells.get((instance_id, method, budget))
                if cell is None or cell["cell_id"] not in terminals:
                    continue
                product_width = cell["product_width"]
                terminal = terminals[cell["cell_id"]]
                states.append(terminal.get("state"))
                x = terminal.get("metrics", {}).get("frontier_size")
                y = metric_value(terminal, metric)
                if terminal.get("state") == "complete" and isinstance(x, (int, float)) and x > 0 and y is not None:
                    points.append((float(x), y))
            series[method] = {"points": points, "states": states}
        region_points = series["region-cc"]["points"]
        blind_points = series["structure-blind-cc"]["points"]
        if not region_points or not blind_points:
            support = None
            adjustment_status = "incomplete-series"
        else:
            lower = max(min(x for x, _ in region_points), min(x for x, _ in blind_points))
            upper = min(max(x for x, _ in region_points), max(x for x, _ in blind_points))
            support = (lower, upper) if lower <= upper else None
            adjustment_status = "common-support" if support is not None else "no-common-support"
        grids = (
            {}
            if support is None
            else {
                "overlap-low": support[0],
                "overlap-mid": math.sqrt(support[0] * support[1]),
                "overlap-high": support[1],
            }
        )
        for grid in ("overlap-low", "overlap-mid", "overlap-high"):
            frontier_size = grids.get(grid)
            region_value = (
                interpolate_log(region_points, frontier_size)
                if frontier_size is not None
                else None
            )
            blind_value = (
                interpolate_log(blind_points, frontier_size)
                if frontier_size is not None
                else None
            )
            complete = region_value is not None and blind_value is not None
            observations.append(
                {
                    "analysis": "frontier-budget-adjusted",
                    "instance_id": instance_id,
                    "product_width": product_width,
                    "grid": grid,
                    "common_frontier_size": frontier_size,
                    "metric": metric,
                    "adjustment_status": adjustment_status,
                    "region_states": series["region-cc"]["states"],
                    "blind_states": series["structure-blind-cc"]["states"],
                    "region_value": region_value,
                    "blind_value": blind_value,
                    "ratio": region_value / blind_value if complete else None,
                    "complete_pair": complete,
                }
            )
    return observations


def summarize_observations(
    observations: list[dict[str, Any]],
    analysis: str,
    metric: str,
    budget_key: str,
    budget_value: str,
    product_width: int | None,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    selected = [
        row
        for row in observations
        if row["analysis"] == analysis
        and row["metric"] == metric
        and row.get(budget_key) == budget_value
        and (product_width is None or row["product_width"] == product_width)
    ]
    complete = [row for row in selected if row["complete_pair"]]
    ratios = [float(row["ratio"]) for row in complete]
    instance_ids = [row["instance_id"] for row in complete]
    declared_instance_ids = [row["instance_id"] for row in selected]
    if len(instance_ids) != len(set(instance_ids)):
        raise AggregateError("an aggregate group contains duplicate instance observations")
    group = "overall" if product_width is None else f"p{product_width}"
    key = f"{analysis}:{metric}:{budget_key}={budget_value}:{group}"
    geometric_ratio = geometric_mean(ratios) if ratios else None
    if metric == "conquer_work_cpu_s":
        criterion = "ratio-no-worse-than-1.10"
        meets_criterion = geometric_ratio is not None and geometric_ratio <= 1.10
    elif metric == "conquer_span_s":
        criterion = "ratio-below-1.00"
        meets_criterion = geometric_ratio is not None and geometric_ratio < 1.0
    else:
        criterion = None
        meets_criterion = None
    return {
        "analysis": analysis,
        "metric": metric,
        budget_key: budget_value,
        "group": group,
        "statistical_unit": "held-out-instance",
        "declared_pairs": len(selected),
        "complete_pairs": len(complete),
        "censored_or_failed_pairs": len(selected) - len(complete),
        "declared_instance_ids": declared_instance_ids,
        "censored_or_failed_instance_ids": [
            row["instance_id"] for row in selected if not row["complete_pair"]
        ],
        "instance_ids": instance_ids,
        "geometric_mean_ratio": geometric_ratio,
        "region_win_rate": (
            sum(ratio < 1.0 for ratio in ratios) / len(ratios) if ratios else None
        ),
        "bootstrap_95_ci": bootstrap_ci(
            ratios,
            int(bootstrap["samples"]),
            float(bootstrap["confidence"]),
            int(bootstrap["seed"]),
            key,
        ),
        "scientific_target": criterion,
        "meets_scientific_target": meets_criterion,
    }


def terminal_cell_rows(
    matrix: dict[str, Any], terminals: dict[str, dict[str, Any]], scope: str
) -> list[dict[str, Any]]:
    rows = []
    for cell in selected_cells(matrix, scope):
        terminal = terminals[cell["cell_id"]]
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "instance_id": cell["instance_id"],
                "split": cell["split"],
                "product_width": cell["product_width"],
                "factor_input_width": cell["factor_input_width"],
                "pilot": cell["pilot"],
                "method": cell["method"],
                "budget": cell["budget"],
                "state": terminal["state"],
                "verdict": terminal.get("verdict"),
                "metrics": terminal.get("metrics"),
            }
        )
    return rows


def method_summaries(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    method_budgets = sorted({(row["method"], row["budget"]) for row in cell_rows})
    for method, budget in method_budgets:
        for width in (None, 64, 72, 80):
            rows = [
                row
                for row in cell_rows
                if row["method"] == method
                and row["budget"] == budget
                and row["split"] == "held_out"
                and (width is None or row["product_width"] == width)
            ]
            if not rows:
                continue
            states: dict[str, int] = {}
            for row in rows:
                states[row["state"]] = states.get(row["state"], 0) + 1
            numeric = {}
            metric_names = (
                "solver_wall_s",
                "conquer_work_cpu_s",
                "conquer_span_s",
                "measured_32_worker_makespan_s",
                "end_to_end_wall_s",
            )
            for metric in metric_names:
                values = [
                    float(row["metrics"][metric])
                    for row in rows
                    if row["state"] == "complete"
                    and isinstance(row.get("metrics"), dict)
                    and isinstance(row["metrics"].get(metric), (int, float))
                    and row["metrics"][metric] > 0
                ]
                numeric[metric] = {
                    "complete_values": len(values),
                    "geometric_mean": geometric_mean(values) if values else None,
                }
            summaries.append(
                {
                    "method": method,
                    "budget": budget,
                    "group": "overall" if width is None else f"p{width}",
                    "declared_instances": len(rows),
                    "terminal_states": states,
                    "metrics": numeric,
                }
            )
    return summaries


def hardness_summary(matrix: dict[str, Any], terminals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for width in (64, 72, 80):
        cells = [
            cell
            for cell in matrix["cells"]
            if cell["method"] == "monolithic-kissat"
            and cell["split"] == "held_out"
            and cell["product_width"] == width
        ]
        states = [terminals[cell["cell_id"]]["state"] for cell in cells]
        timeouts = sum(state == "monolithic-timeout" for state in states)
        rows.append(
            {
                "product_width": width,
                "held_out_instances": len(cells),
                "monolithic_timeouts": timeouts,
                "cnc_regime": timeouts > len(cells) / 2,
            }
        )
    return rows


def pilot_gate(
    matrix: dict[str, Any], terminals: dict[str, dict[str, Any]], hardness: list[dict[str, Any]]
) -> dict[str, Any]:
    regime_widths = {
        row["product_width"] for row in hardness if row["cnc_regime"]
    }
    frontier_methods = []
    for width in sorted(regime_widths):
        for method in ("march-cu-dynamic", "region-cc", "structure-blind-cc"):
            cells = [
                cell
                for cell in matrix["cells"]
                if cell.get("pilot")
                and cell["product_width"] == width
                and cell["method"] == method
            ]
            if cells and all(
                terminals[cell["cell_id"]].get("stages", {}).get("cubing", {}).get("complete")
                for cell in cells
            ):
                frontier_methods.append({"product_width": width, "method": method})
    artifact_reconstruction = all(
        terminal.get("state") != "harness-error" for terminal in terminals.values()
    )
    return {
        "at_least_two_hard_widths": len(regime_widths) >= 2,
        "hard_widths": sorted(regime_widths),
        "complete_frontier_methods_in_regime": frontier_methods,
        "at_least_one_complete_frontier_method": bool(frontier_methods),
        "artifact_reconstruction": artifact_reconstruction,
        "passed": (
            len(regime_widths) >= 2
            and bool(frontier_methods)
            and artifact_reconstruction
        ),
    }


def write_primary_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    relevant = [
        row
        for row in summaries
        if row["analysis"] == "raw-nominal-budget"
        and row["metric"] in {"conquer_work_cpu_s", "conquer_span_s"}
        and row["group"] != "overall"
    ]
    lookup = {
        (row["group"], row["budget"], row["metric"]): row for row in relevant
    }
    lines = [
        "| Product width | Budget | Complete pairs | Work ratio [95% CI] | Span ratio [95% CI] | Span win rate |",
        "|---:|:---|---:|:---|:---|---:|",
    ]

    def ratio(row: dict[str, Any] | None) -> str:
        if row is None or row["geometric_mean_ratio"] is None:
            return "NA"
        interval = row["bootstrap_95_ci"]
        return f"{row['geometric_mean_ratio']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"

    for width in (64, 72, 80):
        for budget in ("low", "medium", "high"):
            work = lookup.get((f"p{width}", budget, "conquer_work_cpu_s"))
            span = lookup.get((f"p{width}", budget, "conquer_span_s"))
            complete = min(
                work["complete_pairs"] if work else 0,
                span["complete_pairs"] if span else 0,
            )
            win = "NA" if span is None or span["region_win_rate"] is None else f"{span['region_win_rate']:.1%}"
            lines.append(
                f"| {width} | {budget} | {complete} | {ratio(work)} | {ratio(span)} | {win} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_csv(path: Path, observations: list[dict[str, Any]]) -> None:
    rows = [row for row in observations if row["analysis"] == "raw-nominal-budget"]
    fields = [
        "instance_id",
        "product_width",
        "budget",
        "metric",
        "region_state",
        "blind_state",
        "region_value",
        "blind_value",
        "ratio",
        "complete_pair",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    scope: str,
    hardness: list[dict[str, Any]],
    gate: dict[str, Any],
    summaries: list[dict[str, Any]],
    methods: list[dict[str, Any]],
) -> None:
    complete_raw = sum(
        row["complete_pairs"]
        for row in summaries
        if row["analysis"] == "raw-nominal-budget" and row["group"] == "overall"
    )
    complete_adjusted = sum(
        row["complete_pairs"]
        for row in summaries
        if row["analysis"] == "frontier-budget-adjusted" and row["group"] == "overall"
    )
    lines = [
        "# Hard-regime CnC work/span report",
        "",
        f"Scope: `{scope}`. Statistical unit: held-out factoring instance.",
        "",
        "## Hardness gate",
        "",
    ]
    lines.extend(
        f"- Product width {row['product_width']}: {row['monolithic_timeouts']}/{row['held_out_instances']} monolithic timeouts; CnC regime = {row['cnc_regime']}."
        for row in hardness
    )
    lines.extend(
        [
            "",
            f"Pilot gate passed: **{gate['passed']}**.",
            "",
            "## Raw paired results",
            "",
            f"Complete overall instance-pairs across metrics/budgets: {complete_raw}. Censored and failed pairs remain in the machine-readable rows.",
            "",
            "## Frontier-budget-adjusted results",
            "",
            f"Complete interpolated overall instance-pairs across metrics/grid points: {complete_adjusted}. Interpolation is log-log within common support only; no extrapolation is used.",
            "",
            "## Finite-worker results",
            "",
            "LPT makespans are reported for 32, 128, and 512 workers. The directly measured 32-worker schedule remains separate from replayed LPT estimates.",
        ]
    )
    lines.extend(["", "## Method terminal overview", ""])
    for row in methods:
        if row["group"] != "overall":
            continue
        end_to_end = row["metrics"]["end_to_end_wall_s"]["geometric_mean"]
        end_text = "NA" if end_to_end is None else f"{end_to_end:.3f}s"
        lines.append(
            f"- `{row['method']}` / `{row['budget']}`: states={row['terminal_states']}; complete-case geometric mean end-to-end={end_text}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(
    contract: dict[str, Any],
    matrix: dict[str, Any],
    terminals: dict[str, dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    cells = terminal_cell_rows(matrix, terminals, scope)
    observations = []
    for metric in METRICS:
        observations.extend(paired_raw_observations(matrix, terminals, metric))
        observations.extend(adjusted_observations(matrix, terminals, metric))
    bootstrap = contract["statistics"]["bootstrap"]
    summaries = []
    for analysis, budget_key, budget_values in (
        ("raw-nominal-budget", "budget", ("low", "medium", "high")),
        (
            "frontier-budget-adjusted",
            "grid",
            ("overlap-low", "overlap-mid", "overlap-high"),
        ),
    ):
        for metric in METRICS:
            for budget_value in budget_values:
                for width in (None, 64, 72, 80):
                    summaries.append(
                        summarize_observations(
                            observations,
                            analysis,
                            metric,
                            budget_key,
                            budget_value,
                            width,
                            bootstrap,
                        )
                    )
    hardness = hardness_summary(matrix, terminals)
    gate = pilot_gate(matrix, terminals, hardness)
    return {
        "schema_version": 1,
        "kind": "hard-regime-instance-level-aggregate",
        "contract_sha256": contract_sha256(contract),
        "scope": scope,
        "statistical_unit": "held-out-instance",
        "bootstrap": bootstrap,
        "hardness": hardness,
        "pilot_gate": gate,
        "method_summaries": method_summaries(cells),
        "summaries": summaries,
        "terminal_cells": cells,
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), default="full")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        contract = load_contract(args.contract)
        matrix = load_json(args.matrix)
        terminals = load_terminals(matrix, args.runs_root, args.scope)
        result = aggregate(contract, matrix, terminals, args.scope)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "aggregate.json", result)
        write_jsonl(args.out_dir / "terminal-cells.jsonl", result["terminal_cells"])
        write_jsonl(args.out_dir / "paired-observations.jsonl", result["observations"])
        write_raw_csv(args.out_dir / "raw-paired-results.csv", result["observations"])
        write_primary_table(args.out_dir / "primary-table.md", result["summaries"])
        write_report(
            args.out_dir / "report.md",
            args.scope,
            result["hardness"],
            result["pilot_gate"],
            result["summaries"],
            result["method_summaries"],
        )
    except (AggregateError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"PASS aggregate: {args.out_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
