#!/usr/bin/env python3
"""Verify issue #51 contracts, calibration locks, terminal cells, and accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.cnc.conquer_parallel import distribution, lpt_makespan, read_cubes
from benchmarks.cnc.hard_regime import HardRegimeError, contract_sha256, load_contract
from benchmarks.cnc.hard_regime_matrix import (
    MatrixError,
    build_matrix,
    validate_instance_manifest,
    verify_toolchain,
)
from benchmarks.cnc.run_hard_regime_cell import verify_region_trace
from benchmarks.pipeline.circuit import (
    canonical_bytes,
    load_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)


class VerificationError(HardRegimeError):
    """The hard-regime evidence bundle is incomplete or inconsistent."""


def close(actual: object, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise VerificationError(f"{label} is not numeric")
    if math.isnan(expected):
        if not math.isnan(float(actual)):
            raise VerificationError(f"{label} should be NaN")
    elif not math.isclose(float(actual), expected, rel_tol=1e-8, abs_tol=1e-8):
        raise VerificationError(f"{label} mismatch: {actual} != {expected}")


def close_optional(actual: object, expected: float | None, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise VerificationError(f"{label} should be null")
        return
    close(actual, expected, label)


def checked_relative(root: Path, spec: object, label: str) -> Path:
    if not isinstance(spec, dict):
        raise VerificationError(f"{label} artifact spec is missing")
    relative = spec.get("path")
    expected = spec.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise VerificationError(f"{label} artifact provenance is incomplete")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise VerificationError(f"{label} artifact escapes its cell directory")
    if sha256_file(path) != expected:
        raise VerificationError(f"{label} artifact SHA-256 mismatch")
    if spec.get("bytes") != path.stat().st_size:
        raise VerificationError(f"{label} artifact byte count mismatch")
    return path


def verify_stage_logs(cell_dir: Path, stage: dict[str, Any], label: str) -> None:
    for stream in ("stdout", "stderr"):
        relative = stage.get(stream)
        expected = stage.get(f"{stream}_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise VerificationError(f"{label} stage has incomplete {stream} provenance")
        path = cell_dir / relative
        if sha256_file(path) != expected:
            raise VerificationError(f"{label} stage {stream} hash mismatch")


def verify_conquer_records(
    frontier: Path,
    results: Path,
    summary: dict[str, Any],
    workers: int,
    replay_workers: list[int],
) -> None:
    cubes = list(read_cubes(frontier))
    rows = read_jsonl(results)
    if len(rows) != len(cubes) or summary.get("cubes") != len(cubes):
        raise VerificationError("conquer records do not cover every frontier cube")
    by_index = {}
    intervals = []
    per_worker: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        index = row.get("cube_index")
        if not isinstance(index, int) or index < 0 or index >= len(cubes) or index in by_index:
            raise VerificationError("conquer records contain invalid or duplicate cube indices")
        by_index[index] = row
        expected_cube_hash = hashlib.sha256(
            (" ".join(map(str, cubes[index])) + " 0\n").encode()
        ).hexdigest()
        if row.get("cube_sha256") != expected_cube_hash:
            raise VerificationError(f"cube {index} assumption hash mismatch")
        released = row.get("released_monotonic_ns")
        started = row.get("started_monotonic_ns")
        finished = row.get("finished_monotonic_ns")
        collected = row.get("collected_monotonic_ns")
        worker = row.get("worker_pid")
        if not all(isinstance(value, int) for value in (released, started, finished, collected, worker)):
            raise VerificationError(f"cube {index} has malformed scheduling events")
        if not released <= started <= finished <= collected:
            raise VerificationError(f"cube {index} has illegal scheduling event order")
        intervals.append((started, 1))
        intervals.append((finished, -1))
        per_worker.setdefault(worker, []).append((started, finished))
    if set(by_index) != set(range(len(cubes))):
        raise VerificationError("conquer cube indices are not exhaustive")
    active = maximum = 0
    for _, delta in sorted(intervals, key=lambda event: (event[0], event[1])):
        active += delta
        if active < 0:
            raise VerificationError("conquer schedule has a finish before its start")
        maximum = max(maximum, active)
    if active or maximum > workers:
        raise VerificationError("conquer schedule exceeds the declared worker count")
    for worker, assigned in per_worker.items():
        assigned.sort()
        if any(left[1] > right[0] for left, right in zip(assigned, assigned[1:])):
            raise VerificationError(f"worker {worker} has overlapping cube assignments")

    ordered = [by_index[index] for index in range(len(cubes))]
    durations = [float(row["elapsed_s"]) for row in ordered]
    cpu = [float(row["user_s"]) + float(row["system_s"]) for row in ordered]
    conflicts = [float(row["conflicts"]) for row in ordered if row.get("conflicts") is not None]
    counts = {
        "timeouts": sum(bool(row.get("censored")) for row in ordered),
        "errors": sum(row.get("result") == "error" for row in ordered),
        "sat": sum(row.get("result") == "sat" for row in ordered),
        "unsat": sum(row.get("result") == "unsat" for row in ordered),
    }
    counts["completed"] = counts["sat"] + counts["unsat"]
    for name, value in counts.items():
        if summary.get(name) != value:
            raise VerificationError(f"conquer {name} count does not reconstruct")
    if summary.get("terminal_records") != len(ordered):
        raise VerificationError("conquer terminal-record count does not reconstruct")
    time_stats = distribution(durations)
    conflict_stats = distribution(conflicts)
    close(summary.get("total_solver_s"), sum(durations), "conquer total solver work")
    close(summary.get("total_cpu_s"), sum(cpu), "conquer total CPU work")
    close(summary.get("max_s"), time_stats["max"], "conquer span")
    close(summary.get("p99_s"), time_stats["p99"], "conquer p99")
    close_optional(
        summary.get("conflicts_max"),
        conflict_stats["max"],
        "conquer maximum conflicts",
    )
    measured = (
        0.0
        if not rows
        else (
            max(int(row["collected_monotonic_ns"]) for row in rows)
            - min(int(row["released_monotonic_ns"]) for row in rows)
        )
        / 1e9
    )
    close(summary.get("measured_makespan_s"), measured, "measured makespan")
    expected_lpt = {str(count): lpt_makespan(durations, count) for count in replay_workers}
    actual_lpt = summary.get("lpt_makespan_by_workers_s")
    if not isinstance(actual_lpt, dict) or set(actual_lpt) != set(expected_lpt):
        raise VerificationError("LPT replay worker set differs from the contract")
    for count, value in expected_lpt.items():
        close(actual_lpt[count], value, f"LPT {count}-worker makespan")
    if summary.get("lpt_is_lower_bound") != bool(counts["timeouts"] or counts["errors"]):
        raise VerificationError("LPT censoring designation is incorrect")


def verify_terminal(
    contract: dict[str, Any],
    matrix_path: Path,
    matrix: dict[str, Any],
    toolchain: dict[str, Any],
    cell: dict[str, Any],
    terminal: dict[str, Any],
    cell_dir: Path,
) -> str:
    expected = {
        "kind": "hard-regime-terminal-cell",
        "contract_sha256": contract_sha256(contract),
        "matrix_sha256": sha256_file(matrix_path),
        "toolchain_sha256": toolchain["toolchain_sha256"],
        "cell_id": cell["cell_id"],
        "cell_sha256": sha256_bytes(canonical_bytes(cell)),
        "instance_id": cell["instance_id"],
        "method": cell["method"],
        "budget": cell["budget"],
        "product_width": cell["product_width"],
        "factor_input_width": cell["factor_input_width"],
    }
    for field, value in expected.items():
        if terminal.get(field) != value:
            raise VerificationError(f"{cell['cell_id']}: terminal {field} mismatch")
    if terminal.get("state") not in {
        "complete",
        "monolithic-timeout",
        "monolithic-error",
        "cubing-timeout",
        "cubing-error",
        "conquer-timeout",
        "conquer-error",
        "wrong-answer",
        "harness-error",
    }:
        raise VerificationError(f"{cell['cell_id']}: unknown terminal state")
    if terminal.get("state") == "wrong-answer":
        raise VerificationError(f"{cell['cell_id']}: solver returned SAT for a prime target")
    inputs = terminal.get("input_artifacts")
    if not isinstance(inputs, dict):
        raise VerificationError(f"{cell['cell_id']}: terminal input hashes are missing")
    if inputs.get("global_cnf", {}).get("sha256") != cell["global_cnf_sha256"]:
        raise VerificationError(f"{cell['cell_id']}: mixed conquer encoding")
    if inputs.get("circuitsat", {}).get("sha256") != cell["circuitsat_sha256"]:
        raise VerificationError(f"{cell['cell_id']}: CircuitSAT input hash mismatch")

    stages = terminal.get("stages", {})
    metrics = terminal.get("metrics", {})
    artifacts = terminal.get("artifacts", {})
    if terminal["state"] == "harness-error":
        if not isinstance(terminal.get("error"), str) or not terminal["error"]:
            raise VerificationError(f"{cell['cell_id']}: harness error has no diagnostic")
        return "harness-terminal-only"
    if cell["method"] == "monolithic-kissat":
        stage = stages.get("monolithic")
        if not isinstance(stage, dict):
            raise VerificationError(f"{cell['cell_id']}: missing monolithic stage")
        verify_stage_logs(cell_dir, stage, "monolithic")
        close(
            metrics.get("end_to_end_wall_s"),
            float(cell["encoding_wall_s"]) + float(stage["wall_s"]),
            "monolithic end-to-end wall",
        )
        close(
            metrics.get("end_to_end_cpu_s"),
            float(cell["encoding_cpu_s"]) + float(stage["cpu_s"]),
            "monolithic end-to-end CPU",
        )
        return "monolithic-stage-reconstructed"

    cubing = stages.get("cubing")
    if not isinstance(cubing, dict):
        raise VerificationError(f"{cell['cell_id']}: missing cubing stage")
    verify_stage_logs(cell_dir, cubing, "cubing")
    if not cubing.get("complete"):
        if terminal["state"] not in {"cubing-timeout", "cubing-error"}:
            raise VerificationError(f"{cell['cell_id']}: incomplete cubing has wrong terminal state")
        return "cubing-stage-reconstructed"
    frontier = checked_relative(cell_dir, artifacts.get("frontier"), "frontier")
    if cubing.get("frontier_size") != sum(1 for _ in read_cubes(frontier)):
        raise VerificationError(f"{cell['cell_id']}: frontier size mismatch")
    if cell["method"] in {"region-cc", "structure-blind-cc"}:
        trace = checked_relative(cell_dir, artifacts.get("trace"), "trace")
        verify_region_trace(frontier, trace)
    conquer = stages.get("conquer")
    if not isinstance(conquer, dict):
        if terminal["state"] not in {"cubing-timeout", "cubing-error"}:
            raise VerificationError(f"{cell['cell_id']}: completed frontier has no conquer records")
        return "cubing-stage-reconstructed"
    results = checked_relative(cell_dir, artifacts.get("cube_results"), "cube results")
    verify_conquer_records(
        frontier,
        results,
        conquer,
        int(contract["scheduling"]["measured_workers"]),
        list(contract["scheduling"]["lpt_replay_workers"]),
    )
    close(metrics.get("conquer_work_cpu_s"), float(conquer["total_cpu_s"]), "conquer work")
    close(metrics.get("conquer_span_s"), float(conquer["max_s"]), "conquer span")
    expected_wall = (
        float(cell["encoding_wall_s"])
        + float(cubing["wall_s"])
        + float(conquer["measured_makespan_s"])
    )
    close(metrics.get("end_to_end_wall_s"), expected_wall, "CnC end-to-end wall")
    return "conquer-records-reconstructed"


def selected_cells(matrix: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    cells = matrix["cells"]
    if scope == "full":
        return cells
    return [
        cell
        for cell in cells
        if cell["method"] == "monolithic-kissat" or cell.get("pilot") is True
    ]


def verify_bundle(
    contract_path: Path,
    manifest_path: Path,
    calibration_root: Path,
    toolchain_path: Path,
    matrix_path: Path,
    runs_root: Path,
    scope: str,
    aggregate_path: Path | None = None,
) -> list[str]:
    contract = load_contract(contract_path)
    manifest = read_jsonl(manifest_path)
    validate_instance_manifest(contract, manifest)
    toolchain = load_json(toolchain_path)
    verify_toolchain(contract, toolchain, check_paths=False)
    matrix = load_json(matrix_path)
    regenerated = build_matrix(contract, manifest, calibration_root, toolchain)
    if matrix != regenerated:
        raise VerificationError("run matrix does not regenerate from frozen inputs")
    cells = selected_cells(matrix, scope)
    missing = []
    reconstruction = Counter()
    for cell in cells:
        cell_dir = runs_root / "cells" / cell["cell_id"]
        terminal_path = cell_dir / "terminal.json"
        if not terminal_path.is_file():
            missing.append(cell["cell_id"])
            continue
        reconstruction[
            verify_terminal(
            contract,
            matrix_path,
            matrix,
            toolchain,
            cell,
            load_json(terminal_path),
            cell_dir,
            )
        ] += 1
    if missing:
        raise VerificationError(
            f"missing terminal cells ({len(missing)}): {', '.join(missing[:3])}"
        )
    thresholds = {}
    for cell in matrix["cells"]:
        if "cc_threshold" not in cell:
            continue
        key = (cell["product_width"], cell["method"], cell["budget"])
        previous = thresholds.setdefault(key, cell["cc_threshold"])
        if previous != cell["cc_threshold"]:
            raise VerificationError("held-out cells contain per-instance threshold tuning")
    messages = [
        f"PASS contract: {contract_sha256(contract)}",
        "PASS widths/splits: factor-product semantics and calibration holdout are frozen",
        "PASS toolchain/encoding: source revisions, executables, and global CNFs are fixed",
        f"PASS completeness: {len(cells)} {scope} cells have terminal records",
        (
            "PASS reconstruction: "
            f"{reconstruction['conquer-records-reconstructed']} conquer cells reconstruct "
            "per-cube work, span, scheduling, LPT replay, and hashes; "
            f"{reconstruction['monolithic-stage-reconstructed'] + reconstruction['cubing-stage-reconstructed']} "
            "stage-terminal cells reconstruct their available logs/metrics; "
            f"{reconstruction['harness-terminal-only']} harness errors are explicitly "
            "terminal-only and are not claimed as work/span reconstruction"
        ),
        "PASS tuning: every width/method/budget uses one calibration-frozen threshold",
    ]
    if aggregate_path is not None:
        from benchmarks.cnc.aggregate_hard_regime import aggregate, load_terminals

        reported = load_json(aggregate_path)
        regenerated_aggregate = aggregate(
            contract, matrix, load_terminals(matrix, runs_root, scope), scope
        )
        if reported != regenerated_aggregate:
            raise VerificationError(
                "aggregate does not regenerate from instance-level terminal records"
            )
        if reported.get("statistical_unit") != "held-out-instance":
            raise VerificationError("aggregate uses cube-level pseudoreplication")
        for summary in reported.get("summaries", []):
            declared_ids = summary.get("declared_instance_ids")
            instance_ids = summary.get("instance_ids")
            if not isinstance(declared_ids, list) or len(declared_ids) != len(
                set(declared_ids)
            ):
                raise VerificationError(
                    "aggregate summary duplicates a declared held-out instance"
                )
            if summary.get("declared_pairs") != len(declared_ids):
                raise VerificationError(
                    "aggregate declared pair count does not match instance IDs"
                )
            if not isinstance(instance_ids, list) or len(instance_ids) != len(
                set(instance_ids)
            ):
                raise VerificationError(
                    "aggregate summary duplicates a held-out instance"
                )
            if summary.get("complete_pairs") != len(instance_ids):
                raise VerificationError(
                    "aggregate pair count does not match instance IDs"
                )
        messages.append(
            "PASS statistics: paired ratios and bootstrap samples use held-out instances"
        )
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), default="full")
    parser.add_argument("--aggregate", type=Path)
    args = parser.parse_args()
    try:
        messages = verify_bundle(
            args.contract,
            args.manifest,
            args.calibration_root,
            args.toolchain,
            args.matrix,
            args.runs_root,
            args.scope,
            args.aggregate,
        )
    except (VerificationError, MatrixError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("\n".join(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
