#!/usr/bin/env python3
"""Calibrate the classical online CC difficulty cutoff by emitted task count."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


_STATS = re.compile(r"status=OK cubes=(\d+).*cutoff=CcDifficulty\((\d+)\)")


class CalibrationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def task_lines(path: Path) -> int:
    return sum(
        line.startswith("a ") or line == "a 0\n"
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
    )


def run_cuber(
    cuber: Path,
    instance: Path,
    threshold: int,
    cubes: Path,
    log: Path,
    max_rows: int,
    trace: Path | None = None,
    selector: str = "region",
    timeout_s: float | None = None,
) -> dict[str, int | float]:
    if selector not in {"region", "structure-blind"}:
        raise CalibrationError(f"unknown selector {selector!r}")
    command = [
        str(cuber.resolve()),
        str(instance.resolve()),
        "--cc-threshold",
        str(threshold),
        "-o",
        str(cubes),
        "--max-rows",
        str(max_rows),
        "--selector",
        selector,
    ]
    if trace:
        command.extend(("--trace", str(trace)))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        log.write_text(stderr, encoding="utf-8")
        raise CalibrationError(
            f"cuber selector={selector} threshold={threshold} timed out after {elapsed:.3f}s"
        ) from exc
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    log.write_text(process.stderr, encoding="utf-8")
    match = _STATS.search(process.stderr)
    if process.returncode or not match or int(match.group(2)) != threshold:
        raise CalibrationError(
            f"cuber threshold={threshold} failed: {process.stderr[-500:]}"
        )
    tasks = int(match.group(1))
    if task_lines(cubes) != tasks:
        raise CalibrationError("reported and emitted task counts differ")
    return {
        "threshold": threshold,
        "tasks": tasks,
        "elapsed_s": elapsed,
        "user_s": after.ru_utime - before.ru_utime,
        "system_s": after.ru_stime - before.ru_stime,
    }


def choose(
    rows: list[dict[str, int | float]], target: int, minimum: int, maximum: int
) -> dict[str, int | float]:
    inside = [row for row in rows if minimum <= int(row["tasks"]) <= maximum]
    pool = inside or rows
    if not pool:
        raise CalibrationError("empty cutoff response")
    return min(
        pool,
        key=lambda row: (
            abs(math.log2(max(1, int(row["tasks"])) / target)),
            abs(int(row["tasks"]) - target),
            int(row["threshold"]),
        ),
    )


def calibrate(
    instance: Path,
    cuber: Path,
    out_dir: Path,
    target: int,
    minimum: int,
    maximum: int,
    initial: int,
    maximum_threshold: int,
    max_rows: int,
    selector: str = "region",
    timeout_s: float | None = None,
) -> dict[str, Any]:
    if target <= 0 or minimum <= 0 or maximum < minimum or initial <= 0:
        raise CalibrationError("invalid task range or threshold")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = out_dir / "candidates"
    candidates.mkdir(exist_ok=True)
    observed: dict[int, dict[str, int | float]] = {}

    def probe(threshold: int) -> dict[str, int | float]:
        if threshold not in observed:
            observed[threshold] = run_cuber(
                cuber,
                instance,
                threshold,
                candidates / f"threshold-{threshold}.icnf",
                candidates / f"threshold-{threshold}.log",
                max_rows,
                selector=selector,
                timeout_s=timeout_s,
            )
        return observed[threshold]

    lower, upper = 0, initial
    probe(lower)
    while int(probe(upper)["tasks"]) < target and upper < maximum_threshold:
        lower, upper = upper, min(upper * 2, maximum_threshold)
    if int(probe(upper)["tasks"]) < target:
        raise CalibrationError("maximum threshold did not reach target task count")
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if int(probe(middle)["tasks"]) >= target:
            upper = middle
        else:
            lower = middle

    response = sorted(observed.values(), key=lambda row: int(row["threshold"]))
    selected = choose(response, target, minimum, maximum)
    threshold = int(selected["threshold"])
    final = run_cuber(
        cuber,
        instance,
        threshold,
        out_dir / "frontier.icnf",
        out_dir / "final.log",
        max_rows,
        out_dir / "nodes.jsonl",
        selector,
        timeout_s,
    )
    candidate = candidates / f"threshold-{threshold}.icnf"
    if sha256_file(candidate) != sha256_file(out_dir / "frontier.icnf"):
        raise CalibrationError("traced rerun changed frontier bytes")
    record = {
        "schema_version": 1,
        "method": "classical-cc-difficulty-task-count-calibration",
        "selector": selector,
        "formula": "D^2*(D+I)/N > threshold",
        "instance": str(instance),
        "instance_sha256": sha256_file(instance),
        "cuber_sha256": sha256_file(cuber),
        "target_tasks": target,
        "accepted_task_range": [minimum, maximum],
        "selected_threshold": threshold,
        "tasks": int(final["tasks"]),
        "cubing_elapsed_s": float(final["elapsed_s"]),
        "cubing_cpu_s": float(final["user_s"]) + float(final["system_s"]),
        "frontier_sha256": sha256_file(out_dir / "frontier.icnf"),
        "trace_sha256": sha256_file(out_dir / "nodes.jsonl"),
        "within_target_range": minimum <= int(final["tasks"]) <= maximum,
        "response": response,
    }
    (out_dir / "selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--cuber", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=512)
    parser.add_argument("--min-tasks", type=int, default=384)
    parser.add_argument("--max-tasks", type=int, default=640)
    parser.add_argument("--initial-threshold", type=int, default=1024)
    parser.add_argument("--max-threshold", type=int, default=1 << 60)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument(
        "--selector", choices=("region", "structure-blind"), default="region"
    )
    parser.add_argument("--timeout-s", type=float)
    args = parser.parse_args()
    try:
        result = calibrate(
            args.instance,
            args.cuber,
            args.out_dir,
            args.target,
            args.min_tasks,
            args.max_tasks,
            args.initial_threshold,
            args.max_threshold,
            args.max_rows,
            args.selector,
            args.timeout_s,
        )
    except (CalibrationError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
