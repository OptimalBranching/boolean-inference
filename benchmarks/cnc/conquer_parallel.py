#!/usr/bin/env python3
"""Conquer frozen cube frontiers in parallel with auditable per-cube records."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from collections.abc import Iterator
from typing import Any


_CNF_HEADER = re.compile(rb"^p cnf (\d+) (\d+)\s*$", re.MULTILINE)
_STAT = re.compile(r"^c\s+(decisions|conflicts):\s+(\d+)", re.MULTILINE)
_BASE_HEADER: bytes
_BASE_BODY: bytes
_KISSAT: str
_TIMEOUT_S: float
_TMPDIR: str | None


def parse_cnf(data: bytes) -> tuple[int, int, bytes]:
    match = _CNF_HEADER.search(data)
    if not match:
        raise ValueError("DIMACS header not found")
    variables, clauses = map(int, match.groups())
    body = data[match.end() :].lstrip(b"\r\n")
    return variables, clauses, body


def read_cubes(path: Path) -> Iterator[list[int]]:
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0] == "c":
                continue
            if fields[0] != "a" or fields[-1] != "0":
                raise ValueError(f"{path}:{lineno}: expected 'a <literals> 0'")
            literals = [int(field) for field in fields[1:-1]]
            if any(literal == 0 for literal in literals):
                raise ValueError(f"{path}:{lineno}: embedded zero literal")
            yield literals


def parse_stats(stdout: str) -> tuple[int, int]:
    values = {name: int(value) for name, value in _STAT.findall(stdout)}
    return values.get("decisions", 0), values.get("conflicts", 0)


def child_cpu_seconds(before: resource.struct_rusage) -> tuple[float, float]:
    """Return user/system CPU consumed by children since ``before``."""

    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return after.ru_utime - before.ru_utime, after.ru_stime - before.ru_stime


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "total": 0.0,
            "mean": math.nan,
            "cv": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "p99_over_p95": math.nan,
            "max": math.nan,
        }
    ordered = sorted(values)

    def ordered_percentile(quantile: float) -> float:
        index = min(
            len(ordered) - 1,
            max(0, round(quantile * (len(ordered) - 1))),
        )
        return ordered[index]

    total = sum(values)
    mean = total / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    p95 = ordered_percentile(0.95)
    p99 = ordered_percentile(0.99)
    return {
        "total": total,
        "mean": mean,
        "cv": math.sqrt(variance) / mean if mean else math.nan,
        "p50": ordered_percentile(0.50),
        "p95": p95,
        "p99": p99,
        "p99_over_p95": p99 / p95 if p95 > 0 else math.nan,
        "max": ordered[-1],
    }


def lpt_makespan(durations: list[float], workers: int) -> float:
    loads = [0.0] * workers
    for duration in sorted(durations, reverse=True):
        slot = min(range(workers), key=loads.__getitem__)
        loads[slot] += duration
    return max(loads, default=0.0)


def _configure_worker(
    variables: int,
    clauses: int,
    body: bytes,
    kissat: str,
    timeout_s: float,
    tmpdir: str | None,
) -> None:
    global _BASE_HEADER, _BASE_BODY, _KISSAT, _TIMEOUT_S, _TMPDIR
    _BASE_HEADER = f"p cnf {variables} {clauses}".encode()
    _BASE_BODY = body
    _KISSAT = kissat
    _TIMEOUT_S = timeout_s
    _TMPDIR = tmpdir


def _solve_cube(task: tuple[str, int, list[int]]) -> dict[str, Any]:
    arm, index, cube = task
    header_fields = _BASE_HEADER.split()
    clause_count = int(header_fields[3]) + len(cube)
    header = b" ".join((*header_fields[:3], str(clause_count).encode())) + b"\n"
    units = b"".join(f"{literal} 0\n".encode() for literal in cube)
    payload = header + _BASE_BODY + units
    started_ns = time.monotonic_ns()
    temporary = tempfile.NamedTemporaryFile(
        prefix=f"cube-{arm}-{index}-", suffix=".cnf", dir=_TMPDIR, delete=False
    )
    try:
        with temporary:
            temporary.write(payload)
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            process = subprocess.run(
                [_KISSAT, "--statistics", "--relaxed", temporary.name],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
                check=False,
            )
            elapsed_s = (time.monotonic_ns() - started_ns) / 1e9
            user_s, system_s = child_cpu_seconds(usage_before)
            decisions, conflicts = parse_stats(process.stdout)
            result = {10: "sat", 20: "unsat"}.get(process.returncode, "error")
            return {
                "schema_version": 1,
                "arm": arm,
                "cube_index": index,
                "cube_literals": len(cube),
                "result": result,
                "returncode": process.returncode,
                "elapsed_s": elapsed_s,
                "user_s": user_s,
                "system_s": system_s,
                "decisions": decisions,
                "conflicts": conflicts,
                "censored": False,
                "stderr_tail": process.stderr[-500:] if result == "error" else "",
            }
        except subprocess.TimeoutExpired:
            user_s, system_s = child_cpu_seconds(usage_before)
            return {
                "schema_version": 1,
                "arm": arm,
                "cube_index": index,
                "cube_literals": len(cube),
                "result": "timeout",
                "returncode": None,
                "elapsed_s": (time.monotonic_ns() - started_ns) / 1e9,
                "user_s": user_s,
                "system_s": system_s,
                "decisions": None,
                "conflicts": None,
                "censored": True,
                "stderr_tail": "",
            }
    finally:
        os.unlink(temporary.name)


def summarize(
    *,
    cubes: int,
    completed: int,
    timeouts: int,
    errors: int,
    sat: int,
    unsat: int,
    durations: list[float],
    cpu_durations: list[float],
    decisions: list[float],
    conflicts: list[float],
    workers: int,
    wall_s: float,
) -> dict[str, Any]:
    time_stats = distribution(durations)
    cpu_stats = distribution(cpu_durations)
    decision_stats = distribution(decisions)
    conflict_stats = distribution(conflicts)
    return {
        "cubes": cubes,
        "completed": completed,
        "timeouts": timeouts,
        "errors": errors,
        "sat": sat,
        "unsat": unsat,
        "total_solver_s": time_stats["total"],
        "total_cpu_s": cpu_stats["total"],
        "cpu_time": cpu_stats,
        "observed_parallel_wall_s": wall_s,
        "lpt_makespan_s": lpt_makespan(durations, workers),
        "cpu_lpt_makespan_s": lpt_makespan(cpu_durations, workers),
        "p50_s": time_stats["p50"],
        "p95_s": time_stats["p95"],
        "p99_s": time_stats["p99"],
        "max_s": time_stats["max"],
        "p99_over_p95": time_stats["p99_over_p95"],
        "time_cv": time_stats["cv"],
        "total_decisions": int(decision_stats["total"]),
        "total_conflicts": int(conflict_stats["total"]),
        "conflicts_p50": conflict_stats["p50"],
        "conflicts_p95": conflict_stats["p95"],
        "conflicts_p99": conflict_stats["p99"],
        "conflicts_max": conflict_stats["max"],
        "conflicts_cv": conflict_stats["cv"],
        "conflicts_p99_over_p95": conflict_stats["p99_over_p95"],
    }


def run_arm(
    arm: str,
    cubes: Iterator[list[int]],
    total_cubes: int,
    workers: int,
    output: Path,
    worker_args: tuple[Any, ...],
) -> dict[str, Any]:
    durations: list[float] = []
    cpu_durations: list[float] = []
    decisions: list[float] = []
    conflicts: list[float] = []
    counts = {"completed": 0, "timeouts": 0, "errors": 0, "sat": 0, "unsat": 0}
    progress_every = 10_000 if total_cubes > 10_000 else 200
    started = time.monotonic()
    with output.open("w", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_configure_worker,
            initargs=worker_args,
        ) as pool:
            indexed_cubes = enumerate(cubes)
            pending: set[concurrent.futures.Future[dict[str, Any]]] = set()

            def submit_one() -> bool:
                try:
                    index, cube = next(indexed_cubes)
                except StopIteration:
                    return False
                pending.add(pool.submit(_solve_cube, (arm, index, cube)))
                return True

            for _ in range(workers * 4):
                if not submit_one():
                    break

            done_count = 0
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    pending.remove(future)
                    row = future.result()
                    row["completed_monotonic_ns"] = time.monotonic_ns()
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    done_count += 1
                    if row["censored"]:
                        counts["timeouts"] += 1
                    elif row["result"] == "error":
                        counts["errors"] += 1
                    else:
                        counts["completed"] += 1
                        counts[row["result"]] += 1
                        durations.append(float(row["elapsed_s"]))
                        cpu_durations.append(float(row["user_s"]) + float(row["system_s"]))
                        decisions.append(float(row["decisions"]))
                        conflicts.append(float(row["conflicts"]))
                    submit_one()
                    if done_count % progress_every == 0 or done_count == total_cubes:
                        print(f"{arm}: {done_count}/{total_cubes}", flush=True)
    if done_count != total_cubes:
        raise RuntimeError(f"{arm}: expected {total_cubes} cubes, completed {done_count}")
    return summarize(
        cubes=total_cubes,
        durations=durations,
        cpu_durations=cpu_durations,
        decisions=decisions,
        conflicts=conflicts,
        workers=workers,
        wall_s=time.monotonic() - started,
        **counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cnf", type=Path)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=CUBES")
    parser.add_argument("--kissat", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_s <= 0:
        parser.error("workers and timeout-s must be positive")

    variables, clauses, body = parse_cnf(args.cnf.read_bytes())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
    worker_args = (
        variables,
        clauses,
        body,
        str(args.kissat.resolve()),
        args.timeout_s,
        str(args.tmp_dir.resolve()) if args.tmp_dir else None,
    )
    summaries: dict[str, Any] = {}
    for spec in args.arm:
        try:
            arm, cube_path = spec.split("=", 1)
        except ValueError as error:
            raise SystemExit(f"invalid --arm {spec!r}; expected NAME=PATH") from error
        cube_file = Path(cube_path)
        total_cubes = sum(1 for _ in read_cubes(cube_file))
        summaries[arm] = run_arm(
            arm,
            read_cubes(cube_file),
            total_cubes,
            args.workers,
            args.out_dir / f"{arm}.jsonl",
            worker_args,
        )
    bundle = {
        "schema_version": 1,
        "workers": args.workers,
        "timeout_s": args.timeout_s,
        "cnf": str(args.cnf),
        "kissat": str(args.kissat),
        "arms": summaries,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
