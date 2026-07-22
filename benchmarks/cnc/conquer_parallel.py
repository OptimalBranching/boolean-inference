#!/usr/bin/env python3
"""Conquer frozen cube frontiers in parallel with auditable per-cube records."""

from __future__ import annotations

import concurrent.futures
import hashlib
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
_BASE_VARIABLES: int
_BASE_CLAUSES: int
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


def read_cubes(path: Path, variables: int | None = None) -> Iterator[list[int]]:
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
            if variables is not None and any(abs(literal) > variables for literal in literals):
                raise ValueError(f"{path}:{lineno}: literal exceeds CNF variable range")
            if len(set(literals)) != len(literals):
                raise ValueError(f"{path}:{lineno}: duplicate literal")
            if any(-literal in literals for literal in literals):
                raise ValueError(f"{path}:{lineno}: contradictory literals")
            yield literals


def parse_stats(stdout: str) -> tuple[int, int]:
    values = {name: int(value) for name, value in _STAT.findall(stdout)}
    return values.get("decisions", 0), values.get("conflicts", 0)


def child_cpu_seconds(before: resource.struct_rusage) -> tuple[float, float]:
    """Return user/system CPU consumed by children since ``before``."""

    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return after.ru_utime - before.ru_utime, after.ru_stime - before.ru_stime


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "total": 0.0,
            "mean": None,
            "cv": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "p99_over_p95": None,
            "max": None,
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
        "cv": math.sqrt(variance) / mean if mean else None,
        "p50": ordered_percentile(0.50),
        "p95": p95,
        "p99": p99,
        "p99_over_p95": p99 / p95 if p95 > 0 else None,
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
    global _BASE_VARIABLES, _BASE_CLAUSES, _BASE_BODY, _KISSAT, _TIMEOUT_S, _TMPDIR
    _BASE_VARIABLES = variables
    _BASE_CLAUSES = clauses
    _BASE_BODY = body
    _KISSAT = kissat
    _TIMEOUT_S = timeout_s
    _TMPDIR = tmpdir


def _solve_cube(task: tuple[str, int, list[int], int]) -> dict[str, Any]:
    arm, index, cube, released_ns = task
    started_ns = time.monotonic_ns()
    common = {
        "schema_version": 1,
        "arm": arm,
        "cube_index": index,
        "cube_literals": len(cube),
        "cube_sha256": hashlib.sha256(
            (" ".join(map(str, cube)) + " 0\n").encode()
        ).hexdigest(),
        "released_monotonic_ns": released_ns,
        "started_monotonic_ns": started_ns,
        "worker_pid": os.getpid(),
    }
    temporary = tempfile.NamedTemporaryFile(
        prefix=f"cube-{arm}-{index}-", suffix=".cnf", dir=_TMPDIR, delete=False
    )
    try:
        with temporary:
            temporary.write(
                f"p cnf {_BASE_VARIABLES} {_BASE_CLAUSES + len(cube)}\n".encode()
            )
            temporary.write(_BASE_BODY)
            for literal in cube:
                temporary.write(f"{literal} 0\n".encode())
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            process = subprocess.run(
                [_KISSAT, "--statistics", "--relaxed", temporary.name],
                capture_output=True,
                text=True,
                timeout=None if _TIMEOUT_S == 0 else _TIMEOUT_S,
                check=False,
            )
            elapsed_s = (time.monotonic_ns() - started_ns) / 1e9
            finished_ns = time.monotonic_ns()
            user_s, system_s = child_cpu_seconds(usage_before)
            decisions, conflicts = parse_stats(process.stdout)
            result = {10: "sat", 20: "unsat"}.get(process.returncode, "error")
            return {
                **common,
                "result": result,
                "returncode": process.returncode,
                "elapsed_s": elapsed_s,
                "user_s": user_s,
                "system_s": system_s,
                "decisions": decisions,
                "conflicts": conflicts,
                "censored": False,
                "finished_monotonic_ns": finished_ns,
                "stderr_tail": process.stderr[-500:] if result == "error" else "",
            }
        except subprocess.TimeoutExpired:
            finished_ns = time.monotonic_ns()
            user_s, system_s = child_cpu_seconds(usage_before)
            return {
                **common,
                "result": "timeout",
                "returncode": None,
                "elapsed_s": (finished_ns - started_ns) / 1e9,
                "user_s": user_s,
                "system_s": system_s,
                "decisions": None,
                "conflicts": None,
                "censored": True,
                "finished_monotonic_ns": finished_ns,
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
    replay_workers: list[int],
    wall_s: float,
    measured_makespan_s: float,
    not_started: int = 0,
) -> dict[str, Any]:
    time_stats = distribution(durations)
    cpu_stats = distribution(cpu_durations)
    decision_stats = distribution(decisions)
    conflict_stats = distribution(conflicts)
    result = (
        "sat"
        if sat
        else "error"
        if errors
        else "timeout"
        if timeouts
        else "unsat"
        if unsat == cubes
        else "incomplete"
    )
    complete = result == "sat" or (result == "unsat" and completed == cubes)
    lpt_wall = {str(count): lpt_makespan(durations, count) for count in replay_workers}
    lpt_cpu = {
        str(count): lpt_makespan(cpu_durations, count) for count in replay_workers
    }
    return {
        "cubes": cubes,
        "terminal_records": completed + timeouts + errors,
        "completed": completed,
        "timeouts": timeouts,
        "errors": errors,
        "not_started": not_started,
        "sat": sat,
        "unsat": unsat,
        "result": result,
        "complete": complete,
        "censored": bool(timeouts),
        "total_solver_s": time_stats["total"],
        "total_cpu_s": cpu_stats["total"],
        "cpu_time": cpu_stats,
        "observed_parallel_wall_s": wall_s,
        "measured_makespan_s": measured_makespan_s,
        "lpt_makespan_s": lpt_makespan(durations, workers),
        "cpu_lpt_makespan_s": lpt_makespan(cpu_durations, workers),
        "lpt_makespan_by_workers_s": lpt_wall,
        "cpu_lpt_makespan_by_workers_s": lpt_cpu,
        "lpt_is_lower_bound": bool(timeouts or errors or not_started),
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
    replay_workers: list[int],
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
    earliest_release_ns: int | None = None
    latest_collection_ns: int | None = None
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
                pending.add(
                    pool.submit(_solve_cube, (arm, index, cube, time.monotonic_ns()))
                )
                return True

            for _ in range(workers):
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
                    row["collected_monotonic_ns"] = time.monotonic_ns()
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    row_released = int(row["released_monotonic_ns"])
                    row_collected = int(row["collected_monotonic_ns"])
                    earliest_release_ns = (
                        row_released
                        if earliest_release_ns is None
                        else min(earliest_release_ns, row_released)
                    )
                    latest_collection_ns = (
                        row_collected
                        if latest_collection_ns is None
                        else max(latest_collection_ns, row_collected)
                    )
                    done_count += 1
                    durations.append(float(row["elapsed_s"]))
                    cpu_durations.append(float(row["user_s"]) + float(row["system_s"]))
                    if row["decisions"] is not None:
                        decisions.append(float(row["decisions"]))
                    if row["conflicts"] is not None:
                        conflicts.append(float(row["conflicts"]))
                    if row["censored"]:
                        counts["timeouts"] += 1
                    elif row["result"] == "error":
                        counts["errors"] += 1
                    else:
                        counts["completed"] += 1
                        counts[row["result"]] += 1
                    if not counts["sat"]:
                        submit_one()
                    if done_count % progress_every == 0 or done_count == total_cubes:
                        print(f"{arm}: {done_count}/{total_cubes}", flush=True)
    if done_count != total_cubes and not counts["sat"]:
        raise RuntimeError(f"{arm}: expected {total_cubes} cubes, completed {done_count}")
    return summarize(
        cubes=total_cubes,
        durations=durations,
        cpu_durations=cpu_durations,
        decisions=decisions,
        conflicts=conflicts,
        workers=workers,
        replay_workers=replay_workers,
        wall_s=time.monotonic() - started,
        measured_makespan_s=(
            0.0
            if earliest_release_ns is None or latest_collection_ns is None
            else (latest_collection_ns - earliest_release_ns) / 1e9
        ),
        not_started=total_cubes - done_count,
        **counts,
    )
