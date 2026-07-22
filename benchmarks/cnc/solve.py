#!/usr/bin/env python3
"""Solve factoring CNFs directly with Kissat or conquer a cube frontier."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmarks.cnc.conquer_parallel import (
    parse_cnf,
    parse_stats,
    read_cubes,
    run_arm,
)
from benchmarks.pipeline.circuit import write_json


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def run_process(
    command: list[str],
    *,
    timeout_s: float,
    out_dir: Path,
    label: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=None if timeout_s == 0 else timeout_s,
            check=False,
        )
        stdout = process.stdout
        stderr = process.stderr
        returncode = process.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = _text(error.stdout)
        stderr = _text(error.stderr)
        returncode = None
        timed_out = True
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (out_dir / f"{label}.stdout").write_text(stdout, encoding="utf-8")
    (out_dir / f"{label}.stderr").write_text(stderr, encoding="utf-8")
    return {
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_s": time.monotonic() - started,
        "user_s": after.ru_utime - before.ru_utime,
        "system_s": after.ru_stime - before.ru_stime,
    }


def run_kissat(
    cnf: Path,
    kissat: Path,
    *,
    timeout_s: float,
    out_dir: Path,
) -> dict[str, Any]:
    process = run_process(
        [str(kissat), "--statistics", "--relaxed", str(cnf)],
        timeout_s=timeout_s,
        out_dir=out_dir,
        label="kissat",
    )
    decisions, conflicts = parse_stats(process["stdout"])
    result = (
        "timeout"
        if process["timed_out"]
        else {10: "sat", 20: "unsat"}.get(process["returncode"], "error")
    )
    record = {
        "schema_version": 1,
        "mode": "direct-kissat",
        "result": result,
        "returncode": process["returncode"],
        "timed_out": process["timed_out"],
        "wall_s": process["wall_s"],
        "user_s": process["user_s"],
        "system_s": process["system_s"],
        "decisions": decisions,
        "conflicts": conflicts,
        "cnf": str(cnf.resolve()),
        "kissat": str(kissat.resolve()),
    }
    write_json(out_dir / "summary.json", record)
    return record


def conquer_frontier(
    cnf: Path,
    frontier: Path,
    kissat: Path,
    *,
    workers: int,
    timeout_s: float,
    out_dir: Path,
    tmp_dir: Path | None = None,
    total_cubes: int | None = None,
) -> dict[str, Any]:
    variables, clauses, body = parse_cnf(cnf.read_bytes())
    out_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    if total_cubes is None:
        total_cubes = sum(1 for _ in read_cubes(frontier, variables))
    summary = run_arm(
        "factoring",
        read_cubes(frontier, variables),
        total_cubes,
        workers,
        [workers],
        out_dir / "cubes.jsonl",
        (
            variables,
            clauses,
            body,
            str(kissat.resolve()),
            timeout_s,
            str(tmp_dir.resolve()) if tmp_dir else None,
        ),
    )
    record = {
        "schema_version": 1,
        "mode": "parallel-conquer",
        "cnf": str(cnf.resolve()),
        "frontier": str(frontier.resolve()),
        "kissat": str(kissat.resolve()),
        "workers": workers,
        "per_cube_timeout_s": timeout_s,
        **summary,
    }
    write_json(out_dir / "summary.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cnf", type=Path)
    parser.add_argument("--kissat", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout_s < 0:
        parser.error("timeout must be non-negative")
    record = run_kissat(
        args.cnf,
        args.kissat,
        timeout_s=args.timeout_s,
        out_dir=args.out_dir,
    )
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
