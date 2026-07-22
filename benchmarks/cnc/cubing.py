#!/usr/bin/env python3
"""Run March or analyze a frozen cube frontier with parallel Kissat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.cnc.conquer_parallel import read_cubes
from benchmarks.cnc.solve import conquer_frontier, run_process
from benchmarks.pipeline.circuit import sha256_file, write_json


def run_cuber(
    command: list[str],
    frontier: Path,
    *,
    timeout_s: float,
    out_dir: Path,
) -> dict[str, Any]:
    frontier.parent.mkdir(parents=True, exist_ok=True)
    process = run_process(
        command,
        timeout_s=timeout_s,
        out_dir=out_dir,
        label="cubing",
    )
    if process["timed_out"]:
        raise TimeoutError("cuber timed out")
    if process["returncode"] != 0:
        raise RuntimeError(f"cuber exited with status {process['returncode']}")
    if not frontier.is_file():
        raise RuntimeError("cuber reported success without producing a frontier")
    cubes = sum(1 for _ in read_cubes(frontier))
    record = {
        "command": command,
        "returncode": process["returncode"],
        "wall_s": process["wall_s"],
        "cubes": cubes,
        "frontier": str(frontier.resolve()),
        "frontier_sha256": sha256_file(frontier),
        "frontier_bytes": frontier.stat().st_size,
    }
    write_json(out_dir / "cubing.json", record)
    return record


def march_then_conquer(
    cnf: Path,
    march_cu: Path,
    kissat: Path,
    *,
    workers: int,
    cube_timeout_s: float,
    cubing_timeout_s: float,
    out_dir: Path,
    remaining_vars: int | None = None,
    tmp_dir: Path | None = None,
) -> dict[str, Any]:
    frontier = out_dir / "frontier.icnf"
    command = [str(march_cu), str(cnf)]
    if remaining_vars is not None:
        command.extend(["-n", str(remaining_vars)])
    command.extend(["-o", str(frontier)])
    cubing = run_cuber(
        command,
        frontier,
        timeout_s=cubing_timeout_s,
        out_dir=out_dir,
    )
    conquer = conquer_frontier(
        cnf,
        frontier,
        kissat,
        workers=workers,
        timeout_s=cube_timeout_s,
        out_dir=out_dir / "conquer",
        tmp_dir=tmp_dir,
        total_cubes=cubing["cubes"],
    )
    record = {"schema_version": 1, "mode": "march-cu", "cubing": cubing, "conquer": conquer}
    write_json(out_dir / "summary.json", record)
    return record


def frozen_then_conquer(
    cnf: Path,
    frontier: Path,
    kissat: Path,
    *,
    workers: int,
    cube_timeout_s: float,
    out_dir: Path,
    tmp_dir: Path | None = None,
) -> dict[str, Any]:
    cubes = sum(1 for _ in read_cubes(frontier))
    conquer = conquer_frontier(
        cnf,
        frontier,
        kissat,
        workers=workers,
        timeout_s=cube_timeout_s,
        out_dir=out_dir / "conquer",
        tmp_dir=tmp_dir,
        total_cubes=cubes,
    )
    record = {
        "schema_version": 1,
        "mode": "frozen-frontier",
        "cnf": str(cnf.resolve()),
        "cnf_sha256": sha256_file(cnf),
        "frontier": str(frontier.resolve()),
        "frontier_sha256": sha256_file(frontier),
        "cubes": cubes,
        "conquer": conquer,
    }
    write_json(out_dir / "summary.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    march = subparsers.add_parser("march")
    march.add_argument("cnf", type=Path)
    march.add_argument("--march-cu", type=Path, required=True)
    march.add_argument("--remaining-vars", type=int)
    frozen = subparsers.add_parser("frontier")
    frozen.add_argument("cnf", type=Path)
    frozen.add_argument("frontier", type=Path)
    for command in (march, frozen):
        command.add_argument("--kissat", type=Path, required=True)
        command.add_argument("--workers", type=int, required=True)
        command.add_argument("--cube-timeout-s", type=float, default=600.0)
        command.add_argument("--out-dir", type=Path, required=True)
        command.add_argument("--tmp-dir", type=Path)
    march.add_argument("--cubing-timeout-s", type=float, default=3600.0)
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.cube_timeout_s < 0
        or getattr(args, "cubing_timeout_s", 0) < 0
    ):
        parser.error("workers must be positive and timeouts non-negative")
    if args.mode == "march":
        record = march_then_conquer(
            args.cnf,
            args.march_cu,
            args.kissat,
            workers=args.workers,
            cube_timeout_s=args.cube_timeout_s,
            cubing_timeout_s=args.cubing_timeout_s,
            out_dir=args.out_dir,
            remaining_vars=args.remaining_vars,
            tmp_dir=args.tmp_dir,
        )
    else:
        record = frozen_then_conquer(
            args.cnf,
            args.frontier,
            args.kissat,
            workers=args.workers,
            cube_timeout_s=args.cube_timeout_s,
            out_dir=args.out_dir,
            tmp_dir=args.tmp_dir,
        )
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
