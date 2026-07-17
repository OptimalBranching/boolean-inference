#!/usr/bin/env python3
"""Verify a self-contained Cube-and-Conquer measurement evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path


class BundleError(ValueError):
    """The bundle is incomplete, inconsistent, or unauditable."""


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise BundleError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def bundle_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BundleError("bundle artifact path must be a non-empty string")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise BundleError(f"bundle artifact escapes its root: {relative}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_artifact(root: Path, spec: object, label: str) -> Path:
    if not isinstance(spec, dict):
        raise BundleError(f"manifest.{label} must be an object")
    path = bundle_path(root, spec.get("path"))
    expected = spec.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise BundleError(f"manifest.{label}.sha256 must be a lowercase SHA-256")
    if sha256_file(path) != expected:
        raise BundleError(f"{label} SHA-256 mismatch")
    return path


def parse_dimacs(path: Path) -> tuple[int, list[list[int]]]:
    variables = None
    clauses: list[list[int]] = []
    pending: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("c"):
            continue
        if stripped.startswith("p "):
            fields = stripped.split()
            if len(fields) != 4 or fields[:2] != ["p", "cnf"]:
                raise BundleError(f"{path}: malformed DIMACS header")
            variables = int(fields[2])
            declared_clauses = int(fields[3])
            continue
        for token in stripped.split():
            literal = int(token)
            if literal == 0:
                clauses.append(pending)
                pending = []
            else:
                pending.append(literal)
    if variables is None or pending or len(clauses) != declared_clauses:
        raise BundleError(f"{path}: malformed DIMACS body")
    if any(abs(literal) > variables for clause in clauses for literal in clause):
        raise BundleError(f"{path}: literal exceeds declared variable count")
    return variables, clauses


def parse_cube(record: dict, variables: list[int]) -> tuple[str, dict[int, bool]]:
    cube_id = record.get("cube_id")
    literals = record.get("literals")
    if not isinstance(cube_id, str) or not cube_id or not isinstance(literals, list):
        raise BundleError("every frontier record needs cube_id and literals")
    assignment: dict[int, bool] = {}
    for literal in literals:
        if not isinstance(literal, int) or literal == 0 or abs(literal) not in variables:
            raise BundleError(f"cube {cube_id}: invalid literal {literal!r}")
        variable = abs(literal)
        value = literal > 0
        if variable in assignment and assignment[variable] != value:
            raise BundleError(f"cube {cube_id}: contradictory literal for {variable}")
        assignment[variable] = value
    return cube_id, assignment


def verify_frontier(records: list[dict], variables: list[int]) -> tuple[int, int]:
    cubes: list[tuple[str, dict[int, bool]]] = []
    ids: set[str] = set()
    for record in records:
        cube_id, cube = parse_cube(record, variables)
        if cube_id in ids:
            raise BundleError(f"duplicate cube id {cube_id!r}")
        ids.add(cube_id)
        cubes.append((cube_id, cube))
    total = 1 << len(variables)
    covered = 0
    for bits in itertools.product((False, True), repeat=len(variables)):
        assignment = dict(zip(variables, bits, strict=True))
        owners = [
            cube_id
            for cube_id, cube in cubes
            if all(assignment[var] == value for var, value in cube.items())
        ]
        label = "".join("1" if bit else "0" for bit in bits)
        if not owners:
            raise BundleError(f"frontier: assignment {label} is uncovered")
        if len(owners) > 1:
            raise BundleError(
                f"frontier: assignment {label} is covered by {len(owners)} cubes"
            )
        covered += 1
    return covered, total


def solver_verdict(raw_output: object) -> str | None:
    if not isinstance(raw_output, str):
        raise BundleError("raw_solver_output must be a string")
    statuses = {
        line.strip().upper()
        for line in raw_output.splitlines()
        if line.strip().upper().startswith("S ")
    }
    mapping = {
        "S SATISFIABLE": "sat",
        "S UNSATISFIABLE": "unsat",
        "S UNKNOWN": "unknown",
    }
    parsed = {mapping[status] for status in statuses if status in mapping}
    if len(parsed) > 1:
        raise BundleError("raw solver output contains contradictory verdicts")
    return next(iter(parsed), None)


def number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def verify_events(
    events: list[dict], results: list[dict], cube_ids: set[str], worker_count: int
) -> dict[str, float]:
    if [event.get("seq") for event in events] != list(range(len(events))):
        raise BundleError("events must have contiguous sequence numbers from zero")
    times = [event.get("monotonic_seconds") for event in events]
    if not all(isinstance(value, (int, float)) for value in times):
        raise BundleError("every event needs a numeric monotonic_seconds")
    if any(left > right for left, right in zip(times, times[1:])):
        raise BundleError("event timestamps are not monotonic")

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        kind = event.get("event")
        if not isinstance(kind, str):
            raise BundleError("every event needs an event name")
        by_kind[kind].append(event)
    for marker in ("run_started", "cubing_started", "cubing_finished", "run_finished"):
        if len(by_kind[marker]) != 1:
            raise BundleError(f"event log needs exactly one {marker}")

    result_by_id: dict[str, dict] = {}
    for record in results:
        cube_id = record.get("cube_id")
        termination = record.get("termination")
        verdict = record.get("verdict")
        cpu_seconds = record.get("cpu_seconds")
        if not isinstance(cube_id, str) or cube_id in result_by_id:
            raise BundleError(f"duplicate or malformed result cube id {cube_id!r}")
        if termination not in {"solved", "cancelled", "timed_out", "never_started"}:
            raise BundleError(f"cube {cube_id}: invalid termination state")
        if verdict not in {"sat", "unsat", "unknown", None}:
            raise BundleError(f"cube {cube_id}: invalid verdict")
        if not isinstance(cpu_seconds, (int, float)) or cpu_seconds < 0:
            raise BundleError(f"cube {cube_id}: invalid cpu_seconds")
        parsed = solver_verdict(record.get("raw_solver_output"))
        if parsed != verdict:
            raise BundleError(f"cube {cube_id}: raw output disagrees with verdict")
        if termination == "solved" and verdict not in {"sat", "unsat"}:
            raise BundleError(f"cube {cube_id}: solved cube needs SAT or UNSAT verdict")
        if termination != "solved" and verdict not in {"unknown", None}:
            raise BundleError(f"cube {cube_id}: non-solved cube has definitive verdict")
        result_by_id[cube_id] = record
    if set(result_by_id) != cube_ids:
        missing = sorted(cube_ids - set(result_by_id))
        extra = sorted(set(result_by_id) - cube_ids)
        raise BundleError(f"result coverage mismatch: missing={missing}, extra={extra}")

    started: dict[str, dict] = {}
    terminal: dict[str, dict] = {}
    intervals = []
    for event in events:
        if event["event"] not in {"cube_started", "cube_terminal"}:
            continue
        cube_id = event.get("cube_id")
        worker = event.get("worker")
        if cube_id not in cube_ids or not isinstance(worker, int) or not 0 <= worker < worker_count:
            raise BundleError("cube event has invalid cube_id or worker")
        target = started if event["event"] == "cube_started" else terminal
        if cube_id in target:
            raise BundleError(f"cube {cube_id}: duplicate {event['event']}")
        target[cube_id] = event

    for cube_id, result in result_by_id.items():
        termination = result["termination"]
        if termination == "never_started":
            if cube_id in started or cube_id in terminal:
                raise BundleError(f"cube {cube_id}: never-started cube has events")
            continue
        if cube_id not in started or cube_id not in terminal:
            raise BundleError(f"cube {cube_id}: missing lifecycle event")
        begin = started[cube_id]
        end = terminal[cube_id]
        if begin["worker"] != end["worker"] or end.get("termination") != termination:
            raise BundleError(f"cube {cube_id}: inconsistent terminal event")
        if begin["monotonic_seconds"] >= end["monotonic_seconds"]:
            raise BundleError(f"cube {cube_id}: terminal event must follow start")
        intervals.append(
            (
                begin["monotonic_seconds"],
                end["monotonic_seconds"],
                cube_id,
                begin["worker"],
            )
        )

    by_worker: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for begin, end, cube_id, worker in intervals:
        by_worker[worker].append((begin, end, cube_id))
    for worker, work in by_worker.items():
        previous_end = None
        for begin, end, cube_id in sorted(work):
            if previous_end is not None and begin < previous_end:
                raise BundleError(
                    f"worker {worker} runs overlapping cube {cube_id}"
                )
            previous_end = end

    points = []
    for begin, end, _, _ in intervals:
        points.extend(((begin, 1), (end, -1)))
    active = maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise BundleError("worker interval accounting became negative")
        maximum = max(maximum, active)
    if maximum > worker_count:
        raise BundleError(
            f"observed concurrency {maximum} exceeds worker limit {worker_count}"
        )

    run_start = float(by_kind["run_started"][0]["monotonic_seconds"])
    run_end = float(by_kind["run_finished"][0]["monotonic_seconds"])
    cube_start = float(by_kind["cubing_started"][0]["monotonic_seconds"])
    cube_end = float(by_kind["cubing_finished"][0]["monotonic_seconds"])
    cpu_start = by_kind["cubing_started"][0].get("process_cpu_seconds")
    cpu_end = by_kind["cubing_finished"][0].get("process_cpu_seconds")
    if not isinstance(cpu_start, (int, float)) or not isinstance(cpu_end, (int, float)):
        raise BundleError("cubing events need process_cpu_seconds counters")
    if not run_start <= cube_start <= cube_end <= run_end or cpu_end < cpu_start:
        raise BundleError("run or cubing counters are not properly nested")
    conquer_start = min((begin for begin, _, _, _ in intervals), default=cube_end)
    conquer_end = max((end for _, end, _, _ in intervals), default=cube_end)
    if conquer_start < cube_end or conquer_end > run_end:
        raise BundleError("conquer intervals fall outside the scheduled run phase")
    metrics = {
        "cubing_wall": cube_end - cube_start,
        "cubing_cpu": float(cpu_end - cpu_start),
        "conquer_cpu": float(sum(record["cpu_seconds"] for record in results)),
        "conquer_makespan": float(conquer_end - conquer_start),
        "end_to_end_wall": run_end - run_start,
    }
    metrics["orchestration"] = (
        metrics["end_to_end_wall"]
        - metrics["cubing_wall"]
        - metrics["conquer_makespan"]
    )
    if any(value < 0 for value in metrics.values()):
        raise BundleError("derived accounting contains a negative duration")
    metrics["maximum_concurrency"] = float(maximum)
    return metrics


def verify_model(
    clauses: list[list[int]], model: list[int], variables: int
) -> dict[int, bool]:
    assignment: dict[int, bool] = {}
    for literal in model:
        if not isinstance(literal, int) or literal == 0 or abs(literal) > variables:
            raise BundleError(f"witness contains invalid literal {literal!r}")
        variable = abs(literal)
        value = literal > 0
        if variable in assignment and assignment[variable] != value:
            raise BundleError(f"witness contradicts variable {variable}")
        assignment[variable] = value
    missing = set(range(1, variables + 1)) - assignment.keys()
    if missing:
        raise BundleError(f"witness omits variables: {sorted(missing)}")
    if not all(
        any(assignment[abs(literal)] == (literal > 0) for literal in clause)
        for clause in clauses
    ):
        raise BundleError("returned model does not satisfy the input")
    return assignment


def verify(bundle: Path) -> list[str]:
    manifest = load_json(bundle / "bundle.json")
    if manifest.get("format_version") != 1:
        raise BundleError("unsupported bundle format_version")
    worker_count = manifest.get("worker_count")
    if not isinstance(worker_count, int) or worker_count < 1:
        raise BundleError("worker_count must be positive")
    for key in ("scheduler", "termination_policy"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise BundleError(f"manifest needs {key}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise BundleError("manifest needs provenance")
    for tool in ("cuber", "conquer"):
        record = provenance.get(tool)
        if not isinstance(record, dict) or not all(
            isinstance(record.get(key), str) and record[key]
            for key in ("id", "version", "path", "executable_sha256")
        ):
            raise BundleError(f"provenance needs complete {tool} identity")
        if not re.fullmatch(r"[0-9a-f]{64}", record["executable_sha256"]):
            raise BundleError(f"{tool} executable_sha256 is malformed")
        executable = bundle_path(bundle, record["path"])
        if sha256_file(executable) != record["executable_sha256"]:
            raise BundleError(f"{tool} executable SHA-256 mismatch")

    input_path = checked_artifact(bundle, manifest.get("input"), "input")
    frontier_path = checked_artifact(bundle, manifest.get("frontier"), "frontier")
    events_path = checked_artifact(bundle, manifest.get("events"), "events")
    results_path = checked_artifact(bundle, manifest.get("results"), "results")
    variables, clauses = parse_dimacs(input_path)
    frontier_spec = manifest["frontier"]
    frontier_variables = frontier_spec.get("variables")
    if frontier_spec.get("mode") != "exhaustive" or not isinstance(
        frontier_variables, list
    ):
        raise BundleError("frontier must declare exhaustive mode and variables")
    if frontier_variables != list(range(1, variables + 1)):
        raise BundleError("exhaustive frontier variables must match the input")
    frontier = read_jsonl(frontier_path)
    covered, total = verify_frontier(frontier, frontier_variables)
    results = read_jsonl(results_path)
    events = read_jsonl(events_path)
    cube_ids = {record["cube_id"] for record in frontier}
    metrics = verify_events(events, results, cube_ids, worker_count)

    declared = manifest.get("accounting")
    if not isinstance(declared, dict):
        raise BundleError("manifest needs declared accounting")
    for key in (
        "cubing_wall",
        "cubing_cpu",
        "conquer_cpu",
        "conquer_makespan",
        "orchestration",
        "end_to_end_wall",
    ):
        if declared.get(key) != metrics[key]:
            raise BundleError(
                f"accounting mismatch for {key}: declared={declared.get(key)!r}, "
                f"derived={metrics[key]!r}"
            )

    verdicts = {record["verdict"] for record in results}
    if "sat" in verdicts:
        aggregate = "sat"
    elif all(record["termination"] == "solved" for record in results) and verdicts == {"unsat"}:
        aggregate = "unsat"
    else:
        aggregate = "unknown"
    if manifest.get("aggregate_verdict") != aggregate:
        raise BundleError("aggregate verdict disagrees with cube records")
    verdict_message = "all cube outcomes justify the aggregate verdict"
    if aggregate == "sat":
        witness_path = checked_artifact(bundle, manifest.get("witness"), "witness")
        witness = load_json(witness_path).get("model")
        if not isinstance(witness, list):
            raise BundleError("witness needs a model list")
        model = verify_model(clauses, witness, variables)
        model_owner = next(
            (
                cube_id
                for cube_id, cube in (
                    parse_cube(record, frontier_variables) for record in frontier
                )
                if all(model[variable] == value for variable, value in cube.items())
            ),
            None,
        )
        sat_cubes = {
            record["cube_id"] for record in results if record["verdict"] == "sat"
        }
        if model_owner not in sat_cubes:
            raise BundleError("returned model does not belong to a SAT cube")
        verdict_message = "returned model satisfies the input"

    return [
        f"PASS frontier: {covered}/{total} assignments are covered exactly once",
        f"PASS verdict: {verdict_message}",
        "PASS accounting: "
        f"cubing_wall={number(metrics['cubing_wall'])} "
        f"conquer_cpu={number(metrics['conquer_cpu'])} "
        f"conquer_makespan={number(metrics['conquer_makespan'])} "
        f"end_to_end_wall={number(metrics['end_to_end_wall'])}",
        f"PASS workers: observed concurrency does not exceed {worker_count}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    try:
        messages = verify(args.bundle)
    except (BundleError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL {exc}")
        return 1
    print("\n".join(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
