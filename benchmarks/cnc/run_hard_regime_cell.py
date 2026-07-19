#!/usr/bin/env python3
"""Execute one frozen hard-regime run-matrix cell and write a terminal record."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import socket
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.cnc.conquer_parallel import (
    parse_cnf,
    parse_stats,
    read_cubes,
    run_arm,
)
from benchmarks.cnc.hard_regime import HardRegimeError, contract_sha256, load_contract
from benchmarks.cnc.hard_regime_matrix import MatrixError, verify_toolchain
from benchmarks.pipeline.circuit import (
    canonical_bytes,
    load_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
)


_CUBER_STATS = re.compile(r"status=OK cubes=(\d+)")


class CellError(HardRegimeError):
    """A frozen cell cannot be executed or its output is inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def run_process(
    command: list[str], timeout_s: float, stdout_path: Path, stderr_path: Path
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    started_utc = utc_now()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = process.stdout
        stderr = process.stderr
        returncode = process.returncode
        state = "finished"
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        returncode = None
        state = "timeout"
    elapsed_s = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "command": command,
        "state": state,
        "returncode": returncode,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "wall_s": elapsed_s,
        "user_s": after.ru_utime - before.ru_utime,
        "system_s": after.ru_stime - before.ru_stime,
        "cpu_s": (after.ru_utime - before.ru_utime)
        + (after.ru_stime - before.ru_stime),
        "stdout": stdout_path.name,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr": stderr_path.name,
        "stderr_sha256": sha256_file(stderr_path),
    }


def verify_region_trace(frontier: Path, trace: Path) -> dict[str, int]:
    cubes = list(read_cubes(frontier))
    nodes = read_jsonl(trace)
    if not nodes:
        raise CellError("region trace is empty")
    children: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cutoff_literals = []
    refutation_counts: dict[str, int] = defaultdict(int)
    variable_count = None
    for expected_id, node in enumerate(nodes):
        if node.get("node_id") != expected_id:
            raise CellError("region trace node IDs are not contiguous")
        literals = node.get("literals")
        if not isinstance(literals, list) or not all(
            isinstance(literal, int) and not isinstance(literal, bool) and literal != 0
            for literal in literals
        ):
            raise CellError(f"region trace node {expected_id} has invalid literals")
        if len({abs(literal) for literal in literals}) != len(literals):
            raise CellError(f"region trace node {expected_id} repeats a decision variable")
        sigma_dec = node.get("sigma_dec")
        sigma_all = node.get("sigma_all")
        freevars = node.get("freevars")
        if (
            sigma_dec != len(literals)
            or not isinstance(sigma_all, int)
            or isinstance(sigma_all, bool)
            or not isinstance(freevars, int)
            or isinstance(freevars, bool)
            or sigma_all < sigma_dec
            or freevars < 0
        ):
            raise CellError(f"region trace node {expected_id} has invalid assignment counts")
        node_variables = sigma_all + freevars
        if variable_count is None:
            variable_count = node_variables
        elif node_variables != variable_count:
            raise CellError("region trace changes the declared variable count")

        parent = node.get("parent_id")
        if expected_id == 0:
            if parent is not None or node.get("child_index") is not None or node.get("depth") != 0:
                raise CellError("region trace has an invalid root")
        else:
            if not isinstance(parent, int) or parent < 0 or parent >= expected_id:
                raise CellError(f"region trace node {expected_id} has an invalid parent")
            if node.get("depth") != nodes[parent].get("depth") + 1:
                raise CellError(f"region trace node {expected_id} has an invalid depth")
            children[parent].append(node)
        kind = node.get("kind")
        if kind not in {"branch", "cutoff", "refuted", "sat"}:
            raise CellError(f"region trace node {expected_id} has invalid kind {kind!r}")
        reason = node.get("refutation_reason")
        if kind == "refuted":
            if reason not in {
                "root-propagation-contradiction",
                "selector-no-feasible-config",
                "branch-propagation-contradiction",
            }:
                raise CellError(f"region refuted node {expected_id} has no closure reason")
            refutation_counts[reason] += 1
        elif reason is not None:
            raise CellError(f"region non-refuted node {expected_id} has a closure reason")
        if kind == "sat":
            raise CellError("region trace contains a SAT leaf for an expected-UNSAT target")
        if kind == "cutoff":
            cutoff_literals.append(literals)
    for node in nodes:
        node_id = node["node_id"]
        actual_children = children.get(node_id, [])
        if node["kind"] == "branch":
            variables = node.get("rule_variables")
            clauses = node.get("rule_clauses")
            if (
                not isinstance(variables, list)
                or not variables
                or len(variables) > 64
                or not all(isinstance(variable, int) and variable > 0 for variable in variables)
                or len(set(variables)) != len(variables)
                or any(variable in {abs(literal) for literal in node["literals"]} for variable in variables)
            ):
                raise CellError(f"region branch node {node_id} has invalid rule variables")
            if not isinstance(clauses, list) or not clauses or len(actual_children) != len(clauses):
                raise CellError(f"region branch node {node_id} has incomplete children")
            indices = sorted(child.get("child_index") for child in actual_children)
            if indices != list(range(len(clauses))):
                raise CellError(f"region branch node {node_id} has invalid child indices")
            by_index = {child["child_index"]: child for child in actual_children}
            variable_mask = (1 << len(variables)) - 1
            for index, clause in enumerate(clauses):
                if not isinstance(clause, dict):
                    raise CellError(f"region branch node {node_id} has a malformed rule clause")
                mask = clause.get("mask")
                value = clause.get("value")
                if (
                    not isinstance(mask, int)
                    or isinstance(mask, bool)
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    or mask <= 0
                    or mask & ~variable_mask
                    or value & ~mask
                ):
                    raise CellError(f"region branch node {node_id} has an invalid rule clause")
                suffix = [
                    variable if (value >> bit) & 1 else -variable
                    for bit, variable in enumerate(variables)
                    if (mask >> bit) & 1
                ]
                if by_index[index].get("literals") != node["literals"] + suffix:
                    raise CellError(
                        f"region child {by_index[index]['node_id']} does not implement rule clause {index}"
                    )
        elif actual_children:
            raise CellError(f"region leaf node {node_id} unexpectedly has children")
        elif node.get("rule_clauses") not in ([], None):
            raise CellError(f"region leaf node {node_id} unexpectedly has rule clauses")
    if cutoff_literals != cubes:
        raise CellError("region trace cutoff leaves do not reproduce frontier bytes")
    return {
        "nodes": len(nodes),
        "branches": sum(node["kind"] == "branch" for node in nodes),
        "cutoffs": len(cutoff_literals),
        "refuted": sum(node["kind"] == "refuted" for node in nodes),
        "sat_leaves": sum(node["kind"] == "sat" for node in nodes),
        "root_refutations": refutation_counts["root-propagation-contradiction"],
        "selector_refutations": refutation_counts["selector-no-feasible-config"],
        "branch_refutations": refutation_counts["branch-propagation-contradiction"],
    }


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def select_cell(matrix: dict[str, Any], cell_id: str) -> dict[str, Any]:
    if matrix.get("schema_version") != 1 or matrix.get("kind") != "hard-regime-run-matrix":
        raise CellError("unsupported run matrix")
    matches = [cell for cell in matrix.get("cells", []) if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise CellError(f"run matrix contains {len(matches)} matches for {cell_id!r}")
    return matches[0]


def resolve_input(root: Path, relative: object, expected_sha256: object) -> Path:
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise CellError("cell input provenance is incomplete")
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise CellError(f"cell input escapes instance root: {relative}")
    if sha256_file(path) != expected_sha256:
        raise CellError(f"cell input hash mismatch: {relative}")
    return path


def slurm_context() -> dict[str, str]:
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_NTASKS",
        "SLURM_MEM_PER_NODE",
        "SLURM_TIMELIMIT",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def base_terminal(
    contract: dict[str, Any],
    matrix_path: Path,
    matrix: dict[str, Any],
    toolchain: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "hard-regime-terminal-cell",
        "contract_sha256": contract_sha256(contract),
        "matrix_sha256": sha256_file(matrix_path),
        "toolchain_sha256": toolchain["toolchain_sha256"],
        "cell_id": cell["cell_id"],
        "cell_sha256": sha256_bytes(canonical_bytes(cell)),
        "instance_id": cell["instance_id"],
        "split": cell["split"],
        "product_width": cell["product_width"],
        "factor_input_width": cell["factor_input_width"],
        "method": cell["method"],
        "budget": cell["budget"],
        "expected_outcome": cell["expected_outcome"],
        "input_artifacts": {
            "circuitsat": {
                "path": cell["circuitsat"],
                "sha256": cell["circuitsat_sha256"],
            },
            "global_cnf": {
                "path": cell["global_cnf"],
                "sha256": cell["global_cnf_sha256"],
            },
        },
        "host": socket.gethostname(),
        "slurm": slurm_context(),
        "started_utc": utc_now(),
    }


def run_monolithic(
    cell: dict[str, Any], cnf: Path, kissat: Path, cell_dir: Path
) -> dict[str, Any]:
    stage = run_process(
        [str(kissat), "--statistics", "--relaxed", str(cnf)],
        float(cell["time_limit_s"]),
        cell_dir / "monolithic.stdout",
        cell_dir / "monolithic.stderr",
    )
    stdout = (cell_dir / "monolithic.stdout").read_text(encoding="utf-8")
    decisions, conflicts = parse_stats(stdout)
    verdict = {10: "sat", 20: "unsat"}.get(stage["returncode"])
    state = (
        "complete"
        if verdict == "unsat"
        else "wrong-answer"
        if verdict == "sat"
        else "monolithic-timeout"
        if stage["state"] == "timeout"
        else "monolithic-error"
    )
    return {
        "state": state,
        "verdict": verdict,
        "stages": {"monolithic": stage},
        "metrics": {
            "encoding_wall_s": cell["encoding_wall_s"],
            "encoding_cpu_s": cell["encoding_cpu_s"],
            "solver_wall_s": stage["wall_s"],
            "solver_cpu_s": stage["cpu_s"],
            "decisions": decisions,
            "conflicts": conflicts,
            "end_to_end_wall_s": cell["encoding_wall_s"] + stage["wall_s"],
            "end_to_end_cpu_s": cell["encoding_cpu_s"] + stage["cpu_s"],
            "censored": stage["state"] == "timeout",
        },
    }


def run_cubing(
    cell: dict[str, Any],
    circuitsat: Path,
    cnf: Path,
    tools: dict[str, Any],
    cell_dir: Path,
) -> tuple[dict[str, Any], Path, Path | None]:
    frontier = cell_dir / "frontier.icnf"
    trace: Path | None = None
    if cell["method"] == "march-cu-dynamic":
        command = [tools["march_cu"]["path"], str(cnf), "-o", str(frontier)]
    else:
        trace = cell_dir / "nodes.jsonl"
        command = [
            tools["cnc_cuber"]["path"],
            str(circuitsat),
            "--cc-threshold",
            str(cell["cc_threshold"]),
            "-o",
            str(frontier),
            "--selector",
            cell["selector"],
            "--max-rows",
            str(cell["max_rows"]),
            "--trace",
            str(trace),
        ]
    stage = run_process(
        command,
        float(cell["cubing_time_limit_s"]),
        cell_dir / "cubing.stdout",
        cell_dir / "cubing.stderr",
    )
    stage["complete"] = stage["state"] == "finished" and stage["returncode"] == 0
    if not stage["complete"]:
        return stage, frontier, trace
    if not frontier.is_file():
        raise CellError("cuber reported success without a frontier")
    cubes = sum(1 for _ in read_cubes(frontier))
    stage["frontier_size"] = cubes
    stage["frontier_bytes"] = frontier.stat().st_size
    stage["frontier_sha256"] = sha256_file(frontier)
    if trace is not None:
        trace_summary = verify_region_trace(frontier, trace)
        stderr = (cell_dir / "cubing.stderr").read_text(encoding="utf-8")
        match = _CUBER_STATS.search(stderr)
        if not match or int(match.group(1)) != cubes:
            raise CellError("region cuber log/frontier task counts disagree")
        stage["completeness"] = {
            "complete": True,
            "evidence": "verified-branch-assignments-and-refutation-reasons",
            **trace_summary,
        }
    else:
        stage["completeness"] = {
            "complete": True,
            "evidence": "upstream-march-cu-successful-partition-output",
        }
    return stage, frontier, trace


def run_cnc(
    contract: dict[str, Any],
    cell: dict[str, Any],
    circuitsat: Path,
    cnf: Path,
    tools: dict[str, Any],
    cell_dir: Path,
    temp_root: Path | None,
) -> dict[str, Any]:
    cubing, frontier, trace = run_cubing(cell, circuitsat, cnf, tools, cell_dir)
    artifacts = {}
    if frontier.is_file():
        artifacts["frontier"] = artifact(frontier, cell_dir)
    if trace is not None and trace.is_file():
        artifacts["trace"] = artifact(trace, cell_dir)
    if not cubing["complete"]:
        state = "cubing-timeout" if cubing["state"] == "timeout" else "cubing-error"
        return {
            "state": state,
            "verdict": None,
            "stages": {"cubing": cubing},
            "artifacts": artifacts,
            "metrics": {
                "encoding_wall_s": cell["encoding_wall_s"],
                "encoding_cpu_s": cell["encoding_cpu_s"],
                "cubing_wall_s": cubing["wall_s"],
                "cubing_cpu_s": cubing["cpu_s"],
                "end_to_end_wall_s": cell["encoding_wall_s"] + cubing["wall_s"],
                "end_to_end_cpu_s": cell["encoding_cpu_s"] + cubing["cpu_s"],
                "censored": cubing["state"] == "timeout",
            },
        }

    cnf_data = cnf.read_bytes()
    variables, clauses, body = parse_cnf(cnf_data)
    raw_results = cell_dir / "cube-results.jsonl"
    temp_dir = (
        temp_root / cell["cell_id"] if temp_root is not None else cell_dir / "tmp"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    workers = int(contract["scheduling"]["measured_workers"])
    replay_workers = list(contract["scheduling"]["lpt_replay_workers"])
    total_cubes = sum(1 for _ in read_cubes(frontier))
    conquer = run_arm(
        cell["cell_id"],
        read_cubes(frontier),
        total_cubes,
        workers,
        replay_workers,
        raw_results,
        (
            variables,
            clauses,
            body,
            tools["kissat"]["path"],
            float(cell["per_cube_time_limit_s"]),
            str(temp_dir),
        ),
    )
    artifacts["cube_results"] = artifact(raw_results, cell_dir)
    state = (
        "complete"
        if conquer["complete"] and conquer["result"] == "unsat"
        else "wrong-answer"
        if conquer["result"] == "sat"
        else "conquer-timeout"
        if conquer["timeouts"]
        else "conquer-error"
    )
    end_wall = (
        float(cell["encoding_wall_s"])
        + float(cubing["wall_s"])
        + float(conquer["measured_makespan_s"])
    )
    end_cpu = (
        float(cell["encoding_cpu_s"])
        + float(cubing["cpu_s"])
        + float(conquer["total_cpu_s"])
    )
    return {
        "state": state,
        "verdict": conquer["result"] if conquer["complete"] else None,
        "stages": {"cubing": cubing, "conquer": conquer},
        "artifacts": artifacts,
        "metrics": {
            "encoding_wall_s": cell["encoding_wall_s"],
            "encoding_cpu_s": cell["encoding_cpu_s"],
            "cubing_wall_s": cubing["wall_s"],
            "cubing_cpu_s": cubing["cpu_s"],
            "frontier_size": cubing["frontier_size"],
            "frontier_bytes": cubing["frontier_bytes"],
            "conquer_work_cpu_s": conquer["total_cpu_s"],
            "conquer_span_s": conquer["max_s"],
            "p99_s": conquer["p99_s"],
            "maximum_conflicts": conquer["conflicts_max"],
            "timeout_count": conquer["timeouts"],
            "measured_32_worker_makespan_s": conquer["measured_makespan_s"],
            "lpt_makespan_by_workers_s": conquer["lpt_makespan_by_workers_s"],
            "end_to_end_wall_s": end_wall,
            "end_to_end_cpu_s": end_cpu,
            "censored": conquer["censored"],
        },
    }


def atomic_terminal(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def run_cell(
    contract: dict[str, Any],
    matrix_path: Path,
    matrix: dict[str, Any],
    toolchain: dict[str, Any],
    cell: dict[str, Any],
    instance_root: Path,
    output_root: Path,
    temp_root: Path | None = None,
) -> dict[str, Any]:
    if matrix.get("contract_sha256") != contract_sha256(contract):
        raise CellError("run matrix contract hash mismatch")
    if matrix.get("toolchain_sha256") != toolchain.get("toolchain_sha256"):
        raise CellError("run matrix toolchain hash mismatch")
    verify_toolchain(contract, toolchain, check_paths=True)
    required_cpus = int(cell["required_cpus"])
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is not None and int(allocated) < required_cpus:
        raise CellError(
            f"cell needs {required_cpus} CPUs but SLURM_CPUS_PER_TASK={allocated}"
        )
    cell_dir = output_root / "cells" / cell["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = cell_dir / "terminal.json"
    base = base_terminal(contract, matrix_path, matrix, toolchain, cell)
    if terminal_path.is_file():
        existing = load_json(terminal_path)
        if existing.get("cell_sha256") != base["cell_sha256"]:
            raise CellError("existing terminal record belongs to a different cell lock")
        return existing

    try:
        circuitsat = resolve_input(
            instance_root, cell["circuitsat"], cell["circuitsat_sha256"]
        )
        cnf = resolve_input(
            instance_root, cell["global_cnf"], cell["global_cnf_sha256"]
        )
        tools = toolchain["tools"]
        if cell["method"] == "monolithic-kissat":
            result = run_monolithic(cell, cnf, Path(tools["kissat"]["path"]), cell_dir)
        else:
            result = run_cnc(
                contract, cell, circuitsat, cnf, tools, cell_dir, temp_root
            )
        terminal = {
            **base,
            **result,
            "finished_utc": utc_now(),
        }
    except Exception as exc:
        terminal = {
            **base,
            "state": "harness-error",
            "verdict": None,
            "error": f"{type(exc).__name__}: {exc}",
            "finished_utc": utc_now(),
        }
        atomic_terminal(terminal_path, terminal)
        raise
    atomic_terminal(terminal_path, terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("toolchain", type=Path)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()
    try:
        contract = load_contract(args.contract)
        matrix = load_json(args.matrix)
        toolchain = load_json(args.toolchain)
        cell = select_cell(matrix, args.cell_id)
        terminal = run_cell(
            contract,
            args.matrix,
            matrix,
            toolchain,
            cell,
            args.instance_root,
            args.output_root,
            args.temp_root,
        )
    except (CellError, MatrixError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL cell {args.cell_id}: {exc}")
        return 2
    print(f"TERMINAL {terminal['cell_id']}: {terminal['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
