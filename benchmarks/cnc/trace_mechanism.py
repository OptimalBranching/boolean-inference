"""Validate and summarize cnc_cuber mechanism traces (schema v2).

This deliberately aggregates raw local evidence without claiming that local
gamma predicts conquer cost. Join the output to per-cube residual/conquer data
at the *instance* level for the paper's mediation analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


class TraceError(ValueError):
    """Trace violates the mechanism schema or a requested invariant."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise TraceError(f"{path}:{line_number}: record is not an object")
            records.append(value)
    if not records:
        raise TraceError(f"{path}: empty trace")
    return records


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def _finite_gamma(value: Any) -> float | None:
    if (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _nonnegative_int(value: Any, label: str, index: int) -> int:
    if type(value) is not int or value < 0:
        raise TraceError(f"record {index}: {label} must be a non-negative integer")
    return value


def _branching_vector(value: Any, label: str, index: int) -> list[float]:
    if not isinstance(value, list):
        raise TraceError(f"record {index}: {label} must be an array")
    vector: list[float] = []
    for reduction in value:
        if (
            isinstance(reduction, bool)
            or not isinstance(reduction, (int, float))
            or not math.isfinite(reduction)
            or reduction < 0
        ):
            raise TraceError(
                f"record {index}: {label} contains an invalid measure reduction"
            )
        vector.append(float(reduction))
    return vector


def _validate_gamma(
    gamma_value: Any, vector: list[float], label: str, index: int
) -> None:
    if gamma_value is None:
        return
    gamma = _finite_gamma(gamma_value)
    if gamma is None or gamma < 1.0 or not vector:
        raise TraceError(f"record {index}: {label} is invalid")
    recurrence = sum(gamma ** (-reduction) for reduction in vector)
    if not math.isclose(recurrence, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise TraceError(
            f"record {index}: {label} does not match its branching vector"
        )


def _validate_evaluation(
    evaluation: Any,
    label: str,
    index: int,
    *,
    expected_branches: int | None = None,
) -> None:
    if not isinstance(evaluation, dict):
        raise TraceError(f"record {index}: {label} replay is not an object")
    branches = _nonnegative_int(evaluation.get("branches"), f"{label}.branches", index)
    vector = _branching_vector(
        evaluation.get("branching_vector"), f"{label}.branching_vector", index
    )
    if branches != len(vector):
        raise TraceError(f"record {index}: {label} branch/vector count mismatch")
    if expected_branches is not None and branches != expected_branches:
        raise TraceError(f"record {index}: {label} has the wrong branch count")
    _nonnegative_int(
        evaluation.get("decision_literals"), f"{label}.decision_literals", index
    )
    _nonnegative_int(evaluation.get("solver_ns"), f"{label}.solver_ns", index)
    _validate_gamma(evaluation.get("gamma"), vector, f"{label}.gamma", index)


def _quantile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for offset in range(cursor, end):
            ranks[order[offset]] = average_rank
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_norm = sum((x - left_mean) ** 2 for x in left)
    right_norm = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_norm * right_norm)
    return numerator / denominator if denominator else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _clauses_may_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether two partial assignments lack a syntactic contradiction.

    A false result proves disjointness.  A true result only means that overlap
    is possible; the native constraints can still make their intersection
    infeasible.
    """

    common = int(left["mask"]) & int(right["mask"])
    return ((int(left["value"]) ^ int(right["value"])) & common) == 0


def _validate_record(record: dict[str, Any], index: int) -> None:
    if record.get("schema_version") != 2:
        raise TraceError(f"record {index}: expected schema_version 2")
    if record.get("search_semantics") != "sat-decision":
        raise TraceError(f"record {index}: expected sat-decision semantics")
    if record.get("selector") not in (None, "region", "structure-blind"):
        raise TraceError(f"record {index}: invalid selector provenance")
    if record.get("branch_solver") not in (None, "greedy", "tail-greedy", "naive"):
        raise TraceError(f"record {index}: invalid branch-solver provenance")
    if record.get("input_kind") not in (
        None,
        "circuit-sat",
        "extensional-csp",
        "dimacs",
    ):
        raise TraceError(f"record {index}: invalid input-kind provenance")
    if type(record.get("node_id")) is not int:
        raise TraceError(f"record {index}: missing integer node_id")
    _nonnegative_int(record.get("depth"), "depth", index)
    if record.get("kind") not in {"branch", "cutoff", "refuted", "sat"}:
        raise TraceError(f"record {index}: invalid node kind")

    clauses = record.get("rule_clauses")
    if not isinstance(clauses, list):
        raise TraceError(f"record {index}: rule_clauses must be an array")
    for clause in clauses:
        if not isinstance(clause, dict):
            raise TraceError(f"record {index}: invalid rule clause")
        mask = clause.get("mask")
        value = clause.get("value")
        if type(mask) is not int or type(value) is not int or mask < 0 or value < 0:
            raise TraceError(f"record {index}: invalid rule clause mask/value")
        if value & ~mask:
            raise TraceError(f"record {index}: rule clause value exceeds its mask")

    diagnostics = record.get("rule_diagnostics")
    if diagnostics is None:
        return
    if not isinstance(diagnostics, dict):
        raise TraceError(f"record {index}: rule_diagnostics is not an object")

    semantics = diagnostics.get("rule_semantics")
    if semantics not in {
        "cover",
        "closed-witness",
        "local-refutation",
    }:
        raise TraceError(f"record {index}: invalid rule semantics")
    for field in (
        "region_tensors",
        "region_variables",
        "boundary_variables",
        "joined_rows",
        "feasible_rows",
    ):
        _nonnegative_int(diagnostics.get(field), field, index)
    region_variables = diagnostics["region_variables"]
    boundary_variables = diagnostics["boundary_variables"]
    joined_rows = diagnostics["joined_rows"]
    feasible_rows = diagnostics["feasible_rows"]
    if boundary_variables > region_variables:
        raise TraceError(f"record {index}: boundary exceeds region variables")
    if feasible_rows > joined_rows:
        raise TraceError(f"record {index}: feasible rows exceed joined rows")
    if not isinstance(diagnostics.get("closed"), bool):
        raise TraceError(f"record {index}: closed must be Boolean")

    timing = diagnostics.get("timing_ns")
    if not isinstance(timing, dict):
        raise TraceError(f"record {index}: timing_ns is not an object")
    for field in ("region_growth", "feasibility_probe", "rule_solver"):
        _nonnegative_int(timing.get(field), f"timing_ns.{field}", index)

    vector = _branching_vector(
        diagnostics.get("branching_vector"), "branching_vector", index
    )
    gamma = diagnostics.get("gamma")
    cover_verified = diagnostics.get("cover_verified")
    if cover_verified is not None and not isinstance(cover_verified, bool):
        raise TraceError(f"record {index}: cover_verified must be Boolean or null")
    closed = diagnostics["closed"]
    if semantics == "cover":
        if closed or feasible_rows == 0 or record["kind"] != "branch":
            raise TraceError(f"record {index}: inconsistent cover semantics")
        if len(clauses) != len(vector) or not clauses:
            raise TraceError(f"record {index}: selected branch/vector count mismatch")
        _validate_gamma(gamma, vector, "gamma", index)
    elif semantics == "closed-witness":
        if (
            not closed
            or feasible_rows == 0
            or record["kind"] != "branch"
            or len(clauses) != 1
            or vector
            or gamma != 1.0
        ):
            raise TraceError(f"record {index}: inconsistent closed-witness semantics")
    elif (
        feasible_rows != 0
        or record["kind"] != "refuted"
        or clauses
        or vector
        or gamma is not None
    ):
        raise TraceError(f"record {index}: inconsistent local-refutation semantics")

    replay = diagnostics.get("same_state_replay")
    if replay is not None:
        if not isinstance(replay, dict):
            raise TraceError(f"record {index}: same_state_replay is not an object")
        _validate_evaluation(replay.get("binary"), "binary", index, expected_branches=2)
        _validate_evaluation(
            replay.get("naive"), "naive", index, expected_branches=feasible_rows
        )


def summarize(
    records: list[dict[str, Any]], *, require_replay: bool = False
) -> dict[str, Any]:
    for index, record in enumerate(records):
        _validate_record(record, index)

    ids = [record["node_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise TraceError("duplicate node_id")

    nodes = {record["node_id"]: record for record in records}
    roots = [record for record in records if record.get("parent_id") is None]
    if len(roots) != 1 or roots[0]["depth"] != 0:
        raise TraceError("trace must contain exactly one depth-zero root")
    child_slots: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        parent_id = record.get("parent_id")
        child_index = record.get("child_index")
        if parent_id is None:
            if child_index is not None:
                raise TraceError(f"record {index}: root has a child_index")
            continue
        if type(parent_id) is not int or parent_id not in nodes:
            raise TraceError(f"record {index}: missing parent node {parent_id}")
        parent = nodes[parent_id]
        if parent["kind"] != "branch" or record["depth"] != parent["depth"] + 1:
            raise TraceError(f"record {index}: broken parent/depth linkage")
        if type(child_index) is not int or not 0 <= child_index < len(
            parent["rule_clauses"]
        ):
            raise TraceError(f"record {index}: invalid child_index")
        slot = (parent_id, child_index)
        if slot in child_slots:
            raise TraceError(f"record {index}: duplicate parent child_index")
        child_slots.add(slot)

    diagnostics = [
        (record, record["rule_diagnostics"])
        for record in records
        if isinstance(record.get("rule_diagnostics"), dict)
    ]
    rule_nodes = [
        (record, diag)
        for record, diag in diagnostics
        if diag.get("rule_semantics") in {"cover", "closed-witness"}
    ]
    cover_nodes = [
        (record, diag)
        for record, diag in rule_nodes
        if diag["rule_semantics"] == "cover"
    ]
    closed_nodes = [
        (record, diag)
        for record, diag in rule_nodes
        if diag["rule_semantics"] == "closed-witness"
    ]
    local_refutations = sum(
        diag.get("rule_semantics") == "local-refutation" for _, diag in diagnostics
    )

    if require_replay:
        missing = [
            record["node_id"]
            for record, diag in rule_nodes
            if not isinstance(diag.get("same_state_replay"), dict)
        ]
        if missing:
            raise TraceError(f"missing same-state replay at nodes {missing[:8]}")
        unverified = [
            record["node_id"]
            for record, diag in cover_nodes
            if diag.get("cover_verified") is not True
        ]
        if unverified:
            raise TraceError(f"cover not verified at nodes {unverified[:8]}")

    selected_branches = sum(len(record.get("rule_clauses", [])) for record, _ in rule_nodes)
    selected_literals = sum(
        int(clause["mask"]).bit_count()
        for record, _ in rule_nodes
        for clause in record.get("rule_clauses", [])
    )
    single_branch_nodes = sum(
        len(record.get("rule_clauses", [])) == 1 for record, _ in rule_nodes
    )
    sibling_pairs = 0
    potentially_overlapping_pairs = 0
    syntactically_disjoint_cover_nodes = 0
    for record, _ in cover_nodes:
        clauses = record["rule_clauses"]
        node_may_overlap = False
        for left_index, left in enumerate(clauses):
            for right in clauses[left_index + 1 :]:
                sibling_pairs += 1
                if _clauses_may_overlap(left, right):
                    potentially_overlapping_pairs += 1
                    node_may_overlap = True
        syntactically_disjoint_cover_nodes += not node_may_overlap

    compression_bits = [
        float(diag["region_variables"]) - math.log2(float(diag["feasible_rows"]))
        for _, diag in rule_nodes
        if diag.get("feasible_rows", 0) > 0
    ]
    boundary_ratios = [
        float(diag["boundary_variables"]) / float(diag["region_variables"])
        for _, diag in rule_nodes
        if diag.get("region_variables", 0) > 0
    ]
    joined_total = sum(int(diag["joined_rows"]) for _, diag in diagnostics)
    feasible_total = sum(int(diag["feasible_rows"]) for _, diag in diagnostics)

    timing_keys = ("region_growth", "feasibility_probe", "rule_solver")
    timing_ms = {
        key: sum(int(diag.get("timing_ns", {}).get(key, 0)) for _, diag in diagnostics)
        / 1_000_000.0
        for key in timing_keys
    }

    replay_nodes = [
        (record, diag, diag["same_state_replay"])
        for record, diag in rule_nodes
        if isinstance(diag.get("same_state_replay"), dict)
    ]
    open_replays = [item for item in replay_nodes if item[1]["rule_semantics"] == "cover"]

    selected_gammas: list[float] = []
    binary_gammas: list[float] = []
    naive_gammas: list[float] = []
    selected_better_binary = 0
    selected_better_naive = 0
    for _, diag, replay in open_replays:
        selected = _finite_gamma(diag.get("gamma"))
        binary = _finite_gamma(replay["binary"].get("gamma"))
        naive = _finite_gamma(replay["naive"].get("gamma"))
        if selected is not None:
            selected_gammas.append(selected)
        if binary is not None:
            binary_gammas.append(binary)
        if naive is not None:
            naive_gammas.append(naive)
        if selected is not None and binary is not None and selected < binary - 1e-12:
            selected_better_binary += 1
        if selected is not None and naive is not None and selected < naive - 1e-12:
            selected_better_naive += 1

    replay_naive_branches = sum(
        int(replay["naive"]["branches"]) for _, _, replay in replay_nodes
    )
    replay_decision_literals = {
        name: sum(
            int(replay[name]["decision_literals"]) for _, _, replay in replay_nodes
        )
        for name in ("binary", "naive")
    }
    replay_solver_ms = {
        name: sum(int(replay[name]["solver_ns"]) for _, _, replay in replay_nodes)
        / 1_000_000.0
        for name in ("binary", "naive")
    }

    return {
        "schema_version": 1,
        "trace_schema_version": 2,
        "nodes": len(records),
        "rule_nodes": len(rule_nodes),
        "cover_nodes": len(cover_nodes),
        "closed_witness_nodes": len(closed_nodes),
        "local_refutations": local_refutations,
        "selected": {
            "branches": selected_branches,
            "decision_literals": selected_literals,
            "mean_literals_per_branch": (
                selected_literals / selected_branches if selected_branches else None
            ),
            "single_branch_nodes": single_branch_nodes,
            "single_branch_fraction": (
                single_branch_nodes / len(rule_nodes) if rule_nodes else None
            ),
            "median_gamma_open": _median(selected_gammas),
            "sibling_pairs": sibling_pairs,
            "potentially_overlapping_sibling_pairs": potentially_overlapping_pairs,
            "syntactically_disjoint_cover_nodes": syntactically_disjoint_cover_nodes,
            "syntactically_disjoint_cover_fraction": (
                syntactically_disjoint_cover_nodes / len(cover_nodes)
                if cover_nodes
                else None
            ),
        },
        "region": {
            "median_variables": _median(
                float(diag["region_variables"]) for _, diag in rule_nodes
            ),
            "median_tensors": _median(
                float(diag["region_tensors"]) for _, diag in rule_nodes
            ),
            "median_boundary_ratio": _median(boundary_ratios),
            "median_compression_bits": _median(compression_bits),
            "joined_rows": joined_total,
            "feasible_rows": feasible_total,
            "probe_survival_fraction": (
                feasible_total / joined_total if joined_total else None
            ),
        },
        "same_state_replay": {
            "nodes": len(replay_nodes),
            "open_cover_nodes": len(open_replays),
            "naive_branches": replay_naive_branches,
            "decision_literals": replay_decision_literals,
            "median_gamma_binary_open": _median(binary_gammas),
            "median_gamma_naive_open": _median(naive_gammas),
            "selected_better_than_binary": selected_better_binary,
            "selected_better_than_naive": selected_better_naive,
            "solver_ms": replay_solver_ms,
        },
        "actual_timing_ms": timing_ms,
    }


def summarize_conquer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    uncensored = [
        row
        for row in rows
        if not row.get("censored", False)
        and isinstance(row.get("conflicts"), (int, float))
        and isinstance(row.get("elapsed_s"), (int, float))
    ]
    conflicts = [float(row["conflicts"]) for row in uncensored]
    elapsed = [float(row["elapsed_s"]) for row in uncensored]

    def distribution(values: list[float]) -> dict[str, Any]:
        mean = statistics.fmean(values) if values else None
        return {
            "total": sum(values),
            "median": _median(values),
            "p95": _quantile(values, 0.95),
            "max": max(values) if values else None,
            "cv": (
                statistics.pstdev(values) / mean
                if values and mean is not None and mean != 0
                else None
            ),
        }

    return {
        "rows": len(rows),
        "uncensored": len(uncensored),
        "conflicts": distribution(conflicts),
        "elapsed_s": distribution(elapsed),
    }


def link_conquer(
    trace: list[dict[str, Any]], conquer_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Join DFS-ordered cutoff leaves to per-cube outcomes by cube_index.

    This is within-instance exploratory evidence only. It does not turn cubes
    into independent samples and must not be used for paper-level confidence
    intervals.
    """

    nodes = {record["node_id"]: record for record in trace}
    leaves = [record for record in trace if record.get("kind") == "cutoff"]
    outcomes = {int(row["cube_index"]): row for row in conquer_rows}
    if set(outcomes) != set(range(len(leaves))):
        raise TraceError(
            "conquer cube_index set does not match DFS-ordered cutoff leaves"
        )

    linked: list[dict[str, float]] = []
    for cube_index, leaf in enumerate(leaves):
        outcome = outcomes[cube_index]
        if len(leaf.get("literals", [])) != int(outcome["cube_literals"]):
            raise TraceError(f"cube {cube_index}: literal count mismatch")
        if outcome.get("censored", False):
            continue
        if not isinstance(outcome.get("conflicts"), (int, float)) or not isinstance(
            outcome.get("elapsed_s"), (int, float)
        ):
            continue

        path_edges: list[tuple[dict[str, Any], int]] = []
        child = leaf
        parent_id = leaf.get("parent_id")
        while parent_id is not None:
            parent = nodes.get(parent_id)
            if parent is None:
                raise TraceError(f"cube {cube_index}: missing parent node {parent_id}")
            diagnostics = parent.get("rule_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get("rule_semantics") in {
                "cover",
                "closed-witness",
            }:
                child_index = child.get("child_index")
                if type(child_index) is not int:
                    raise TraceError(
                        f"cube {cube_index}: child of rule {parent_id} has no child_index"
                    )
                path_edges.append((parent, child_index))
            child = parent
            parent_id = parent.get("parent_id")
        path_edges.reverse()
        if not path_edges:
            raise TraceError(f"cube {cube_index}: cutoff has no rule path")

        compression: list[float] = []
        boundary_ratio: list[float] = []
        selected_gammas: list[float] = []
        selected_reductions: list[float] = []
        gamma_advantage_naive = 0.0
        single_branch_nodes = 0
        for node, child_index in path_edges:
            diagnostics = node["rule_diagnostics"]
            variables = float(diagnostics["region_variables"])
            feasible = float(diagnostics["feasible_rows"])
            if feasible > 0:
                compression.append(variables - math.log2(feasible))
            if variables > 0:
                boundary_ratio.append(
                    float(diagnostics["boundary_variables"]) / variables
                )
            selected = _finite_gamma(diagnostics.get("gamma"))
            if selected is not None:
                selected_gammas.append(selected)
            vector = _branching_vector(
                diagnostics.get("branching_vector"),
                "rule_diagnostics.branching_vector",
                int(node["node_id"]),
            )
            if not vector and diagnostics.get("rule_semantics") == "closed-witness":
                if child_index != 0:
                    raise TraceError(
                        f"cube {cube_index}: closed witness has a nonzero child_index"
                    )
                selected_reductions.append(0.0)
            elif child_index < 0 or child_index >= len(vector):
                raise TraceError(
                    f"cube {cube_index}: child_index exceeds branching vector"
                )
            else:
                selected_reductions.append(vector[child_index])
            replay = diagnostics.get("same_state_replay")
            if isinstance(replay, dict) and selected is not None and selected > 0:
                naive = _finite_gamma(replay["naive"].get("gamma"))
                if naive is not None and naive > 0:
                    gamma_advantage_naive += math.log(naive) - math.log(selected)
            if len(node.get("rule_clauses", [])) == 1:
                single_branch_nodes += 1
        root_node, root_child_index = path_edges[0]
        root_reduction = selected_reductions[0]

        linked.append(
            {
                "conflicts": float(outcome["conflicts"]),
                "conflicts_log1p": math.log1p(float(outcome["conflicts"])),
                "elapsed_log": math.log(max(float(outcome["elapsed_s"]), 1e-12)),
                "decision_literals": float(len(leaf.get("literals", []))),
                "branch_depth": float(len(path_edges)),
                "single_branch_fraction": single_branch_nodes / len(path_edges),
                "median_compression_bits": statistics.median(compression),
                "median_boundary_ratio": statistics.median(boundary_ratio),
                "median_selected_gamma": statistics.median(selected_gammas),
                "gamma_advantage_naive_sum": gamma_advantage_naive,
                "selected_measure_reduction_sum": sum(selected_reductions),
                "selected_measure_reduction_min": min(selected_reductions),
                "root_node_id": float(root_node["node_id"]),
                "root_child_index": float(root_child_index),
                "root_child_measure_reduction": root_reduction,
            }
        )

    feature_names = [
        "decision_literals",
        "branch_depth",
        "single_branch_fraction",
        "median_compression_bits",
        "median_boundary_ratio",
        "median_selected_gamma",
        "gamma_advantage_naive_sum",
        "selected_measure_reduction_sum",
        "selected_measure_reduction_min",
        "root_child_measure_reduction",
    ]
    correlations: dict[str, dict[str, float | None]] = {}
    conflicts = [row["conflicts_log1p"] for row in linked]
    elapsed = [row["elapsed_log"] for row in linked]
    for feature in feature_names:
        values = [row[feature] for row in linked]
        correlations[feature] = {
            "spearman_log1p_conflicts": _spearman(values, conflicts),
            "spearman_log_elapsed": _spearman(values, elapsed),
        }

    hardest_count = max(1, math.ceil(0.1 * len(linked))) if linked else 0
    hardest = sorted(linked, key=lambda row: row["conflicts_log1p"], reverse=True)[
        :hardest_count
    ]
    remainder = sorted(linked, key=lambda row: row["conflicts_log1p"], reverse=True)[
        hardest_count:
    ]

    def means(rows: list[dict[str, float]]) -> dict[str, float | None]:
        return {
            feature: statistics.fmean(row[feature] for row in rows) if rows else None
            for feature in feature_names
        }

    root_child_attribution = None
    if linked:
        root_ids = {int(row["root_node_id"]) for row in linked}
        if len(root_ids) != 1:
            raise TraceError("linked cutoff paths do not share one root rule")
        root_id = root_ids.pop()
        root_diagnostics = nodes[root_id]["rule_diagnostics"]
        root_vector = _branching_vector(
            root_diagnostics.get("branching_vector"),
            "root.branching_vector",
            root_id,
        )
        if not root_vector:
            root_vector = [0.0]
        weakest_child = min(range(len(root_vector)), key=root_vector.__getitem__)
        weak_rows = [
            row for row in linked if int(row["root_child_index"]) == weakest_child
        ]
        total_conflicts = sum(row["conflicts"] for row in linked)
        hardest_five_count = max(1, math.ceil(0.05 * len(linked)))
        hardest_five = sorted(
            linked, key=lambda row: row["conflicts"], reverse=True
        )[:hardest_five_count]
        root_child_attribution = {
            "root_node_id": root_id,
            "branches": len(root_vector),
            "measure_reduction_min": min(root_vector),
            "measure_reduction_median": statistics.median(root_vector),
            "measure_reduction_max": max(root_vector),
            "weakest_child_index": weakest_child,
            "weakest_child_measure_reduction": root_vector[weakest_child],
            "weakest_child_frontier_cubes": len(weak_rows),
            "weakest_child_frontier_fraction": len(weak_rows) / len(linked),
            "weakest_child_conflict_share": (
                sum(row["conflicts"] for row in weak_rows) / total_conflicts
                if total_conflicts
                else None
            ),
            "hardest_five_percent_cubes": hardest_five_count,
            "hardest_five_percent_in_weakest_child": sum(
                int(row["root_child_index"]) == weakest_child
                for row in hardest_five
            ),
        }

    return {
        "linked_uncensored_cubes": len(linked),
        "within_instance_exploratory_only": True,
        "spearman": correlations,
        "hardest_conflict_decile": means(hardest),
        "remaining_cubes": means(remainder),
        "root_child_attribution": root_child_attribution,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--require-replay", action="store_true")
    parser.add_argument("--conquer", type=Path)
    parser.add_argument("--baseline-conquer", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        trace = read_jsonl(args.trace)
        result = summarize(trace, require_replay=args.require_replay)
        if args.conquer:
            conquer = read_jsonl(args.conquer)
            result["conquer"] = summarize_conquer(conquer)
            result["path_to_conquer"] = link_conquer(trace, conquer)
        if args.baseline_conquer:
            result["baseline_conquer"] = summarize_conquer(
                read_jsonl(args.baseline_conquer)
            )
    except (OSError, TraceError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
