#!/usr/bin/env python3
"""Lock the issue #51 toolchain and materialize its complete run matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.cnc.calibrate_hard_regime import (
    CalibrationError,
    calibration_loss,
    choose_width_response,
    median_tasks,
)

from benchmarks.cnc.hard_regime import (
    HardRegimeError,
    contract_sha256,
    load_contract,
    mapping,
    target_records,
)
from benchmarks.pipeline.circuit import (
    canonical_bytes,
    load_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
)


class MatrixError(HardRegimeError):
    """The tool lock, instance manifest, calibration lock, or matrix is invalid."""


def executable_record(
    path: Path,
    source_revision: str,
    version_args: list[str],
    *,
    allow_nonzero: bool = False,
    required_basename: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise MatrixError(f"tool is not executable: {resolved}")
    if required_basename is not None and resolved.name != required_basename:
        raise MatrixError(
            f"tool {resolved} is not the revision-qualified binary {required_basename}"
        )
    process = subprocess.run(
        [str(resolved), *version_args], capture_output=True, text=True, check=False
    )
    if process.returncode and not allow_nonzero:
        raise MatrixError(f"version command failed for {resolved}: {process.stderr[-500:]}")
    output = (process.stdout + process.stderr).strip()
    if not output:
        raise MatrixError(f"version command produced no identity output for {resolved}")
    return {
        "path": str(resolved),
        "executable_sha256": sha256_file(resolved),
        "source_revision": source_revision,
        "version_command": [str(resolved), *version_args],
        "version_output": output[:4000],
        "version_output_sha256": sha256_bytes(output.encode()),
    }


def lock_toolchain(
    contract: dict[str, Any],
    cuber: Path,
    kissat: Path,
    march_cu: Path,
    repository_revision: str,
) -> dict[str, Any]:
    if not repository_revision or len(repository_revision) < 7:
        raise MatrixError("repository revision is required for cnc_cuber provenance")
    sources = contract["tool_sources"]
    record = {
        "schema_version": 1,
        "kind": "hard-regime-toolchain-lock",
        "contract_sha256": contract_sha256(contract),
        "tools": {
            "cnc_cuber": executable_record(cuber, repository_revision, ["--help"]),
            "kissat": executable_record(
                kissat,
                sources["kissat"]["revision"],
                ["--version"],
                required_basename=f"kissat-{sources['kissat']['revision']}",
            ),
            "march_cu": executable_record(
                march_cu,
                sources["march_cu"]["revision"],
                [],
                allow_nonzero=True,
                required_basename=f"march_cu-{sources['march_cu']['revision']}",
            ),
        },
    }
    record["toolchain_sha256"] = sha256_bytes(canonical_bytes(record))
    return record


def verify_toolchain(
    contract: dict[str, Any], toolchain: dict[str, Any], *, check_paths: bool
) -> None:
    if toolchain.get("schema_version") != 1 or toolchain.get("kind") != "hard-regime-toolchain-lock":
        raise MatrixError("unsupported toolchain lock")
    if toolchain.get("contract_sha256") != contract_sha256(contract):
        raise MatrixError("toolchain lock uses a different contract")
    tools = mapping(toolchain.get("tools"), "toolchain.tools")
    if set(tools) != {"cnc_cuber", "kissat", "march_cu"}:
        raise MatrixError("toolchain must contain cnc_cuber, kissat, and march_cu")
    expected_revisions = {
        "kissat": contract["tool_sources"]["kissat"]["revision"],
        "march_cu": contract["tool_sources"]["march_cu"]["revision"],
    }
    for name, raw in tools.items():
        spec = mapping(raw, f"toolchain tool {name}")
        digest = spec.get("executable_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise MatrixError(f"toolchain tool {name} has no executable SHA-256")
        if name in expected_revisions and spec.get("source_revision") != expected_revisions[name]:
            raise MatrixError(f"toolchain tool {name} source revision mismatch")
        if check_paths:
            path = Path(str(spec.get("path"))).resolve()
            if sha256_file(path) != digest:
                raise MatrixError(f"toolchain tool {name} executable hash mismatch")
    unsigned = {key: value for key, value in toolchain.items() if key != "toolchain_sha256"}
    if toolchain.get("toolchain_sha256") != sha256_bytes(canonical_bytes(unsigned)):
        raise MatrixError("toolchain lock fingerprint mismatch")


def validate_instance_manifest(
    contract: dict[str, Any], manifest: list[dict[str, Any]]
) -> None:
    expected = {record["id"]: record for record in target_records(contract)}
    if len(manifest) != len(expected):
        raise MatrixError(f"instance manifest needs {len(expected)} records")
    seen = set()
    digest = contract_sha256(contract)
    for record in manifest:
        instance_id = record.get("id")
        if instance_id in seen:
            raise MatrixError(f"duplicate manifest instance {instance_id!r}")
        seen.add(instance_id)
        if instance_id not in expected:
            raise MatrixError(f"undeclared manifest instance {instance_id!r}")
        target = expected[instance_id]
        for field in (
            "target",
            "product_width",
            "factor_input_width",
            "split",
            "split_index",
            "seed",
            "architecture",
            "expected_outcome",
        ):
            if record.get(field) != target[field]:
                raise MatrixError(f"{instance_id}: manifest field {field} differs from target lock")
        if record.get("contract_sha256") != digest:
            raise MatrixError(f"{instance_id}: manifest contract hash mismatch")
        if record["product_width"] != 2 * record["factor_input_width"]:
            raise MatrixError(f"{instance_id}: factor/product width confusion")
        for artifact in ("circuitsat", "cnf"):
            if not isinstance(record.get(artifact), str) or not isinstance(
                record.get(f"{artifact}_sha256"), str
            ):
                raise MatrixError(f"{instance_id}: incomplete {artifact} provenance")
        for metric in ("encoding_wall_s", "encoding_cpu_s"):
            value = record.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise MatrixError(f"{instance_id}: invalid {metric}")
    if seen != set(expected):
        raise MatrixError("instance manifest does not cover the declared target set")


def load_calibration_lock(
    contract: dict[str, Any],
    calibration_root: Path,
    product_width: int,
    selector: str,
    toolchain: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    path = calibration_root / f"p{product_width}" / selector / "calibration-lock.json"
    lock = load_json(path)
    if lock.get("schema_version") != 1 or lock.get("kind") != "width-level-cc-calibration-lock":
        raise MatrixError(f"{path}: unsupported calibration lock")
    for field, value in {
        "contract_sha256": contract_sha256(contract),
        "product_width": product_width,
        "selector": selector,
        "cuber_sha256": toolchain["tools"]["cnc_cuber"]["executable_sha256"],
    }.items():
        if lock.get(field) != value:
            raise MatrixError(f"{path}: calibration {field} mismatch")
    expected_instances = sorted(
        [
            {"id": record["id"], "sha256": record["circuitsat_sha256"]}
            for record in manifest
            if record["product_width"] == product_width
            and record["split"] == "calibration"
        ],
        key=lambda item: item["id"],
    )
    actual_instances = lock.get("calibration_instances")
    if not isinstance(actual_instances, list) or not all(
        isinstance(item, dict) for item in actual_instances
    ):
        raise MatrixError(f"{path}: calibration instance provenance is malformed")
    actual_instances = sorted(actual_instances, key=lambda item: str(item.get("id")))
    if actual_instances != expected_instances:
        raise MatrixError(f"{path}: calibration instances are not the frozen split")
    method = "region-cc" if selector == "region" else "structure-blind-cc"
    if lock.get("method") != method:
        raise MatrixError(f"{path}: calibration method mismatch")
    if lock.get("max_rows") != contract["methods"][method]["max_rows"]:
        raise MatrixError(f"{path}: calibration max_rows mismatch")

    search = lock.get("search")
    if not isinstance(search, dict):
        raise MatrixError(f"{path}: calibration search provenance is missing")
    initial_threshold = search.get("initial_threshold")
    maximum_threshold = search.get("maximum_threshold")
    if (
        not isinstance(initial_threshold, int)
        or isinstance(initial_threshold, bool)
        or initial_threshold != 1
        or not isinstance(maximum_threshold, int)
        or isinstance(maximum_threshold, bool)
        or maximum_threshold <= initial_threshold
        or not isinstance(search.get("probe_checkpoint_schema_version"), int)
        or isinstance(search.get("probe_checkpoint_schema_version"), bool)
        or search["probe_checkpoint_schema_version"] <= 0
    ):
        raise MatrixError(f"{path}: calibration search provenance is malformed")

    response = lock.get("response")
    if not isinstance(response, list) or not response:
        raise MatrixError(f"{path}: calibration response is empty")
    expected_ids = [item["id"] for item in expected_instances]
    response_by_threshold = {}
    for row in response:
        if not isinstance(row, dict):
            raise MatrixError(f"{path}: malformed calibration response row")
        threshold = row.get("threshold")
        if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
            raise MatrixError(f"{path}: malformed calibration response threshold")
        if threshold in response_by_threshold:
            raise MatrixError(f"{path}: duplicate calibration response threshold {threshold}")
        instances = row.get("instances")
        response_ids = (
            [item.get("id") for item in instances]
            if isinstance(instances, list)
            and all(isinstance(item, dict) for item in instances)
            else []
        )
        if not all(isinstance(instance_id, str) for instance_id in response_ids) or sorted(
            response_ids
        ) != expected_ids:
            raise MatrixError(f"{path}: response threshold {threshold} uses the wrong instances")
        for item in instances:
            if not isinstance(item, dict):
                raise MatrixError(f"{path}: malformed response instance")
            tasks = item.get("tasks")
            if not isinstance(tasks, int) or isinstance(tasks, bool) or tasks < 0:
                raise MatrixError(f"{path}: response task count is invalid")
            for field in ("elapsed_s", "cpu_s"):
                value = item.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    raise MatrixError(f"{path}: response {field} is invalid")
        response_by_threshold[threshold] = row

    bands = mapping(lock.get("bands"), f"{path} bands")
    if set(bands) != set(contract["frontier_bands"]):
        raise MatrixError(f"{path}: calibration bands are incomplete")
    for band, spec in bands.items():
        if not isinstance(spec, dict):
            raise MatrixError(f"{path}: band {band} is malformed")
        contract_band = contract["frontier_bands"][band]
        target = int(contract_band["center_cubes"])
        minimum = math.ceil(target * float(contract_band["accepted_ratio"][0]))
        maximum = math.floor(target * float(contract_band["accepted_ratio"][1]))
        threshold = spec.get("selected_threshold")
        if not isinstance(threshold, int) or threshold < 0:
            raise MatrixError(f"{path}: band {band} has no frozen threshold")
        if spec.get("target_tasks") != target:
            raise MatrixError(f"{path}: band {band} target differs from contract")
        if spec.get("accepted_task_range") != [minimum, maximum]:
            raise MatrixError(f"{path}: band {band} accepted range differs from contract")
        bracket = spec.get("search_bracket")
        if (
            not isinstance(bracket, list)
            or len(bracket) != 2
            or not all(isinstance(value, int) and value >= 0 for value in bracket)
            or bracket[0] >= bracket[1]
            or any(value not in response_by_threshold for value in bracket)
        ):
            raise MatrixError(f"{path}: band {band} search bracket is invalid")
        try:
            selected = choose_width_response(response, target, minimum, maximum)
        except CalibrationError as error:
            raise MatrixError(f"{path}: band {band} has no acceptable calibration") from error
        if threshold != selected["threshold"] or threshold not in response_by_threshold:
            raise MatrixError(f"{path}: band {band} threshold is not selected from the response")
        expected_median = median_tasks(selected)
        expected_loss = calibration_loss(selected, target)[0]
        recorded_median = spec.get("median_tasks")
        if not isinstance(recorded_median, (int, float)) or isinstance(
            recorded_median, bool
        ) or not math.isclose(
            float(recorded_median), expected_median, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise MatrixError(f"{path}: band {band} median task count is inconsistent")
        recorded_loss = spec.get("calibration_loss")
        if not isinstance(recorded_loss, (int, float)) or isinstance(
            recorded_loss, bool
        ) or not math.isclose(
            float(recorded_loss), expected_loss, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise MatrixError(f"{path}: band {band} selection loss is inconsistent")
        if spec.get("within_target_range") is not True:
            raise MatrixError(f"{path}: band {band} is outside its accepted task range")
        final_instances = spec.get("instances")
        final_ids = (
            [item.get("id") for item in final_instances]
            if isinstance(final_instances, list)
            and all(isinstance(item, dict) for item in final_instances)
            else []
        )
        if not all(isinstance(instance_id, str) for instance_id in final_ids) or sorted(
            final_ids
        ) != expected_ids:
            raise MatrixError(f"{path}: band {band} final rerun uses the wrong instances")
        response_tasks = {
            item["id"]: item["tasks"] for item in response_by_threshold[threshold]["instances"]
        }
        for item in final_instances:
            if not isinstance(item, dict) or item.get("tasks") != response_tasks.get(item.get("id")):
                raise MatrixError(f"{path}: band {band} final task count changed on rerun")
            for field in ("frontier_sha256", "trace_sha256"):
                digest = item.get(field)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise MatrixError(f"{path}: band {band} has invalid {field}")
            for field in ("cubing_elapsed_s", "cubing_cpu_s"):
                value = item.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    raise MatrixError(f"{path}: band {band} has invalid {field}")
    return lock


def cell_id(instance: str, method: str, budget: str) -> str:
    return f"{instance}__{method}__{budget}"


def build_matrix(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    calibration_root: Path,
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    validate_instance_manifest(contract, manifest)
    verify_toolchain(contract, toolchain, check_paths=False)
    locks = {
        (width, selector): load_calibration_lock(
            contract,
            calibration_root,
            width,
            selector,
            toolchain,
            manifest,
        )
        for width in (64, 72, 80)
        for selector in ("region", "structure-blind")
    }
    cells = []
    for instance in sorted(manifest, key=lambda record: record["id"]):
        common = {
            "schema_version": 1,
            "instance_id": instance["id"],
            "split": instance["split"],
            "split_index": instance["split_index"],
            "product_width": instance["product_width"],
            "factor_input_width": instance["factor_input_width"],
            "expected_outcome": "unsat",
            "circuitsat": instance["circuitsat"],
            "circuitsat_sha256": instance["circuitsat_sha256"],
            "global_cnf": instance["cnf"],
            "global_cnf_sha256": instance["cnf_sha256"],
            "encoding_wall_s": instance["encoding_wall_s"],
            "encoding_cpu_s": instance["encoding_cpu_s"],
            "pilot": instance["split"] == "held_out" and instance["split_index"] < 3,
        }

        cells.append(
            {
                **common,
                "cell_id": cell_id(instance["id"], "monolithic-kissat", "none"),
                "method": "monolithic-kissat",
                "budget": "none",
                "required_cpus": 1,
                "time_limit_s": contract["limits_seconds"]["monolithic"],
            }
        )
        if instance["split"] != "held_out":
            continue
        cells.append(
            {
                **common,
                "cell_id": cell_id(instance["id"], "march-cu-dynamic", "dynamic-default"),
                "method": "march-cu-dynamic",
                "budget": "dynamic-default",
                "required_cpus": contract["scheduling"]["measured_workers"],
                "cubing_time_limit_s": contract["limits_seconds"]["cubing"],
                "per_cube_time_limit_s": contract["limits_seconds"]["per_cube_conquer"],
                "cutoff_policy": "upstream-default-dynamic",
            }
        )
        for method, selector in (
            ("region-cc", "region"),
            ("structure-blind-cc", "structure-blind"),
        ):
            lock = locks[(instance["product_width"], selector)]
            for budget in ("low", "medium", "high"):
                cells.append(
                    {
                        **common,
                        "cell_id": cell_id(instance["id"], method, budget),
                        "method": method,
                        "budget": budget,
                        "required_cpus": contract["scheduling"]["measured_workers"],
                        "cubing_time_limit_s": contract["limits_seconds"]["cubing"],
                        "per_cube_time_limit_s": contract["limits_seconds"]["per_cube_conquer"],
                        "selector": selector,
                        "max_rows": contract["methods"][method]["max_rows"],
                        "cc_threshold": lock["bands"][budget]["selected_threshold"],
                        "calibration_lock": str(
                            Path(f"p{instance['product_width']}")
                            / selector
                            / "calibration-lock.json"
                        ),
                    }
                )
    counts = Counter(cell["method"] for cell in cells)
    expected_counts = {
        "monolithic-kissat": 39,
        "march-cu-dynamic": 30,
        "region-cc": 90,
        "structure-blind-cc": 90,
    }
    if dict(counts) != expected_counts:
        raise MatrixError(f"internal run-matrix count mismatch: {dict(counts)}")
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise MatrixError("run matrix contains duplicate cell IDs")
    return {
        "schema_version": 1,
        "kind": "hard-regime-run-matrix",
        "contract_sha256": contract_sha256(contract),
        "toolchain_sha256": toolchain["toolchain_sha256"],
        "cell_counts": expected_counts,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("lock-toolchain")
    command.add_argument("contract", type=Path)
    command.add_argument("--cuber", type=Path, required=True)
    command.add_argument("--kissat", type=Path, required=True)
    command.add_argument("--march-cu", type=Path, required=True)
    command.add_argument("--repository-revision", required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser("build-matrix")
    command.add_argument("contract", type=Path)
    command.add_argument("manifest", type=Path)
    command.add_argument("--calibration-root", type=Path, required=True)
    command.add_argument("--toolchain", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser("list-cells")
    command.add_argument("matrix", type=Path)
    command.add_argument(
        "--set",
        choices=("monolithic", "pilot-cnc", "full-cnc", "remaining-cnc"),
        required=True,
    )
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "list-cells":
            matrix = load_json(args.matrix)
            cells = matrix.get("cells")
            if not isinstance(cells, list):
                raise MatrixError("run matrix has no cells")
            selected = []
            for cell in cells:
                monolithic = cell.get("method") == "monolithic-kissat"
                include = {
                    "monolithic": monolithic,
                    "pilot-cnc": not monolithic and cell.get("pilot") is True,
                    "full-cnc": not monolithic,
                    "remaining-cnc": not monolithic and cell.get("pilot") is not True,
                }[args.set]
                if include:
                    selected.append(cell["cell_id"])
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("".join(f"{cell_id}\n" for cell_id in selected), encoding="utf-8")
            print(f"PASS cell-list: {len(selected)} cells -> {args.out}")
            return 0
        contract = load_contract(args.contract)
        if args.command == "lock-toolchain":
            result = lock_toolchain(
                contract,
                args.cuber,
                args.kissat,
                args.march_cu,
                args.repository_revision,
            )
        elif args.command == "build-matrix":
            result = build_matrix(
                contract,
                read_jsonl(args.manifest),
                args.calibration_root,
                load_json(args.toolchain),
            )
        else:
            raise MatrixError(f"unsupported command {args.command}")
        write_json(args.out, result)
    except (MatrixError, HardRegimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"PASS {result['kind']}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
