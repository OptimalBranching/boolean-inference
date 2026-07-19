import tempfile
import unittest
from collections import Counter
from pathlib import Path

from benchmarks.cnc.hard_regime import contract_sha256, load_contract, target_records
from benchmarks.cnc.hard_regime_matrix import MatrixError, build_matrix, lock_toolchain
from benchmarks.pipeline.circuit import canonical_bytes, sha256_bytes, write_json


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmarks/cnc/contracts/hard-regime-v1.yaml"


def fake_manifest(contract):
    digest = contract_sha256(contract)
    return [
        {
            **record,
            "contract_sha256": digest,
            "circuitsat": f"instances/{record['id']}.json",
            "circuitsat_sha256": f"{index + 1:064x}",
            "cnf": f"instances/{record['id']}.cnf",
            "cnf_sha256": f"{index + 100:064x}",
            "encoding_wall_s": 1.0,
            "encoding_cpu_s": 0.9,
        }
        for index, record in enumerate(target_records(contract))
    ]


def fake_toolchain(contract):
    record = {
        "schema_version": 1,
        "kind": "hard-regime-toolchain-lock",
        "contract_sha256": contract_sha256(contract),
        "tools": {
            "cnc_cuber": {
                "path": "/tools/cnc_cuber",
                "executable_sha256": "a" * 64,
                "source_revision": "repo-revision",
            },
            "kissat": {
                "path": "/tools/kissat",
                "executable_sha256": "b" * 64,
                "source_revision": contract["tool_sources"]["kissat"]["revision"],
            },
            "march_cu": {
                "path": "/tools/march_cu",
                "executable_sha256": "c" * 64,
                "source_revision": contract["tool_sources"]["march_cu"]["revision"],
            },
        },
    }
    record["toolchain_sha256"] = sha256_bytes(canonical_bytes(record))
    return record


def write_calibration_locks(root, contract, manifest, cuber_sha):
    for width in (64, 72, 80):
        calibration = [
            {"id": record["id"], "sha256": record["circuitsat_sha256"]}
            for record in manifest
            if record["product_width"] == width and record["split"] == "calibration"
        ]
        for selector in ("region", "structure-blind"):
            method = "region-cc" if selector == "region" else "structure-blind-cc"
            responses = []
            bands = {}
            for index, (name, spec) in enumerate(contract["frontier_bands"].items(), 1):
                threshold = width * 1000 + index
                target = spec["center_cubes"]
                response_instances = [
                    {"id": item["id"], "tasks": target, "elapsed_s": 1.0, "cpu_s": 0.9}
                    for item in calibration
                ]
                responses.append(
                    {"threshold": threshold, "instances": response_instances}
                )
                minimum = int(target * spec["accepted_ratio"][0])
                maximum = int(target * spec["accepted_ratio"][1])
                bands[name] = {
                    "target_tasks": target,
                    "accepted_task_range": [minimum, maximum],
                    "selected_threshold": threshold,
                    "search_bracket": (
                        [threshold, threshold + 1]
                        if index == 1
                        else [threshold - 1, threshold]
                    ),
                    "median_tasks": float(target),
                    "calibration_loss": 0.0,
                    "within_target_range": True,
                    "instances": [
                        {
                            **item,
                            "frontier_sha256": f"{index:064x}",
                            "trace_sha256": f"{index + 10:064x}",
                            "cubing_elapsed_s": 1.0,
                            "cubing_cpu_s": 0.9,
                        }
                        for item in response_instances
                    ],
                }
            write_json(
                root / f"p{width}" / selector / "calibration-lock.json",
                {
                    "schema_version": 1,
                    "kind": "width-level-cc-calibration-lock",
                    "contract_sha256": contract_sha256(contract),
                    "product_width": width,
                    "selector": selector,
                    "method": method,
                    "max_rows": contract["methods"][method]["max_rows"],
                    "cuber_sha256": cuber_sha,
                    "calibration_instances": calibration,
                    "bands": bands,
                    "response": responses,
                },
            )


class HardRegimeMatrixTests(unittest.TestCase):
    def test_matrix_declares_every_required_cell_once(self):
        contract = load_contract(CONTRACT_PATH)
        manifest = fake_manifest(contract)
        tools = fake_toolchain(contract)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_calibration_locks(
                root, contract, manifest, tools["tools"]["cnc_cuber"]["executable_sha256"]
            )
            matrix = build_matrix(contract, manifest, root, tools)
        counts = Counter(cell["method"] for cell in matrix["cells"])
        self.assertEqual(
            counts,
            {
                "monolithic-kissat": 39,
                "march-cu-dynamic": 30,
                "region-cc": 90,
                "structure-blind-cc": 90,
            },
        )
        self.assertEqual(sum(cell["pilot"] for cell in matrix["cells"]), 72)
        held_out = [cell for cell in matrix["cells"] if cell["method"] == "region-cc"]
        thresholds = {
            (cell["product_width"], cell["budget"]): cell["cc_threshold"]
            for cell in held_out
        }
        self.assertEqual(len(thresholds), 9)
        for cell in held_out:
            self.assertEqual(
                cell["cc_threshold"],
                thresholds[(cell["product_width"], cell["budget"])],
            )

    def test_matrix_rejects_width_confusion(self):
        contract = load_contract(CONTRACT_PATH)
        manifest = fake_manifest(contract)
        manifest[0]["factor_input_width"] = manifest[0]["product_width"]
        with self.assertRaisesRegex(MatrixError, "factor_input_width differs"):
            with tempfile.TemporaryDirectory() as directory:
                build_matrix(contract, manifest, Path(directory), fake_toolchain(contract))

    def test_matrix_rejects_calibration_test_leakage(self):
        contract = load_contract(CONTRACT_PATH)
        manifest = fake_manifest(contract)
        tools = fake_toolchain(contract)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_calibration_locks(
                root, contract, manifest, tools["tools"]["cnc_cuber"]["executable_sha256"]
            )
            path = root / "p64/region/calibration-lock.json"
            lock = __import__("json").loads(path.read_text())
            held_out = next(
                record
                for record in manifest
                if record["product_width"] == 64 and record["split"] == "held_out"
            )
            lock["calibration_instances"][0]["id"] = held_out["id"]
            write_json(path, lock)
            with self.assertRaisesRegex(MatrixError, "not the frozen split"):
                build_matrix(contract, manifest, root, tools)

    def test_matrix_recomputes_the_selected_calibration_threshold(self):
        contract = load_contract(CONTRACT_PATH)
        manifest = fake_manifest(contract)
        tools = fake_toolchain(contract)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_calibration_locks(
                root, contract, manifest, tools["tools"]["cnc_cuber"]["executable_sha256"]
            )
            path = root / "p64/region/calibration-lock.json"
            lock = __import__("json").loads(path.read_text())
            lock["bands"]["low"]["selected_threshold"] += 999
            write_json(path, lock)
            with self.assertRaisesRegex(MatrixError, "not selected from the response"):
                build_matrix(contract, manifest, root, tools)

    def test_toolchain_lock_requires_revision_qualified_upstream_binaries(self):
        contract = load_contract(CONTRACT_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cuber = root / "cnc_cuber"
            kissat_revision = contract["tool_sources"]["kissat"]["revision"]
            march_revision = contract["tool_sources"]["march_cu"]["revision"]
            kissat = root / f"kissat-{kissat_revision}"
            march = root / f"march_cu-{march_revision}"
            for path in (cuber, kissat, march):
                path.write_text("#!/bin/sh\necho identity\n", encoding="utf-8")
                path.chmod(0o755)
            record = lock_toolchain(contract, cuber, kissat, march, "abcdef0123456789")
            self.assertEqual(
                record["tools"]["kissat"]["source_revision"], kissat_revision
            )
            generic = root / "kissat"
            generic.write_text("#!/bin/sh\necho stale\n", encoding="utf-8")
            generic.chmod(0o755)
            with self.assertRaisesRegex(MatrixError, "revision-qualified binary"):
                lock_toolchain(contract, cuber, generic, march, "abcdef0123456789")


if __name__ == "__main__":
    unittest.main()
