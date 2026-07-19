import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.cnc.conquer_parallel import summarize
from benchmarks.cnc.hard_regime import contract_sha256, load_contract
from benchmarks.cnc.verify_hard_regime import (
    VerificationError,
    verify_conquer_records,
    verify_terminal,
)
from benchmarks.pipeline.circuit import canonical_bytes, sha256_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmarks/cnc/contracts/hard-regime-v1.yaml"


class HardRegimeVerifierTests(unittest.TestCase):
    def make_records(self, root: Path):
        frontier = root / "frontier.icnf"
        results = root / "results.jsonl"
        frontier.write_text("a 1 0\na -1 0\n", encoding="utf-8")
        rows = []
        for index, literal in enumerate((1, -1)):
            started = 100 + index * 20
            rows.append(
                {
                    "cube_index": index,
                    "cube_sha256": hashlib.sha256(f"{literal} 0\n".encode()).hexdigest(),
                    "released_monotonic_ns": started - 5,
                    "started_monotonic_ns": started,
                    "finished_monotonic_ns": started + 10,
                    "collected_monotonic_ns": started + 15,
                    "worker_pid": index + 1,
                    "elapsed_s": 1.0 + index,
                    "user_s": 0.5 + index,
                    "system_s": 0.25,
                    "decisions": 4,
                    "conflicts": 3 + index,
                    "result": "unsat",
                    "censored": False,
                }
            )
        results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary = summarize(
            cubes=2,
            completed=2,
            timeouts=0,
            errors=0,
            sat=0,
            unsat=2,
            durations=[1.0, 2.0],
            cpu_durations=[0.75, 1.75],
            decisions=[4.0, 4.0],
            conflicts=[3.0, 4.0],
            workers=2,
            replay_workers=[2, 4],
            wall_s=2.1,
            measured_makespan_s=40 / 1e9,
        )
        return frontier, results, summary, rows

    def test_per_cube_records_reconstruct_work_and_scheduling(self):
        with tempfile.TemporaryDirectory() as directory:
            frontier, results, summary, _ = self.make_records(Path(directory))
            verify_conquer_records(frontier, results, summary, 2, [2, 4])

    def test_duplicate_cube_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontier, results, summary, rows = self.make_records(root)
            rows[1]["cube_index"] = 0
            results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(VerificationError, "duplicate cube indices"):
                verify_conquer_records(frontier, results, summary, 2, [2, 4])

    def test_harness_error_is_terminal_only_not_reconstruction(self):
        contract = load_contract(CONTRACT_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.json"
            matrix_path.write_text("{}\n", encoding="utf-8")
            toolchain = {"toolchain_sha256": "toolchain"}
            cell = {
                "cell_id": "cell",
                "instance_id": "instance",
                "method": "region-cc",
                "budget": "low",
                "product_width": 64,
                "factor_input_width": 32,
                "global_cnf_sha256": "a" * 64,
                "circuitsat_sha256": "b" * 64,
            }
            terminal = {
                "kind": "hard-regime-terminal-cell",
                "contract_sha256": contract_sha256(contract),
                "matrix_sha256": sha256_file(matrix_path),
                "toolchain_sha256": "toolchain",
                "cell_id": "cell",
                "cell_sha256": sha256_bytes(canonical_bytes(cell)),
                "instance_id": "instance",
                "method": "region-cc",
                "budget": "low",
                "product_width": 64,
                "factor_input_width": 32,
                "state": "harness-error",
                "error": "CellError: input hash mismatch",
                "input_artifacts": {
                    "global_cnf": {"sha256": "a" * 64},
                    "circuitsat": {"sha256": "b" * 64},
                },
            }
            self.assertEqual(
                verify_terminal(
                    contract,
                    matrix_path,
                    {},
                    toolchain,
                    cell,
                    terminal,
                    root,
                ),
                "harness-terminal-only",
            )


if __name__ == "__main__":
    unittest.main()
