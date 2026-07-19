import json
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.cnc.run_hard_regime_cell import CellError, run_process, verify_region_trace


class HardRegimeCellTests(unittest.TestCase):
    def test_region_trace_reconstructs_complete_frontier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontier = root / "frontier.icnf"
            trace = root / "nodes.jsonl"
            frontier.write_text("a 1 0\na -1 0\n", encoding="utf-8")
            rows = [
                {
                    "node_id": 0,
                    "parent_id": None,
                    "child_index": None,
                    "depth": 0,
                    "kind": "branch",
                    "refutation_reason": None,
                    "literals": [],
                    "sigma_dec": 0,
                    "sigma_all": 0,
                    "freevars": 1,
                    "rule_variables": [1],
                    "rule_clauses": [
                        {"mask": 1, "value": 1},
                        {"mask": 1, "value": 0},
                    ],
                },
                {
                    "node_id": 1,
                    "parent_id": 0,
                    "child_index": 0,
                    "depth": 1,
                    "kind": "cutoff",
                    "refutation_reason": None,
                    "literals": [1],
                    "sigma_dec": 1,
                    "sigma_all": 1,
                    "freevars": 0,
                    "rule_variables": [],
                    "rule_clauses": [],
                },
                {
                    "node_id": 2,
                    "parent_id": 0,
                    "child_index": 1,
                    "depth": 1,
                    "kind": "cutoff",
                    "refutation_reason": None,
                    "literals": [-1],
                    "sigma_dec": 1,
                    "sigma_all": 1,
                    "freevars": 0,
                    "rule_variables": [],
                    "rule_clauses": [],
                },
            ]
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = verify_region_trace(frontier, trace)
            self.assertEqual(
                summary,
                {
                    "nodes": 3,
                    "branches": 1,
                    "cutoffs": 2,
                    "refuted": 0,
                    "sat_leaves": 0,
                    "root_refutations": 0,
                    "selector_refutations": 0,
                    "branch_refutations": 0,
                },
            )

            rows.pop()
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(CellError, "incomplete children"):
                verify_region_trace(frontier, trace)

    def test_region_trace_rejects_child_that_does_not_apply_its_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontier = root / "frontier.icnf"
            trace = root / "nodes.jsonl"
            frontier.write_text("a -1 0\n", encoding="utf-8")
            rows = [
                {
                    "node_id": 0,
                    "parent_id": None,
                    "child_index": None,
                    "depth": 0,
                    "kind": "branch",
                    "refutation_reason": None,
                    "literals": [],
                    "sigma_dec": 0,
                    "sigma_all": 0,
                    "freevars": 1,
                    "rule_variables": [1],
                    "rule_clauses": [{"mask": 1, "value": 1}],
                },
                {
                    "node_id": 1,
                    "parent_id": 0,
                    "child_index": 0,
                    "depth": 1,
                    "kind": "cutoff",
                    "refutation_reason": None,
                    "literals": [-1],
                    "sigma_dec": 1,
                    "sigma_all": 1,
                    "freevars": 0,
                    "rule_variables": [],
                    "rule_clauses": [],
                },
            ]
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(CellError, "does not implement rule clause"):
                verify_region_trace(frontier, trace)

    def test_process_timeout_is_a_terminal_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_process(
                [sys.executable, "-c", "import time; time.sleep(1)"],
                0.01,
                root / "stdout",
                root / "stderr",
            )
            self.assertEqual(result["state"], "timeout")
            self.assertIsNone(result["returncode"])
            self.assertGreaterEqual(result["wall_s"], 0.01)
            self.assertEqual(len(result["stdout_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
