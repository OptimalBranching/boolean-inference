import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "benchmarks" / "cnc" / "verify_measurements.py"
FIXTURES = ROOT / "tests" / "fixtures" / "cnc"


class CncMeasurementsTest(unittest.TestCase):
    def run_bundle(self, path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(VERIFIER), "--bundle", str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_fixture(self, name: str) -> subprocess.CompletedProcess[str]:
        return self.run_bundle(FIXTURES / name)

    def test_valid_bundle_recomputes_issue_41_metrics(self):
        result = self.run_fixture("measurement-valid")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "PASS frontier: 8/8 assignments are covered exactly once",
                "PASS verdict: returned model satisfies the input",
                "PASS accounting: cubing_wall=2 conquer_cpu=9 "
                "conquer_makespan=5 end_to_end_wall=7",
                "PASS workers: observed concurrency does not exceed 2",
            ],
        )

    def test_missing_cube_fails_with_uncovered_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            shutil.copytree(FIXTURES / "measurement-valid", bundle)
            frontier_path = bundle / "frontier.jsonl"
            frontier = "\n".join(frontier_path.read_text().splitlines()[1:]) + "\n"
            frontier_path.write_text(frontier, encoding="utf-8")
            manifest_path = bundle / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["frontier"]["sha256"] = hashlib.sha256(
                frontier.encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self.run_bundle(bundle)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL frontier: assignment 000 is uncovered", result.stdout)

    def test_accounting_is_recomputed_instead_of_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            shutil.copytree(FIXTURES / "measurement-valid", bundle)
            manifest_path = bundle / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["accounting"]["conquer_cpu"] = 8.0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self.run_bundle(bundle)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("accounting mismatch for conquer_cpu", result.stdout)

    def test_worker_identity_cannot_hide_overlapping_work(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            shutil.copytree(FIXTURES / "measurement-valid", bundle)
            events_path = bundle / "events.jsonl"
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            for event in events:
                if event.get("cube_id") == "010":
                    event["worker"] = 1
            encoded = "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                for event in events
            )
            events_path.write_text(encoded, encoding="utf-8")
            manifest_path = bundle / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["events"]["sha256"] = hashlib.sha256(
                encoded.encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self.run_bundle(bundle)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("worker 1 runs overlapping cube 010", result.stdout)


if __name__ == "__main__":
    unittest.main()
