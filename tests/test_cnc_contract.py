import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "experiments" / "cnc" / "contract.py"
CONTRACT = ROOT / "experiments" / "cnc-study.yaml"
LOCK = ROOT / "experiments" / "cnc-study.lock"
LEAKAGE = ROOT / "tests" / "fixtures" / "cnc" / "contract-with-split-leakage.yaml"


class ContractAuditTest(unittest.TestCase):
    def run_audit(self, *paths: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(AUDITOR), "audit", *(str(path) for path in paths)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_frozen_contract_passes_every_audit(self):
        result = self.run_audit(CONTRACT, LOCK)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "PASS completeness: no required field is unresolved",
                "PASS split: tuning and held-out instances are disjoint",
                "PASS comparisons: methods and controlled contrasts are explicit",
                "PASS accounting: every reported metric has a declared definition",
                "PASS freeze: canonical digest matches the lock",
            ],
        )

    def test_split_leakage_fails(self):
        result = self.run_audit(LEAKAGE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "FAIL split: an instance appears in both tuning and held-out data",
            result.stdout,
        )

    def test_contract_change_without_new_lock_fails(self):
        contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
        contract["execution"]["workers"] += 1
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.yaml"
            changed.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            result = self.run_audit(changed, LOCK)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL freeze: canonical digest does not match the lock", result.stdout)

    def test_unknown_field_fails_completeness(self):
        contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
        contract["protocol"]["after_the_fact"] = True
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.yaml"
            changed.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            result = self.run_audit(changed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL completeness", result.stdout)
        self.assertIn("unknown field 'after_the_fact'", result.stdout)

    def test_duplicate_instance_id_fails_completeness(self):
        contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
        instances = contract["benchmarks"]["families"][0]["instances"]
        instances.append(dict(instances[0]))
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.yaml"
            changed.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
            result = self.run_audit(changed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate instance IDs", result.stdout)

    def test_changed_contract_requires_a_new_protocol_version(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            contract_path = repo / "cnc-study.yaml"
            lock_path = repo / "cnc-study.lock"
            contract_path.write_text(CONTRACT.read_text(encoding="utf-8"), encoding="utf-8")
            lock_path.write_text(LOCK.read_text(encoding="utf-8"), encoding="utf-8")
            for command in (
                ["git", "init", "-q"],
                ["git", "config", "user.name", "Contract Test"],
                ["git", "config", "user.email", "contract-test@example.invalid"],
                ["git", "add", "cnc-study.yaml", "cnc-study.lock"],
                ["git", "commit", "-qm", "Freeze v1"],
            ):
                subprocess.run(command, cwd=repo, check=True)

            contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
            contract["execution"]["workers"] += 1
            contract_path.write_text(
                yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
            )
            digest = subprocess.run(
                [sys.executable, str(AUDITOR), "digest", str(contract_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
            lock["contract_digest"] = digest
            lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")

            result = self.run_audit(contract_path, lock_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "FAIL freeze: protocol version 1 is already frozen to a different digest",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
