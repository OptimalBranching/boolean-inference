import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "benchmarks" / "scope" / "audit.py"
SCOPE = ROOT / "benchmarks" / "scope" / "benchmark-scope.yaml"
LOCK = ROOT / "benchmarks" / "scope" / "benchmark-scope.lock"


class BenchmarkScopeAuditTest(unittest.TestCase):
    def run_audit(self, *paths: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(AUDITOR), "audit", *(str(path) for path in paths)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def changed_scope(self, mutate) -> Path:
        scope = yaml.safe_load(SCOPE.read_text(encoding="utf-8"))
        mutate(scope)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "scope.yaml"
        path.write_text(yaml.safe_dump(scope, sort_keys=False), encoding="utf-8")
        return path

    def test_frozen_scope_passes_every_audit(self):
        result = self.run_audit(SCOPE, LOCK)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "PASS completeness: scope matches the schema and has no unresolved fields",
                "PASS multiplier-coverage: required multiplier structures and matched targets are explicit",
                "PASS breadth: external miters and non-multiplication arithmetic are included",
                "PASS boundaries: conditional families and justified exclusions are explicit",
                "PASS freeze: canonical digest matches the scope lock",
            ],
        )

    def test_missing_multiplier_architecture_fails(self):
        path = self.changed_scope(
            lambda scope: scope["controlled_multiplier_factoring"][
                "architectures"
            ].pop()
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "FAIL multiplier-coverage: missing required multiplier architectures: karatsuba",
            result.stdout,
        )

    def test_missing_non_multiplier_circuit_fails(self):
        path = self.changed_scope(
            lambda scope: scope["non_multiplier_arithmetic"]["circuits"].pop()
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "missing required non-multiplier arithmetic circuits: square-root",
            result.stdout,
        )

    def test_evaluation_policy_does_not_belong_in_scope(self):
        path = self.changed_scope(
            lambda scope: scope.update({"tuning": {"trials": 20}})
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL completeness", result.stdout)
        self.assertIn("unknown field 'tuning'", result.stdout)

    def test_scope_change_without_new_lock_fails(self):
        path = self.changed_scope(
            lambda scope: scope["scope"].update(
                {"claim_boundary": "Changed after freeze."}
            )
        )
        result = self.run_audit(path, LOCK)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "FAIL freeze: canonical digest does not match the lock", result.stdout
        )

    def test_duplicate_source_id_fails(self):
        def duplicate(scope):
            scope["sources"].append(dict(scope["sources"][0]))

        result = self.run_audit(self.changed_scope(duplicate))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate IDs in sources", result.stdout)


if __name__ == "__main__":
    unittest.main()
