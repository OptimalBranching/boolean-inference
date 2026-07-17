import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "benchmarks" / "scope" / "audit.py"
SCOPE = ROOT / "benchmarks" / "scope" / "benchmark-scope.yaml"


class BenchmarkScopeAuditTest(unittest.TestCase):
    def run_audit(self, path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(AUDITOR), str(path)],
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

    def test_scope_passes_every_audit(self):
        result = self.run_audit(SCOPE)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "PASS completeness: scope is complete and versioned",
                "PASS multiplier-coverage: required structures and widths are explicit",
                "PASS breadth: external miters and non-multiplier arithmetic are included",
                "PASS boundaries: evaluation choices remain out of scope",
            ],
        )

    def test_missing_architecture_fails(self):
        path = self.changed_scope(
            lambda scope: scope["controlled_multiplier_factoring"][
                "architectures"
            ].pop()
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required architectures: karatsuba", result.stdout)

    def test_changing_formal_widths_fails(self):
        path = self.changed_scope(
            lambda scope: scope["controlled_multiplier_factoring"]["scale_policy"][
                "benchmark_widths"
            ].pop()
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("formal factor-width ladder must be 64, 96, and 128 bits", result.stdout)

    def test_missing_non_multiplier_circuit_fails(self):
        path = self.changed_scope(
            lambda scope: scope["non_multiplier_arithmetic"]["circuits"].pop()
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required arithmetic circuits: square-root", result.stdout)

    def test_evaluation_policy_does_not_belong_in_scope(self):
        path = self.changed_scope(
            lambda scope: scope.update({"tuning": {"trials": 20}})
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown top-level fields: tuning", result.stdout)

    def test_duplicate_source_id_fails(self):
        def duplicate(scope):
            scope["sources"].append(dict(scope["sources"][0]))

        result = self.run_audit(self.changed_scope(duplicate))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate IDs in sources", result.stdout)

    def test_unresolved_placeholder_fails(self):
        path = self.changed_scope(
            lambda scope: scope["scope"].update({"claim_boundary": "TBD"})
        )
        result = self.run_audit(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scope contains an unresolved placeholder", result.stdout)


if __name__ == "__main__":
    unittest.main()
