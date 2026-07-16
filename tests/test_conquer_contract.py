import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "conquer_cubes.py"
SPEC = importlib.util.spec_from_file_location("conquer_cubes", MODULE_PATH)
CONQUER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONQUER)


class ConquerContractDigestTest(unittest.TestCase):
    def test_reads_exactly_one_valid_digest(self):
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "study.lock"
            lock.write_text(f"algorithm: sha256\ncontract_digest: {digest}\n", encoding="utf-8")
            self.assertEqual(CONQUER.read_contract_digest(lock), digest)

    def test_rejects_missing_or_malformed_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "study.lock"
            lock.write_text("contract_digest: not-a-digest\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected one lowercase SHA-256"):
                CONQUER.read_contract_digest(lock)


if __name__ == "__main__":
    unittest.main()
