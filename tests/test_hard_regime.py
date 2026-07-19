import copy
import unittest
from pathlib import Path

from benchmarks.cnc.hard_regime import (
    EXPECTED_METHOD_BUDGETS,
    HardRegimeError,
    load_contract,
    strong_miller_rabin,
    target_records,
    validate_contract,
    verify_target_records,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks/cnc/contracts/hard-regime-v1.yaml"


class HardRegimeTests(unittest.TestCase):
    def test_contract_freezes_issue_51_matrix(self):
        contract = load_contract(CONTRACT)
        self.assertEqual(
            {
                name: tuple(spec["budgets"])
                for name, spec in contract["methods"].items()
            },
            EXPECTED_METHOD_BUDGETS,
        )
        self.assertEqual(contract["limits_seconds"]["per_cube_conquer"], 1800)
        self.assertEqual(contract["statistics"]["unit"], "held-out-instance")

    def test_targets_are_deterministic_prime_in_range_and_disjoint(self):
        contract = load_contract(CONTRACT)
        first = target_records(contract)
        second = target_records(contract)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 39)
        self.assertTrue(all(record["expected_outcome"] == "unsat" for record in first))
        self.assertEqual(len({record["id"] for record in first}), 39)
        messages = verify_target_records(contract, first)
        self.assertTrue(any("39 deterministic prime" in message for message in messages))

    def test_contract_rejects_width_confusion_and_split_seed_overlap(self):
        contract = load_contract(CONTRACT)
        confused = copy.deepcopy(contract)
        confused["widths"][0]["factor_input_width"] = 64
        with self.assertRaisesRegex(HardRegimeError, "twice factor-input width"):
            validate_contract(confused)

        overlapping = copy.deepcopy(contract)
        overlapping["widths"][0]["held_out"]["seed"] = overlapping["widths"][0][
            "calibration"
        ]["seed"]
        with self.assertRaisesRegex(HardRegimeError, "seeds overlap"):
            validate_contract(overlapping)

    def test_contract_rejects_march_budget_tuning(self):
        contract = load_contract(CONTRACT)
        contract["methods"]["march-cu-dynamic"]["budgets"] = [
            "low",
            "medium",
            "high",
        ]
        with self.assertRaisesRegex(HardRegimeError, "march-cu-dynamic budgets"):
            validate_contract(contract)

    def test_independent_primality_check_rejects_composites(self):
        self.assertTrue(strong_miller_rabin(37))
        self.assertFalse(strong_miller_rabin(39))

if __name__ == "__main__":
    unittest.main()
