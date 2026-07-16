import copy
import itertools
import tempfile
import unittest
from pathlib import Path

from benchmarks.pipeline.circuit import (
    CircuitError,
    assignment,
    nary,
    pin_port_values,
    simulate,
    validate_circuit,
    var,
    write_json,
    write_jsonl,
)
from benchmarks.pipeline.cnf import Cnf, encode_circuit
from benchmarks.pipeline.collect_manifest import collect
from benchmarks.pipeline.generate_multiplier_instances import generate
from benchmarks.pipeline.generate_structural_multiplier import (
    generate as generate_multiplier,
)
from benchmarks.pipeline.generate_targets import records
from benchmarks.pipeline.import_verilog import import_verilog
from benchmarks.pipeline.make_miter import build_miter
from benchmarks.pipeline.make_preimages import generate as generate_preimages
from benchmarks.pipeline.validate import validate_dimacs
from benchmarks.pipeline.validate_multiplier_witnesses import (
    validate as validate_multiplier_witnesses,
)
from benchmarks.pipeline.yosys_json_to_circuitsat import convert


def half_adder():
    return {
        "variables": ["a", "b", "sum", "carry"],
        "circuit": {
            "assignments": [
                assignment("sum", nary("Xor", [var("a"), var("b")])),
                assignment("carry", nary("And", [var("a"), var("b")])),
            ]
        },
        "metadata": {
            "ports": {
                "a": {"direction": "input", "bits": ["a"], "lsb_first": True},
                "b": {"direction": "input", "bits": ["b"], "lsb_first": True},
                "product": {
                    "direction": "output",
                    "bits": ["sum", "carry"],
                    "lsb_first": True,
                },
            }
        },
    }


def cnf_satisfiable(cnf: Cnf, fixed: dict[int, bool] | None = None) -> bool:
    fixed = fixed or {}
    free = [
        index for index in range(1, len(cnf.variable_names) + 1) if index not in fixed
    ]
    for bits in itertools.product((False, True), repeat=len(free)):
        values = {**fixed, **dict(zip(free, bits, strict=True))}
        if all(
            any(values[abs(literal)] == (literal > 0) for literal in clause)
            for clause in cnf.clauses
        ):
            return True
    return False


class BenchmarkPipelineTest(unittest.TestCase):
    def test_targets_are_deterministic_balanced_semiprimes(self):
        first = list(records([8], 3, 100))
        second = list(records([8], 3, 100))
        self.assertEqual(first, second)
        for public, oracle in first:
            self.assertEqual(
                public["target"], oracle["left_factor"] * oracle["right_factor"]
            )
            self.assertEqual(oracle["left_factor"].bit_length(), 8)
            self.assertEqual(oracle["right_factor"].bit_length(), 8)
            self.assertNotIn("left_factor", public)
            self.assertEqual(public["generator"], "balanced-semiprime-v1")
        with self.assertRaises(ValueError):
            list(records([8, 8], 1, 100))

    def test_yosys_simple_gate_conversion_and_simulation(self):
        yosys = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2]},
                        "b": {"direction": "input", "bits": [3]},
                        "product": {"direction": "output", "bits": [4]},
                    },
                    "cells": {
                        "scope": {
                            "type": "$scopeinfo",
                            "port_directions": {},
                            "connections": {},
                        },
                        "gate": {
                            "type": "$_AND_",
                            "port_directions": {
                                "A": "input",
                                "B": "input",
                                "Y": "output",
                            },
                            "connections": {"A": [2], "B": [3], "Y": [4]},
                        },
                    },
                    "netnames": {},
                }
            }
        }
        data = convert(yosys)
        for left, right in itertools.product((False, True), repeat=2):
            values = simulate(data, {"a[0]": left, "b[0]": right})
            self.assertEqual(values["product[0]"], left and right)

    def test_validation_and_simulation_handle_ports_and_dependency_order(self):
        malformed = copy.deepcopy(half_adder())
        malformed["metadata"]["ports"]["a"]["bits"] = ["missing"]
        with self.assertRaises(CircuitError):
            validate_circuit(malformed)

        chain = {
            "variables": ["a", "x", "y"],
            "circuit": {
                "assignments": [
                    assignment("y", var("x")),
                    assignment("x", var("a")),
                ]
            },
        }
        self.assertTrue(simulate(chain, {"a": True})["y"])

    def test_import_requires_complete_source_provenance(self):
        with self.assertRaises(CircuitError):
            import_verilog(
                Path("missing.v"),
                "top",
                "yosys",
                source_id="source-without-revision",
            )

    def test_native_array_and_karatsuba_multipliers_are_correct(self):
        for architecture in ("array-ripple", "karatsuba"):
            data = generate_multiplier(5, architecture, base_case=3)
            for left, right in itertools.product(range(32), repeat=2):
                inputs = {
                    **{f"a[{index}]": bool((left >> index) & 1) for index in range(5)},
                    **{f"b[{index}]": bool((right >> index) & 1) for index in range(5)},
                }
                values = simulate(data, inputs)
                product = sum(
                    int(values[f"product[{index}]"]) << index for index in range(10)
                )
                self.assertEqual(product, left * right)

    def test_tseitin_encoding_matches_half_adder_truth_table(self):
        cnf = encode_circuit(half_adder())
        ids = {name: index for index, name in enumerate(cnf.variable_names, 1)}
        for left, right, sum_bit, carry in itertools.product((False, True), repeat=4):
            expected = sum_bit == (left ^ right) and carry == (left and right)
            fixed = {
                ids["a"]: left,
                ids["b"]: right,
                ids["sum"]: sum_bit,
                ids["carry"]: carry,
            }
            self.assertEqual(cnf_satisfiable(cnf, fixed), expected)

    def test_pinned_instance_and_identical_miter_have_expected_status(self):
        raw = half_adder()
        pinned = pin_port_values(raw, {"product": 2})
        self.assertTrue(cnf_satisfiable(encode_circuit(pinned)))
        miter = build_miter(raw, raw, "product", "product", None)
        self.assertFalse(cnf_satisfiable(encode_circuit(miter)))
        self.assertLessEqual(max(map(len, encode_circuit(miter).clauses)), 3)

    def test_preimage_generation_writes_paired_artifacts_and_valid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_out = generate_preimages(half_adder(), 2, 7, root, "half-adder")
            self.assertEqual(len(records_out), 2)
            self.assertEqual(len(collect(root)), 2)
            self.assertEqual(
                len(
                    {
                        tuple(sorted(record["pinned_outputs"].items()))
                        for record in records_out
                    }
                ),
                2,
            )
            for record in records_out:
                self.assertEqual(record["expected_outcome"], "sat")
                self.assertTrue((root / record["circuitsat"]).is_file())
                self.assertTrue((root / record["cnf"]).is_file())

    def test_multiplier_targets_are_reused_across_architectures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw-2.json"
            raw_circuit = generate_multiplier(1, "array-ripple")
            del raw_circuit["metadata"]["architecture"]
            write_json(raw, raw_circuit)
            generated = generate(
                [{"id": "fact-1-0000", "factor_bits": 1, "target": 1}],
                {"array-ripple": str(raw), "wallace-ripple": str(raw)},
                {},
                "product",
                root / "out",
            )
            self.assertEqual({record["target"] for record in generated}, {1})
            self.assertEqual(
                {record["architecture"] for record in generated},
                {"array-ripple", "wallace-ripple"},
            )
            self.assertEqual(
                {record["source_provenance"]["generator"] for record in generated},
                {"boolean-inference-structural-multiplier-v1"},
            )

    def test_multiplier_witness_validation_fails_on_wrong_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            write_json(
                raw_dir / "array-ripple-2.json",
                generate_multiplier(2, "array-ripple"),
            )
            manifest = root / "manifest.jsonl"
            oracle = root / "oracle.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "target_id": "fact-2-0000",
                        "target": 6,
                        "raw_circuit": "array-ripple-2.json",
                    }
                ],
            )
            write_jsonl(
                oracle,
                [{"id": "fact-2-0000", "left_factor": 2, "right_factor": 3}],
            )
            self.assertEqual(
                validate_multiplier_witnesses(manifest, oracle, raw_dir), 1
            )

            write_jsonl(
                oracle,
                [{"id": "fact-2-0000", "left_factor": 2, "right_factor": 2}],
            )
            with self.assertRaises(CircuitError):
                validate_multiplier_witnesses(manifest, oracle, raw_dir)

    def test_manifest_and_dimacs_validation_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(CircuitError):
                collect(root)
            cnf = root / "wide.cnf"
            cnf.write_text("p cnf 3 1\n1 2 3 0\n", encoding="utf-8")
            self.assertEqual(validate_dimacs(cnf, 3), (3, 1, 3))
            with self.assertRaises(CircuitError):
                validate_dimacs(cnf, 2)
            cnf.write_text("p cnf 2 1\n3 0\n", encoding="utf-8")
            with self.assertRaises(CircuitError):
                validate_dimacs(cnf)

    def test_multiplier_generation_rejects_duplicate_target_ids(self):
        with self.assertRaises(CircuitError):
            generate(
                [
                    {"id": "duplicate", "factor_bits": 1, "target": 1},
                    {"id": "duplicate", "factor_bits": 1, "target": 1},
                ],
                {},
                {},
                "product",
                Path("unused"),
            )


if __name__ == "__main__":
    unittest.main()
