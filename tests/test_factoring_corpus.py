import json

import pytest

from benchmarks.cnc.factoring import materialize, sat_targets, unsat_targets
from benchmarks.pipeline.circuit import load_json, read_jsonl, sha256_file
from benchmarks.pipeline.multipliers import is_prime


def test_sat_and_unsat_targets_are_deterministic_and_paired():
    first_sat, _ = sat_targets(12, 3)
    second_sat, _ = sat_targets(12, 3)
    unsat, oracles = unsat_targets(12, first_sat)

    assert first_sat == second_sat
    assert all(target.paired_sat_id == sat.instance_id for target, sat in zip(unsat, first_sat))
    assert all(is_prime(target.target) for target in unsat)
    assert all(oracle["target_exceeds_max_factor"] for oracle in oracles)


def test_tiny_width_rejects_more_distinct_instances_than_exist():
    with pytest.raises(ValueError, match="only 3 distinct"):
        sat_targets(4, 4)


def test_materialize_writes_matching_circuit_and_cnf(tmp_path):
    manifest = materialize([4], 1, tmp_path)

    assert len(manifest) == 2
    assert {row["expected_outcome"] for row in manifest} == {"sat", "unsat"}
    for row in manifest:
        circuit_path = tmp_path / row["circuit"]
        cnf_path = tmp_path / row["cnf"]
        circuit = load_json(circuit_path)
        assert circuit["metadata"]["factoring"]["target"] == row["target"]
        assert circuit["metadata"]["pinned_outputs"]["product"] == row["target"]
        assert cnf_path.read_text(encoding="utf-8").startswith("c var ")
        assert "\np cnf " in cnf_path.read_text(encoding="utf-8")
        assert sha256_file(circuit_path) == row["circuit_sha256"]
        assert sha256_file(cnf_path) == row["cnf_sha256"]

    assert read_jsonl(tmp_path / "manifest.jsonl") == manifest
    oracles = read_jsonl(tmp_path / "oracles.jsonl")
    assert len(oracles) == 2
    assert json.loads((tmp_path / manifest[0]["metadata"]).read_text())["target"] == manifest[0]["target"]
