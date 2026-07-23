import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks.cnc.trace_mechanism import (
    TraceError,
    link_conquer,
    summarize,
    summarize_conquer,
)
from benchmarks.cnc.full_tree_attribution import add_cubing_seconds, compare_arms
from benchmarks.cnc.residual_mediation import (
    analyze_arm as analyze_residual_arm,
    compare_arms as compare_residual_arms,
    cube_sha256,
)
from benchmarks.cnc.residual_mediation_cohort import (
    compare_instance as compare_residual_instance,
    summarize_cohort as summarize_residual_cohort,
)
from benchmarks.cnc.trace_mechanism_cohort import summarize_cohort


def rule_record(
    node_id,
    semantics="cover",
    vector=None,
    replay_value=None,
    *,
    parent_id=None,
    child_index=None,
    depth=0,
):
    variables = 4
    feasible = 4
    vector = [2.0, 2.0] if vector is None else vector
    branches = len(vector)
    gamma = math.sqrt(2.0) if vector == [2.0, 2.0] else 1.0
    if semantics == "closed-witness":
        branches = 1
        vector = []
        gamma = 1.0
    return {
        "search_semantics": "sat-decision",
        "propagation": "ct",
        "cdcl_mode": "off",
        "node_id": node_id,
        "parent_id": parent_id,
        "child_index": child_index,
        "depth": depth,
        "kind": "branch",
        "optimized_rule_clauses": [{"mask": 0b0011, "value": 0}] * branches,
        "rule_clauses": [{"mask": 0b0011, "value": 0}] * branches,
        "rule_partition_sources": list(range(branches)),
        "rule_diagnostics": {
            "rule_semantics": semantics,
            "region_tensors": 3,
            "region_variables": variables,
            "boundary_variables": 1,
            "joined_rows": 8,
            "feasible_rows": feasible,
            "closed": semantics == "closed-witness",
            "branching_vector": vector,
            "gamma": gamma,
            "cover_verified": semantics == "cover" and replay_value is not None,
            "timing_ns": {
                "region_growth": 1_000_000,
                "feasibility_probe": 2_000_000,
                "rule_solver": 3_000_000,
            },
            "same_state_replay": replay_value,
        },
    }


def replay():
    return {
        "binary": {
            "branches": 2,
            "decision_literals": 2,
            "branching_vector": [1.0, 1.0],
            "gamma": 2.0,
            "solver_ns": 100,
        },
        "naive": {
            "branches": 4,
            "decision_literals": 16,
            "branching_vector": [2.0] * 4,
            "gamma": 2.0,
            "solver_ns": 200,
        },
    }


class TraceMechanismTest(unittest.TestCase):
    def test_residual_cohort_keeps_instances_as_units(self):
        def arm(cubes, fixed, active, conflicts, p95, cv, tail_share):
            return {
                "cubes": cubes,
                "cube_literals": {"mean": 10.0},
                "native_residual": {
                    "fixed_variables": {"mean": fixed},
                    "active_tensors": {"mean": active},
                },
                "conflicts": {
                    "total": conflicts,
                    "mean": conflicts / cubes,
                    "p95": p95,
                    "cv": cv,
                },
                "hardest_five_percent": {"conflict_share": tail_share},
                "native_cnf_propagation_equivalent_on_frontier": True,
            }

        rows = []
        for index, cube_ratio in enumerate((0.5, 2.0)):
            reference = arm(100, 10, 20, 1_000, 20, 1.0, 0.2)
            treatment = arm(
                int(100 * cube_ratio),
                15,
                18,
                1_200,
                10,
                0.8,
                0.1,
            )
            rows.append(
                compare_residual_instance(f"i{index}", treatment, reference)
            )
        result = summarize_residual_cohort(rows, "region", "march")
        self.assertEqual(result["instances"], 2)
        self.assertTrue(result["all_native_cnf_propagation_equivalent"])
        self.assertAlmostEqual(
            result["geometric_mean_ratios"]["cube_ratio"], 1.0
        )
        self.assertTrue(result["inverse_count_scaled_p95_is_sensitivity_only"])

    def test_residual_mediation_requires_exact_cube_join(self):
        residual = {
            "fixed_variables": 3,
            "unfixed_variables": 7,
            "active_tensors": 5,
            "entailed_tensors": 2,
            "constrained_variables": 7,
            "free_variables": 0,
            "constrained_components": 1,
            "largest_component_variables": 7,
            "largest_component_tensors": 5,
            "active_incidence_edges": 14,
            "active_degree_mean": 2.0,
            "residual_arity_mean": 2.8,
            "live_rows_total": 20,
            "tensor_compression_mean_bits": 1.0,
        }

        def audit(index, literals, fixed_mask, input_kind):
            return {
                "schema_version": 2,
                "cube_index": index,
                "literals": literals,
                "cube_literals": len(literals),
                "input_kind": input_kind,
                "gac_additional_fixed_variables": 2 + index,
                "fixed_mask_hex": fixed_mask,
                "fixed_value_hex": "1",
                "residual": {**residual, "fixed_variables": 3 + index},
            }

        literals = [[1, -2], [-1, 2]]
        native = [
            audit(index, cube, str(index + 1), "circuit-sat")
            for index, cube in enumerate(literals)
        ]
        cnf = [
            audit(index, cube, str(index + 1), "dimacs")
            for index, cube in enumerate(literals)
        ]
        conquer = [
            {
                "cube_index": index,
                "cube_literals": len(cube),
                "cube_sha256": cube_sha256(cube),
                "conflicts": 10 * (index + 1),
                "censored": False,
            }
            for index, cube in enumerate(literals)
        ]

        def write_jsonl(path, rows):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native_path = root / "native.jsonl"
            cnf_path = root / "cnf.jsonl"
            conquer_path = root / "conquer.jsonl"
            write_jsonl(native_path, native)
            write_jsonl(cnf_path, cnf)
            write_jsonl(conquer_path, list(reversed(conquer)))
            result = analyze_residual_arm(
                "greedy", native_path, cnf_path, conquer_path
            )
            self.assertTrue(result["cube_hash_join_verified"])
            self.assertTrue(result["native_cnf_propagation_equivalent_on_frontier"])
            self.assertEqual(result["conflicts"]["total"], 30.0)
            comparison = compare_residual_arms([result], "greedy")
            self.assertTrue(comparison["all_cube_hash_joins_verified"])

            conquer[0]["cube_sha256"] = "corrupt"
            write_jsonl(conquer_path, conquer)
            with self.assertRaisesRegex(TraceError, "SHA-256 differs"):
                analyze_residual_arm("greedy", native_path, cnf_path, conquer_path)

    def test_full_tree_comparison_uses_one_reference(self):
        def arm(name, scale):
            distribution = {
                "total": 100.0 * scale,
                "median": 10.0 * scale,
                "p95": 20.0 * scale,
                "max": 30.0 * scale,
                "cv": 0.5 * scale,
            }
            return {
                "name": name,
                "complete": True,
                "cubes": int(10 * scale),
                "decision_literals": {"mean": 12.0 * scale},
                "conflicts": distribution,
                "elapsed_s": distribution,
                "measured_makespan_s": 50.0 * scale,
            }

        arms = [arm("greedy", 1.0), arm("naive", 2.0)]
        add_cubing_seconds(arms, {"greedy": 1.0, "naive": 2.0})
        result = compare_arms(
            arms, "greedy", compare_elapsed=True
        )
        self.assertTrue(result["all_complete"])
        self.assertEqual(result["ratios_to_reference"]["naive"]["cubes"], 2.0)
        self.assertEqual(
            result["ratios_to_reference"]["naive"]["conflicts_p95"], 2.0
        )
        self.assertEqual(
            result["ratios_to_reference"]["naive"]["measured_makespan_s"], 2.0
        )
        self.assertEqual(
            result["ratios_to_reference"]["naive"]["end_to_end_s"], 2.0
        )

    def test_summarizes_open_and_closed_rules_without_pseudoclaims(self):
        records = [
            rule_record(0, replay_value=replay()),
            rule_record(
                1,
                "closed-witness",
                replay_value=replay(),
                parent_id=0,
                child_index=0,
                depth=1,
            ),
        ]
        result = summarize(records, require_replay=True)
        self.assertEqual(result["rule_nodes"], 2)
        self.assertEqual(result["cover_nodes"], 1)
        self.assertEqual(result["closed_witness_nodes"], 1)
        self.assertEqual(result["selected"]["single_branch_fraction"], 0.5)
        self.assertEqual(result["region"]["median_compression_bits"], 2.0)
        self.assertEqual(result["region"]["probe_survival_fraction"], 0.5)
        # Gamma comparisons deliberately use only ordinary cover nodes: a
        # closed witness is an existential shortcut, not Greedy rule evidence.
        self.assertEqual(result["same_state_replay"]["open_cover_nodes"], 1)
        self.assertEqual(result["same_state_replay"]["selected_better_than_naive"], 1)
        self.assertEqual(
            result["same_state_replay"]["decision_literals"]["naive"], 32
        )
        self.assertEqual(result["actual_timing_ms"]["rule_solver"], 6.0)
        # The fixture's two identical open-cover clauses may overlap.  This is
        # a syntactic warning, not a claim that their native intersection has a
        # satisfying completion.
        self.assertEqual(result["selected"]["sibling_pairs"], 1)
        self.assertEqual(
            result["selected"]["potentially_overlapping_sibling_pairs"], 1
        )
        self.assertEqual(result["selected"]["syntactically_disjoint_cover_fraction"], 0)

    def test_requires_explicit_replay_when_requested(self):
        with self.assertRaisesRegex(TraceError, "missing same-state replay"):
            summarize([rule_record(0)], require_replay=True)

    def test_requires_runtime_cover_verification_with_replay(self):
        record = rule_record(0, replay_value=replay())
        record["rule_diagnostics"]["cover_verified"] = False
        with self.assertRaisesRegex(TraceError, "cover not verified"):
            summarize([record], require_replay=True)

    def test_rejects_semantic_contract_corruption(self):
        record = rule_record(0, replay_value=replay())
        record["search_semantics"] = "partition"
        with self.assertRaisesRegex(TraceError, "sat-decision"):
            summarize([record])

    def test_accepts_only_branch_learning_cdcl_for_hybrid_provenance(self):
        record = rule_record(0, replay_value=replay())
        record["propagation"] = "hybrid"
        record["cdcl_mode"] = "branch-learning"
        self.assertEqual(summarize([record])["rule_nodes"], 1)

        record["cdcl_mode"] = "off"
        with self.assertRaisesRegex(TraceError, "invalid CDCL search provenance"):
            summarize([record])

    def test_links_cutoff_paths_without_treating_cubes_as_instances(self):
        root = rule_record(0, replay_value=replay())
        leaves = [
            {
                "search_semantics": "sat-decision",
                "propagation": "ct",
                "cdcl_mode": "off",
                "node_id": index + 1,
                "parent_id": 0,
                "child_index": index,
                "depth": 1,
                "kind": "cutoff",
                "literals": [-1, index + 2],
                "optimized_rule_clauses": [],
                "rule_clauses": [],
                "rule_partition_sources": [],
                "rule_diagnostics": None,
            }
            for index in range(2)
        ]
        outcomes = [
            {
                "cube_index": 0,
                "cube_literals": 2,
                "conflicts": 10,
                "elapsed_s": 1.0,
                "censored": False,
            },
            {
                "cube_index": 1,
                "cube_literals": 2,
                "conflicts": 100,
                "elapsed_s": 2.0,
                "censored": False,
            },
        ]
        linked = link_conquer([root, *leaves], outcomes)
        self.assertEqual(linked["linked_uncensored_cubes"], 2)
        self.assertTrue(linked["within_instance_exploratory_only"])
        self.assertEqual(linked["root_child_attribution"]["branches"], 2)
        self.assertEqual(
            linked["root_child_attribution"]["weakest_child_conflict_share"],
            10 / 110,
        )
        conquer = summarize_conquer(outcomes)
        self.assertEqual(conquer["conflicts"]["total"], 110.0)
        self.assertEqual(conquer["elapsed_s"]["max"], 2.0)

    def test_rejects_gamma_and_tree_link_corruption(self):
        bad_gamma = rule_record(0, replay_value=replay())
        bad_gamma["rule_diagnostics"]["gamma"] = 1.5
        with self.assertRaisesRegex(TraceError, "does not match"):
            summarize([bad_gamma])

        root = rule_record(0, replay_value=replay())
        child = rule_record(
            1,
            replay_value=replay(),
            parent_id=0,
            child_index=0,
            depth=3,
        )
        with self.assertRaisesRegex(TraceError, "broken parent/depth"):
            summarize([root, child])

    def test_cohort_summary_uses_instances_as_units(self):
        rows = []
        for index, ratio in enumerate((0.5, 2.0)):
            rows.append(
                {
                    "instance_id": f"i{index}",
                    "frontier_verified": True,
                    "single_branch_fraction": 0.1 + index * 0.1,
                    "selected_better_binary_fraction": 0.8,
                    "selected_better_naive_fraction": 0.7,
                    "selected_better_binary_nodes": 2,
                    "selected_better_naive_nodes": 1,
                    "median_compression_bits": 10.0,
                    "median_boundary_ratio": 0.25,
                    "probe_survival_fraction": 0.9,
                    "cover_nodes": 2,
                    "selected_branches": 3,
                    "naive_branches": 8,
                    "selected_decision_literals": 12,
                    "naive_decision_literals": 32,
                    "selected_over_naive_branches": 3 / 8,
                    "sibling_pairs": 3,
                    "potentially_overlapping_sibling_pairs": index,
                    "syntactically_disjoint_cover_nodes": 2 - index,
                    "syntactically_disjoint_cover_fraction": 1.0 - index * 0.5,
                    "rule_solver_ms": 2.0,
                    "region_growth_ms": 1.0,
                    "feasibility_probe_ms": 1.0,
                    "region_over_baseline": {
                        feature: ratio
                        for feature in (
                            "total_conflicts",
                            "conflicts_cv",
                            "conflicts_p95",
                            "conflicts_max",
                            "elapsed_p95",
                            "elapsed_max",
                        )
                    },
                }
            )
        summary = summarize_cohort(rows)
        self.assertEqual(summary["instances"], 2)
        self.assertTrue(summary["all_frontiers_verified"])
        self.assertAlmostEqual(
            summary["region_over_baseline"]["conflicts_p95"]["geometric_mean"],
            1.0,
        )
        self.assertEqual(
            summary["region_over_baseline"]["conflicts_p95"]["region_wins"], 1
        )
        self.assertEqual(summary["same_state_totals"]["cover_nodes"], 4)
        self.assertEqual(summary["same_state_totals"]["selected_branches"], 6)
        self.assertEqual(summary["same_state_totals"]["naive_branches"], 16)
        self.assertEqual(
            summary["same_state_totals"]["selected_decision_literals"], 24
        )
        self.assertEqual(
            summary["same_state_totals"]["naive_decision_literals"], 64
        )
        self.assertEqual(
            summary["same_state_totals"]["selected_better_binary_nodes"], 4
        )
        self.assertEqual(
            summary["same_state_totals"]["selected_better_naive_nodes"], 2
        )
        self.assertEqual(summary["same_state_totals"]["sibling_pairs"], 6)
        self.assertEqual(
            summary["same_state_totals"]["potentially_overlapping_sibling_pairs"],
            1,
        )
        self.assertEqual(
            summary["same_state_totals"]["syntactically_disjoint_cover_fraction"],
            0.75,
        )
        self.assertEqual(
            summary["rule_solver_fraction_of_recorded_structural_time"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
