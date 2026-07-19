import unittest

from benchmarks.cnc.aggregate_hard_regime import (
    adjusted_observations,
    bootstrap_ci,
    geometric_mean,
    interpolate_log,
    summarize_observations,
)


class HardRegimeAggregateTests(unittest.TestCase):
    def test_log_interpolation_has_no_extrapolation(self):
        points = [(100.0, 10.0), (400.0, 40.0)]
        self.assertAlmostEqual(interpolate_log(points, 200.0), 20.0)
        self.assertIsNone(interpolate_log(points, 50.0))
        self.assertIsNone(interpolate_log(points, 800.0))

    def test_geometric_mean_and_bootstrap_are_deterministic(self):
        self.assertAlmostEqual(geometric_mean([0.5, 2.0]), 1.0)
        first = bootstrap_ci([0.8, 1.0, 1.2], 100, 0.95, 51, "work:low")
        second = bootstrap_ci([0.8, 1.0, 1.2], 100, 0.95, 51, "work:low")
        self.assertEqual(first, second)

    def test_summary_counts_instances_not_cubes(self):
        observations = [
            {
                "analysis": "raw-nominal-budget",
                "metric": "conquer_work_cpu_s",
                "budget": "low",
                "product_width": 64,
                "instance_id": "a",
                "complete_pair": True,
                "ratio": 0.8,
            },
            {
                "analysis": "raw-nominal-budget",
                "metric": "conquer_work_cpu_s",
                "budget": "low",
                "product_width": 64,
                "instance_id": "b",
                "complete_pair": False,
                "ratio": None,
            },
        ]
        summary = summarize_observations(
            observations,
            "raw-nominal-budget",
            "conquer_work_cpu_s",
            "budget",
            "low",
            64,
            {"samples": 100, "confidence": 0.95, "seed": 51},
        )
        self.assertEqual(summary["declared_pairs"], 2)
        self.assertEqual(summary["complete_pairs"], 1)
        self.assertEqual(summary["instance_ids"], ["a"])
        self.assertAlmostEqual(summary["geometric_mean_ratio"], 0.8)

    def test_adjusted_rows_retain_instances_without_common_support(self):
        cells = []
        terminals = {}
        for method in ("region-cc", "structure-blind-cc"):
            for budget in ("low", "medium", "high"):
                cell_id = f"instance__{method}__{budget}"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "instance_id": "instance",
                        "split": "held_out",
                        "product_width": 64,
                        "method": method,
                        "budget": budget,
                    }
                )
                terminals[cell_id] = {
                    "state": "conquer-timeout",
                    "metrics": {"frontier_size": 100},
                }
        rows = adjusted_observations(
            {"cells": cells}, terminals, "conquer_work_cpu_s"
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["instance_id"] for row in rows}, {"instance"})
        self.assertTrue(all(row["adjustment_status"] == "incomplete-series" for row in rows))
        self.assertTrue(all(row["complete_pair"] is False for row in rows))
        self.assertTrue(all(row["common_frontier_size"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
