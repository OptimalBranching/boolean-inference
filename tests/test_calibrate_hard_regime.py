import inspect
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.cnc.calibrate_cc_difficulty import CalibrationError, sha256_file
from benchmarks.cnc.calibrate_hard_regime import (
    calibrate_width,
    calibration_loss,
    choose_width_response,
    median_tasks,
    ProbeContext,
    run_or_resume_probe,
)


def row(threshold, *counts):
    return {
        "threshold": threshold,
        "instances": [
            {"id": f"cal-{index}", "tasks": count}
            for index, count in enumerate(counts)
        ],
    }


class HardRegimeCalibrationTests(unittest.TestCase):
    def probe_fixture(self, root: Path, *, phase: str = "candidate"):
        cuber = root / "cnc_cuber"
        cuber.write_text("fake cuber\n", encoding="utf-8")
        instance = root / "instance.json"
        instance.write_text("{}\n", encoding="utf-8")
        frontier = root / "frontier.icnf"
        log = root / "cuber.log"
        checkpoint = root / "probe.json"
        trace = root / "nodes.jsonl" if phase != "candidate" else None
        calls = []

        def runner(
            cuber_path,
            instance_path,
            threshold,
            frontier_path,
            log_path,
            max_rows,
            *,
            trace,
            selector,
            timeout_s,
        ):
            calls.append((cuber_path, instance_path, threshold, selector, timeout_s))
            frontier_path.write_text("a 1 0\n", encoding="utf-8")
            log_path.write_text("status=OK cubes=1\n", encoding="utf-8")
            if trace is not None:
                trace.write_text('{"node":0}\n', encoding="utf-8")
            return {
                "threshold": threshold,
                "tasks": 1,
                "elapsed_s": 2.0,
                "user_s": 1.5,
                "system_s": 0.25,
            }

        arguments = {
            "context": ProbeContext(
                probe_runner=runner,
                cuber=cuber,
                cuber_sha256=sha256_file(cuber),
                contract_digest="a" * 64,
                product_width=64,
                selector="region",
                max_rows=512,
                timeout_s=7200.0,
            ),
            "instance": instance,
            "instance_id": "cal-0",
            "instance_sha256": sha256_file(instance),
            "threshold": 1,
            "phase": phase,
            "frontier": frontier,
            "log": log,
            "checkpoint": checkpoint,
            "trace": trace,
        }
        return arguments, calls

    def test_completed_probe_checkpoint_is_reused(self):
        with TemporaryDirectory() as directory:
            arguments, calls = self.probe_fixture(Path(directory))
            first = run_or_resume_probe(**arguments)

            def unexpected_runner(*args, **kwargs):
                self.fail("valid checkpoint should avoid rerunning the cuber")

            arguments["context"] = replace(
                arguments["context"], probe_runner=unexpected_runner
            )
            second = run_or_resume_probe(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_checkpoint_rejects_tampered_frontier(self):
        with TemporaryDirectory() as directory:
            arguments, _ = self.probe_fixture(Path(directory))
            run_or_resume_probe(**arguments)
            arguments["frontier"].write_text("a 1 0\na 2 0\n", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "artifact hash mismatch"):
                run_or_resume_probe(**arguments)

    def test_checkpoint_rejects_changed_provenance(self):
        with TemporaryDirectory() as directory:
            arguments, _ = self.probe_fixture(Path(directory))
            run_or_resume_probe(**arguments)
            arguments["context"] = replace(
                arguments["context"], selector="structure-blind"
            )
            with self.assertRaisesRegex(CalibrationError, "provenance mismatch"):
                run_or_resume_probe(**arguments)

    def test_selected_checkpoint_requires_trace(self):
        with TemporaryDirectory() as directory:
            arguments, _ = self.probe_fixture(Path(directory), phase="selected-low")
            run_or_resume_probe(**arguments)
            arguments["trace"].unlink()
            with self.assertRaisesRegex(CalibrationError, "artifact is missing"):
                run_or_resume_probe(**arguments)

    def test_hard_regime_search_starts_at_one(self):
        default = inspect.signature(calibrate_width).parameters["initial_threshold"].default
        self.assertEqual(default, 1)

    def test_hard_regime_rejects_a_different_search_start(self):
        with self.assertRaisesRegex(CalibrationError, "must start at 1"):
            calibrate_width(
                contract={"methods": {}},
                instances=[{}, {}, {}],
                product_width=64,
                selector="region",
                cuber=Path("cuber"),
                out_dir=Path("out"),
                initial_threshold=1024,
            )

    def test_hard_regime_search_probes_one_then_two_and_never_zero(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cuber = root / "cnc_cuber"
            cuber.write_text("fake cuber\n", encoding="utf-8")
            instances = []
            for index in range(3):
                path = root / f"cal-{index}.json"
                path.write_text("{}\n", encoding="utf-8")
                instances.append(
                    {"id": f"cal-{index}", "path": path, "sha256": sha256_file(path)}
                )
            candidate_thresholds = []

            def runner(
                cuber_path,
                instance_path,
                threshold,
                frontier_path,
                log_path,
                max_rows,
                *,
                trace,
                selector,
                timeout_s,
            ):
                if trace is None:
                    candidate_thresholds.append(threshold)
                frontier_path.write_text(
                    "".join(f"a {index + 1} 0\n" for index in range(threshold)),
                    encoding="utf-8",
                )
                log_path.write_text("status=OK\n", encoding="utf-8")
                if trace is not None:
                    trace.write_text('{"node":0}\n', encoding="utf-8")
                return {
                    "threshold": threshold,
                    "tasks": threshold,
                    "elapsed_s": 1.0,
                    "user_s": 0.5,
                    "system_s": 0.25,
                }

            calibrate_width(
                contract={
                    "methods": {"region-cc": {"max_rows": 512}},
                    "limits_seconds": {"cubing": 60},
                    "frontier_bands": {
                        "low": {"center_cubes": 2, "accepted_ratio": [0.5, 1.5]}
                    },
                },
                instances=instances,
                product_width=64,
                selector="region",
                cuber=cuber,
                out_dir=root / "calibration",
                maximum_threshold=8,
                probe_runner=runner,
            )

        self.assertEqual(candidate_thresholds, [1, 1, 1, 2, 2, 2])
        self.assertNotIn(0, candidate_thresholds)

    def test_width_selection_uses_all_three_calibration_instances(self):
        response = [
            row(100, 300, 400, 500),
            row(200, 480, 520, 560),
            row(300, 500, 700, 4000),
        ]
        selected = choose_width_response(response, 512, 384, 640)
        self.assertEqual(selected["threshold"], 200)
        self.assertEqual(median_tasks(selected), 520)

    def test_loss_penalizes_cross_instance_mismatch(self):
        balanced = row(100, 480, 512, 544)
        skewed = row(200, 128, 512, 2048)
        self.assertLess(calibration_loss(balanced, 512), calibration_loss(skewed, 512))

    def test_empty_response_fails_closed(self):
        with self.assertRaises(CalibrationError):
            choose_width_response([], 512, 384, 640)

    def test_response_without_an_in_band_threshold_fails_closed(self):
        with self.assertRaisesRegex(CalibrationError, "accepted task range"):
            choose_width_response([row(100, 10, 20, 30)], 512, 384, 640)


if __name__ == "__main__":
    unittest.main()
