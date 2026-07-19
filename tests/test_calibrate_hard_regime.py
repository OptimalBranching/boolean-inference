import unittest

from benchmarks.cnc.calibrate_cc_difficulty import CalibrationError
from benchmarks.cnc.calibrate_hard_regime import (
    calibration_loss,
    choose_width_response,
    median_tasks,
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
