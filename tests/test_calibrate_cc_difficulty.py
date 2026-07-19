import unittest

from benchmarks.cnc.calibrate_cc_difficulty import CalibrationError, choose


class CalibrateCcDifficultyTests(unittest.TestCase):
    def test_choose_prefers_closest_count_in_accepted_range(self):
        rows = [
            {"threshold": 100, "tasks": 300},
            {"threshold": 200, "tasks": 480},
            {"threshold": 300, "tasks": 540},
            {"threshold": 400, "tasks": 900},
        ]
        self.assertEqual(choose(rows, 512, 384, 640)["threshold"], 300)

    def test_choose_falls_back_when_range_is_skipped(self):
        rows = [
            {"threshold": 100, "tasks": 100},
            {"threshold": 200, "tasks": 1000},
        ]
        self.assertEqual(choose(rows, 512, 384, 640)["threshold"], 200)

    def test_tie_breaks_toward_lower_threshold(self):
        rows = [
            {"threshold": 200, "tasks": 512},
            {"threshold": 100, "tasks": 512},
        ]
        self.assertEqual(choose(rows, 512, 1, 2000)["threshold"], 100)

    def test_empty_response_fails_closed(self):
        with self.assertRaises(CalibrationError):
            choose([], 512, 384, 640)


if __name__ == "__main__":
    unittest.main()
