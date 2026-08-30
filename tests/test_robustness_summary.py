from __future__ import annotations

import unittest

from robustness_benchmark import RobustnessScenario, summarize


def _row(
    scenario: str,
    *,
    lateral: float,
    maximum: float,
    heading: float,
    speed: float,
    violations: int = 0,
    failures: int = 0,
) -> dict:
    return {
        "scenario": scenario,
        "lateral_rmse_m": lateral,
        "max_position_error_m": maximum,
        "heading_rmse_deg": heading,
        "speed_rmse_mps": speed,
        "constraint_violations": violations,
        "solver_failures": failures,
    }


class RobustnessSummaryRegressionTests(unittest.TestCase):
    def test_summarize_preserves_aggregation_and_acceptance_logic(self) -> None:
        scenario = RobustnessScenario(name="noise")
        rows = [
            _row("noise", lateral=0.10, maximum=0.30, heading=1.0, speed=0.2),
            _row("noise", lateral=0.20, maximum=0.40, heading=3.0, speed=0.4),
        ]

        summary = summarize(rows, (scenario,), 0.25, 0.50)[0]

        self.assertEqual(summary["runs"], 2)
        self.assertAlmostEqual(summary["lateral_rmse_mean_m"], 0.15)
        self.assertAlmostEqual(summary["lateral_rmse_worst_m"], 0.20)
        self.assertAlmostEqual(summary["max_position_error_worst_m"], 0.40)
        self.assertAlmostEqual(summary["heading_rmse_mean_deg"], 2.0)
        self.assertAlmostEqual(summary["speed_rmse_mean_mps"], 0.3)
        self.assertTrue(summary["acceptance_passed"])

    def test_summarize_rejects_constraint_or_solver_failures(self) -> None:
        scenarios = (
            RobustnessScenario(name="constraint"),
            RobustnessScenario(name="solver"),
        )
        rows = [
            _row(
                "constraint",
                lateral=0.01,
                maximum=0.02,
                heading=0.1,
                speed=0.1,
                violations=1,
            ),
            _row(
                "solver",
                lateral=0.01,
                maximum=0.02,
                heading=0.1,
                speed=0.1,
                failures=1,
            ),
        ]

        summaries = summarize(rows, scenarios, 1.0, 1.0)

        self.assertEqual(summaries[0]["constraint_violations_total"], 1)
        self.assertFalse(summaries[0]["acceptance_passed"])
        self.assertEqual(summaries[1]["solver_failures_total"], 1)
        self.assertFalse(summaries[1]["acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
