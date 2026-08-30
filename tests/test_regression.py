import unittest

import numpy as np

from config import MPCConfig, VehicleConfig, load_project_config
from control import NonlinearMPC
from dynamic_model_benchmark import _load_scenarios, run_dynamic_trial
from models import (
    DynamicBicycle,
    DynamicVehicleConfig,
    DynamicVehicleState,
    KinematicBicycle,
)
from planning import candidate_cost_weights_from_mapping
from simulation import run_closed_loop


class QuantitativeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dynamic_project = load_project_config("configs/dynamic_model_benchmark.json")
        cls.dynamic_scenarios = {
            scenario.name: scenario
            for scenario in _load_scenarios(
                cls.dynamic_project.raw["dynamic_benchmark"]["scenarios"],
                cls.dynamic_project,
            )
        }

    def test_default_nmpc_acceptance_gate(self):
        project = load_project_config("configs/default.json")
        result = run_closed_loop(
            "nmpc", project.simulation, project.road.amplitude_m,
            project.road.initial_lateral_offset_m, project.vehicle, project.mpc,
            project.road.wavelength_m, project.road.length_m,
        )
        self.assertLessEqual(result.metrics["lateral_rmse_m"], 0.08)
        self.assertEqual(result.metrics["constraint_violations"], 0)
        self.assertEqual(result.metrics["solver_failures"], 0)

    def test_mpc_enforces_predicted_speed_limit(self):
        limits = VehicleConfig()
        controller = NonlinearMPC(KinematicBicycle(limits, 0.1), limits, MPCConfig())
        state = np.array([0.0, 0.0, 0.0, limits.max_speed - 0.05])
        references = np.tile([2.0, 0.0, 0.0, limits.max_speed + 10.0], (8, 1))
        controller.control(state, references)
        predicted_speed = controller._rollout(state, controller.previous_solution)[:, 3]
        self.assertTrue(controller.last_success)
        self.assertTrue(np.all(predicted_speed <= limits.max_speed + 1e-5))
        self.assertTrue(np.all(predicted_speed >= limits.min_speed - 1e-5))

    def test_combined_tire_force_stays_inside_friction_circle(self):
        limits = VehicleConfig()
        plant = DynamicBicycle(
            DynamicVehicleConfig(friction_coefficient=0.3), limits, 0.1,
        )
        state = DynamicVehicleState(0.0, 0.0, 0.0, 12.0)
        for _ in range(20):
            state = plant.step(state, limits.min_accel, limits.max_steer)
        self.assertLessEqual(plant.max_tire_friction_utilization, 1.0 + 1e-9)

    def test_planner_cost_weights_are_config_driven(self):
        planner = load_project_config("configs/planner_demo.json").raw["planner"]
        weights = candidate_cost_weights_from_mapping(planner["cost_weights"])
        self.assertEqual(weights.obstacle_risk, 2.0)
        self.assertEqual(planner["min_normalized_clearance"], 1.25)

    def test_rate_aware_dynamic_correction_gate(self):
        metrics, _, _ = run_dynamic_trial(
            self.dynamic_scenarios["steer_rate_aware_12"], self.dynamic_project,
        )
        self.assertLessEqual(metrics["lateral_rmse_m"], 0.30)
        self.assertLessEqual(metrics["max_sideslip_deg"], 5.0)
        self.assertLessEqual(metrics["max_steering_rate_deg_s"], np.rad2deg(0.6) + 1e-6)
        self.assertEqual(metrics["constraint_violations"], 0)
        self.assertEqual(metrics["solver_failures"], 0)

    def test_friction_aware_dynamic_correction_gate(self):
        metrics, _, _ = run_dynamic_trial(
            self.dynamic_scenarios["friction_aware_low_mu_12"], self.dynamic_project,
        )
        self.assertLess(metrics["selected_target_speed_mps"], 12.0)
        self.assertLessEqual(
            metrics["reference_peak_combined_accel_mps2"],
            metrics["friction_accel_budget_mps2"],
        )
        self.assertLessEqual(metrics["max_tire_friction_utilization"], 1.0 + 1e-9)
        self.assertLessEqual(metrics["lateral_rmse_m"], 0.30)
        self.assertEqual(metrics["constraint_violations"], 0)
        self.assertEqual(metrics["solver_failures"], 0)


if __name__ == "__main__":
    unittest.main()
