import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import SimulationConfig, VehicleConfig, load_project_config
from control import EmergencyBrakeController, StanleyPIDController
from models import (
    DynamicBicycle,
    DynamicVehicleConfig,
    DynamicVehicleState,
    KinematicBicycle,
    VehicleState,
)
from planning import (
    FrenetObstacle,
    ReferencePath,
    generate_and_score_candidates,
    generate_emergency_stop_trajectory,
    generate_lane_change,
    select_best_candidate,
    select_friction_aware_trajectory,
    select_with_emergency_fallback,
)
from replanning_demo import run_replanning_demo
from robustness_benchmark import DelayedActuator


class ModelAndPlannerTests(unittest.TestCase):
    def test_versioned_config_inheritance(self):
        default = load_project_config("configs/default.json")
        fallback = load_project_config("configs/fallback_demo.json")
        self.assertEqual(default.schema_version, 1)
        self.assertEqual(default.vehicle, fallback.vehicle)
        self.assertEqual(default.mpc, fallback.mpc)
        self.assertEqual(fallback.simulation.duration, 5.0)
        self.assertEqual(len(fallback.raw["planner"]["obstacles"]), 2)
        for name in (
            "benchmark.json", "planner_demo.json", "replanning_demo.json",
            "replanning_blocked_validation.json", "robustness_benchmark.json",
            "dynamic_model_benchmark.json", "sensitivity_benchmark.json",
        ):
            self.assertEqual(load_project_config(f"configs/{name}").schema_version, 1)

    def test_straight_vehicle_step(self):
        model = KinematicBicycle(VehicleConfig(), 0.1)
        next_state = model.step(VehicleState(0.0, 0.0, 0.0, 10.0), 0.0, 0.0)
        self.assertTrue(np.allclose(next_state.as_array(), [1.0, 0.0, 0.0, 10.0]))

    def test_dynamic_vehicle_straight_step(self):
        limits = VehicleConfig()
        model = DynamicBicycle(DynamicVehicleConfig(), limits, 0.1)
        next_state = model.step(DynamicVehicleState(0.0, 0.0, 0.0, 10.0), 0.0, 0.0)
        self.assertAlmostEqual(next_state.x, 1.0, places=6)
        self.assertAlmostEqual(next_state.y, 0.0, places=6)
        self.assertAlmostEqual(next_state.yaw_rate, 0.0, places=6)

    def test_lane_change_boundary_conditions(self):
        cfg = SimulationConfig()
        path = generate_lane_change(ReferencePath.sinusoidal(), cfg.duration, cfg.dt, cfg.target_speed, cfg.lane_width, cfg.lane_change_duration)
        self.assertTrue(np.isclose(path.d[0], 0.0))
        self.assertTrue(np.isclose(path.d[-1], cfg.lane_width))
        self.assertTrue(np.all(np.diff(path.s) > 0.0))

    def test_cartesian_frenet_round_trip(self):
        road = ReferencePath.sinusoidal()
        expected_s, expected_d = 35.0, 3.5
        x, y = road.frenet_to_cartesian(np.array([expected_s]), np.array([expected_d]))
        actual_s, actual_d = road.cartesian_to_frenet(float(x[0]), float(y[0]))
        self.assertAlmostEqual(actual_s, expected_s, delta=0.15)
        self.assertAlmostEqual(actual_d, expected_d, delta=0.05)

    def test_stanley_pid_respects_input_limits(self):
        limits = VehicleConfig()
        controller = StanleyPIDController(limits, 0.1)
        state = np.array([0.0, -20.0, -2.0, 0.0])
        references = np.array([[1.0, 20.0, 2.0, 30.0]])
        accel, steer = controller.control(state, references)
        self.assertLessEqual(accel, limits.max_accel)
        self.assertGreaterEqual(accel, limits.min_accel)
        self.assertLessEqual(abs(steer), limits.max_steer)

    def test_actuation_delay_fifo(self):
        actuator = DelayedActuator(delay_steps=2)
        first = np.array([1.0, 0.1])
        second = np.array([2.0, 0.2])
        self.assertTrue(np.allclose(actuator.apply(first), [0.0, 0.0]))
        self.assertTrue(np.allclose(actuator.apply(second), [0.0, 0.0]))
        self.assertTrue(np.allclose(actuator.pending_controls()[0], first))
        self.assertTrue(np.allclose(actuator.apply(np.array([3.0, 0.3])), first))

    def test_friction_aware_plan_respects_budget(self):
        project = load_project_config("configs/dynamic_model_benchmark.json")
        settings = project.raw["dynamic_benchmark"]["friction_planner"]
        road = ReferencePath.sinusoidal(amplitude=4.0)
        plan = select_friction_aware_trajectory(
            road, duration=8.8, dt=0.1, current_speed=12.0, desired_speed=12.0,
            target_d=3.5, friction_coefficient=0.3,
            target_speeds=tuple(settings["target_speeds_mps"]),
            lane_change_durations=tuple(settings["lane_change_durations_s"]),
            lane_change_start_times=tuple(settings["lane_change_start_times_s"]),
            speed_transition_durations=tuple(settings["speed_transition_durations_s"]),
            safety_factor=settings["safety_factor"],
            min_longitudinal_acceleration=project.vehicle.min_accel,
            max_longitudinal_acceleration=project.vehicle.max_accel,
        )
        self.assertLessEqual(plan.peak_combined_acceleration, plan.friction_acceleration_budget)
        self.assertLessEqual(plan.peak_longitudinal_acceleration, abs(project.vehicle.min_accel))
        self.assertLess(plan.target_speed, 12.0)

    def test_candidate_planner_avoids_blocked_lane(self):
        road = ReferencePath.sinusoidal(amplitude=0.0)
        candidates = generate_and_score_candidates(
            road, duration=6.0, dt=0.1,
            target_offsets=(0.0, 3.5), lane_change_durations=(3.0, 4.0),
            target_speeds=(10.0,), obstacles=[FrenetObstacle(s=30.0, d=0.0)],
            desired_speed=10.0,
        )
        selected = select_best_candidate(candidates)
        self.assertTrue(selected.feasible)
        self.assertEqual(selected.target_d, 3.5)
        self.assertGreater(selected.min_normalized_clearance, 1.0)
        self.assertTrue(all(not candidate.feasible for candidate in candidates if candidate.target_d == 0.0))

    def test_emergency_stop_profile(self):
        road = ReferencePath.sinusoidal(amplitude=0.0)
        plan = generate_emergency_stop_trajectory(
            road, duration=5.0, dt=0.1, s0=0.0, d0=0.0,
            current_speed=10.0, deceleration=-3.0,
        )
        self.assertTrue(np.all(np.diff(plan.trajectory.speed) <= 1e-12))
        self.assertTrue(np.all(plan.trajectory.speed >= 0.0))
        self.assertAlmostEqual(plan.trajectory.speed[-1], 0.0)
        self.assertAlmostEqual(plan.analytical_stop_distance, 10.0**2 / (2.0 * 3.0), delta=0.03)
        self.assertAlmostEqual(plan.stop_distance, 17.17, delta=0.03)

    def test_emergency_controller_overrides_longitudinal_tracking(self):
        limits = VehicleConfig()
        controller = EmergencyBrakeController(limits, 0.1)
        state = np.array([0.0, 0.0, 0.0, 10.0])
        references = np.array([[1.0, 0.0, 0.0, 10.0]])
        control = controller.control(state, references)
        self.assertEqual(control[0], limits.min_accel)

    def test_replanning_loop_integrates_emergency_fallback(self):
        with TemporaryDirectory() as directory:
            summary = run_replanning_demo(
                Path(directory), Path("configs/replanning_blocked_validation.json"), False,
            )
        self.assertGreater(summary["emergency_fallback_events"], 0)
        self.assertEqual(summary["final_speed_mps"], 0.0)
        self.assertGreater(summary["minimum_actual_normalized_clearance"], 1.25)

    def test_no_candidate_uses_emergency_fallback(self):
        road = ReferencePath.sinusoidal(amplitude=0.0)
        obstacles = [FrenetObstacle(s=27.0, d=0.0), FrenetObstacle(s=27.0, d=3.5)]
        candidates = generate_and_score_candidates(
            road, duration=5.8, dt=0.1,
            target_offsets=(0.0, 3.5), lane_change_durations=(3.0, 4.0),
            target_speeds=(6.0, 10.0), obstacles=obstacles, desired_speed=10.0,
            current_speed=10.0,
        )
        decision = select_with_emergency_fallback(
            candidates, road, duration=5.8, dt=0.1, s0=0.0, d0=0.0,
            current_speed=10.0, obstacles=obstacles,
        )
        self.assertFalse(any(candidate.feasible for candidate in candidates))
        self.assertEqual(decision.mode, "emergency_fallback")
        self.assertTrue(decision.emergency_plan.collision_avoidable)
        self.assertEqual(decision.transition_log, "normal_planning -> emergency_fallback")

    def test_emergency_stop_reports_unavoidable_collision(self):
        road = ReferencePath.sinusoidal(amplitude=0.0)
        plan = generate_emergency_stop_trajectory(
            road, duration=5.0, dt=0.1, s0=0.0, d0=0.0,
            current_speed=10.0, deceleration=-3.0,
            obstacles=[FrenetObstacle(s=10.0, d=0.0)],
        )
        self.assertFalse(plan.collision_avoidable)
        self.assertLessEqual(plan.min_normalized_clearance, 1.25)


if __name__ == "__main__":
    unittest.main()
