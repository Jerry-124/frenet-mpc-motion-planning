import unittest

import numpy as np

from config import VehicleConfig
from models import DynamicBicycle, DynamicVehicleConfig, DynamicVehicleState


class DynamicBicyclePhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = VehicleConfig()

    def test_zero_input_straight_motion_is_force_free(self):
        plant = DynamicBicycle(DynamicVehicleConfig(), self.limits, 0.1)
        derivative = plant._derivative(
            DynamicVehicleState(0.0, 0.0, 0.0, 10.0).as_array(),
            np.array([0.0, 0.0]),
        )
        np.testing.assert_allclose(
            derivative,
            np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            atol=1e-12,
        )

    def test_body_frame_longitudinal_equation_retains_vy_yaw_rate_coupling(self):
        config = DynamicVehicleConfig(
            cornering_stiffness_front=1e-9,
            cornering_stiffness_rear=1e-9,
        )
        plant = DynamicBicycle(config, self.limits, 0.1)
        state = DynamicVehicleState(
            0.0,
            0.0,
            0.0,
            vx=10.0,
            vy=1.0,
            yaw_rate=0.2,
        )
        derivative = plant._derivative(state.as_array(), np.array([0.0, 0.0]))
        self.assertAlmostEqual(derivative[3], state.vy * state.yaw_rate, places=10)

    def test_front_tire_force_is_rotated_into_vehicle_body_frame(self):
        plant = DynamicBicycle(DynamicVehicleConfig(), self.limits, 0.1)
        state = DynamicVehicleState(0.0, 0.0, 0.0, 10.0)
        derivative = plant._derivative(state.as_array(), np.array([0.0, 0.1]))

        self.assertLess(derivative[3], 0.0)
        self.assertGreater(derivative[4], 0.0)
        self.assertGreater(derivative[5], 0.0)

    def test_left_and_right_steering_are_dynamically_symmetric(self):
        plant = DynamicBicycle(DynamicVehicleConfig(), self.limits, 0.1)
        state = DynamicVehicleState(0.0, 0.0, 0.0, 10.0).as_array()
        left = plant._derivative(state, np.array([0.0, 0.1]))
        right = plant._derivative(state, np.array([0.0, -0.1]))

        self.assertAlmostEqual(left[3], right[3], places=12)
        self.assertAlmostEqual(left[4], -right[4], places=12)
        self.assertAlmostEqual(left[5], -right[5], places=12)

    def test_zero_steer_longitudinal_acceleration_matches_requested_acceleration(self):
        plant = DynamicBicycle(DynamicVehicleConfig(), self.limits, 0.1)
        derivative = plant._derivative(
            DynamicVehicleState(0.0, 0.0, 0.0, 10.0).as_array(),
            np.array([1.5, 0.0]),
        )
        self.assertAlmostEqual(derivative[3], 1.5, places=12)
        self.assertAlmostEqual(derivative[4], 0.0, places=12)
        self.assertAlmostEqual(derivative[5], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
