import numpy as np

from config import VehicleConfig
from .stanley_pid import StanleyPIDController


class EmergencyBrakeController:
    """Independent safety controller: maximum braking plus lane-holding steering."""

    def __init__(self, vehicle: VehicleConfig, dt: float):
        self.vehicle = vehicle
        self.lateral_controller = StanleyPIDController(vehicle, dt)
        self.last_success = True

    def control(self, state: np.ndarray, references: np.ndarray) -> np.ndarray:
        lateral_control = self.lateral_controller.control(state, references)
        acceleration = self.vehicle.min_accel if state[3] > self.vehicle.min_speed + 1e-9 else 0.0
        return np.array([acceleration, lateral_control[1]], dtype=float)
