import numpy as np

from config import VehicleConfig


class StanleyPIDController:
    """Stanley lateral tracking plus PID longitudinal speed control."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        dt: float,
        stanley_gain: float = 1.4,
        softening_speed: float = 1.0,
        kp: float = 1.0,
        ki: float = 0.08,
        kd: float = 0.05,
    ):
        self.vehicle = vehicle
        self.dt = dt
        self.stanley_gain = stanley_gain
        self.softening_speed = softening_speed
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_error = 0.0
        self.previous_speed_error = 0.0
        self.last_success = True

    @staticmethod
    def _wrap(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def control(self, state: np.ndarray, references: np.ndarray) -> np.ndarray:
        target = references[0]
        heading_error = self._wrap(target[2] - state[2])
        normal = np.array([-np.sin(target[2]), np.cos(target[2])])
        cross_track_error = float(np.dot(target[:2] - state[:2], normal))
        correction = np.arctan2(
            self.stanley_gain * cross_track_error,
            max(state[3], 0.0) + self.softening_speed,
        )
        steer = np.clip(
            heading_error + correction,
            -self.vehicle.max_steer,
            self.vehicle.max_steer,
        )

        speed_error = float(target[3] - state[3])
        self.integral_error = np.clip(self.integral_error + speed_error * self.dt, -5.0, 5.0)
        derivative = (speed_error - self.previous_speed_error) / self.dt
        self.previous_speed_error = speed_error
        accel = np.clip(
            self.kp * speed_error + self.ki * self.integral_error + self.kd * derivative,
            self.vehicle.min_accel,
            self.vehicle.max_accel,
        )
        return np.array([accel, steer], dtype=float)

