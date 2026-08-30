from dataclasses import dataclass

import numpy as np

from config import VehicleConfig


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v], dtype=float)


class KinematicBicycle:
    """Rear-axle kinematic bicycle model with hard input/state clipping."""

    def __init__(self, config: VehicleConfig, dt: float):
        self.config = config
        self.dt = dt

    def step_array_unclipped(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Prediction step without state clipping, for explicit MPC constraints."""
        x, y, yaw, speed = state
        accel = np.clip(control[0], self.config.min_accel, self.config.max_accel)
        steer = np.clip(control[1], -self.config.max_steer, self.config.max_steer)
        next_state = np.array(
            [
                x + speed * np.cos(yaw) * self.dt,
                y + speed * np.sin(yaw) * self.dt,
                yaw + speed / self.config.wheelbase * np.tan(steer) * self.dt,
                speed + accel * self.dt,
            ]
        )
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        return next_state

    def step_array(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        next_state = self.step_array_unclipped(state, control)
        next_state[3] = np.clip(next_state[3], self.config.min_speed, self.config.max_speed)
        return next_state

    def step(self, state: VehicleState, accel: float, steer: float) -> VehicleState:
        values = self.step_array(state.as_array(), np.array([accel, steer]))
        return VehicleState(*values)
