from dataclasses import dataclass

import numpy as np

from config import VehicleConfig


@dataclass(frozen=True)
class DynamicVehicleConfig:
    mass: float = 1500.0
    yaw_inertia: float = 2500.0
    lf: float = 1.2
    lr: float = 1.6
    cornering_stiffness_front: float = 80000.0
    cornering_stiffness_rear: float = 80000.0
    friction_coefficient: float = 1.0
    gravity: float = 9.81
    integration_substeps: int = 10
    max_steer_rate_rad_s: float | None = None


@dataclass
class DynamicVehicleState:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float = 0.0
    yaw_rate: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.vx, self.vy, self.yaw_rate], dtype=float)

    def controller_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.vx], dtype=float)


class DynamicBicycle:
    """Six-state single-track plant with smooth nonlinear tires and friction circles."""

    def __init__(self, config: DynamicVehicleConfig, limits: VehicleConfig, dt: float):
        self.config = config
        self.limits = limits
        self.dt = dt
        self.actual_steer = 0.0
        self.max_tire_friction_utilization = 0.0

    def _derivative(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        _x, _y, yaw, vx, vy, yaw_rate = state
        requested_accel = float(np.clip(control[0], self.limits.min_accel, self.limits.max_accel))
        steer = float(np.clip(control[1], -self.limits.max_steer, self.limits.max_steer))
        safe_vx = max(abs(vx), 0.5)
        alpha_front = steer - np.arctan2(vy + self.config.lf * yaw_rate, safe_vx)
        alpha_rear = -np.arctan2(vy - self.config.lr * yaw_rate, safe_vx)

        wheelbase = self.config.lf + self.config.lr
        front_load = self.config.mass * self.config.gravity * self.config.lr / wheelbase
        rear_load = self.config.mass * self.config.gravity * self.config.lf / wheelbase
        total_longitudinal_force = np.clip(
            self.config.mass * requested_accel,
            -self.config.friction_coefficient * self.config.mass * self.config.gravity,
            self.config.friction_coefficient * self.config.mass * self.config.gravity,
        )
        front_longitudinal = total_longitudinal_force * self.config.lr / wheelbase
        rear_longitudinal = total_longitudinal_force * self.config.lf / wheelbase

        front_circle = self.config.friction_coefficient * front_load
        rear_circle = self.config.friction_coefficient * rear_load
        front_lateral_limit = np.sqrt(max(front_circle**2 - front_longitudinal**2, 0.0))
        rear_lateral_limit = np.sqrt(max(rear_circle**2 - rear_longitudinal**2, 0.0))
        front_force = self._smooth_tire_force(
            alpha_front, self.config.cornering_stiffness_front, front_lateral_limit,
        )
        rear_force = self._smooth_tire_force(
            alpha_rear, self.config.cornering_stiffness_rear, rear_lateral_limit,
        )
        front_utilization = np.hypot(front_longitudinal, front_force) / max(front_circle, 1e-9)
        rear_utilization = np.hypot(rear_longitudinal, rear_force) / max(rear_circle, 1e-9)
        self.max_tire_friction_utilization = max(
            self.max_tire_friction_utilization,
            float(front_utilization), float(rear_utilization),
        )
        accel = total_longitudinal_force / self.config.mass

        x_dot = vx * np.cos(yaw) - vy * np.sin(yaw)
        y_dot = vx * np.sin(yaw) + vy * np.cos(yaw)
        yaw_dot = yaw_rate
        vx_dot = accel
        vy_dot = (front_force + rear_force) / self.config.mass - vx * yaw_rate
        yaw_rate_dot = (self.config.lf * front_force - self.config.lr * rear_force) / self.config.yaw_inertia
        return np.array([x_dot, y_dot, yaw_dot, vx_dot, vy_dot, yaw_rate_dot])

    @staticmethod
    def _smooth_tire_force(slip_angle: float, cornering_stiffness: float, limit: float) -> float:
        if limit <= 1e-9:
            return 0.0
        return float(limit * np.tanh(cornering_stiffness * slip_angle / limit))

    def step(self, state: DynamicVehicleState, accel: float, steer: float) -> DynamicVehicleState:
        values = state.as_array()
        desired_steer = float(np.clip(steer, -self.limits.max_steer, self.limits.max_steer))
        if self.config.max_steer_rate_rad_s is None:
            self.actual_steer = desired_steer
        else:
            steer_change = np.clip(
                desired_steer - self.actual_steer,
                -self.config.max_steer_rate_rad_s * self.dt,
                self.config.max_steer_rate_rad_s * self.dt,
            )
            self.actual_steer += steer_change
        control = np.array([accel, self.actual_steer], dtype=float)
        sub_dt = self.dt / self.config.integration_substeps
        for _ in range(self.config.integration_substeps):
            k1 = self._derivative(values, control)
            k2 = self._derivative(values + 0.5 * sub_dt * k1, control)
            k3 = self._derivative(values + 0.5 * sub_dt * k2, control)
            k4 = self._derivative(values + sub_dt * k3, control)
            values = values + sub_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            values[3] = np.clip(values[3], self.limits.min_speed, self.limits.max_speed)
            values[2] = np.arctan2(np.sin(values[2]), np.cos(values[2]))
        return DynamicVehicleState(*values)
