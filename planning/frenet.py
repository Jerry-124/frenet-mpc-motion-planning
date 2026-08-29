from dataclasses import dataclass
import numpy as np

from .reference_path import ReferencePath


@dataclass
class FrenetTrajectory:
    time: np.ndarray
    s: np.ndarray
    d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray

    @property
    def states(self) -> np.ndarray:
        return np.column_stack((self.x, self.y, self.yaw, self.speed))


def _quintic_blend(tau: np.ndarray) -> np.ndarray:
    """Minimum-jerk position blend with zero end velocity/acceleration."""
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def generate_lane_change(
    road: ReferencePath,
    duration: float,
    dt: float,
    speed: float,
    lane_width: float,
    lane_change_duration: float,
) -> FrenetTrajectory:
    return generate_frenet_trajectory(
        road=road,
        duration=duration,
        dt=dt,
        s0=0.0,
        d0=0.0,
        current_speed=speed,
        target_speed=speed,
        target_d=lane_width,
        lane_change_duration=lane_change_duration,
    )


def generate_frenet_trajectory(
    road: ReferencePath,
    duration: float,
    dt: float,
    s0: float,
    d0: float,
    current_speed: float,
    target_speed: float,
    target_d: float,
    lane_change_duration: float,
    lane_change_start_time: float = 0.0,
    speed_transition_duration: float | None = None,
) -> FrenetTrajectory:
    """Generate a smooth trajectory from the current Frenet state."""
    time = np.arange(0.0, duration + 0.5 * dt, dt)
    speed_duration = lane_change_duration if speed_transition_duration is None else speed_transition_duration
    speed_tau = np.clip(time / max(speed_duration, dt), 0.0, 1.0)
    longitudinal_speed = current_speed + (target_speed - current_speed) * _quintic_blend(speed_tau)
    s = np.empty_like(time)
    s[0] = s0
    s[1:] = s0 + np.cumsum(0.5 * (longitudinal_speed[:-1] + longitudinal_speed[1:]) * dt)
    lateral_tau = np.clip((time - lane_change_start_time) / max(lane_change_duration, dt), 0.0, 1.0)
    d = d0 + (target_d - d0) * _quintic_blend(lateral_tau)
    x, y = road.frenet_to_cartesian(s, d)
    yaw = np.unwrap(np.arctan2(np.gradient(y, time), np.gradient(x, time)))
    trajectory_speed = np.hypot(np.gradient(x, time), np.gradient(y, time))
    return FrenetTrajectory(time, s, d, x, y, yaw, trajectory_speed)
