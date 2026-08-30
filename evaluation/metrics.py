import csv
from pathlib import Path

import numpy as np


def calculate_metrics(
    states: np.ndarray,
    reference: np.ndarray,
    controls: np.ndarray,
    solver_failures: int,
    limits,
    dt: float | None = None,
    max_steer_rate: float | None = None,
) -> dict:
    error_xy = states[:, :2] - reference[:, :2]
    position_error = np.linalg.norm(error_xy, axis=1)
    tangent = np.column_stack((np.cos(reference[:, 2]), np.sin(reference[:, 2])))
    normal = np.column_stack((-np.sin(reference[:, 2]), np.cos(reference[:, 2])))
    longitudinal_error = np.sum(error_xy * tangent, axis=1)
    lateral_error = np.sum(error_xy * normal, axis=1)
    yaw_error = np.arctan2(np.sin(states[:, 2] - reference[:, 2]), np.cos(states[:, 2] - reference[:, 2]))
    steering_violations = int(np.sum(np.abs(controls[:, 1]) > limits.max_steer + 1e-9))
    acceleration_violations = int(np.sum(controls[:, 0] > limits.max_accel + 1e-9))
    acceleration_violations += int(np.sum(controls[:, 0] < limits.min_accel - 1e-9))
    speed_violations = int(np.sum(states[:, 3] > limits.max_speed + 1e-9))
    speed_violations += int(np.sum(states[:, 3] < limits.min_speed - 1e-9))
    steering_rate_violations = 0
    max_actual_steer_rate = float("nan")
    if dt is not None and len(controls):
        steering_rates = np.abs(np.diff(np.r_[0.0, controls[:, 1]])) / dt
        max_actual_steer_rate = float(np.max(steering_rates))
        if max_steer_rate is not None:
            steering_rate_violations = int(np.sum(steering_rates > max_steer_rate + 1e-9))
    violations = acceleration_violations + steering_violations + speed_violations + steering_rate_violations
    return {
        "lateral_rmse_m": float(np.sqrt(np.mean(lateral_error**2))),
        "longitudinal_rmse_m": float(np.sqrt(np.mean(longitudinal_error**2))),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "max_position_error_m": float(np.max(position_error)),
        "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_error**2)))),
        "speed_rmse_mps": float(np.sqrt(np.mean((states[:, 3] - reference[:, 3]) ** 2))),
        "mean_abs_steering_deg": float(np.rad2deg(np.mean(np.abs(controls[:, 1])))),
        "max_steering_rate_deg_s": float(np.rad2deg(max_actual_steer_rate)),
        "acceleration_limit_violations": acceleration_violations,
        "steering_limit_violations": steering_violations,
        "steering_rate_violations": steering_rate_violations,
        "speed_state_violations": speed_violations,
        "constraint_violations": violations,
        "solver_failures": int(solver_failures),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
