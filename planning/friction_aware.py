from dataclasses import dataclass
import numpy as np

from .frenet import FrenetTrajectory, generate_frenet_trajectory
from .reference_path import ReferencePath


@dataclass(frozen=True)
class FrictionPlanCostWeights:
    speed_error: float = 2.0
    lane_change_duration: float = 0.08
    lane_change_delay: float = 0.15
    friction_utilization: float = 0.1


def friction_plan_cost_weights_from_mapping(values: dict | None) -> FrictionPlanCostWeights:
    return FrictionPlanCostWeights(**(values or {}))


@dataclass
class FrictionAwarePlan:
    trajectory: FrenetTrajectory
    target_speed: float
    lane_change_duration: float
    lane_change_start_time: float
    peak_lateral_acceleration: float
    lateral_acceleration_budget: float
    evaluated_candidates: int
    feasible_candidates: int
    peak_longitudinal_acceleration: float
    peak_combined_acceleration: float
    speed_transition_duration: float

    @property
    def friction_acceleration_budget(self) -> float:
        return self.lateral_acceleration_budget


def trajectory_peak_lateral_acceleration(trajectory: FrenetTrajectory) -> float:
    _, lateral, _ = trajectory_acceleration_components(trajectory)
    return float(np.max(np.abs(lateral)))


def trajectory_acceleration_components(
    trajectory: FrenetTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interior longitudinal, lateral, and combined Cartesian acceleration."""
    dt = float(trajectory.time[1] - trajectory.time[0])
    vx = np.gradient(trajectory.x, dt)
    vy = np.gradient(trajectory.y, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    longitudinal = ax * np.cos(trajectory.yaw) + ay * np.sin(trajectory.yaw)
    lateral = -ax * np.sin(trajectory.yaw) + ay * np.cos(trajectory.yaw)
    combined = np.hypot(ax, ay)
    interior = slice(2, -2) if len(lateral) > 4 else slice(None)
    return longitudinal[interior], lateral[interior], combined[interior]


def select_friction_aware_trajectory(
    road: ReferencePath,
    duration: float,
    dt: float,
    current_speed: float,
    desired_speed: float,
    target_d: float,
    friction_coefficient: float,
    target_speeds: tuple[float, ...],
    lane_change_durations: tuple[float, ...],
    lane_change_start_times: tuple[float, ...] = (0.0,),
    safety_factor: float = 0.7,
    gravity: float = 9.81,
    min_longitudinal_acceleration: float = -3.0,
    max_longitudinal_acceleration: float = 2.0,
    speed_transition_durations: tuple[float, ...] | None = None,
    cost_weights: FrictionPlanCostWeights | None = None,
) -> FrictionAwarePlan:
    weights = cost_weights or FrictionPlanCostWeights()
    budget = safety_factor * friction_coefficient * gravity
    feasible = []
    evaluated = 0
    for target_speed in target_speeds:
        for change_duration in lane_change_durations:
            for start_time in lane_change_start_times:
                transition_options = speed_transition_durations or (
                    max(start_time, 1.0) if start_time > 0.0 else change_duration,
                )
                for speed_transition_duration in transition_options:
                    evaluated += 1
                    trajectory = generate_frenet_trajectory(
                        road, duration, dt, 0.0, 0.0, current_speed, target_speed,
                        target_d, change_duration, start_time, speed_transition_duration,
                    )
                    longitudinal, lateral, combined = trajectory_acceleration_components(trajectory)
                    peak_lateral = float(np.max(np.abs(lateral)))
                    peak_longitudinal = float(np.max(np.abs(longitudinal)))
                    peak_combined = float(np.max(combined))
                    longitudinal_feasible = bool(
                        np.min(longitudinal) >= min_longitudinal_acceleration - 1e-9
                        and np.max(longitudinal) <= max_longitudinal_acceleration + 1e-9
                    )
                    friction_feasible = peak_combined <= budget + 1e-9
                    if longitudinal_feasible and friction_feasible:
                        cost = (
                            weights.speed_error * abs(desired_speed - target_speed)
                            + weights.lane_change_duration * change_duration
                            + weights.lane_change_delay * start_time
                            + weights.friction_utilization * peak_combined / budget
                        )
                        feasible.append((
                            cost, trajectory, target_speed, change_duration, start_time,
                            peak_lateral, peak_longitudinal, peak_combined,
                            speed_transition_duration,
                        ))
    if not feasible:
        raise RuntimeError("No trajectory satisfies longitudinal acceleration and friction-circle limits")
    (
        _, trajectory, target_speed, change_duration, start_time,
        peak_lateral, peak_longitudinal, peak_combined, speed_transition_duration,
    ) = min(feasible, key=lambda item: item[0])
    return FrictionAwarePlan(
        trajectory, target_speed, change_duration, start_time, peak_lateral, budget,
        evaluated, len(feasible), peak_longitudinal, peak_combined, speed_transition_duration,
    )
