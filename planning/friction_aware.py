from dataclasses import dataclass
from itertools import product

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


@dataclass(frozen=True)
class _CandidateEvaluation:
    cost: float
    trajectory: FrenetTrajectory
    target_speed: float
    lane_change_duration: float
    lane_change_start_time: float
    peak_lateral_acceleration: float
    peak_longitudinal_acceleration: float
    peak_combined_acceleration: float
    speed_transition_duration: float


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


def _iter_candidate_settings(
    target_speeds: tuple[float, ...],
    lane_change_durations: tuple[float, ...],
    lane_change_start_times: tuple[float, ...],
    speed_transition_durations: tuple[float, ...] | None,
):
    for target_speed, change_duration, start_time in product(
        target_speeds,
        lane_change_durations,
        lane_change_start_times,
    ):
        transition_options = speed_transition_durations or (
            max(start_time, 1.0) if start_time > 0.0 else change_duration,
        )
        for speed_transition_duration in transition_options:
            yield target_speed, change_duration, start_time, speed_transition_duration


def _acceleration_peaks(
    trajectory: FrenetTrajectory,
) -> tuple[np.ndarray, float, float, float]:
    longitudinal, lateral, combined = trajectory_acceleration_components(trajectory)
    return (
        longitudinal,
        float(np.max(np.abs(lateral))),
        float(np.max(np.abs(longitudinal))),
        float(np.max(combined)),
    )


def _longitudinal_feasible(
    longitudinal: np.ndarray,
    minimum: float,
    maximum: float,
) -> bool:
    return bool(
        np.min(longitudinal) >= minimum - 1e-9
        and np.max(longitudinal) <= maximum + 1e-9
    )


def _candidate_cost(
    *,
    desired_speed: float,
    target_speed: float,
    change_duration: float,
    start_time: float,
    peak_combined: float,
    budget: float,
    weights: FrictionPlanCostWeights,
) -> float:
    return float(
        weights.speed_error * abs(desired_speed - target_speed)
        + weights.lane_change_duration * change_duration
        + weights.lane_change_delay * start_time
        + weights.friction_utilization * peak_combined / budget
    )


def _evaluate_candidate(
    *,
    road: ReferencePath,
    duration: float,
    dt: float,
    current_speed: float,
    desired_speed: float,
    target_d: float,
    target_speed: float,
    change_duration: float,
    start_time: float,
    speed_transition_duration: float,
    budget: float,
    min_longitudinal_acceleration: float,
    max_longitudinal_acceleration: float,
    weights: FrictionPlanCostWeights,
) -> _CandidateEvaluation | None:
    trajectory = generate_frenet_trajectory(
        road,
        duration,
        dt,
        0.0,
        0.0,
        current_speed,
        target_speed,
        target_d,
        change_duration,
        start_time,
        speed_transition_duration,
    )
    longitudinal, peak_lateral, peak_longitudinal, peak_combined = (
        _acceleration_peaks(trajectory)
    )
    if not _longitudinal_feasible(
        longitudinal,
        min_longitudinal_acceleration,
        max_longitudinal_acceleration,
    ) or peak_combined > budget + 1e-9:
        return None
    return _CandidateEvaluation(
        cost=_candidate_cost(
            desired_speed=desired_speed,
            target_speed=target_speed,
            change_duration=change_duration,
            start_time=start_time,
            peak_combined=peak_combined,
            budget=budget,
            weights=weights,
        ),
        trajectory=trajectory,
        target_speed=target_speed,
        lane_change_duration=change_duration,
        lane_change_start_time=start_time,
        peak_lateral_acceleration=peak_lateral,
        peak_longitudinal_acceleration=peak_longitudinal,
        peak_combined_acceleration=peak_combined,
        speed_transition_duration=speed_transition_duration,
    )


def _to_plan(
    best: _CandidateEvaluation,
    *,
    budget: float,
    evaluated: int,
    feasible_count: int,
) -> FrictionAwarePlan:
    return FrictionAwarePlan(
        trajectory=best.trajectory,
        target_speed=best.target_speed,
        lane_change_duration=best.lane_change_duration,
        lane_change_start_time=best.lane_change_start_time,
        peak_lateral_acceleration=best.peak_lateral_acceleration,
        lateral_acceleration_budget=budget,
        evaluated_candidates=evaluated,
        feasible_candidates=feasible_count,
        peak_longitudinal_acceleration=best.peak_longitudinal_acceleration,
        peak_combined_acceleration=best.peak_combined_acceleration,
        speed_transition_duration=best.speed_transition_duration,
    )


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
    settings = list(
        _iter_candidate_settings(
            target_speeds,
            lane_change_durations,
            lane_change_start_times,
            speed_transition_durations,
        )
    )
    feasible = [
        candidate
        for target_speed, change_duration, start_time, transition_duration in settings
        if (
            candidate := _evaluate_candidate(
                road=road,
                duration=duration,
                dt=dt,
                current_speed=current_speed,
                desired_speed=desired_speed,
                target_d=target_d,
                target_speed=target_speed,
                change_duration=change_duration,
                start_time=start_time,
                speed_transition_duration=transition_duration,
                budget=budget,
                min_longitudinal_acceleration=min_longitudinal_acceleration,
                max_longitudinal_acceleration=max_longitudinal_acceleration,
                weights=weights,
            )
        )
        is not None
    ]
    if not feasible:
        raise RuntimeError(
            "No trajectory satisfies longitudinal acceleration and friction-circle limits"
        )
    best = min(feasible, key=lambda item: item.cost)
    return _to_plan(
        best,
        budget=budget,
        evaluated=len(settings),
        feasible_count=len(feasible),
    )
