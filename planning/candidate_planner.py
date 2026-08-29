from dataclasses import dataclass
import numpy as np

from .frenet import FrenetTrajectory, generate_frenet_trajectory
from .reference_path import ReferencePath

MIN_NORMALIZED_CLEARANCE = 1.25


@dataclass(frozen=True)
class CandidateCostWeights:
    lateral_jerk: float = 0.03
    lateral_acceleration: float = 0.08
    speed_error: float = 0.8
    lane_offset: float = 0.25
    obstacle_risk: float = 2.0


def candidate_cost_weights_from_mapping(values: dict | None) -> CandidateCostWeights:
    """Build planner weights from config while retaining documented defaults."""
    return CandidateCostWeights(**(values or {}))


@dataclass(frozen=True)
class FrenetObstacle:
    s: float
    d: float
    speed: float = 0.0
    longitudinal_clearance: float = 5.0
    lateral_clearance: float = 1.25


@dataclass
class ScoredTrajectory:
    trajectory: FrenetTrajectory
    target_d: float
    lane_change_duration: float
    target_speed: float
    feasible: bool
    total_cost: float
    comfort_cost: float
    efficiency_cost: float
    lane_cost: float
    risk_cost: float
    min_normalized_clearance: float
    rejection_reason: str = ""


def _score_candidate(
    trajectory: FrenetTrajectory,
    target_d: float,
    lane_change_duration: float,
    target_speed: float,
    obstacles: list[FrenetObstacle],
    desired_speed: float,
    preferred_lane_d: float,
    road_bounds: tuple[float, float],
    prediction_start_time: float,
    cost_weights: CandidateCostWeights,
    min_normalized_clearance: float,
) -> ScoredTrajectory:
    dt = float(trajectory.time[1] - trajectory.time[0])
    lateral_acceleration = np.gradient(np.gradient(trajectory.d, dt), dt)
    lateral_jerk = np.gradient(lateral_acceleration, dt)
    comfort_cost = float(
        cost_weights.lateral_jerk * np.sum(lateral_jerk**2) * dt
        + cost_weights.lateral_acceleration * np.sum(lateral_acceleration**2) * dt
    )
    efficiency_cost = float(cost_weights.speed_error * abs(target_speed - desired_speed))
    lane_cost = float(cost_weights.lane_offset * abs(target_d - preferred_lane_d))

    within_road = np.all((trajectory.d >= road_bounds[0]) & (trajectory.d <= road_bounds[1]))
    min_clearance = np.inf
    for obstacle in obstacles:
        obstacle_s = obstacle.s + obstacle.speed * (prediction_start_time + trajectory.time)
        normalized_distance = np.hypot(
            (trajectory.s - obstacle_s) / obstacle.longitudinal_clearance,
            (trajectory.d - obstacle.d) / obstacle.lateral_clearance,
        )
        min_clearance = min(min_clearance, float(np.min(normalized_distance)))

    collision_free = min_clearance > min_normalized_clearance
    risk_cost = 0.0 if not obstacles else float(cost_weights.obstacle_risk / max(min_clearance, 1e-6))
    feasible = bool(within_road and collision_free)
    reason = ""
    if not within_road:
        reason = "road_boundary"
    elif not collision_free:
        reason = "collision"
    total = comfort_cost + efficiency_cost + lane_cost + risk_cost if feasible else float("inf")
    return ScoredTrajectory(
        trajectory, target_d, lane_change_duration, target_speed, feasible, total,
        comfort_cost, efficiency_cost, lane_cost, risk_cost, min_clearance, reason,
    )


def generate_and_score_candidates(
    road: ReferencePath,
    duration: float,
    dt: float,
    target_offsets: tuple[float, ...],
    lane_change_durations: tuple[float, ...],
    target_speeds: tuple[float, ...],
    obstacles: list[FrenetObstacle],
    desired_speed: float,
    preferred_lane_d: float = 0.0,
    road_bounds: tuple[float, float] = (-1.75, 5.25),
    s0: float = 0.0,
    d0: float = 0.0,
    current_speed: float | None = None,
    prediction_start_time: float = 0.0,
    cost_weights: CandidateCostWeights | None = None,
    min_normalized_clearance: float = MIN_NORMALIZED_CLEARANCE,
) -> list[ScoredTrajectory]:
    weights = cost_weights or CandidateCostWeights()
    candidates = []
    for target_d in target_offsets:
        for change_duration in lane_change_durations:
            for speed in target_speeds:
                trajectory = generate_frenet_trajectory(
                    road, duration, dt, s0, d0,
                    speed if current_speed is None else current_speed,
                    speed, target_d, change_duration,
                )
                candidates.append(
                    _score_candidate(
                        trajectory, target_d, change_duration, speed, obstacles,
                        desired_speed, preferred_lane_d, road_bounds, prediction_start_time,
                        weights, min_normalized_clearance,
                    )
                )
    return candidates


def select_best_candidate(candidates: list[ScoredTrajectory]) -> ScoredTrajectory:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        raise RuntimeError("No collision-free Frenet trajectory satisfies the road boundaries")
    return min(feasible, key=lambda candidate: candidate.total_cost)
