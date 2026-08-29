from dataclasses import dataclass
import numpy as np

from .candidate_planner import (
    MIN_NORMALIZED_CLEARANCE,
    FrenetObstacle,
    ScoredTrajectory,
    select_best_candidate,
)
from .frenet import FrenetTrajectory
from .reference_path import ReferencePath


@dataclass(frozen=True)
class EmergencyStopPlan:
    trajectory: FrenetTrajectory
    deceleration: float
    stop_time: float
    stop_distance: float
    analytical_stop_distance: float
    min_normalized_clearance: float
    collision_avoidable: bool


@dataclass(frozen=True)
class PlanningDecision:
    mode: str
    trajectory: FrenetTrajectory
    selected_candidate: ScoredTrajectory | None = None
    emergency_plan: EmergencyStopPlan | None = None
    transition_log: str = "normal_planning"


def minimum_obstacle_clearance(
    trajectory: FrenetTrajectory,
    obstacles: list[FrenetObstacle],
    prediction_start_time: float = 0.0,
) -> float:
    """Return the minimum distance in the planner's normalized safety ellipse."""
    if not obstacles:
        return float("inf")
    minimum = float("inf")
    for obstacle in obstacles:
        obstacle_s = obstacle.s + obstacle.speed * (prediction_start_time + trajectory.time)
        normalized = np.hypot(
            (trajectory.s - obstacle_s) / obstacle.longitudinal_clearance,
            (trajectory.d - obstacle.d) / obstacle.lateral_clearance,
        )
        minimum = min(minimum, float(np.min(normalized)))
    return minimum


def generate_emergency_stop_trajectory(
    road: ReferencePath,
    duration: float,
    dt: float,
    s0: float,
    d0: float,
    current_speed: float,
    deceleration: float = -3.0,
    obstacles: list[FrenetObstacle] | None = None,
    prediction_start_time: float = 0.0,
    min_normalized_clearance: float = MIN_NORMALIZED_CLEARANCE,
) -> EmergencyStopPlan:
    """Generate a maximum-deceleration, lane-holding emergency stop reference."""
    if deceleration >= 0.0:
        raise ValueError("Emergency deceleration must be negative")
    if current_speed < 0.0:
        raise ValueError("Current speed must be non-negative")

    time = np.arange(0.0, duration + 0.5 * dt, dt)
    speed = np.maximum(current_speed + deceleration * time, 0.0)
    s = np.empty_like(time)
    s[0] = s0
    # Match the plant's forward-Euler position update so the fallback preview
    # and closed-loop stopping distance use the same discrete-time model.
    s[1:] = s0 + np.cumsum(speed[:-1] * dt)
    d = np.full_like(time, d0)
    x, y = road.frenet_to_cartesian(s, d)
    _, _, yaw = road.sample(s)
    trajectory = FrenetTrajectory(time, s, d, x, y, yaw, speed)

    stop_time = current_speed / abs(deceleration) if current_speed > 0.0 else 0.0
    stop_distance = float(s[-1] - s0)
    analytical_stop_distance = current_speed**2 / (2.0 * abs(deceleration))
    clearance = minimum_obstacle_clearance(trajectory, obstacles or [], prediction_start_time)
    return EmergencyStopPlan(
        trajectory=trajectory,
        deceleration=deceleration,
        stop_time=stop_time,
        stop_distance=stop_distance,
        analytical_stop_distance=analytical_stop_distance,
        min_normalized_clearance=clearance,
        collision_avoidable=clearance > min_normalized_clearance,
    )


def select_with_emergency_fallback(
    candidates: list[ScoredTrajectory],
    road: ReferencePath,
    duration: float,
    dt: float,
    s0: float,
    d0: float,
    current_speed: float,
    obstacles: list[FrenetObstacle],
    deceleration: float = -3.0,
    prediction_start_time: float = 0.0,
    min_normalized_clearance: float = MIN_NORMALIZED_CLEARANCE,
) -> PlanningDecision:
    """Select a normal candidate, or explicitly transition to emergency braking."""
    try:
        selected = select_best_candidate(candidates)
        return PlanningDecision(
            mode="normal",
            trajectory=selected.trajectory,
            selected_candidate=selected,
        )
    except RuntimeError:
        emergency = generate_emergency_stop_trajectory(
            road, duration, dt, s0, d0, current_speed, deceleration,
            obstacles, prediction_start_time, min_normalized_clearance,
        )
        return PlanningDecision(
            mode="emergency_fallback",
            trajectory=emergency.trajectory,
            emergency_plan=emergency,
            transition_log="normal_planning -> emergency_fallback",
        )
