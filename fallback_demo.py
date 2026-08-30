import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import load_project_config
from evaluation import save_metrics
from planning import (
    FrenetObstacle,
    ReferencePath,
    candidate_cost_weights_from_mapping,
    generate_and_score_candidates,
    select_with_emergency_fallback,
)
from simulation import track_reference


def _actual_minimum_clearance(road, states, obstacles, dt):
    frenet = np.asarray([road.cartesian_to_frenet(state[0], state[1]) for state in states])
    times = np.arange(len(states)) * dt
    minimum = float("inf")
    for obstacle in obstacles:
        obstacle_s = obstacle.s + obstacle.speed * times
        normalized = np.hypot(
            (frenet[:, 0] - obstacle_s) / obstacle.longitudinal_clearance,
            (frenet[:, 1] - obstacle.d) / obstacle.lateral_clearance,
        )
        minimum = min(minimum, float(np.min(normalized)))
    return minimum, frenet


def run_fallback_demo(output_dir: Path = Path("results"), config_path: Path = Path("configs/fallback_demo.json")) -> dict:
    project = load_project_config(config_path)
    sim_cfg, mpc_cfg, vehicle_cfg, settings = project.simulation, project.mpc, project.vehicle, project.raw["planner"]
    road = ReferencePath.sinusoidal(
        length=project.road.length_m, amplitude=project.road.amplitude_m,
        wavelength=project.road.wavelength_m,
    )
    obstacles = [FrenetObstacle(
        s=item["s_m"], d=item["d_m"], speed=item.get("speed_mps", 0.0),
        longitudinal_clearance=item.get("longitudinal_clearance_m", 5.0),
        lateral_clearance=item.get("lateral_clearance_m", 1.25),
    ) for item in settings["obstacles"]]
    planning_duration = sim_cfg.duration + mpc_cfg.horizon * sim_cfg.dt
    candidates = generate_and_score_candidates(
        road=road,
        duration=planning_duration,
        dt=sim_cfg.dt,
        target_offsets=tuple(settings["target_offsets_m"]),
        lane_change_durations=tuple(settings["lane_change_durations_s"]),
        target_speeds=tuple(settings["target_speeds_mps"]),
        obstacles=obstacles,
        desired_speed=sim_cfg.target_speed,
        current_speed=sim_cfg.target_speed,
        cost_weights=candidate_cost_weights_from_mapping(settings.get("cost_weights")),
        min_normalized_clearance=settings.get("min_normalized_clearance", 1.25),
        road_bounds=tuple(settings.get("road_bounds_m", (-1.75, 5.25))),
    )
    decision = select_with_emergency_fallback(
        candidates=candidates,
        road=road,
        duration=planning_duration,
        dt=sim_cfg.dt,
        s0=0.0,
        d0=0.0,
        current_speed=sim_cfg.target_speed,
        obstacles=obstacles,
        deceleration=vehicle_cfg.min_accel,
        min_normalized_clearance=settings.get("min_normalized_clearance", 1.25),
    )
    if decision.mode != "emergency_fallback":
        raise RuntimeError("The blocked-road validation scenario did not trigger fallback")

    tracking = track_reference(
        "emergency_brake", road, decision.trajectory.states, sim_cfg,
        initial_lateral_offset=project.road.initial_lateral_offset_m,
        vehicle_cfg=project.vehicle, mpc_cfg=project.mpc,
    )
    actual_clearance, actual_frenet = _actual_minimum_clearance(road, tracking.states, obstacles, sim_cfg.dt)
    emergency = decision.emergency_plan
    summary = {
        "mode": decision.mode,
        "transition_log": decision.transition_log,
        "candidate_count": len(candidates),
        "feasible_candidate_count": sum(candidate.feasible for candidate in candidates),
        "rejected_candidate_count": sum(not candidate.feasible for candidate in candidates),
        "reference_stop_time_s": emergency.stop_time,
        "analytical_continuous_stop_distance_m": emergency.analytical_stop_distance,
        "reference_stop_distance_m": emergency.stop_distance,
        "reference_min_normalized_clearance": emergency.min_normalized_clearance,
        "reference_collision_avoidable": emergency.collision_avoidable,
        "actual_travel_distance_m": float(actual_frenet[-1, 0] - actual_frenet[0, 0]),
        "actual_stop_distance_error_m": float(
            actual_frenet[-1, 0] - actual_frenet[0, 0] - emergency.stop_distance
        ),
        "actual_final_speed_mps": float(tracking.states[-1, 3]),
        "actual_min_normalized_clearance": actual_clearance,
        "tracking_lateral_rmse_m": tracking.metrics["lateral_rmse_m"],
        "tracking_constraint_violations": tracking.metrics["constraint_violations"],
        "tracking_solver_failures": tracking.metrics["solver_failures"],
        "minimum_commanded_accel_mps2": float(np.min(tracking.controls[:, 0])),
    }
    save_metrics(summary, output_dir / "metrics" / "emergency_fallback.csv")
    _save_plot(
        output_dir / "figures" / "emergency_fallback.png",
        road, candidates, decision.trajectory, obstacles, tracking.states, sim_cfg.dt,
    )
    print(json.dumps(summary, indent=2))
    return summary


def _save_plot(path, road, candidates, emergency, obstacles, states, dt):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.plot(road.x, road.y, "--", color="0.7", label="road centerline")
    for candidate in candidates:
        ax.plot(candidate.trajectory.x, candidate.trajectory.y, color="tab:red", alpha=0.18, linewidth=1.0)
    ax.plot([], [], color="tab:red", alpha=0.55, label="rejected normal candidates")
    ax.plot(emergency.x, emergency.y, color="tab:blue", linewidth=3.0, label="emergency-stop reference")
    ax.plot(states[:, 0], states[:, 1], color="tab:orange", linewidth=2.0, label="safety-controller tracking")
    for index, obstacle in enumerate(obstacles):
        x, y = road.frenet_to_cartesian(np.array([obstacle.s]), np.array([obstacle.d]))
        ax.scatter(x, y, marker="X", s=170, color="black", zorder=5, label="blocked lanes" if index == 0 else None)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="No feasible candidate: emergency fallback")
    ax.set_xlim(-2.0, 38.0); ax.set_ylim(-4.0, 10.0); ax.set_aspect("equal", adjustable="box")
    ax.grid(True); ax.legend(loc="upper left", ncol=2)

    reference_time = emergency.time
    actual_time = np.arange(len(states)) * dt
    axes[1].plot(reference_time, emergency.speed, linewidth=2.5, label="emergency reference")
    axes[1].plot(actual_time, states[:, 3], linewidth=2.0, label="actual speed")
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set(xlabel="time [s]", ylabel="speed [m/s]", title="Maximum-deceleration stop profile")
    axes[1].grid(True); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured emergency-fallback validation")
    parser.add_argument("--config", type=Path, default=Path("configs/fallback_demo.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_fallback_demo(args.output, args.config)
