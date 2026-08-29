from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from config import load_project_config
from planning import (
    FrenetObstacle,
    ReferencePath,
    candidate_cost_weights_from_mapping,
    generate_and_score_candidates,
    select_best_candidate,
)
from simulation import track_reference


def run_planner_demo(output_dir: Path = Path("results"), config_path: Path = Path("configs/planner_demo.json")):
    project = load_project_config(config_path)
    sim_cfg, mpc_cfg, settings = project.simulation, project.mpc, project.raw["planner"]
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
        cost_weights=candidate_cost_weights_from_mapping(settings.get("cost_weights")),
        min_normalized_clearance=settings.get("min_normalized_clearance", 1.25),
        road_bounds=tuple(settings.get("road_bounds_m", (-1.75, 5.25))),
    )
    selected = select_best_candidate(candidates)
    tracking = track_reference(
        "nmpc", road, selected.trajectory.states, sim_cfg,
        project.road.initial_lateral_offset_m, project.vehicle, project.mpc,
    )
    save_candidate_table(output_dir / "metrics" / "frenet_candidates.csv", candidates, selected)
    save_planning_plot(
        output_dir / "figures" / "frenet_candidate_selection.png", road, candidates,
        selected, obstacles, tracking.states,
        tuple(settings.get("road_bounds_m", (-1.75, 5.25))),
    )
    summary = {
        "candidate_count": len(candidates),
        "feasible_count": sum(candidate.feasible for candidate in candidates),
        "selected_target_d_m": selected.target_d,
        "selected_lane_change_duration_s": selected.lane_change_duration,
        "selected_target_speed_mps": selected.target_speed,
        "selected_total_cost": selected.total_cost,
        "selected_min_normalized_clearance": selected.min_normalized_clearance,
        "tracking_lateral_rmse_m": tracking.metrics["lateral_rmse_m"],
        "tracking_constraint_violations": tracking.metrics["constraint_violations"],
        "tracking_solver_failures": tracking.metrics["solver_failures"],
    }
    print(json.dumps(summary, indent=2))
    return summary


def save_candidate_table(path: Path, candidates, selected):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "selected", "target_d_m", "lane_change_duration_s", "target_speed_mps",
        "feasible", "total_cost", "comfort_cost", "efficiency_cost", "lane_cost",
        "risk_cost", "min_normalized_clearance", "rejection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                "selected": candidate is selected,
                "target_d_m": candidate.target_d,
                "lane_change_duration_s": candidate.lane_change_duration,
                "target_speed_mps": candidate.target_speed,
                "feasible": candidate.feasible,
                "total_cost": candidate.total_cost,
                "comfort_cost": candidate.comfort_cost,
                "efficiency_cost": candidate.efficiency_cost,
                "lane_cost": candidate.lane_cost,
                "risk_cost": candidate.risk_cost,
                "min_normalized_clearance": candidate.min_normalized_clearance,
                "rejection_reason": candidate.rejection_reason,
            })


def save_planning_plot(path: Path, road, candidates, selected, obstacles, tracked_states, road_bounds):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(road.x, road.y, "--", color="0.65", label="road centerline")
    for offset, label in ((road_bounds[0], "road boundaries"), (road_bounds[1], None)):
        bx, by = road.frenet_to_cartesian(road.s, np.full_like(road.s, offset))
        ax.plot(bx, by, ":", color="0.5", linewidth=1.0, label=label)
    for candidate in candidates:
        color = "tab:blue" if candidate.feasible else "tab:red"
        alpha = 0.16 if candidate is not selected else 1.0
        width = 0.8 if candidate is not selected else 3.0
        ax.plot(candidate.trajectory.x, candidate.trajectory.y, color=color, alpha=alpha, linewidth=width)
    ax.plot(selected.trajectory.x, selected.trajectory.y, color="tab:blue", linewidth=3.0, label="selected Frenet path")
    ax.plot(tracked_states[:, 0], tracked_states[:, 1], color="tab:orange", linewidth=2.0, label="NMPC tracking")
    for index, obstacle in enumerate(obstacles):
        x, y = road.frenet_to_cartesian(np.array([obstacle.s]), np.array([obstacle.d]))
        ax.scatter(x, y, s=160, marker="X", color="black", label="slow obstacle (initial)" if index == 0 else None, zorder=5)
        obstacle_s = obstacle.s + obstacle.speed * selected.trajectory.time
        ox, oy = road.frenet_to_cartesian(obstacle_s, np.full_like(obstacle_s, obstacle.d))
        ax.plot(ox, oy, "k:", linewidth=1.5, label="obstacle prediction" if index == 0 else None)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Frenet candidate generation, rejection, and selection")
    ax.set_xlim(-5.0, float(np.max(tracked_states[:, 0])) + 10.0)
    ax.set_ylim(-5.0, 12.0)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured Frenet candidate planning demo")
    parser.add_argument("--config", type=Path, default=Path("configs/planner_demo.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_planner_demo(args.output, args.config)
