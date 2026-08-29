from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from config import load_project_config
from control import EmergencyBrakeController, NonlinearMPC
from models import KinematicBicycle, VehicleState
from planning import (
    FrenetObstacle,
    ReferencePath,
    candidate_cost_weights_from_mapping,
    generate_and_score_candidates,
    select_with_emergency_fallback,
)


def _decision_label(current_d: float, target_d: float, target_speed: float, desired_speed: float) -> str:
    if target_d > current_d + 0.7:
        return "lane_change_left"
    if target_d < current_d - 0.7:
        return "return_right"
    if target_speed < desired_speed - 1.0:
        return "slow_or_follow"
    return "keep_lane"


def run_replanning_demo(
    output_dir: Path = Path("results"),
    config_path: Path = Path("configs/replanning_demo.json"),
    print_summary: bool = True,
) -> dict:
    project = load_project_config(config_path)
    sim_cfg, vehicle_cfg, mpc_cfg, settings = project.simulation, project.vehicle, project.mpc, project.raw["replanning"]
    road = ReferencePath.sinusoidal(
        length=project.road.length_m, amplitude=project.road.amplitude_m,
        wavelength=project.road.wavelength_m,
    )
    model = KinematicBicycle(vehicle_cfg, sim_cfg.dt)
    controller = NonlinearMPC(model, vehicle_cfg, mpc_cfg)
    emergency_controller = EmergencyBrakeController(vehicle_cfg, sim_cfg.dt)
    obstacles = [FrenetObstacle(
        s=item["s_m"], d=item["d_m"], speed=item.get("speed_mps", 0.0),
        longitudinal_clearance=item.get("longitudinal_clearance_m", 5.0),
        lateral_clearance=item.get("lateral_clearance_m", 1.25),
    ) for item in settings["obstacles"]]
    x0, y0 = road.frenet_to_cartesian(np.array([0.0]), np.array([0.0]))
    _, _, yaw0 = road.sample(np.array([0.0]))
    state = VehicleState(float(x0[0]), float(y0[0]), float(yaw0[0]), sim_cfg.target_speed)

    replan_interval = float(settings["interval_s"])
    replan_steps = int(round(replan_interval / sim_cfg.dt))
    planning_duration = float(settings["planning_duration_s"])
    states, controls, decisions = [state.as_array()], [], []
    active_plan = None
    active_trajectory = None
    plan_step = 0
    solver_failures = 0
    emergency_fallback_events = 0
    emergency_mode = False
    committed_lane = None
    remaining_maneuver_time = 0.0

    for step in range(int(round(sim_cfg.duration / sim_cfg.dt))):
        current_time = step * sim_cfg.dt
        if step % replan_steps == 0:
            ego_s, ego_d = road.cartesian_to_frenet(state.x, state.y)
            if committed_lane is not None and abs(ego_d - committed_lane) < 0.25:
                committed_lane = None
                remaining_maneuver_time = 0.0
            if committed_lane is None:
                target_offsets = tuple(settings["target_offsets_m"])
                maneuver_durations = tuple(settings["lane_change_durations_s"])
                preferred_lane = 0.0
            else:
                target_offsets = (committed_lane,)
                maneuver_durations = (max(remaining_maneuver_time, 1.0),)
                preferred_lane = committed_lane
            candidates = generate_and_score_candidates(
                road=road,
                duration=planning_duration,
                dt=sim_cfg.dt,
                target_offsets=target_offsets,
                lane_change_durations=maneuver_durations,
                target_speeds=tuple(settings["target_speeds_mps"]),
                obstacles=obstacles,
                desired_speed=sim_cfg.target_speed,
                s0=ego_s,
                d0=ego_d,
                current_speed=state.v,
                prediction_start_time=current_time,
                preferred_lane_d=preferred_lane,
                cost_weights=candidate_cost_weights_from_mapping(settings.get("cost_weights")),
                min_normalized_clearance=settings.get("min_normalized_clearance", 1.25),
                road_bounds=tuple(settings.get("road_bounds_m", (-1.75, 5.25))),
            )
            planning_decision = select_with_emergency_fallback(
                candidates=candidates,
                road=road,
                duration=planning_duration,
                dt=sim_cfg.dt,
                s0=ego_s,
                d0=ego_d,
                current_speed=state.v,
                obstacles=obstacles,
                deceleration=vehicle_cfg.min_accel,
                prediction_start_time=current_time,
                min_normalized_clearance=settings.get("min_normalized_clearance", 1.25),
            )
            active_plan = planning_decision.selected_candidate
            active_trajectory = planning_decision.trajectory
            emergency_mode = planning_decision.mode == "emergency_fallback"
            emergency_fallback_events += int(emergency_mode)
            plan_step = 0
            if emergency_mode:
                committed_lane = None
                remaining_maneuver_time = 0.0
            elif committed_lane is None and abs(active_plan.target_d - ego_d) > 0.7:
                committed_lane = active_plan.target_d
                remaining_maneuver_time = active_plan.lane_change_duration
            if committed_lane is not None:
                remaining_maneuver_time = max(replan_interval, remaining_maneuver_time - replan_interval)
            emergency_plan = planning_decision.emergency_plan
            decisions.append({
                "time_s": current_time,
                "ego_s_m": ego_s,
                "ego_d_m": ego_d,
                "ego_speed_mps": state.v,
                "planner_mode": planning_decision.mode,
                "decision": "emergency_stop" if emergency_mode else _decision_label(
                    ego_d, active_plan.target_d, active_plan.target_speed, sim_cfg.target_speed,
                ),
                "target_d_m": ego_d if emergency_mode else active_plan.target_d,
                "target_speed_mps": 0.0 if emergency_mode else active_plan.target_speed,
                "lane_change_duration_s": 0.0 if emergency_mode else active_plan.lane_change_duration,
                "feasible_candidates": sum(candidate.feasible for candidate in candidates),
                "selected_cost": float("nan") if emergency_mode else active_plan.total_cost,
                "predicted_min_clearance": (
                    emergency_plan.min_normalized_clearance
                    if emergency_mode else active_plan.min_normalized_clearance
                ),
                "collision_avoidable": True if not emergency_mode else emergency_plan.collision_avoidable,
            })

        references = active_trajectory.states[plan_step + 1 : plan_step + 1 + mpc_cfg.horizon]
        active_controller = emergency_controller if emergency_mode else controller
        control = active_controller.control(state.as_array(), references)
        solver_failures += int(not active_controller.last_success)
        state = model.step(state, control[0], control[1])
        controls.append(control)
        states.append(state.as_array())
        plan_step += 1

    states, controls = np.asarray(states), np.asarray(controls)
    actual_clearance = _minimum_actual_clearance(states, road, obstacles, sim_cfg.dt)
    _save_decisions(output_dir / "metrics" / "replanning_decisions.csv", decisions)
    _save_replanning_plot(output_dir / "figures" / "receding_horizon_replanning.png", states, controls, decisions, road, obstacles, sim_cfg)
    constraint_violations = int(np.sum(np.abs(controls[:, 1]) > vehicle_cfg.max_steer + 1e-9))
    constraint_violations += int(np.sum(controls[:, 0] > vehicle_cfg.max_accel + 1e-9))
    constraint_violations += int(np.sum(controls[:, 0] < vehicle_cfg.min_accel - 1e-9))
    summary = {
        "replanning_events": len(decisions),
        "emergency_fallback_events": emergency_fallback_events,
        "decision_sequence": [row["decision"] for row in decisions],
        "minimum_actual_normalized_clearance": actual_clearance,
        "constraint_violations": constraint_violations,
        "solver_failures": solver_failures,
        "final_speed_mps": float(states[-1, 3]),
    }
    if print_summary:
        print(json.dumps(summary, indent=2))
    return summary


def _minimum_actual_clearance(states, road, obstacles, dt):
    minimum = np.inf
    for index, state in enumerate(states):
        ego_s, ego_d = road.cartesian_to_frenet(state[0], state[1])
        time = index * dt
        for obstacle in obstacles:
            distance = np.hypot(
                (ego_s - (obstacle.s + obstacle.speed * time)) / obstacle.longitudinal_clearance,
                (ego_d - obstacle.d) / obstacle.lateral_clearance,
            )
            minimum = min(minimum, float(distance))
    return minimum


def _save_decisions(path, decisions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=decisions[0].keys())
        writer.writeheader(); writer.writerows(decisions)


def _save_replanning_plot(path, states, controls, decisions, road, obstacles, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(len(states)) * config.dt
    ego_s_d = np.array([road.cartesian_to_frenet(x, y) for x, y in states[:, :2]])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(road.x, road.y, "--", color="0.65", label="road centerline")
    axes[0].plot(states[:, 0], states[:, 1], color="tab:blue", linewidth=2.5, label="ego trajectory")
    for index, obstacle in enumerate(obstacles):
        obstacle_s = obstacle.s + obstacle.speed * time
        ox, oy = road.frenet_to_cartesian(obstacle_s, np.full_like(time, obstacle.d))
        axes[0].plot(ox, oy, ":", linewidth=2, label=f"obstacle {index + 1} prediction")
        axes[0].scatter(ox[0], oy[0], marker="X", s=100, color="black")
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Closed-loop trajectory with repeated Frenet replanning")
    axes[0].set_xlim(float(np.min(states[:, 0])) - 3.0, float(np.max(states[:, 0])) + 15.0)
    axes[0].set_ylim(-5.0, 10.0)
    axes[0].grid(True); axes[0].legend(ncol=2); axes[0].set_aspect("equal", adjustable="box")

    axes[1].plot(time, states[:, 3], label="ego speed [m/s]", color="tab:blue")
    axes[1].plot(time, ego_s_d[:, 1], label="lateral offset d [m]", color="tab:orange")
    for row in decisions:
        axes[1].axvline(row["time_s"], color="0.75", linewidth=0.8)
        axes[1].scatter(row["time_s"], row["target_d_m"], color="tab:red", s=22)
    axes[1].set(xlabel="time [s]", title="State evolution and replanning instants")
    axes[1].grid(True); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured receding-horizon replanning demo")
    parser.add_argument("--config", type=Path, default=Path("configs/replanning_demo.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_replanning_demo(args.output, args.config)
