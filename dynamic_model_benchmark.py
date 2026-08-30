import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import load_project_config
from control import NonlinearMPC
from evaluation import calculate_metrics
from models import (
    DynamicBicycle,
    DynamicVehicleConfig,
    DynamicVehicleState,
    KinematicBicycle,
)
from planning import (
    ReferencePath,
    friction_plan_cost_weights_from_mapping,
    generate_lane_change,
    select_friction_aware_trajectory,
)
from simulation import run_closed_loop


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    speed_mps: float
    friction_coefficient: float
    plant: str = "dynamic"
    max_steer_rate_rad_s: float | None = None
    lane_change_duration_s: float = 4.0
    road_amplitude_m: float = 2.0
    rate_aware_controller: bool = False
    friction_aware_planner: bool = False


def _load_scenarios(items, project) -> tuple[DynamicScenario, ...]:
    return tuple(
        DynamicScenario(
            name=item["name"],
            speed_mps=item["speed_mps"],
            friction_coefficient=item["friction_coefficient"],
            plant=item.get("plant", "dynamic"),
            max_steer_rate_rad_s=item.get("max_steer_rate_rad_s"),
            lane_change_duration_s=item.get(
                "lane_change_duration_s", project.simulation.lane_change_duration
            ),
            road_amplitude_m=item.get("road_amplitude_m", project.road.amplitude_m),
            rate_aware_controller=item.get("rate_aware_controller", False),
            friction_aware_planner=item.get("friction_aware_planner", False),
        )
        for item in items
    )


def _run_kinematic_trial(scenario: DynamicScenario, project, sim_cfg, settings):
    result = run_closed_loop(
        "nmpc",
        sim_cfg,
        road_amplitude=scenario.road_amplitude_m,
        initial_lateral_offset=settings["initial_lateral_offset_m"],
        vehicle_cfg=project.vehicle,
        mpc_cfg=project.mpc,
        road_wavelength=project.road.wavelength_m,
        road_length=project.road.length_m,
    )
    metrics = {
        **result.metrics,
        "max_sideslip_deg": 0.0,
        "max_yaw_rate_deg_s": float("nan"),
        "max_lateral_accel_mps2": float("nan"),
        "max_tire_friction_utilization": float("nan"),
        "selected_target_speed_mps": scenario.speed_mps,
        "selected_lane_change_duration_s": scenario.lane_change_duration_s,
        "selected_lane_change_start_time_s": 0.0,
        "selected_speed_transition_duration_s": 0.0,
        "friction_accel_budget_mps2": float("nan"),
        "reference_peak_lateral_accel_mps2": float("nan"),
        "reference_peak_longitudinal_accel_mps2": float("nan"),
        "reference_peak_combined_accel_mps2": float("nan"),
    }
    return metrics, result.states, result.reference


def _build_reference_trajectory(scenario, project, sim_cfg, settings, road):
    horizon_duration = sim_cfg.duration + project.mpc.horizon * sim_cfg.dt
    if not scenario.friction_aware_planner:
        trajectory = generate_lane_change(
            road,
            horizon_duration,
            sim_cfg.dt,
            sim_cfg.target_speed,
            sim_cfg.lane_width,
            scenario.lane_change_duration_s,
        )
        return trajectory, None

    planner_cfg = settings["friction_planner"]
    planning_result = select_friction_aware_trajectory(
        road=road,
        duration=horizon_duration,
        dt=sim_cfg.dt,
        current_speed=scenario.speed_mps,
        desired_speed=scenario.speed_mps,
        target_d=sim_cfg.lane_width,
        friction_coefficient=scenario.friction_coefficient,
        target_speeds=tuple(planner_cfg["target_speeds_mps"]),
        lane_change_durations=tuple(planner_cfg["lane_change_durations_s"]),
        lane_change_start_times=tuple(planner_cfg["lane_change_start_times_s"]),
        speed_transition_durations=tuple(planner_cfg["speed_transition_durations_s"]),
        safety_factor=planner_cfg["safety_factor"],
        min_longitudinal_acceleration=project.vehicle.min_accel,
        max_longitudinal_acceleration=project.vehicle.max_accel,
        cost_weights=friction_plan_cost_weights_from_mapping(planner_cfg.get("cost_weights")),
    )
    return planning_result.trajectory, planning_result


def _build_dynamic_plant(scenario, settings, limits, dt):
    vehicle_model = settings["vehicle_model"]
    if vehicle_model["tire_model"] != "smooth_tanh_friction_circle":
        raise ValueError("Unsupported dynamic tire model")
    plant_config = DynamicVehicleConfig(
        mass=vehicle_model["mass_kg"],
        yaw_inertia=vehicle_model["yaw_inertia_kgm2"],
        lf=vehicle_model["lf_m"],
        lr=vehicle_model["lr_m"],
        cornering_stiffness_front=vehicle_model["cornering_stiffness_front_n_rad"],
        cornering_stiffness_rear=vehicle_model["cornering_stiffness_rear_n_rad"],
        friction_coefficient=scenario.friction_coefficient,
        gravity=vehicle_model["gravity_mps2"],
        integration_substeps=vehicle_model["integration_substeps"],
        max_steer_rate_rad_s=scenario.max_steer_rate_rad_s,
    )
    return DynamicBicycle(plant_config, limits, dt)


def _initial_dynamic_state(trajectory, scenario, settings):
    initial = trajectory.states[0].copy()
    initial[1] += (
        settings["friction_aware_initial_lateral_offset_m"]
        if scenario.friction_aware_planner
        else settings["initial_lateral_offset_m"]
    )
    return DynamicVehicleState(initial[0], initial[1], initial[2], initial[3])


def _simulate_dynamic_closed_loop(
    scenario,
    trajectory,
    controller,
    plant,
    state,
    sim_cfg,
    mpc_cfg,
):
    states, controls = [state.as_array()], []
    solver_failures = 0
    simulation_steps = round(sim_cfg.duration / sim_cfg.dt)
    for index in range(simulation_steps):
        references = trajectory.states[index + 1 : index + 1 + mpc_cfg.horizon]
        control = controller.control(
            state.controller_state(),
            references,
            actual_steer=plant.actual_steer if scenario.rate_aware_controller else None,
        )
        solver_failures += int(not controller.last_success)
        state = plant.step(state, control[0], control[1])
        states.append(state.as_array())
        controls.append(control)
    return np.asarray(states), np.asarray(controls), solver_failures, simulation_steps


def _planning_metrics(scenario, planning_result) -> dict[str, float]:
    if planning_result is None:
        return {
            "selected_target_speed_mps": scenario.speed_mps,
            "selected_lane_change_duration_s": scenario.lane_change_duration_s,
            "selected_lane_change_start_time_s": 0.0,
            "selected_speed_transition_duration_s": 0.0,
            "friction_accel_budget_mps2": float("nan"),
            "reference_peak_lateral_accel_mps2": float("nan"),
            "reference_peak_longitudinal_accel_mps2": float("nan"),
            "reference_peak_combined_accel_mps2": float("nan"),
        }
    return {
        "selected_target_speed_mps": planning_result.target_speed,
        "selected_lane_change_duration_s": planning_result.lane_change_duration,
        "selected_lane_change_start_time_s": planning_result.lane_change_start_time,
        "selected_speed_transition_duration_s": planning_result.speed_transition_duration,
        "friction_accel_budget_mps2": planning_result.lateral_acceleration_budget,
        "reference_peak_lateral_accel_mps2": planning_result.peak_lateral_acceleration,
        "reference_peak_longitudinal_accel_mps2": planning_result.peak_longitudinal_acceleration,
        "reference_peak_combined_accel_mps2": planning_result.peak_combined_acceleration,
    }


def _calculate_dynamic_metrics(
    scenario,
    states,
    controls,
    reference,
    solver_failures,
    limits,
    dt,
    plant,
    planning_result,
):
    controller_states = states[:, :4]
    metrics = calculate_metrics(
        controller_states,
        reference,
        controls,
        solver_failures,
        limits,
        dt,
        scenario.max_steer_rate_rad_s,
    )
    sideslip = np.arctan2(states[:, 4], np.maximum(states[:, 3], 0.1))
    lateral_accel = np.gradient(states[:, 4], dt) + states[:, 3] * states[:, 5]
    metrics.update(
        {
            "max_sideslip_deg": float(np.max(np.abs(np.rad2deg(sideslip)))),
            "max_yaw_rate_deg_s": float(np.max(np.abs(np.rad2deg(states[:, 5])))),
            "max_lateral_accel_mps2": float(np.max(np.abs(lateral_accel))),
            "max_tire_friction_utilization": plant.max_tire_friction_utilization,
            **_planning_metrics(scenario, planning_result),
        }
    )
    return metrics, controller_states


def run_dynamic_trial(scenario: DynamicScenario, project=None):
    project = project or load_project_config("configs/dynamic_model_benchmark.json")
    settings = project.raw["dynamic_benchmark"]
    sim_cfg = replace(
        project.simulation,
        target_speed=scenario.speed_mps,
        lane_change_duration=scenario.lane_change_duration_s,
    )
    if scenario.plant == "kinematic":
        return _run_kinematic_trial(scenario, project, sim_cfg, settings)

    limits, mpc_cfg = project.vehicle, project.mpc
    road = ReferencePath.sinusoidal(
        length=project.road.length_m,
        amplitude=scenario.road_amplitude_m,
        wavelength=project.road.wavelength_m,
    )
    trajectory, planning_result = _build_reference_trajectory(
        scenario,
        project,
        sim_cfg,
        settings,
        road,
    )
    prediction_model = KinematicBicycle(limits, sim_cfg.dt)
    controller = NonlinearMPC(
        prediction_model,
        limits,
        mpc_cfg,
        max_steer_rate=(
            scenario.max_steer_rate_rad_s if scenario.rate_aware_controller else None
        ),
    )
    plant = _build_dynamic_plant(scenario, settings, limits, sim_cfg.dt)
    state = _initial_dynamic_state(trajectory, scenario, settings)
    states, controls, solver_failures, simulation_steps = _simulate_dynamic_closed_loop(
        scenario,
        trajectory,
        controller,
        plant,
        state,
        sim_cfg,
        mpc_cfg,
    )
    reference = trajectory.states[: simulation_steps + 1]
    metrics, controller_states = _calculate_dynamic_metrics(
        scenario,
        states,
        controls,
        reference,
        solver_failures,
        limits,
        sim_cfg.dt,
        plant,
        planning_result,
    )
    return metrics, controller_states, reference


def run_dynamic_benchmark(
    output_dir: Path = Path("results"),
    config_path: Path = Path("configs/dynamic_model_benchmark.json"),
) -> list[dict]:
    project = load_project_config(config_path)
    settings = project.raw["dynamic_benchmark"]
    scenarios = _load_scenarios(settings["scenarios"], project)
    lateral_limit = float(settings["acceptance"]["lateral_rmse_limit_m"])
    sideslip_limit = float(settings["acceptance"]["sideslip_limit_deg"])
    rows, trajectories = [], {}
    for scenario in scenarios:
        metrics, states, reference = run_dynamic_trial(scenario, project)
        passed = bool(
            metrics["lateral_rmse_m"] <= lateral_limit
            and metrics["max_sideslip_deg"] <= sideslip_limit
            and metrics["constraint_violations"] == 0
            and metrics["solver_failures"] == 0
        )
        rows.append(
            {
                "scenario": scenario.name,
                "plant": scenario.plant,
                "speed_mps": scenario.speed_mps,
                "friction_coefficient": scenario.friction_coefficient,
                **metrics,
                "acceptance_passed": passed,
            }
        )
        trajectories[scenario.name] = (states, reference)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "dynamic_model_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _save_metric_plot(
        output_dir / "figures" / "dynamic_model_benchmark.png", rows, lateral_limit
    )
    _save_trajectory_plot(
        output_dir / "figures" / "dynamic_model_trajectories.png",
        trajectories,
        rows,
        project.simulation.dt,
    )
    print(json.dumps(rows, indent=2))
    return rows


def _save_metric_plot(path: Path, rows: list[dict], lateral_limit: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["scenario"].replace("_", "\n") for row in rows]
    values = [row["lateral_rmse_m"] for row in rows]
    colors = ["tab:blue" if row["acceptance_passed"] else "tab:red" for row in rows]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(rows)), values, color=colors)
    ax.axhline(
        lateral_limit,
        color="black",
        linestyle="--",
        label="RMSE acceptance limit",
    )
    ax.set_yscale("log")
    ax.set(
        ylabel="lateral RMSE [m] (log scale)",
        title="Kinematic MPC controlling kinematic vs. dynamic plants",
    )
    ax.set_xticks(np.arange(len(rows)), labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_trajectory_plot(path: Path, trajectories: dict, rows: list[dict], dt: float):
    status = {row["scenario"]: row["acceptance_passed"] for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for name, (states, reference) in trajectories.items():
        error_xy = states[:, :2] - reference[:, :2]
        normal = np.column_stack((-np.sin(reference[:, 2]), np.cos(reference[:, 2])))
        lateral_error = np.sum(error_xy * normal, axis=1)
        time = np.arange(len(states)) * dt
        axes[0 if status[name] else 1].plot(
            time, lateral_error, linewidth=1.5, label=name
        )
    axes[0].set_title("Accepted scenarios")
    axes[1].set_title("Rejected scenarios")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(xlabel="time [s]", ylabel="signed lateral error [m]")
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.suptitle("Dynamic-plant tracking error and failure modes")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured dynamic-plant benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dynamic_model_benchmark.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_dynamic_benchmark(args.output, args.config)
