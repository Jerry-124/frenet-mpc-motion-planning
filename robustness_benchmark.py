from dataclasses import dataclass, replace
from pathlib import Path
from collections import deque
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from config import load_project_config
from control import NonlinearMPC
from evaluation import calculate_metrics
from models import KinematicBicycle, VehicleState
from planning import ReferencePath, generate_lane_change


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    position_noise_std_m: float = 0.0
    yaw_noise_std_rad: float = 0.0
    speed_noise_std_mps: float = 0.0
    actuation_delay_steps: int = 0
    plant_wheelbase_scale: float = 1.0
    delay_compensation: bool = False


def _load_scenarios(items) -> tuple[RobustnessScenario, ...]:
    return tuple(RobustnessScenario(
        name=item["name"],
        position_noise_std_m=item.get("position_noise_std_m", 0.0),
        yaw_noise_std_rad=np.deg2rad(item.get("yaw_noise_std_deg", 0.0)),
        speed_noise_std_mps=item.get("speed_noise_std_mps", 0.0),
        actuation_delay_steps=item.get("actuation_delay_steps", 0),
        plant_wheelbase_scale=item.get("plant_wheelbase_scale", 1.0),
        delay_compensation=item.get("delay_compensation", False),
    ) for item in items)


class DelayedActuator:
    def __init__(self, delay_steps: int):
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.delay_steps = delay_steps
        self.queue = deque([np.zeros(2) for _ in range(delay_steps)])

    def apply(self, command: np.ndarray) -> np.ndarray:
        if self.delay_steps == 0:
            return np.asarray(command, dtype=float).copy()
        self.queue.append(np.asarray(command, dtype=float).copy())
        return self.queue.popleft()

    def pending_controls(self) -> list[np.ndarray]:
        return [control.copy() for control in self.queue]


def run_trial(scenario: RobustnessScenario, seed: int, project=None):
    rng = np.random.default_rng(seed)
    project = project or load_project_config("configs/robustness_benchmark.json")
    sim_cfg, mpc_cfg = project.simulation, project.mpc
    nominal_vehicle = project.vehicle
    plant_vehicle = replace(
        nominal_vehicle,
        wheelbase=nominal_vehicle.wheelbase * scenario.plant_wheelbase_scale,
    )
    road = ReferencePath.sinusoidal(
        length=project.road.length_m, amplitude=project.road.amplitude_m,
        wavelength=project.road.wavelength_m,
    )
    trajectory = generate_lane_change(
        road,
        sim_cfg.duration + (mpc_cfg.horizon + scenario.actuation_delay_steps) * sim_cfg.dt,
        sim_cfg.dt,
        sim_cfg.target_speed,
        sim_cfg.lane_width,
        sim_cfg.lane_change_duration,
    )
    prediction_model = KinematicBicycle(nominal_vehicle, sim_cfg.dt)
    plant = KinematicBicycle(plant_vehicle, sim_cfg.dt)
    controller = NonlinearMPC(prediction_model, nominal_vehicle, mpc_cfg)
    actuator = DelayedActuator(scenario.actuation_delay_steps)

    initial = trajectory.states[0].copy()
    initial[1] += project.road.initial_lateral_offset_m
    state = VehicleState(*initial)
    states, applied_controls = [state.as_array()], []
    solver_failures = 0
    simulation_steps = int(round(sim_cfg.duration / sim_cfg.dt))
    for index in range(simulation_steps):
        measured = state.as_array().copy()
        measured[:2] += rng.normal(0.0, scenario.position_noise_std_m, 2)
        measured[2] += rng.normal(0.0, scenario.yaw_noise_std_rad)
        measured[2] = np.arctan2(np.sin(measured[2]), np.cos(measured[2]))
        measured[3] = max(0.0, measured[3] + rng.normal(0.0, scenario.speed_noise_std_mps))
        controller_state = measured
        reference_shift = 0
        if scenario.delay_compensation and scenario.actuation_delay_steps:
            controller_state = measured.copy()
            for pending_control in actuator.pending_controls():
                controller_state = prediction_model.step_array(controller_state, pending_control)
            reference_shift = scenario.actuation_delay_steps
        references = trajectory.states[
            index + 1 + reference_shift : index + 1 + reference_shift + mpc_cfg.horizon
        ]
        command = controller.control(controller_state, references)
        solver_failures += int(not controller.last_success)
        applied = actuator.apply(command)
        state = plant.step(state, applied[0], applied[1])
        applied_controls.append(applied)
        states.append(state.as_array())

    states, applied_controls = np.asarray(states), np.asarray(applied_controls)
    reference = trajectory.states[: simulation_steps + 1]
    metrics = calculate_metrics(
        states, reference, applied_controls, solver_failures, nominal_vehicle, sim_cfg.dt,
    )
    return metrics, states, reference


def run_robustness_benchmark(
    output_dir: Path = Path("results"),
    config_path: Path = Path("configs/robustness_benchmark.json"),
) -> list[dict]:
    project = load_project_config(config_path)
    settings = project.raw["robustness"]
    scenarios = _load_scenarios(settings["scenarios"])
    seeds = tuple(settings["seeds"])
    acceptance = settings["acceptance"]
    rows, representatives = [], {}
    for scenario in scenarios:
        for seed in seeds:
            metrics, states, reference = run_trial(scenario, int(seed), project)
            rows.append({"scenario": scenario.name, "seed": int(seed), **metrics})
            if seed == 0:
                representatives[scenario.name] = (states, reference)

    summaries = summarize(
        rows, scenarios,
        float(acceptance["lateral_rmse_limit_m"]),
        float(acceptance["max_position_error_limit_m"]),
    )
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_dir / "robustness_runs.csv", rows)
    _write_csv(metrics_dir / "robustness_summary.csv", summaries)
    _save_summary_plot(
        output_dir / "figures" / "robustness_benchmark.png", summaries,
        float(acceptance["lateral_rmse_limit_m"]),
    )
    _save_trajectory_plot(output_dir / "figures" / "robustness_trajectories.png", representatives)
    print(json.dumps(summaries, indent=2))
    return summaries


def summarize(rows: list[dict], scenarios, lateral_rmse_limit_m: float, max_position_error_limit_m: float) -> list[dict]:
    summaries = []
    for scenario in scenarios:
        group = [row for row in rows if row["scenario"] == scenario.name]
        lateral = np.array([row["lateral_rmse_m"] for row in group])
        maximum = np.array([row["max_position_error_m"] for row in group])
        violations = sum(row["constraint_violations"] for row in group)
        failures = sum(row["solver_failures"] for row in group)
        passed = bool(
            np.max(lateral) <= lateral_rmse_limit_m
            and np.max(maximum) <= max_position_error_limit_m
            and violations == 0
            and failures == 0
        )
        summaries.append({
            "scenario": scenario.name,
            "runs": len(group),
            "lateral_rmse_mean_m": float(np.mean(lateral)),
            "lateral_rmse_std_m": float(np.std(lateral)),
            "lateral_rmse_worst_m": float(np.max(lateral)),
            "max_position_error_worst_m": float(np.max(maximum)),
            "heading_rmse_mean_deg": float(np.mean([row["heading_rmse_deg"] for row in group])),
            "speed_rmse_mean_mps": float(np.mean([row["speed_rmse_mps"] for row in group])),
            "constraint_violations_total": int(violations),
            "solver_failures_total": int(failures),
            "acceptance_passed": passed,
        })
    return summaries


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def _save_summary_plot(path: Path, summaries: list[dict], lateral_rmse_limit_m: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["scenario"].replace("_", "\n") for row in summaries]
    means = [row["lateral_rmse_mean_m"] for row in summaries]
    stds = [row["lateral_rmse_std_m"] for row in summaries]
    colors = ["tab:blue" if row["acceptance_passed"] else "tab:red" for row in summaries]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(labels)), means, yerr=stds, capsize=5, color=colors)
    ax.axhline(lateral_rmse_limit_m, color="black", linestyle="--", label="acceptance limit")
    ax.set(ylabel="lateral RMSE [m]", title="NMPC robustness benchmark (mean ± std, 5 seeds)")
    ax.set_xticks(np.arange(len(labels)), labels); ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _save_trajectory_plot(path: Path, representatives: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    reference = next(iter(representatives.values()))[1]
    ax.plot(reference[:, 0], reference[:, 1], "k--", linewidth=2, label="reference")
    for name, (states, _) in representatives.items():
        ax.plot(states[:, 0], states[:, 1], linewidth=1.4, label=name)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Representative trajectories under disturbances")
    ax.axis("equal"); ax.grid(True); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured robustness Monte Carlo benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/robustness_benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_robustness_benchmark(args.output, args.config)
