from pathlib import Path
import argparse
import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import replace

from config import load_project_config
from simulation import run_closed_loop


def run_benchmark(output_dir: Path = Path("results"), config_path: Path = Path("configs/benchmark.json")) -> list[dict]:
    project = load_project_config(config_path)
    controllers = tuple(project.raw["controllers"])
    scenarios = tuple(project.raw["scenarios"])
    comparison_scenario = project.raw["comparison_scenario"]
    rows = []
    nominal_results = {}
    for scenario_data in scenarios:
        scenario = scenario_data["name"]
        speed = float(scenario_data["speed_mps"])
        amplitude = float(scenario_data["road_amplitude_m"])
        config = replace(project.simulation, target_speed=speed)
        for controller in controllers:
            start = time.perf_counter()
            result = run_closed_loop(
                controller, config, amplitude, project.road.initial_lateral_offset_m,
                project.vehicle, project.mpc, project.road.wavelength_m, project.road.length_m,
            )
            elapsed = time.perf_counter() - start
            row = {
                "scenario": scenario,
                "controller": controller,
                "target_speed_mps": speed,
                "road_amplitude_m": amplitude,
                "runtime_s": elapsed,
                **result.metrics,
            }
            rows.append(row)
            if scenario == comparison_scenario:
                nominal_results[controller] = result

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "controller_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_comparison_plots(
        output_dir / "figures", rows, nominal_results, scenarios, controllers,
        comparison_scenario,
    )
    print(json.dumps(rows, indent=2))
    return rows


def save_comparison_plots(directory: Path, rows: list[dict], nominal_results: dict, scenarios, controllers, comparison_scenario):
    directory.mkdir(parents=True, exist_ok=True)
    labels = [scenario["name"] for scenario in scenarios]
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, controller in zip((-width / 2, width / 2), controllers):
        values = [row["lateral_rmse_m"] for row in rows if row["controller"] == controller]
        ax.bar(x + offset, values, width, label=controller)
    ax.set_ylabel("lateral RMSE [m]"); ax.set_xticks(x, labels, rotation=15)
    ax.set_title("Controller robustness across scenarios"); ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(directory / "controller_benchmark.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    reference = nominal_results["nmpc"].reference
    ax.plot(reference[:, 0], reference[:, 1], "k--", label="reference")
    for controller in controllers:
        states = nominal_results[controller].states
        ax.plot(states[:, 0], states[:, 1], label=controller)
    ax.set(xlabel="x [m]", ylabel="y [m]", title=f"Trajectory comparison: {comparison_scenario}")
    ax.axis("equal"); ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(directory / "controller_trajectory_comparison.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NMPC vs Stanley/PID scenario benchmark")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    args = parser.parse_args()
    run_benchmark(args.output, args.config)
