from dataclasses import replace
from pathlib import Path
import argparse
import csv
import json
import time

import matplotlib.pyplot as plt

from config import load_project_config
from simulation import run_closed_loop


def run_sensitivity_benchmark(
    output_dir: Path = Path("results"),
    config_path: Path = Path("configs/sensitivity_benchmark.json"),
) -> list[dict]:
    project = load_project_config(config_path)
    settings = project.raw["sensitivity"]
    lateral_limit = float(settings["acceptance"]["lateral_rmse_limit_m"])
    rows = []
    for parameter, values in settings["sweeps"].items():
        for value in values:
            mpc = replace(project.mpc, **{parameter: int(value) if parameter == "horizon" else float(value)})
            start = time.perf_counter()
            result = run_closed_loop(
                "nmpc", project.simulation, project.road.amplitude_m,
                project.road.initial_lateral_offset_m, project.vehicle, mpc,
                project.road.wavelength_m, project.road.length_m,
            )
            runtime = time.perf_counter() - start
            steps = int(round(project.simulation.duration / project.simulation.dt))
            passed = bool(
                result.metrics["lateral_rmse_m"] <= lateral_limit
                and result.metrics["constraint_violations"] == 0
                and result.metrics["solver_failures"] == 0
            )
            rows.append({
                "parameter": parameter,
                "value": value,
                "horizon_steps": mpc.horizon,
                "q_y": mpc.q_y,
                "rd_steer": mpc.rd_steer,
                "runtime_s": runtime,
                "mean_control_time_ms": 1000.0 * runtime / steps,
                **result.metrics,
                "acceptance_passed": passed,
            })

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "mpc_parameter_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    _save_plot(output_dir / "figures" / "mpc_parameter_sensitivity.png", rows, lateral_limit)
    print(json.dumps(rows, indent=2))
    return rows


def _save_plot(path: Path, rows: list[dict], lateral_limit: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = tuple(dict.fromkeys(row["parameter"] for row in rows))
    fig, axes = plt.subplots(1, len(parameters), figsize=(15, 4.8))
    for ax, parameter in zip(axes, parameters):
        group = [row for row in rows if row["parameter"] == parameter]
        x = [row["value"] for row in group]
        rmse = [row["lateral_rmse_m"] for row in group]
        runtime = [row["mean_control_time_ms"] for row in group]
        ax.plot(x, rmse, "o-", color="tab:blue", label="lateral RMSE")
        ax.axhline(lateral_limit, color="0.35", linestyle="--", linewidth=1.0)
        ax.set(xlabel=parameter, ylabel="lateral RMSE [m]", title=f"Sensitivity: {parameter}")
        ax.grid(True, alpha=0.3)
        runtime_ax = ax.twinx()
        runtime_ax.plot(x, runtime, "s--", color="tab:orange", label="mean control time")
        runtime_ax.set_ylabel("mean control time [ms]")
        lines = ax.get_lines()[:1] + runtime_ax.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured MPC parameter sensitivity sweeps")
    parser.add_argument("--config", type=Path, default=Path("configs/sensitivity_benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_sensitivity_benchmark(args.output, args.config)
