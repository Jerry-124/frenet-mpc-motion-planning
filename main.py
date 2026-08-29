from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from evaluation import save_metrics
from config import load_project_config
from planning import ReferencePath
from simulation import run_closed_loop


def run_simulation(output_dir: Path = Path("results"), config_path: Path = Path("configs/default.json")) -> dict:
    config = load_project_config(config_path)
    result = run_closed_loop(
        "nmpc", config.simulation, config.road.amplitude_m,
        config.road.initial_lateral_offset_m, config.vehicle, config.mpc,
        config.road.wavelength_m, config.road.length_m,
    )
    save_metrics(result.metrics, output_dir / "metrics" / "mpc_evaluation.csv")
    save_plots(
        output_dir / "figures", result.road, result.reference, result.states,
        result.controls, config.simulation.dt,
    )
    print(json.dumps(result.metrics, indent=2))
    return result.metrics


def save_plots(directory: Path, road: ReferencePath, reference: np.ndarray, states: np.ndarray, controls: np.ndarray, dt: float):
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(road.x, road.y, "--", color="0.65", label="road centerline")
    ax.plot(reference[:, 0], reference[:, 1], label="Frenet reference")
    ax.plot(states[:, 0], states[:, 1], label="MPC tracking")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Reference vs. actual trajectory")
    ax.axis("equal"); ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(directory / "reference_vs_actual.png", dpi=160); plt.close(fig)

    time = np.arange(len(states)) * dt
    error = np.linalg.norm(states[:, :2] - reference[:, :2], axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, error); ax.set(xlabel="time [s]", ylabel="position error [m]", title="Tracking error")
    ax.grid(True); fig.tight_layout(); fig.savefig(directory / "tracking_error.png", dpi=160); plt.close(fig)

    control_time = time[:-1]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(control_time, controls[:, 0]); axes[0].set_ylabel("acceleration [m/s²]"); axes[0].grid(True)
    axes[1].plot(control_time, np.rad2deg(controls[:, 1])); axes[1].set_ylabel("steering [deg]"); axes[1].set_xlabel("time [s]"); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(directory / "control_inputs.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the configured NMPC baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_simulation(args.output, args.config)
