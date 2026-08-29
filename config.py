from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase: float = 2.8
    max_steer: float = np.deg2rad(30.0)
    min_accel: float = -3.0
    max_accel: float = 2.0
    min_speed: float = 0.0
    max_speed: float = 20.0


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.1
    duration: float = 10.0
    target_speed: float = 10.0
    lane_width: float = 3.5
    lane_change_duration: float = 4.0


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 8
    q_x: float = 1.0
    q_y: float = 8.0
    q_yaw: float = 3.0
    q_v: float = 1.0
    r_accel: float = 0.15
    r_steer: float = 0.3
    rd_accel: float = 0.4
    rd_steer: float = 2.0
    max_iterations: int = 35


@dataclass(frozen=True)
class RoadConfig:
    amplitude_m: float = 2.0
    wavelength_m: float = 70.0
    length_m: float = 140.0
    initial_lateral_offset_m: float = -0.35


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    vehicle: VehicleConfig
    simulation: SimulationConfig
    mpc: MPCConfig
    road: RoadConfig
    raw: dict[str, Any]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a schema-versioned JSON config with optional relative inheritance."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be an object: {config_path}")
    parent = data.pop("extends", None)
    if parent is not None:
        data = _deep_merge(load_json_config(config_path.parent / parent), data)
    version = data.get("schema_version")
    if version != 1:
        raise ValueError(f"Unsupported configuration schema_version {version!r}; expected 1")
    return data


def _section(data: dict, name: str, allowed: set[str]) -> dict:
    values = data.get(name, {})
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{name}' must be an object")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in '{name}': {sorted(unknown)}")
    return values


def load_project_config(path: str | Path = "configs/default.json") -> ProjectConfig:
    data = load_json_config(path)
    vehicle_data = _section(data, "vehicle", {
        "wheelbase_m", "max_steer_deg", "min_accel_mps2", "max_accel_mps2",
        "min_speed_mps", "max_speed_mps",
    })
    simulation_data = _section(data, "simulation", {
        "dt_s", "duration_s", "target_speed_mps", "lane_width_m", "lane_change_duration_s",
    })
    mpc_data = _section(data, "mpc", {
        "horizon_steps", "q_x", "q_y", "q_yaw", "q_v", "r_accel", "r_steer",
        "rd_accel", "rd_steer", "max_iterations",
    })
    road_data = _section(data, "road", {
        "amplitude_m", "wavelength_m", "length_m", "initial_lateral_offset_m",
    })
    vehicle = VehicleConfig(
        wheelbase=vehicle_data.get("wheelbase_m", 2.8),
        max_steer=np.deg2rad(vehicle_data.get("max_steer_deg", 30.0)),
        min_accel=vehicle_data.get("min_accel_mps2", -3.0),
        max_accel=vehicle_data.get("max_accel_mps2", 2.0),
        min_speed=vehicle_data.get("min_speed_mps", 0.0),
        max_speed=vehicle_data.get("max_speed_mps", 20.0),
    )
    simulation = SimulationConfig(
        dt=simulation_data.get("dt_s", 0.1),
        duration=simulation_data.get("duration_s", 10.0),
        target_speed=simulation_data.get("target_speed_mps", 10.0),
        lane_width=simulation_data.get("lane_width_m", 3.5),
        lane_change_duration=simulation_data.get("lane_change_duration_s", 4.0),
    )
    mpc = MPCConfig(
        horizon=mpc_data.get("horizon_steps", 8),
        q_x=mpc_data.get("q_x", 1.0), q_y=mpc_data.get("q_y", 8.0),
        q_yaw=mpc_data.get("q_yaw", 3.0), q_v=mpc_data.get("q_v", 1.0),
        r_accel=mpc_data.get("r_accel", 0.15), r_steer=mpc_data.get("r_steer", 0.3),
        rd_accel=mpc_data.get("rd_accel", 0.4), rd_steer=mpc_data.get("rd_steer", 2.0),
        max_iterations=mpc_data.get("max_iterations", 35),
    )
    road = RoadConfig(
        amplitude_m=road_data.get("amplitude_m", 2.0),
        wavelength_m=road_data.get("wavelength_m", 70.0),
        length_m=road_data.get("length_m", 140.0),
        initial_lateral_offset_m=road_data.get("initial_lateral_offset_m", -0.35),
    )
    if simulation.dt <= 0.0 or simulation.duration <= 0.0 or mpc.horizon <= 0:
        raise ValueError("dt, duration, and MPC horizon must be positive")
    if vehicle.min_accel >= vehicle.max_accel or vehicle.min_speed > vehicle.max_speed:
        raise ValueError("Vehicle limits are inconsistent")
    return ProjectConfig(1, vehicle, simulation, mpc, road, data)
