from dataclasses import dataclass

import numpy as np

from config import MPCConfig, SimulationConfig, VehicleConfig
from control import EmergencyBrakeController, NonlinearMPC, StanleyPIDController
from evaluation import calculate_metrics
from models import KinematicBicycle, VehicleState
from planning import ReferencePath, generate_lane_change


@dataclass
class SimulationResult:
    road: ReferencePath
    reference: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    metrics: dict


def run_closed_loop(
    controller_name: str = "nmpc",
    sim_cfg: SimulationConfig | None = None,
    road_amplitude: float = 2.0,
    initial_lateral_offset: float = -0.35,
    vehicle_cfg: VehicleConfig | None = None,
    mpc_cfg: MPCConfig | None = None,
    road_wavelength: float = 70.0,
    road_length: float = 140.0,
) -> SimulationResult:
    sim_cfg = sim_cfg or SimulationConfig()
    vehicle_cfg, mpc_cfg = vehicle_cfg or VehicleConfig(), mpc_cfg or MPCConfig()
    road = ReferencePath.sinusoidal(length=road_length, amplitude=road_amplitude, wavelength=road_wavelength)
    preview_duration = mpc_cfg.horizon * sim_cfg.dt
    trajectory = generate_lane_change(
        road,
        sim_cfg.duration + preview_duration,
        sim_cfg.dt,
        sim_cfg.target_speed,
        sim_cfg.lane_width,
        sim_cfg.lane_change_duration,
    )
    return track_reference(
        controller_name, road, trajectory.states, sim_cfg, initial_lateral_offset,
        vehicle_cfg=vehicle_cfg, mpc_cfg=mpc_cfg,
    )


def track_reference(
    controller_name: str,
    road: ReferencePath,
    reference_states: np.ndarray,
    sim_cfg: SimulationConfig,
    initial_lateral_offset: float = -0.35,
    vehicle_cfg: VehicleConfig | None = None,
    mpc_cfg: MPCConfig | None = None,
) -> SimulationResult:
    vehicle_cfg, mpc_cfg = vehicle_cfg or VehicleConfig(), mpc_cfg or MPCConfig()
    model = KinematicBicycle(vehicle_cfg, sim_cfg.dt)
    if controller_name == "nmpc":
        controller = NonlinearMPC(model, vehicle_cfg, mpc_cfg)
    elif controller_name == "stanley_pid":
        controller = StanleyPIDController(vehicle_cfg, sim_cfg.dt)
    elif controller_name == "emergency_brake":
        controller = EmergencyBrakeController(vehicle_cfg, sim_cfg.dt)
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    simulation_steps = round(sim_cfg.duration / sim_cfg.dt)
    required_points = simulation_steps + mpc_cfg.horizon
    if len(reference_states) < required_points:
        raise ValueError(f"Reference needs at least {required_points} points for a full MPC preview")
    initial = reference_states[0].copy()
    initial[1] += initial_lateral_offset
    state = VehicleState(*initial)
    states, controls = [state.as_array()], []
    solver_failures = 0
    for index in range(simulation_steps):
        references = reference_states[index + 1 : index + 1 + mpc_cfg.horizon]
        control = controller.control(state.as_array(), references)
        solver_failures += int(not controller.last_success)
        state = model.step(state, control[0], control[1])
        controls.append(control)
        states.append(state.as_array())

    states, controls = np.asarray(states), np.asarray(controls)
    reference = reference_states[: simulation_steps + 1]
    metrics = calculate_metrics(
        states, reference, controls, solver_failures, vehicle_cfg, sim_cfg.dt,
    )
    return SimulationResult(road, reference, states, controls, metrics)
