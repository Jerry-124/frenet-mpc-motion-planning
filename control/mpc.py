import numpy as np
from scipy.optimize import minimize

from config import MPCConfig, VehicleConfig
from models.vehicle import KinematicBicycle


def _angle_error(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


class NonlinearMPC:
    """Direct-shooting NMPC solved by SLSQP; inputs are acceleration and steering."""

    def __init__(
        self,
        model: KinematicBicycle,
        vehicle: VehicleConfig,
        config: MPCConfig,
        max_steer_rate: float | None = None,
    ):
        self.model = model
        self.vehicle = vehicle
        self.config = config
        self.previous_solution = np.zeros((config.horizon, 2), dtype=float)
        self.last_success = True
        self.max_steer_rate = max_steer_rate

    def _rollout(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        predicted = []
        current = state.copy()
        for control in controls:
            current = self.model.step_array_unclipped(current, control)
            predicted.append(current)
        return np.asarray(predicted)

    def _objective(self, flat_controls: np.ndarray, state: np.ndarray, references: np.ndarray) -> float:
        controls = flat_controls.reshape(self.config.horizon, 2)
        predicted = self._rollout(state, controls)
        position_error = predicted[:, :2] - references[:, :2]
        yaw_error = np.array([_angle_error(a, b) for a, b in zip(predicted[:, 2], references[:, 2])])
        speed_error = predicted[:, 3] - references[:, 3]
        cost = (
            self.config.q_x * np.sum(position_error[:, 0] ** 2)
            + self.config.q_y * np.sum(position_error[:, 1] ** 2)
            + self.config.q_yaw * np.sum(yaw_error**2)
            + self.config.q_v * np.sum(speed_error**2)
            + self.config.r_accel * np.sum(controls[:, 0] ** 2)
            + self.config.r_steer * np.sum(controls[:, 1] ** 2)
        )
        delta = np.diff(np.vstack((self.previous_solution[0], controls)), axis=0)
        cost += self.config.rd_accel * np.sum(delta[:, 0] ** 2)
        cost += self.config.rd_steer * np.sum(delta[:, 1] ** 2)
        return float(cost)

    def _speed_constraints(self, flat_controls: np.ndarray, state: np.ndarray) -> np.ndarray:
        predicted_speed = self._rollout(
            state, flat_controls.reshape(self.config.horizon, 2),
        )[:, 3]
        return np.r_[
            predicted_speed - self.vehicle.min_speed,
            self.vehicle.max_speed - predicted_speed,
        ]

    def control(self, state: np.ndarray, references: np.ndarray, actual_steer: float | None = None) -> np.ndarray:
        if len(references) < self.config.horizon:
            references = np.vstack((references, np.repeat(references[-1][None, :], self.config.horizon - len(references), axis=0)))
        else:
            references = references[: self.config.horizon]
        guess = np.vstack((self.previous_solution[1:], self.previous_solution[-1]))
        constraints = [{
            "type": "ineq",
            "fun": lambda flat: self._speed_constraints(flat, state),
        }]
        if self.max_steer_rate is not None:
            if actual_steer is None:
                raise ValueError("actual_steer is required when steering-rate constraints are enabled")
            steer_step = self.max_steer_rate * self.model.dt
            previous = float(actual_steer)
            for index in range(self.config.horizon):
                guess[index, 1] = np.clip(guess[index, 1], previous - steer_step, previous + steer_step)
                previous = guess[index, 1]
            constraints.append({
                "type": "ineq",
                "fun": lambda flat: steer_step - np.abs(
                    np.diff(np.r_[float(actual_steer), flat.reshape(self.config.horizon, 2)[:, 1]])
                ),
            })
        bounds = []
        for _ in range(self.config.horizon):
            bounds.extend([(self.vehicle.min_accel, self.vehicle.max_accel), (-self.vehicle.max_steer, self.vehicle.max_steer)])
        result = minimize(
            self._objective,
            guess.ravel(),
            args=(state, references),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.config.max_iterations, "ftol": 1e-3, "disp": False},
        )
        self.last_success = bool(result.success)
        if result.success and np.all(np.isfinite(result.x)):
            self.previous_solution = result.x.reshape(self.config.horizon, 2)
        else:
            self.previous_solution = guess
        return self.previous_solution[0].copy()
