# Autonomous Vehicle Motion Planning & MPC

An executable baseline for Frenet-frame lane-change trajectory generation and constrained trajectory tracking with nonlinear model predictive control (NMPC).

## Implemented baseline

- arc-length-parameterized sinusoidal road reference
- Frenet-to-Cartesian transformation
- minimum-jerk quintic lateral lane-change profile
- rear-axle kinematic bicycle model
- direct-shooting NMPC using SciPy SLSQP
- Stanley lateral controller with PID speed control as a classical baseline
- acceleration and steering bounds
- explicit predicted-speed and steering-rate constraints across the MPC horizon
- warm-started receding-horizon simulation
- five-scenario controller benchmark
- multi-candidate Frenet local planner with moving-obstacle prediction
- hard road-boundary/collision rejection and interpretable cost breakdown
- Monte Carlo robustness benchmark with delay compensation
- six-state dynamic bicycle plant with smooth nonlinear tire forces and per-axle friction circles
- speed/friction/model-mismatch operating-envelope benchmark
- hard steering-rate constraints initialized from measured actuator state
- friction-circle-aware speed/timing adaptation with longitudinal acceleration limits
- online no-feasible-candidate fallback with direct maximum-braking override
- schema-versioned JSON configuration for every reproducible experiment
- MPC parameter-sensitivity sweeps
- quantitative constraint metrics, reproducible figures, regression gates, and CI

The state is `z = [x, y, yaw, velocity]`, and the input is `u = [acceleration, steering]`. The controller penalizes Cartesian position, heading and speed errors, control effort, and changes in control input. NMPC deliberately keeps its lightweight kinematic prediction model, while a separate six-state dynamic bicycle plant exposes model-mismatch limits.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python main.py
python benchmark.py
python planner_demo.py
python replanning_demo.py
python robustness_benchmark.py
python dynamic_model_benchmark.py
python fallback_demo.py
python sensitivity_benchmark.py
```

Run the automated unit and quantitative regression suite:

```bash
python -m unittest discover -s tests
```

Outputs are written to `results/metrics/` and `results/figures/`.

For the complete methodology, validation matrix, limitations, and resume/interview material, see:

- [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)
- [`docs/RESUME_INTERVIEW_GUIDE_CN.md`](docs/RESUME_INTERVIEW_GUIDE_CN.md)

Every executable also accepts an explicit configuration and output directory:

```bash
python main.py --config configs/default.json --output results
python dynamic_model_benchmark.py --config configs/dynamic_model_benchmark.json
```

## Versioned experiment configuration

All experiment inputs are stored under `configs/` with `schema_version: 1`. `default.json` is the single source for vehicle limits, simulation timing, road geometry, and MPC weights. Each experiment-specific file uses `extends: "default.json"` and contains only its scenario definitions or overrides.

| Configuration | Controls |
|---|---|
| `default.json` | vehicle, simulation, road, MPC horizon and weights |
| `benchmark.json` | controller comparison scenarios |
| `planner_demo.json` | candidate grid, moving obstacles, cost weights and clearance threshold |
| `replanning_demo.json` | replanning interval, candidates, traffic, cost weights and clearance threshold |
| `replanning_blocked_validation.json` | online fully blocked-road fallback validation |
| `robustness_benchmark.json` | seeds, noise/delay cases and acceptance limits |
| `dynamic_model_benchmark.json` | dynamic vehicle/tire parameters, friction, steering-rate and plant cases |
| `fallback_demo.json` | fully blocked-road emergency scenario and planner costs |
| `sensitivity_benchmark.json` | MPC horizon and weight sweeps with acceptance limit |

The loader resolves relative inheritance, converts human-readable steering degrees to radians, checks core keys and physical limits, and rejects unsupported schema versions. Copy an experiment file, change only the desired values, then pass it with `--config`; no Python source edit is required.

## Baseline result

The checked-in result was generated with Python 3.12, a `0.1 s` sample time, an 8-step horizon, and the default scenario above.

| Metric | Result |
|---|---:|
| Lateral RMSE | 0.056 m |
| Longitudinal RMSE | 0.079 m |
| Position RMSE | 0.097 m |
| Heading RMSE | 1.154 deg |
| Speed RMSE | 0.051 m/s |
| Input constraint violations | 0 |
| Predicted speed-state violations | 0 |
| Solver failures (100 steps) | 0 |

These values are a deterministic software-baseline result from a matched kinematic model. They should not be interpreted as real-vehicle validation; model mismatch, noise, and delay remain future work.

## Controller benchmark

`benchmark.py` compares NMPC with Stanley+PID in five combinations of speed and road curvature. Both controllers use the same plant, reference trajectory, initial `-0.35 m` disturbance, limits, and evaluation code.

| Controller | Mean lateral RMSE | Mean simulation runtime* | Constraint violations |
|---|---:|---:|---:|
| NMPC | 0.064 m | ≈0.8 s | 0 |
| Stanley+PID | 0.234 m | 0.004 s | 0 |

NMPC reduced mean lateral RMSE by approximately 73% in this five-scenario run, while Stanley+PID was substantially cheaper computationally. The largest separation appeared in the high-curvature scenario: `0.063 m` versus `0.352 m` lateral RMSE.

\* Runtime is a local end-to-end Python measurement for an 8-second simulation, not a hard real-time execution guarantee.

## Frenet candidate planning

`planner_demo.py` creates 18 candidate trajectories from two target lanes, three lane-change durations, and three target speeds. A slow vehicle is predicted in Frenet coordinates over the planning horizon. Candidates are processed in two stages:

1. Hard rejection for road-boundary violations or entry into the obstacle safety ellipse.
2. Ranking of feasible trajectories using lateral acceleration/jerk, speed error, lane preference, and obstacle-clearance costs.

In the default scenario, all 9 trajectories that remain in the blocked lane are rejected. The selected trajectory changes to `d = 3.5 m` over `5 s` at `10 m/s`; its normalized obstacle clearance is `2.80` against a conservative feasibility threshold of `1.25`. NMPC then tracks it with `0.063 m` lateral RMSE, zero input-limit violations, and zero solver failures.

Candidate-level decisions and individual cost terms are exported to `results/metrics/frenet_candidates.csv` so rejected paths and tuning choices remain auditable.

## Receding-horizon replanning

`replanning_demo.py` closes the planning-control loop over 10 seconds. Every second it:

1. projects the measured Cartesian vehicle state back into Frenet coordinates;
2. predicts two surrounding vehicles over a 5-second horizon;
3. regenerates and scores lane/speed candidates;
4. keeps an active lane change committed until completion;
5. sends only the next segment of the selected trajectory to NMPC.

The default decision sequence is: slow/follow → change left → hold the adjacent lane → return right after passing. The lane-change commitment state prevents repeated polynomial resets from producing an incomplete maneuver.

| Online-planning metric | Result |
|---|---:|
| Replanning events | 10 |
| Minimum actual normalized clearance | 2.45 |
| Input constraint violations | 0 |
| NMPC solver failures | 0 |
| Final speed | 10.13 m/s |

Every online decision—including ego state, selected lane/speed, feasible-candidate count, cost, and predicted clearance—is exported to `results/metrics/replanning_decisions.csv`.

The same online loop is also tested with `configs/replanning_blocked_validation.json`. Both lanes remain blocked, so all five replanning events transition to `emergency_stop`; the independent brake override stops the vehicle with minimum normalized clearance `1.980`, zero constraint violations, and zero solver failures. This confirms that fallback is integrated into repeated planning rather than existing only in the one-shot demo.

## Robustness benchmark and delay compensation

`robustness_benchmark.py` runs seven scenarios with five deterministic seeds each. Disturbances include position/heading/speed measurement noise, a `200 ms` actuator delay, `15%` wheelbase mismatch, and a combined case. Acceptance requires worst-case lateral RMSE ≤ `0.20 m`, worst-case position error ≤ `0.80 m`, zero input-limit violations, and zero solver failures.

| Scenario | Mean lateral RMSE | Worst lateral RMSE | Result |
|---|---:|---:|---:|
| Nominal | 0.063 m | 0.063 m | Pass |
| Sensor noise | 0.080 m | 0.085 m | Pass |
| Delay, uncompensated | 2.202 m | 2.202 m | Fail |
| Delay, compensated | 0.084 m | 0.084 m | Pass |
| 15% wheelbase mismatch | 0.064 m | 0.064 m | Pass |
| Combined, uncompensated | 2.202 m | 2.330 m | Fail |
| Combined, compensated | 0.115 m | 0.125 m | Pass |

The uncompensated delay destabilizes tracking even though the optimizer itself still converges. The compensator propagates the measured state through queued actuator commands and shifts the reference to the command-application time. It reduces mean RMSE by approximately `96.2%` in the delay-only case and `94.8%` in the combined case.

All 35 run-level results and the aggregated pass/fail table are retained in `results/metrics/robustness_runs.csv` and `results/metrics/robustness_summary.csv`.

## MPC constraints and parameter sensitivity

NMPC now evaluates speed without clipping inside its prediction rollout and imposes lower/upper speed inequalities at every prediction step. Steering-rate constraints likewise span the full horizon and are initialized from measured actuator steering. The exported metrics split acceleration, steering-angle, steering-rate, and speed-state violations instead of reporting only one aggregate count.

`sensitivity_benchmark.py` sweeps 15 configured combinations while holding the experiment otherwise fixed. All 15 pass the `0.20 m` lateral-RMSE gate with zero constraint violations and solver failures.

| Sweep | End-point effect | Interpretation |
|---|---|---|
| Horizon `4 → 16` | RMSE `0.0639 → 0.0625 m`; mean control time `5.4 → 66.2 ms` | Accuracy saturates near 8 steps; the default 8-step horizon is the better compute/accuracy balance |
| `q_y` `2 → 32` | RMSE `0.0682 → 0.0587 m`; max steering rate `93.1 → 261.7°/s` | More lateral weight improves tracking but makes steering substantially more aggressive |
| `rd_steer` `0.5 → 8` | RMSE `0.0594 → 0.0675 m`; max steering rate `234.0 → 96.7°/s` | More steering smoothing trades a small tracking loss for gentler actuation |

Timing is a local Python measurement, not a hard real-time guarantee. Full sweep data and the trade-off plot are exported to `results/metrics/mpc_parameter_sensitivity.csv` and `results/figures/mpc_parameter_sensitivity.png`.

## Dynamic-plant model mismatch

`dynamic_model_benchmark.py` replaces the matched kinematic plant with a six-state dynamic bicycle model while leaving NMPC's prediction model unchanged. The plant adds lateral velocity, yaw rate, mass, yaw inertia, axle geometry, front/rear cornering stiffness, smooth `tanh` tire-force saturation, combined longitudinal/lateral friction circles at each axle, and RK4 integration. Physical parameters are loaded from the experiment configuration.

| Scenario | Lateral RMSE | Max sideslip | Result |
|---|---:|---:|---:|
| Matched kinematic, 12 m/s | 0.035 m | 0.00° | Pass |
| Dynamic dry, 8–16 m/s | 0.037–0.064 m | ≤3.36° | Pass |
| Dynamic wet, 12–16 m/s | 0.045–0.048 m | ≤2.44° | Pass |
| Dynamic low-μ, 8–12 m/s | 0.043–0.064 m | ≤1.96° | Pass |
| Aggressive dry, 16 m/s | 0.164 m | 10.42° | Fail |
| Steering-rate limited, 12 m/s | 11.783 m | 55.60° | Fail |
| Steering-rate aware MPC, 12 m/s | 0.058 m | 1.79° | Pass |
| Aggressive low-μ, 12 m/s | 10.462 m | 43.15° | Fail |
| Friction-aware low-μ planning | 0.051 m | 1.76° | Pass |

The results separate two failure mechanisms. Ordinary tire-dynamics mismatch remains manageable, but an unmodeled `0.6 rad/s` steering-rate limit destabilizes the controller. Adding an actual-steering initial condition and hard rate constraints reduces lateral RMSE by `99.5%` without changing the physical actuator limit. The nonlinear tire model also correctly rejects the aggressive dry case on the `5°` sideslip gate even though its tracking RMSE remains below `0.30 m`; this prevents a numerically close but physically unstable run from being labeled successful.

An aggressive `2.5 s` maneuver on `μ = 0.3` saturates available force and fails. The corrected friction-aware planner checks both configured longitudinal acceleration bounds and the Cartesian combined-acceleration friction circle. It evaluates speed-transition timing independently from lane-change timing and selects: transition from `12` to `9.5 m/s` over `2.25 s` → start changing lanes at `0.5 s` → complete the maneuver over `4 s`. Peak longitudinal acceleration is `1.87 m/s²`; peak combined acceleration is `2.73 m/s²`, below the `0.95 μg = 2.80 m/s²` budget. The dynamic plant reaches at most `0.944` tire-friction utilization and lateral RMSE falls by `99.5%` to `0.051 m` without using a longitudinally infeasible reference.

Both original failures remain in the benchmark as before/after evidence rather than being overwritten.

Scenario-level results are exported to `results/metrics/dynamic_model_benchmark.csv`.

## Emergency fallback when no candidate is feasible

`fallback_demo.py` blocks both available lanes at the same longitudinal position. All 8 ordinary lane/speed candidates enter an obstacle safety ellipse and are rejected. Instead of terminating with `No collision-free Frenet trajectory`, the planner records `normal_planning -> emergency_fallback`. A separate safety controller then bypasses ordinary longitudinal MPC and directly commands the maximum permitted deceleration of `-3 m/s²`, while an independent lateral controller holds the lane.

| Fallback metric | Result |
|---|---:|
| Normal candidates / feasible | 8 / 0 |
| Reference stop time | 3.33 s |
| Analytical continuous stop distance | 16.67 m |
| Discrete-model stop distance | 17.17 m |
| Reference minimum normalized clearance | 1.966 |
| Actual travel distance | 17.148 m |
| Distance error vs. discrete prediction | -0.022 m |
| Actual final speed | 0.00 m/s |
| Actual minimum normalized clearance | 1.974 |
| Minimum commanded acceleration | -3.00 m/s² |
| Input constraint violations | 0 |
| Safety-controller failures | 0 |

The forward-Euler stop preview now matches the simulation plant, leaving only `0.022 m` distance error. The feasibility threshold is `1.25`, so the checked-in blocked-road scenario remains clear while stopping. The fallback is deliberately best-effort rather than a false guarantee: if an obstacle is already inside the physical stopping envelope, `collision_avoidable` is set to `false` while maximum braking is still commanded. Unit tests cover the safe-stop, unavoidable-collision, direct-brake override, and online-replanning integration branches.

Fallback metrics and the planning/tracking plot are exported to `results/metrics/emergency_fallback.csv` and `results/figures/emergency_fallback.png`.

## Structure

```text
config.py                  experiment, vehicle, and MPC parameters
configs/                   schema-v1 reproducible experiment definitions
planning/reference_path.py arc-length road representation
planning/frenet.py         Frenet lane-change trajectory
planning/candidate_planner.py candidate feasibility and cost ranking
planning/friction_aware.py friction-limited speed/timing adaptation
planning/emergency.py      emergency-stop generation and fallback decision
models/vehicle.py          kinematic bicycle plant/prediction model
models/dynamic_vehicle.py  six-state nonlinear bicycle plant
control/mpc.py             constrained nonlinear MPC
control/stanley_pid.py     classical tracking baseline
control/emergency.py       direct maximum braking and lane holding
evaluation/metrics.py      metric calculation and CSV export
simulation.py              shared closed-loop experiment runner
benchmark.py               multi-scenario controller comparison
planner_demo.py            one-shot candidate planning and tracking
replanning_demo.py         online prediction, commitment, and replanning loop
robustness_benchmark.py    noise/delay/mismatch Monte Carlo experiments
dynamic_model_benchmark.py friction and dynamic-plant envelope tests
fallback_demo.py           fully blocked-road emergency-stop validation
sensitivity_benchmark.py    MPC horizon/weight trade-off experiment
main.py                    closed-loop simulation and plots
tests/                     unit and quantitative regression gates
.github/workflows/ci.yml   Python 3.12 compile and regression workflow
docs/PROJECT_REPORT.md     consolidated methodology and results report
docs/RESUME_INTERVIEW_GUIDE_CN.md resume bullets and interview preparation
```

## License

This project is licensed under the [MIT License](LICENSE).

## Current scenario

The ego vehicle starts with a deliberate `-0.35 m` lateral disturbance, tracks a `3.5 m` lane change over `4 s`, and continues along a curved road at approximately `10 m/s`. This is a baseline experiment, not yet evidence of real-world controller performance.

## Current completion status

The initial modeling, P0 safety corrections, P1 controller analysis/constraints, and P2 dynamic-model/configuration/regression work are complete. The suite currently contains 20 passing tests, including deterministic acceptance gates for the baseline controller, predicted speed constraints, per-axle friction circles, steering-rate correction, and friction-aware low-μ correction. GitHub Actions runs the same suite on every push and pull request.

The remaining project milestone is a final consolidated report and resume-ready result summary; hardware-oriented validation remains outside this software-model scope.
