# Frenet-Frame Motion Planning and Nonlinear MPC

[![Version](https://img.shields.io/badge/version-v1.1.1-blueviolet)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#quick-start)
[![CI](https://github.com/Jerry-124/frenet-mpc-motion-planning/actions/workflows/ci.yml/badge.svg)](https://github.com/Jerry-124/frenet-mpc-motion-planning/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-32-brightgreen)](#verification)

A reproducible autonomous-driving software project for Frenet-frame local planning and constrained trajectory tracking with nonlinear model predictive control (NMPC). The repository combines planning, control, robustness evaluation, dynamic-plant model mismatch, actuator constraints, friction-aware adaptation, and emergency fallback in one deterministic benchmark suite.

## Highlights

- Frenet-to-Cartesian trajectory generation with minimum-jerk quintic lane changes.
- Multi-candidate local planning with moving-obstacle prediction and hard collision rejection.
- Direct-shooting NMPC using SciPy SLSQP with explicit speed, acceleration, steering-angle, and steering-rate constraints.
- Stanley lateral control with PID speed control as a classical baseline.
- Receding-horizon replanning with lane-change commitment logic.
- Deterministic robustness benchmarks for measurement noise, actuator delay, wheelbase mismatch, and combined disturbances.
- Six-state dynamic bicycle plant with corrected body-frame force coupling, steering-force projection, nonlinear tire saturation, and per-axle friction circles.
- Steering-rate-aware NMPC and friction-aware speed/timing adaptation for model-mismatch stress cases.
- Independent maximum-braking fallback when no ordinary candidate is feasible.
- Schema-versioned JSON experiment configuration, committed metrics and figures, regression gates, and CI.

NMPC intentionally retains a lightweight rear-axle kinematic prediction model. A separate dynamic bicycle plant is used to expose the operating envelope and quantify model mismatch rather than hiding it inside the controller model.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the main experiments:

```bash
python main.py
python benchmark.py
python planner_demo.py
python replanning_demo.py
python robustness_benchmark.py
python dynamic_model_benchmark.py
python fallback_demo.py
python sensitivity_benchmark.py
```

Run verification:

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

Experiment outputs are written to `results/metrics/` and `results/figures/`.

## Experiment Configuration

All experiment inputs are stored under `configs/` with `schema_version: 1`. `default.json` defines the shared vehicle, simulation, road, and NMPC settings; experiment-specific files contain only scenario definitions or overrides.

| Configuration | Purpose |
|---|---|
| `default.json` | Shared vehicle, road, simulation, and NMPC parameters |
| `benchmark.json` | NMPC versus Stanley+PID comparison |
| `planner_demo.json` | Frenet candidate generation and obstacle scoring |
| `replanning_demo.json` | Closed-loop online replanning |
| `replanning_blocked_validation.json` | Repeated no-feasible-candidate fallback validation |
| `robustness_benchmark.json` | Noise, delay, mismatch, and combined robustness cases |
| `dynamic_model_benchmark.json` | Dynamic bicycle and friction-envelope stress cases |
| `fallback_demo.json` | Fully blocked-road emergency stopping case |
| `sensitivity_benchmark.json` | NMPC horizon and weight sweeps |

## Verification

The current suite contains **32 pytest tests**. GitHub Actions validates Python 3.10 and 3.12 and runs dependency checks, source compilation, pytest, Ruff linting, and Ruff formatting checks.

Regression coverage includes controller constraints, solver-failure fallback, configuration validation, friction-circle enforcement, steering-rate handling, corrected body-frame dynamic-bicycle equations, force projection, left/right steering symmetry, and pure longitudinal acceleration.

## Key Results

### Matched-Model Baseline

The deterministic matched-kinematic benchmark uses a `0.1 s` sample time and an 8-step NMPC horizon.

| Metric | Result |
|---|---:|
| Lateral RMSE | 0.056 m |
| Longitudinal RMSE | 0.079 m |
| Position RMSE | 0.097 m |
| Heading RMSE | 1.154 deg |
| Speed RMSE | 0.051 m/s |
| Constraint violations | 0 |
| Solver failures | 0 |

These are software-model results, not real-vehicle validation. Model mismatch, sensor noise, and actuator delay are evaluated explicitly in the robustness and dynamic-plant benchmarks below.

### Controller Comparison

Across five configured speed/curvature scenarios:

| Controller | Mean Lateral RMSE | Constraint Violations |
|---|---:|---:|
| NMPC | 0.064 m | 0 |
| Stanley+PID | 0.234 m | 0 |

NMPC reduces mean lateral RMSE by approximately **73%** in this benchmark. The comparison is scenario-specific and is not a general claim that NMPC dominates classical control in all operating conditions.

### Robustness and Delay Compensation

A `200 ms` uncompensated actuator delay increases mean lateral RMSE to `2.202 m`. Delay compensation reduces it to `0.084 m`, approximately a **96.2%** reduction. The combined disturbance case falls from `2.202 m` to `0.115 m` after compensation.

All run-level results are retained under `results/metrics/` so failures and corrections remain auditable.

### Dynamic-Plant Model Mismatch

The corrected six-state dynamic bicycle plant separates ordinary tire-model mismatch from two more severe failure mechanisms:

| Scenario | Lateral RMSE | Max Sideslip | Result |
|---|---:|---:|---|
| Dynamic dry/wet/low-μ operating cases | 0.037–0.067 m | ≤3.54° | Pass |
| Aggressive dry, 16 m/s | 0.146 m | 9.09° | Fail |
| Steering-rate limited, 12 m/s | 10.054 m | 5.93° | Fail |
| Steering-rate-aware NMPC | 0.061 m | 1.80° | Pass |
| Aggressive low-μ | 10.116 m | 89.51° | Fail |
| Friction-aware low-μ planning | 0.054 m | 1.71° | Pass |

Making NMPC aware of the physical steering-rate state reduces lateral RMSE by approximately **99.4%** in the configured steering-rate stress case. Friction-aware speed/timing adaptation reduces the aggressive low-μ case by approximately **99.5%** while respecting the configured combined-acceleration envelope.

Stress-case failures are intentionally retained as before/after evidence rather than removed from the benchmark.

### Emergency Fallback

When both candidate lanes are blocked, ordinary planning returns no feasible candidate and the controller transitions to an independent maximum-braking fallback. In the checked-in scenario, the vehicle stops with zero constraint violations and zero safety-controller failures; the discrete stop-preview distance differs from the simulated travel distance by only `0.022 m`.

The fallback is best-effort rather than a formal safety guarantee. If an obstacle is already inside the physical stopping envelope, the implementation records the collision as unavoidable while continuing to command maximum braking.

## Reproducibility

- Deterministic experiment configuration is versioned under `configs/`.
- Quantitative outputs are committed under `results/metrics/`.
- Figures are committed under `results/figures/`.
- Acceptance criteria are exercised by the automated regression suite.
- Release history is documented in [`CHANGELOG.md`](CHANGELOG.md).

## Repository Structure

```text
frenet-mpc-motion-planning/
|-- README.md
|-- CHANGELOG.md
|-- pyproject.toml
|-- configs/                    # Versioned experiment inputs
|-- control/                    # NMPC and classical controllers
|-- models/                     # Kinematic and dynamic vehicle models
|-- planning/                   # Frenet planning and fallback logic
|-- evaluation/                 # Metrics and regression helpers
|-- tests/                      # 32 automated tests
|-- results/
|   |-- metrics/                # Reproducible numerical outputs
|   `-- figures/                # Reproducible plots
|-- main.py
|-- benchmark.py
|-- planner_demo.py
|-- replanning_demo.py
|-- robustness_benchmark.py
|-- dynamic_model_benchmark.py
|-- fallback_demo.py
`-- sensitivity_benchmark.py
```

## Scope

This repository is a reproducible research and algorithm-development project. It is not a production autonomous-driving stack, a certified safety component, or evidence of real-vehicle performance.

The current scope covers software-model planning and control, robustness evaluation, physical-model mismatch, actuator and friction constraints, and deterministic fallback behavior. Hardware-in-the-loop testing, perception uncertainty, real-vehicle integration, and formal safety certification remain outside the current baseline.

Research extensions are kept separate from the stable `v1.1.1` baseline on `research/robust-cbf-nmpc-v2`.

## License

This project is licensed under the [MIT License](LICENSE).