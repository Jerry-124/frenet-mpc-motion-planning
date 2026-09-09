# Changelog

Notable stable releases are documented below. Historical results are preserved as they were reported for each release; later corrections are not retroactively substituted into earlier entries.

## [1.1.1] - 2026-09-08

### Summary

Scientific and physical-fidelity patch for the six-state dynamic bicycle plant used in model-mismatch evaluation.

### Changes

- Corrected front-wheel longitudinal/lateral force rotation into the vehicle body frame.
- Restored the standard body-frame longitudinal and lateral velocity-coupling terms.
- Regenerated the complete dynamic-model benchmark and figures using the corrected plant.
- Added five physics-invariant tests covering force-free straight motion, body-frame velocity coupling, steering-force projection, left/right steering symmetry, and pure longitudinal acceleration.

### Validation

- Automated suite expanded from 27 to 32 tests.
- The corrected plant preserves the main intervention conclusions of the benchmark.
- No controller retuning was required after the physics correction.

### Scope

This patch improves physical fidelity without expanding the stable project scope.

## [1.1.0] - 2026-09-07

### Summary

Reliability and validation hardening for the stable Frenet-NMPC baseline.

### Changes

- Added deterministic bounded fallback control for solver failure or non-finite NMPC output.
- Added physical validation for vehicle, simulation, road, NMPC, dynamic-plant, steering-rate, and integration parameters.
- Added regression coverage for fallback and invalid configurations.
- Updated GitHub Actions to Node 24-compatible `actions/checkout@v5` and `actions/setup-python@v6`.
- Scoped CI triggers to `main` and pull requests targeting `main`, with redundant in-progress runs cancelled.

### Validation

- Automated suite expanded from 23 to 27 tests.
- CI validates Python 3.10 and 3.12 with dependency checks, source compilation, pytest, Ruff linting, and Ruff formatting checks.

### Scope

The release hardens the existing baseline; research-only CBF and robust-NMPC extensions remain isolated from `main`.

## [1.0.0] - 2026-09-07

### Summary

First stable project-level baseline for Frenet-frame motion planning and nonlinear MPC trajectory tracking.

### Changes

- Frenet-frame candidate trajectory generation and Frenet-to-Cartesian transformation.
- Moving-obstacle prediction and receding-horizon replanning.
- Direct-shooting NMPC with explicit speed, acceleration, steering-angle, and steering-rate constraints.
- Stanley lateral control with PID speed control as a classical baseline.
- Six-state dynamic bicycle plant for model-mismatch evaluation.
- Robustness benchmarks covering sensor noise, 200 ms actuator delay, parameter mismatch, and low-friction operation.
- Delay compensation, friction-aware planning, emergency fallback, reproducible experiment configuration, metrics, figures, and CI.

### Validation

- 23 automated pytest tests with Python 3.10 and 3.12 validation.
- Five-scenario NMPC mean lateral RMSE: approximately 0.064 m.
- Mean lateral RMSE approximately 73% lower than Stanley + PID in the configured benchmark.
- In the 200 ms delay benchmark, compensation reduced lateral RMSE from 2.202 m to 0.084 m.

### Scope

`v1.0.0` was restored retrospectively at the original stable V1 commit to preserve release history. At that historical commit, Python package metadata still used the earlier `0.1.0` development identifier; project/package semantic versioning was unified in `v1.1.0`.