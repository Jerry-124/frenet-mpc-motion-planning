# Changelog

All notable portfolio releases are documented here. The `v1.0.0` entry records the first stable project-level portfolio baseline at its historical commit; package-level semantic versioning was unified with the project release line in `v1.1.0`.

## [1.1.0] - 2026-09-07

### Added
- Deterministic bounded fallback control when the NMPC solver fails or returns a non-finite solution.
- Physical validation for vehicle, simulation, road, MPC, dynamic-plant, steering-rate, and integration parameters.
- Regression coverage for solver-failure fallback and invalid configuration inputs.
- Repository metadata linking the Python package to its GitHub source.

### Changed
- Expanded the automated suite from 23 to 27 tests.
- CI now validates Python 3.10 and 3.12 with dependency checks, bytecode compilation, pytest, and Ruff.
- GitHub Actions uses current Node 24-compatible `actions/checkout@v5` and `actions/setup-python@v6`.
- CI triggers are scoped to `main` and pull requests targeting `main`, with redundant in-progress runs cancelled.

### Scope
- This release hardens and freezes the portfolio-oriented V1 baseline. Research-only CBF/robust-NMPC extensions remain isolated from `main`.

## [1.0.0] - 2026-09-07

### Stable Portfolio Baseline
- Frenet-frame candidate trajectory generation and Frenet-to-Cartesian transformation.
- Moving-obstacle prediction and receding-horizon replanning.
- Direct-shooting NMPC using SciPy SLSQP with explicit speed, acceleration, steering-angle, and steering-rate constraints.
- Stanley lateral control with PID speed control as a classical baseline.
- Six-state dynamic bicycle plant with nonlinear tire forces and per-axle friction circles for model-mismatch evaluation.
- Robustness benchmarks covering sensor noise, 200 ms actuator delay, parameter mismatch, and low-friction operation.
- Delay compensation, friction-aware planning, and no-feasible-candidate emergency fallback.
- Reproducible experiment configurations, quantitative metrics, figures, regression gates, and CI.
- 23 automated pytest tests with Python 3.10/3.12 validation.

### Baseline Results
- Five-scenario NMPC mean lateral RMSE: approximately 0.064 m.
- Mean lateral RMSE approximately 73% lower than Stanley + PID in the configured benchmark.
- In the 200 ms delay benchmark, compensation reduced lateral RMSE from 2.202 m to 0.084 m.

### Version History Note
- `v1.0.0` was restored retrospectively at the original stable V1 baseline commit to preserve the project's release history.
- At that historical commit, Python package metadata still used the earlier `0.1.0` development identifier; project/package semantic versioning was unified in `v1.1.0`.
