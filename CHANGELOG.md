# Changelog

All notable portfolio releases are documented here.

## [1.1.0] - 2026-09-07

### Added
- Deterministic bounded fallback control when the NMPC solver fails or returns a non-finite solution.
- Physical validation for vehicle, simulation, road, MPC, dynamic-plant, steering-rate, and integration parameters.
- Regression coverage for solver-failure fallback and invalid configuration inputs.
- Repository metadata linking the Python package to its GitHub source.

### Changed
- Expanded the automated suite to 27 tests.
- CI now validates Python 3.10 and 3.12 with dependency checks, bytecode compilation, pytest, and Ruff.
- GitHub Actions uses current Node 24-compatible `actions/checkout@v5` and `actions/setup-python@v6`.
- CI triggers are scoped to `main` and pull requests targeting `main`, with redundant in-progress runs cancelled.

### Scope
- This release freezes the portfolio-oriented V1 baseline. Research-only CBF/robust-NMPC extensions remain isolated from `main`.
