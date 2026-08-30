# Research V2 Roadmap: Robust Friction-Adaptive CBF-NMPC

## Goal

Upgrade the current Frenet-MPC baseline into a research-oriented planning and control stack that improves safety and feasibility under low friction, actuator limits, model mismatch, and obstacle uncertainty while preserving the existing baseline for ablation.

Proposed method name (working title): **Robust Friction-Adaptive CBF-NMPC**.

## Design principle

Do not replace the current implementation. Treat the existing Frenet-MPC stack as the V1 baseline and add V2 components incrementally. Every new component must support at least one measurable research question, table, figure, or ablation.

## Research questions

1. How much does friction-aware trajectory selection expand the safe operating envelope versus the baseline Frenet-MPC stack?
2. Can CBF safety constraints reduce collision/lane-boundary violations without unacceptable loss of feasibility or comfort?
3. How robust is the controller to friction uncertainty, state noise, model mismatch, and actuation delay?
4. Does risk-aware candidate scoring improve tail-risk behavior beyond mean-cost optimization?
5. What is the runtime cost of each added safety/robustness layer?

## Milestones

### M0 — Baseline freeze and reproducibility
- Preserve current `main` behavior as the reference baseline.
- Record canonical baseline configs and expected benchmark metrics.
- Add deterministic seed handling where needed.
- Add regression tests that prevent accidental degradation of V1 behavior.

Exit criterion: baseline results are reproducible and traceable from configuration to CSV/JSON output.

### M1 — Friction-envelope constraints
- Add a physically meaningful combined longitudinal/lateral friction constraint.
- Support scenario-specific friction coefficient `mu`.
- Expose friction utilization in evaluation outputs.
- Add speed × friction × maneuver benchmark matrix.

Exit criterion: the benchmark can identify the baseline failure boundary and quantify changes in feasible operating region.

### M2 — CBF safety layer
- Introduce control-barrier constraints for obstacle clearance and lane boundaries.
- Keep the CBF layer modular so it can be disabled for ablation.
- Log safety-margin, intervention, infeasibility, and fallback metrics.

Exit criterion: baseline vs friction-aware vs friction+CBF can be compared with the same scenarios and metrics.

### M3 — Robust uncertainty handling
- Add uncertainty representation for friction, state estimation, model mismatch, and actuation delay.
- Begin with constraint tightening; consider tube/chance-constrained formulation only if justified by results.
- Add combined-disturbance robustness scenarios.

Exit criterion: the controller shows measurable robustness gains without hiding failure cases.

### M4 — Risk-aware candidate scoring
- Add tail-risk-aware candidate scoring (e.g. CVaR-style objective or explicit worst-case scenario sampling).
- Keep expected-cost and risk-aware scoring switchable for ablation.

Exit criterion: tail-risk metrics improve on selected uncertainty scenarios with quantified trade-offs in comfort/runtime.

### M5 — Realistic scenario benchmark
- Add CommonRoad-compatible scenario ingestion or a minimal adapter for selected CommonRoad scenarios.
- Preserve a small self-contained synthetic benchmark for CI.
- Separate real-scenario benchmark data from repository source if licensing/size requires it.

Exit criterion: the proposed method is evaluated on both controlled synthetic cases and externally sourced realistic scenarios.

### M6 — Publication-grade evaluation
- Run at least the following methods:
  1. V1 Frenet-MPC baseline
  2. + friction-aware selection/constraints
  3. + CBF
  4. + robust uncertainty handling
  5. full proposed method with risk-aware scoring
- Report lateral RMSE, maximum lateral error, heading RMSE, sideslip, yaw rate, lateral acceleration, friction utilization, constraint violations, collisions, infeasibility, fallback rate, mean/P95/max solve time, and deadline-miss rate.
- Produce operating-envelope plots over speed × friction.
- Produce ablation tables and failure-case analysis.

Exit criterion: all principal paper claims are backed by generated artifacts and can be reproduced from documented commands.

## Optional V3 — Learned residual dynamics

Only after V2 is stable:

`x_next = f_physics(x, u) + f_theta(x, u)`

Use a compact learned residual model to compensate physics-model error. Do not replace the physics model or safety constraints with an end-to-end learned controller. Treat this as a separate study/ablation rather than a required V2 dependency.

## Suggested implementation order

1. Baseline freeze and benchmark contract.
2. Friction-envelope formulation and metrics.
3. CBF abstraction and tests.
4. Robust constraint tightening.
5. Risk-aware scoring.
6. CommonRoad adapter.
7. Full benchmark matrix and publication figures.
8. Only then consider learned residual dynamics.

## Continuity / checkpoint policy

- Commit after every milestone or independently validated sub-milestone.
- Keep work on `research/robust-cbf-nmpc-v2`; do not merge into `main` until regression checks and the intended benchmark for that milestone pass.
- Do not rewrite or delete the existing V1 baseline results.
- Each milestone should leave the branch in a readable, resumable state with tests and a short update to this roadmap.
- If work is interrupted, resume by reading this file, the branch diff against `main`, and the latest CI/test results rather than relying on chat history.

## Publication stop rule

Do not add a feature unless it contributes to a research question, an ablation, a benchmark metric, or a publication figure/table. Once M6 is satisfied, stop adding architecture features and switch to experiment analysis and manuscript preparation.
