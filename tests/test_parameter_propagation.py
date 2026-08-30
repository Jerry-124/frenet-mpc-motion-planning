import numpy as np

from dynamic_model_benchmark import DynamicScenario, run_dynamic_trial


def test_kinematic_lane_change_duration_changes_reference_and_states() -> None:
    """Changing the scenario duration must change the kinematic benchmark itself.

    Regression guard for the bug where ``lane_change_duration_s`` was copied
    into result metadata but never propagated to ``SimulationConfig``. In that
    failure mode 2.5 s and 6.0 s requests produced byte-for-byte equivalent
    reference trajectories and closed-loop states.
    """
    short = DynamicScenario(
        name="regression_short",
        speed_mps=12.0,
        friction_coefficient=1.0,
        plant="kinematic",
        lane_change_duration_s=2.5,
    )
    long = DynamicScenario(
        name="regression_long",
        speed_mps=12.0,
        friction_coefficient=1.0,
        plant="kinematic",
        lane_change_duration_s=6.0,
    )

    short_metrics, short_states, short_reference = run_dynamic_trial(short)
    long_metrics, long_states, long_reference = run_dynamic_trial(long)

    assert short_metrics["selected_lane_change_duration_s"] == 2.5
    assert long_metrics["selected_lane_change_duration_s"] == 6.0
    assert short_reference.shape == long_reference.shape
    assert short_states.shape == long_states.shape
    assert not np.allclose(short_reference, long_reference)
    assert not np.allclose(short_states, long_states)
    assert float(np.max(np.abs(short_reference - long_reference))) > 1e-4
    assert float(np.max(np.abs(short_states - long_states))) > 1e-4
