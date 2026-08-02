"""Forecast metric correctness."""

from __future__ import annotations

import numpy as np
import pytest

from gridpulse.models import metrics


def test_perfect_forecast_scores_zero_error():
    truth = np.array([100.0, 200.0, 300.0])
    assert metrics.mape(truth, truth) == pytest.approx(0.0)
    assert metrics.mae(truth, truth) == pytest.approx(0.0)
    assert metrics.rmse(truth, truth) == pytest.approx(0.0)
    assert metrics.r2(truth, truth) == pytest.approx(1.0)


def test_mape_is_computed_correctly():
    # A uniform 10 percent over-forecast must yield exactly 10 percent MAPE.
    truth = np.array([100.0, 200.0, 400.0])
    assert metrics.mape(truth, truth * 1.1) == pytest.approx(10.0)


def test_rmse_penalises_large_errors_more_than_mae():
    truth = np.array([100.0, 100.0, 100.0, 100.0])
    concentrated = np.array([100.0, 100.0, 100.0, 140.0])  # one big miss
    spread = np.array([110.0, 110.0, 110.0, 110.0])        # four small misses
    assert metrics.mae(truth, concentrated) == pytest.approx(metrics.mae(truth, spread))
    assert metrics.rmse(truth, concentrated) > metrics.rmse(truth, spread)


def test_non_finite_and_nonpositive_values_are_excluded():
    truth = np.array([100.0, np.nan, 0.0, 200.0])
    pred = np.array([110.0, 150.0, 50.0, 180.0])
    assert metrics.evaluate_forecast(truth, pred)["n_obs"] == 2


def test_skill_is_positive_when_model_beats_benchmark():
    assert metrics.skill_vs_benchmark(1.5, 2.0) == pytest.approx(25.0)


def test_skill_is_negative_when_model_loses():
    assert metrics.skill_vs_benchmark(2.5, 2.0) < 0


def test_skill_handles_zero_benchmark():
    assert np.isnan(metrics.skill_vs_benchmark(1.0, 0.0))


def test_coverage_matches_the_share_inside_the_interval():
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([5.0, 15.0, 35.0, 35.0])
    upper = np.array([15.0, 25.0, 38.0, 45.0])
    assert metrics.coverage(truth, lower, upper) == pytest.approx(75.0)


def test_pinball_loss_is_asymmetric():
    """A P90 forecast should be punished harder for being too low than too high."""
    truth = np.array([100.0])
    too_low = metrics.pinball_loss(truth, np.array([90.0]), 0.9)
    too_high = metrics.pinball_loss(truth, np.array([110.0]), 0.9)
    assert too_low > too_high
