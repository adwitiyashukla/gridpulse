"""Configuration and the balancing-authority registry."""

from __future__ import annotations

import pytest

from gridpulse.config import (
    BALANCING_AUTHORITIES,
    EIA_MEASURES,
    WEATHER_VARIABLES,
    active_bas,
)


def test_registry_is_populated():
    assert len(BALANCING_AUTHORITIES) >= 10


def test_every_ba_has_plausible_coordinates():
    """Load centres must sit inside the continental US bounding box."""
    for code, ba in BALANCING_AUTHORITIES.items():
        assert 24 <= ba.latitude <= 50, f"{code} latitude out of range"
        assert -125 <= ba.longitude <= -66, f"{code} longitude out of range"
        assert ba.timezone.startswith("America/"), f"{code} has a non-US timezone"


def test_ba_codes_are_self_consistent():
    for code, ba in BALANCING_AUTHORITIES.items():
        assert code == ba.code
        assert ba.slug == code.lower()


def test_active_bas_respects_env(monkeypatch):
    monkeypatch.setenv("GRIDPULSE_BAS", "PJM,ERCO")
    codes = [ba.code for ba in active_bas()]
    assert codes == ["PJM", "ERCO"]


def test_unknown_ba_is_rejected(monkeypatch):
    monkeypatch.setenv("GRIDPULSE_BAS", "PJM,NOT_A_REAL_BA")
    with pytest.raises(ValueError, match="Unknown balancing authority"):
        active_bas()


def test_benchmark_measure_is_present():
    """DF is the EIA's own forecast and the benchmark the project is built around."""
    assert "DF" in EIA_MEASURES
    assert "D" in EIA_MEASURES


def test_weather_variables_include_temperature():
    assert "temperature_2m" in WEATHER_VARIABLES
