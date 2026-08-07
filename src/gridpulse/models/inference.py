"""Makes a real forward-looking 24-hour forecast for the live app.

Falls back to replaying the most recent stored day when the network is down, so
the public site still shows something instead of an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gridpulse.config import BALANCING_AUTHORITIES, FORECAST_HORIZON, PATHS
from gridpulse.features.build import build_features

logger = logging.getLogger(__name__)

HISTORY_HOURS = 400


@dataclass
class Forecast:
    ba_code: str
    generated_at_utc: datetime
    mode: str
    frame: pd.DataFrame
    model: str = "gbm"
    notes: list[str] = None

    def to_records(self) -> list[dict]:
        out = self.frame.copy()
        out["period_utc"] = out["period_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return out.to_dict(orient="records")


def _app_database():
    slim = PATHS.gold / "gridpulse_app.duckdb"
    return slim if slim.exists() else PATHS.duckdb


def _load_recent(ba_code: str, hours: int = HISTORY_HOURS) -> pd.DataFrame:
    """Recent observed history for one BA from whichever database is present."""
    from gridpulse.warehouse.duck import connect

    with connect(_app_database(), read_only=True) as con:
        return con.execute(
            """
            SELECT period_utc, ba_code, demand_mwh, demand_forecast_mwh,
                   temperature_2m, apparent_temperature, relative_humidity_2m,
                   dew_point_2m, cloud_cover, wind_speed_10m, shortwave_radiation
            FROM fact_demand_hourly
            WHERE ba_code = ?
            ORDER BY period_utc DESC
            LIMIT ?
            """,
            [ba_code, hours],
        ).df().sort_values("period_utc")


def _fetch_future_weather(ba_code: str, hours: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Hourly weather forecast for the BA's load centre."""
    import httpx

    from gridpulse.config import WEATHER_VARIABLES

    ba = BALANCING_AUTHORITIES[ba_code]
    response = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": ba.latitude,
            "longitude": ba.longitude,
            "hourly": ",".join(WEATHER_VARIABLES),
            "forecast_days": 3,
            "past_days": 2,
            "timezone": "UTC",
        },
        timeout=20.0,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]

    frame = pd.DataFrame({"period_utc": pd.to_datetime(hourly["time"], utc=True)})
    for variable in WEATHER_VARIABLES:
        frame[variable] = pd.to_numeric(pd.Series(hourly.get(variable)), errors="coerce")
    return frame


def _calendar_columns(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Derive the local-time calendar attributes for rows that have none yet."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    local = frame["period_utc"].dt.tz_convert(timezone_name)
    frame["date_local"] = local.dt.date
    frame["hour_local"] = local.dt.hour
    frame["day_of_week"] = local.dt.dayofweek
    frame["month"] = local.dt.month
    frame["year"] = local.dt.year

    days = pd.to_datetime(frame["date_local"])
    holidays = set(USFederalHolidayCalendar().holidays(start=days.min(), end=days.max() + pd.Timedelta(days=1)))
    frame["is_holiday"] = days.isin(holidays)
    frame["is_weekend"] = frame["day_of_week"] >= 5
    frame["is_business_day"] = ~(frame["is_weekend"] | frame["is_holiday"])
    frame["is_day_before_holiday"] = days.isin({h - pd.Timedelta(days=1) for h in holidays})
    frame["is_day_after_holiday"] = days.isin({h + pd.Timedelta(days=1) for h in holidays})
    return frame


def forecast(ba_code: str, horizon: int = FORECAST_HORIZON, allow_network: bool = True) -> Forecast:
    """Produce a 24-hour-ahead demand forecast with P10/P90 bands."""
    ba_code = ba_code.upper()
    if ba_code not in BALANCING_AUTHORITIES:
        raise ValueError(f"Unknown balancing authority '{ba_code}'.")

    notes: list[str] = []
    history = _load_recent(ba_code)
    if history.empty:
        raise RuntimeError(f"No stored history for {ba_code}. Run the pipeline first.")

    history["period_utc"] = pd.to_datetime(history["period_utc"], utc=True)
    last_observed = history["period_utc"].max()

    mode = "replay"
    future = pd.DataFrame()

    if allow_network:
        try:
            weather = _fetch_future_weather(ba_code)
            future_start = last_observed + pd.Timedelta(hours=1)
            future = weather[weather["period_utc"] >= future_start].head(horizon).copy()
            if len(future) >= horizon // 2:
                mode = "live"
                gap_hours = (future["period_utc"].min() - last_observed).total_seconds() / 3600
                if gap_hours > 6:
                    notes.append(
                        f"Stored data ends {last_observed:%Y-%m-%d %H:%M} UTC, "
                        f"{gap_hours:.0f}h behind. Forecast bridges the gap."
                    )
            else:
                future = pd.DataFrame()
                notes.append("Weather forecast did not extend past stored history; using replay mode.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live weather fetch failed (%s); falling back to replay", exc)
            notes.append("Live weather unavailable; showing a replay of the most recent 24 hours.")

    if mode == "live":
        future["ba_code"] = ba_code
        future["demand_mwh"] = np.nan
        future["demand_forecast_mwh"] = np.nan
        combined = pd.concat([history, future], ignore_index=True)
        target_periods = future["period_utc"]
    else:
        combined = history.copy()
        target_periods = combined["period_utc"].tail(horizon)
        combined.loc[combined["period_utc"].isin(target_periods), "demand_mwh"] = np.nan

    combined = combined.sort_values("period_utc").reset_index(drop=True)

    combined = _calendar_columns(combined, BALANCING_AUTHORITIES[ba_code].timezone)

    featured = build_features(frame=combined, dropna_target=False)
    horizon_rows = featured[featured["period_utc"].isin(target_periods)].copy()
    if horizon_rows.empty:
        raise RuntimeError(
            "Could not build features for the forecast horizon. "
            "There is likely insufficient contiguous recent history."
        )

    from gridpulse.models.gbm import TrainedGBM

    model = TrainedGBM.load()
    predicted = model.predict(horizon_rows)

    out = pd.DataFrame({
        "period_utc": horizon_rows["period_utc"].to_numpy(),
        "forecast_mwh": predicted["pred_gbm"].to_numpy().round(1),
    })
    if "pred_gbm_p10" in predicted:
        out["p10_mwh"] = predicted["pred_gbm_p10"].to_numpy().round(1)
        out["p90_mwh"] = predicted["pred_gbm_p90"].to_numpy().round(1)

    if mode == "replay":
        actual = history.set_index("period_utc")["demand_mwh"]
        out["actual_mwh"] = out["period_utc"].map(actual).round(1)

    return Forecast(
        ba_code=ba_code,
        generated_at_utc=datetime.now(timezone.utc),
        mode=mode,
        frame=out.sort_values("period_utc").reset_index(drop=True),
        notes=notes,
    )


def artifacts_available() -> bool:
    return (PATHS.artifacts / "gbm" / "meta.json").exists()
