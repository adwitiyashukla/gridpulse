-- Staging view over EIA's own day-ahead forecast error.

select
    period_utc,
    ba_code,
    date_local,
    hour_local,
    actual_mwh,
    eia_forecast_mwh,
    error_mwh,
    abs_error_mwh,
    abs_pct_error,
    bias_direction

from {{ source('gridpulse', 'fact_forecast_accuracy') }}
where actual_mwh > 0
