-- Monthly scorecard of the EIA's own day-ahead forecast accuracy.
-- Bias is reported separately from magnitude: a forecaster that is consistently
-- 2 percent high is a different problem from one that is randomly 2 percent off.

with accuracy as (

    select * from {{ ref('stg_forecast_accuracy') }}

)

select
    ba_code,
    date_trunc('month', date_local)                  as month_local,

    count(*)                                         as hours_scored,
    round(avg(abs_pct_error), 3)                     as eia_mape_pct,
    round(quantile_cont(abs_pct_error, 0.5), 3)      as eia_median_ape_pct,
    round(quantile_cont(abs_pct_error, 0.95), 3)     as eia_p95_ape_pct,
    round(avg(abs_error_mwh))                        as eia_mae_mw,
    round(sqrt(avg(error_mwh * error_mwh)))          as eia_rmse_mw,

    -- Positive mean error means EIA systematically over-forecast.
    round(avg(error_mwh))                            as mean_bias_mw,
    round(100.0 * count(*) filter (where bias_direction = 'over') / count(*), 1)
                                                     as over_forecast_pct

from accuracy
group by ba_code, month_local
