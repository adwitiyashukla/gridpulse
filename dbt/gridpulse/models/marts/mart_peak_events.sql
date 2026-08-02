-- The days each system came closest to its all-time peak.
-- These are the days capacity planning is built around, and the days a forecast
-- miss is most expensive.

with daily as (

    select * from {{ ref('mart_daily_demand') }}

),

ranked as (

    select
        *,
        max(peak_demand_mw) over (partition by ba_code)                     as all_time_peak_mw,
        row_number() over (partition by ba_code order by peak_demand_mw desc) as peak_rank
    from daily

)

select
    ba_code,
    date_local,
    day_type,
    season,
    peak_demand_mw,
    peak_hour_local,
    max_temp_c,
    min_temp_c,
    load_factor,
    peak_rank,
    round(100.0 * peak_demand_mw / nullif(all_time_peak_mw, 0), 2) as pct_of_all_time_peak

from ranked
where peak_rank <= 25
