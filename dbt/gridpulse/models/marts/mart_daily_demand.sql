-- Daily demand aggregate per balancing authority.
-- Load factor (mean / peak) is the headline efficiency metric utilities track:
-- a low load factor means expensive peaking capacity sits idle most of the day.

with hourly as (

    select * from {{ ref('stg_demand_hourly') }}

)

select
    ba_code,
    date_local,
    day_type,
    season,

    count(*)                                        as hours_reported,
    round(sum(demand_mwh))                          as total_demand_mwh,
    round(avg(demand_mwh))                          as mean_demand_mw,
    round(max(demand_mwh))                          as peak_demand_mw,
    round(min(demand_mwh))                          as trough_demand_mw,
    round(max(demand_mwh) - min(demand_mwh))        as daily_swing_mw,
    round(avg(demand_mwh) / nullif(max(demand_mwh), 0), 4) as load_factor,

    round(avg(temperature_2m), 2)                   as mean_temp_c,
    round(max(temperature_2m), 2)                   as max_temp_c,
    round(min(temperature_2m), 2)                   as min_temp_c,
    round(sum(heating_degrees) / 24, 2)             as heating_degree_days,
    round(sum(cooling_degrees) / 24, 2)             as cooling_degree_days,

    -- Hour of the local day at which the system peaked.
    arg_max(hour_local, demand_mwh)                 as peak_hour_local

from hourly
group by ba_code, date_local, day_type, season
