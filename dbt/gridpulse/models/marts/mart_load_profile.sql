-- Average hourly load shape by balancing authority, season and day type.
-- Normalising each profile by its own mean makes shapes comparable across
-- systems that differ by an order of magnitude in absolute size.

with hourly as (

    select * from {{ ref('stg_demand_hourly') }}

),

profile as (

    select
        ba_code,
        season,
        day_type,
        hour_local,
        round(avg(demand_mwh))                                as mean_demand_mw,
        round(quantile_cont(demand_mwh, 0.1))                 as p10_demand_mw,
        round(quantile_cont(demand_mwh, 0.9))                 as p90_demand_mw,
        count(*)                                              as observations
    from hourly
    group by ba_code, season, day_type, hour_local

)

select
    p.*,
    round(
        p.mean_demand_mw
        / nullif(avg(p.mean_demand_mw) over (partition by p.ba_code, p.season, p.day_type), 0),
        4
    ) as shape_index
from profile p
