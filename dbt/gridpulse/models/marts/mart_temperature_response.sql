-- Binned demand response to temperature: the empirical V-curve.
-- Demand is indexed against each BA's own median so curves from systems of
-- wildly different size can be plotted on one axis.

with hourly as (

    select * from {{ ref('stg_demand_hourly') }}
    where temperature_2m is not null

),

baseline as (

    select ba_code, median(demand_mwh) as median_demand_mw
    from hourly group by ba_code

)

select
    h.ba_code,
    h.day_type,
    cast(floor(h.temperature_2m / 2) * 2 as integer)  as temp_bin_c,

    count(*)                                          as observations,
    round(avg(h.demand_mwh))                          as mean_demand_mw,
    round(avg(h.demand_mwh) / nullif(b.median_demand_mw, 0), 4) as demand_index,
    round(stddev_samp(h.demand_mwh))                  as demand_stddev_mw

from hourly h
join baseline b using (ba_code)
group by h.ba_code, h.day_type, temp_bin_c, b.median_demand_mw
having count(*) >= 10
