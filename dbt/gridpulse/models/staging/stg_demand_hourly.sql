-- Staging view over the gold fact built by the Python pipeline.
-- Adds derived measures every downstream mart needs, so the logic lives once.

with source as (

    select * from {{ source('gridpulse', 'fact_demand_hourly') }}

),

enriched as (

    select
        period_utc,
        ba_code,
        date_local,
        hour_local,
        demand_mwh,
        demand_forecast_mwh,
        net_generation_mwh,
        total_interchange_mwh,
        temperature_2m,
        apparent_temperature,
        relative_humidity_2m,
        cloud_cover,
        wind_speed_10m,

        -- Heating and cooling degrees split the V-shaped demand/temperature
        -- response into two monotonic limbs.
        greatest({{ var('balance_point_c') }} - temperature_2m, 0) as heating_degrees,
        greatest(temperature_2m - {{ var('balance_point_c') }}, 0) as cooling_degrees,

        -- Net generation minus demand is the system's surplus position.
        net_generation_mwh - demand_mwh                             as generation_surplus_mwh,

        is_weekend,
        is_holiday,
        is_business_day,
        season,
        month,
        year,

        case
            when is_holiday then 'Holiday'
            when is_weekend then 'Weekend'
            else 'Weekday'
        end                                                          as day_type

    from source
    where demand_mwh is not null
      and demand_mwh > 0

)

select * from enriched
