# dbt project: GridPulse analytics marts

The Python pipeline (`gridpulse build`) materialises the gold star schema.
dbt sits on top of it and builds the analytics marts the dashboard and BI layer
consume, with tests and generated documentation.

## Run

```bash
cd dbt/gridpulse
dbt deps --profiles-dir .
dbt build --profiles-dir .        # run models + tests
dbt docs generate --profiles-dir .
dbt docs serve --profiles-dir .   # interactive lineage graph
```

## Models

| Model | Grain | Purpose |
|---|---|---|
| `stg_demand_hourly` | BA × hour | Cleaned hourly demand with degree days and day type |
| `stg_forecast_accuracy` | BA × hour | EIA's own forecast error per hour |
| `mart_daily_demand` | BA × day | Daily totals, peak, load factor, degree days |
| `mart_load_profile` | BA × season × day type × hour | Average load shape, normalised |
| `mart_forecast_scorecard` | BA × month | EIA accuracy and directional bias |
| `mart_temperature_response` | BA × day type × 2°C bin | Empirical demand/temperature curve |
| `mart_peak_events` | BA × top 25 days | The days capacity planning is built around |

Tests cover not-null and uniqueness on every grain, referential integrity to
`dim_ba`, accepted ranges on load factor, peak hour and percentage columns, and
accepted values on categorical fields.
