# The dbt part of GridPulse

The Python pipeline (`gridpulse build`) creates the gold star schema. dbt then sits
on top of that and builds the summary tables the dashboard uses, along with tests
and documentation it generates itself.

## Running it

```bash
cd dbt/gridpulse
dbt deps --profiles-dir .
dbt build --profiles-dir .        # build the models and run the tests
dbt docs generate --profiles-dir .
dbt docs serve --profiles-dir .   # opens a browsable diagram of how everything connects
```

## The models

| Model | One row per | What it gives you |
|---|---|---|
| `stg_demand_hourly` | region and hour | Cleaned hourly demand, plus degree days and what kind of day it was |
| `stg_forecast_accuracy` | region and hour | How far off EIA's own forecast was, hour by hour |
| `mart_daily_demand` | region and day | Daily total, daily peak, load factor and degree days |
| `mart_load_profile` | region, season, day type and hour | The average shape of a day, scaled so regions can be compared |
| `mart_forecast_scorecard` | region and month | How accurate EIA is, and whether they tend to guess high or low |
| `mart_temperature_response` | region, day type and 2°C band | The actual measured curve between temperature and demand |
| `mart_peak_events` | region, top 25 days | The busiest days, which are the ones capacity planning is based on |

The tests check that no key columns are null and that there is only one row per
key, that every region code matches a real region in `dim_ba`, that things like
load factor and peak hour fall in sensible ranges, and that the text columns only
contain values I expect.
