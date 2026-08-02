# Engineering log

Every non-trivial bug found while building GridPulse, what caused it, and how it
was fixed. Kept in the repository deliberately: the failures are more instructive
than the finished code, and several of them are the kind that pass silently.

---

## 1. Timezone crash when building the hourly spine

**Symptom.** `AssertionError: Inferred time zone not equal to passed time zone`
from deep inside pandas, on every warehouse build.

**Cause.** DuckDB's `.df()` returns timezone-aware timestamps. Passing those to
`pd.date_range` *together with* `tz="UTC"` trips a consistency assertion, because
pandas will not accept a timezone declaration alongside endpoints that already
carry one.

**Fix.** Normalise endpoints through a small `_as_utc()` helper instead of
re-declaring the timezone. `warehouse/build.py`.

---

## 2. Reported sample size did not match the scored sample

**Symptom.** A metrics test failed with `assert 3 == 2`.

**Cause.** `evaluate_forecast` counted every finite actual for `n_obs`, while the
metrics themselves were computed only over rows that were finite **and positive**.
Every number in the bundle described a different sample than the one advertised.

**Fix.** Derive `n_obs` from the same cleaning function the metrics use. The test
was right and the code was wrong. `models/metrics.py`.

---

## 3. A test that was wrong about its own data

**Symptom.** `test_hourly_spine_is_continuous` reported 12 gaps in a series that
was demonstrably continuous.

**Cause.** `date_diff('hour', ...)` is evaluated against the session timezone, so
daylight-saving transitions were counted as discontinuities in a UTC series that
had none.

**Fix.** Measure elapsed epoch seconds, which no calendar rule can distort. The
lesson: when a test fails, confirm which side is wrong before changing either.
`tests/test_warehouse.py`.

---

## 4. A CLI flag that swallowed its own subcommand

**Symptom.** `gridpulse --bas PJM ingest` failed with "the following arguments are
required: command".

**Cause.** `nargs="*"` is greedy. argparse consumed `ingest` as a second balancing
authority code and then found no subcommand.

**Fix.** Comma-separated values (`--bas PJM,ERCO`), which have unambiguous
boundaries. `cli.py`.

---

## 5. Rate limiting handled as though it were a transient error

**Symptom.** Weather ingestion completed for 8 of 12 balancing authorities, then
died with `HTTP 429` after five retries spanning about 14 seconds.

**Cause.** One retry policy for all failures. A 503 clears in seconds; a rate
limit does not. Backing off 2 seconds against a rate limiter is just a slower way
of being refused.

**Fix.** Treat 429 as its own class: honour the `Retry-After` header, otherwise
back off 20 to 90 seconds. Weather requests were also serialised with a 2-second
pause rather than fired concurrently, which removed the 429s at source.
`ingestion/http.py`, `ingestion/weather.py`.

---

## 6. Deep models that would have run for hours

**Symptom.** LSTM training at 4 to 6 minutes per epoch, implying well over an hour
per model on CPU.

**Cause.** Two compounding issues. Roughly 795,000 sliding windows where each
shares 167 of its 168 input hours with its neighbour, so nearly all of that
sampling was redundant. And a 168-step recurrent encoder, whose cost is linear in
sequence length and cannot be parallelised because step *t* depends on *t-1*.

**Fix.** Stride the training windows (keeping one in twelve), and subsample the
encoder input to every third hour, giving 56 steps instead of 168. Hourly demand
is heavily autocorrelated, so the discarded points carried little independent
information. Combined effect: about 13x faster, epochs down to roughly 20 seconds.
`models/deep.py`.

---

## 7. Forty corrupt readings that destroyed the model

**Symptom.** LightGBM early-stopped after **6 trees**. Final MAPE 53.9% with an R²
of -131. Feature importances ranked `doy_sin` and `cloud_cover` above
`demand_lag_24h`, which is nonsense for load forecasting.

**Diagnosis.** Rather than guessing, a diagnostic script dumped the fitted scaler
statistics:

```
PJM   mean 158,481 MW    std 10,739,790 MW    <- std is 10.7 million
TVA   mean  18,514 MW    std     56,844 MW    <- std is 3x the mean
```

PJM's true range is roughly 70,000 to 165,000 MW. Per-balancing-authority error
confirmed it: **PJM 583% MAPE, TVA 20%, and every other BA between 2.9% and 7.3%.**
Exactly the two systems with corrupted statistics. Median absolute percentage error
was 4.2% against a mean of 53.9%, and the worst 1% of rows carried 47.5% of all
error.

**Cause.** Out of 797,677 hourly readings, **40** were physically impossible. The
target was normalised per balancing authority using mean and standard deviation,
both of which have a breakdown point of zero: a single arbitrary value moves them
without limit. Those 40 rows corrupted the normalisation, which corrupted every
prediction for the affected systems.

**Fix.**
- Both target scalers moved to **median and IQR**, which outliers cannot move.
- Implausible readings excluded from modelling (outside 0.2x to 5x the BA median).
- Two new **critical** quality checks: magnitude plausibility, and a dispersion
  check that fails when any BA's standard deviation exceeds its mean.

**Result.** MAPE 53.9% to **3.68%**, R² -131 to **0.994**.

**The real lesson.** The quality suite had 13 checks and passed 13 of 13. It
verified that demand was not *below* zero, and never considered that it might be
absurdly *large*. A quality suite only tests the failures its author imagined.

---

## 8. Charts still showing spikes after the data was clean

**Symptom.** After despiking, the public dashboard still drew vertical excursions
in the final hours of each series.

**First attempt (wrong).** Flag a point as a spike when it deviates more than 25%
from *both* immediate neighbours in the same direction. It caught nothing.

**Diagnosis.** A second diagnostic dumped the last 60 hours with every flag:

```
2026-08-01 03:00   9,104 MW   +21.5% then -26.2%   flag_isolated_spike = False
2026-08-01 06:00   9,551 MW   +47.3%                flag_isolated_spike = False
```

Two independent failure modes. The first excursion moved 21.5% on one side, just
under the threshold. The second was the **final row of the series**, so it had no
"next" neighbour and the condition could never evaluate true. Neighbour comparison
is structurally blind at series boundaries, which is precisely where preliminary,
unsettled data lives.

**Fix.** Compare against a centred 5-hour **rolling median** instead. Aggregate
demand is smooth relative to that window, so a genuine load profile never departs
far from its local median while an isolated excursion departs sharply regardless
of which side it falls on, and partial windows at the boundary still resolve.
`warehouse/build.py`.

---

## 9. Rebuilding data silently destroyed model results

**Symptom.** After `gridpulse build --rebuild`, the app export dropped from 13.4 MB
to 6.3 MB with five tables missing.

**Cause.** The rebuild path dropped `model_scores`, `model_predictions` and
`anomaly_scores` alongside the tables it actually owned, discarding 24 minutes of
training because someone rebuilt the data layer.

**Fix.** Rebuild drops only what it owns, and warns that model outputs are now
stale rather than deleting them. `warehouse/build.py`.

---

## 10. Version conflicts and stale imports in the deployed app

Three smaller issues, each worth a line:

- **`statsmodels` 0.14.4 imports `scipy._lib._util._lazywhere`**, removed in scipy
  1.17. It was pulled in only by a `trendline="lowess"` call. Replaced with a
  binned median computed in pandas, which needs no extra dependency, is robust to
  outliers, and reads more clearly. `statsmodels` moved out of the app requirements
  entirely.
- **`SAMPLE_QUESTIONS` was defined but never re-exported** from the agent package's
  `__init__.py`. Fixed, and every other package `__init__` was swept for the same
  omission.
- **The app export used a hand-written column list** that silently omitted two
  weather columns the feature builder required. The failure only surfaced at
  inference time. The list is now derived from `WEATHER_VARIABLES` so it cannot
  drift.

---

## 11. A chart colour argument that broke sort order

**Symptom.** On the model leaderboard, the EIA benchmark bar rendered at the top of
the chart instead of in its correct rank. Every other bar was ordered correctly.

**Cause.** Plotly Express's `color=` argument splits data into one trace per colour
group. Highlighting a single bar therefore pulled it into a separate trace, which
was drawn independently of the sort.

**Fix.** A single `go.Bar` trace with per-bar `marker_color`, plus an explicit
`categoryarray` on the axis. `app.py`.

---

## Themes

**Robust statistics are not optional on real data.** Mean and standard deviation
have a breakdown point of zero. Median and IQR do not. Forty rows in 797,677 were
enough to make the difference between a 53.9% and a 3.7% MAPE.

**A quality suite only tests the failures its author imagined.** Thirteen checks
passed while the data was corrupt enough to destroy the model. The gap was not
subtle in hindsight, and that is exactly the point.

**Diagnose before fixing.** Three bugs here were fixed on the first attempt because
a script dumped the actual values first. The one bug that took two attempts was the
one where a fix was guessed at instead.

**Boundaries are where data is worst and logic is weakest.** The final rows of a
series carry preliminary data, and neighbour-based logic cannot classify them at
all. Both problems met in the same place.
