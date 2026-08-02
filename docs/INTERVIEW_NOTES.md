# Interview notes

Questions an interviewer is likely to ask about this project, and the honest
answers. Skim before a conversation.

---

**Why DuckDB instead of Postgres, Snowflake or Spark?**

The workload is analytical, single-writer and embarrassingly columnar: scan a few
hundred million hourly rows, aggregate, join two small dimensions. That is exactly
DuckDB's shape, and it removes an entire class of operational work - no server, no
connection pool, no container to keep alive. The SQL is standard enough to port to
Snowflake or Synapse without rewriting. Spark would be the right answer at a data
volume this project does not have; choosing it here would be resume-driven design.

**Why one global model instead of one per balancing authority?**

BAs share physics. The demand response to temperature in Atlanta genuinely informs
the same curve in Charlotte, so pooling regularises the smaller territories using
data from the larger ones. Operationally it is also one artifact to version, deploy
and monitor rather than twelve, and onboarding a thirteenth BA becomes a data
change instead of an infrastructure change. `ba_code` enters as a categorical
feature so the model can still learn system-specific level and shape.

**How do you know you are not leaking the future?**

Three defences. Splits are strictly chronological, never random. Rolling statistics
are shifted by the full forecast horizon, so a feature for hour *t* only uses data
available at *t - 24h*. And `tests/test_features.py` asserts both properties
directly by reconstructing the expected window from the raw series.

The one thing the model *does* see from the future is weather and calendar. That is
deliberate and defensible: a system operator producing a day-ahead forecast
genuinely holds tomorrow's numerical weather prediction. Hiding it would model a
harder problem than the real one.

**Why is beating the EIA forecast meaningful?**

Because it is not a strawman. The EIA publishes each BA's own day-ahead forecast in
the same feed as the actuals - the forecast operators genuinely published and
operated against. Most portfolio projects invent a naive baseline and beat it. This
one is scored against production reality.

Caveat worth volunteering: the EIA figure is submitted ahead of time under
operational constraints, while this model is fit with hindsight over the full
history. It is a fair accuracy comparison, not a claim of operational superiority.

**What breaks first at 100× the data?**

The feature-engineering step, which currently materialises the full frame in
pandas. The fix is to push lag and rolling computation into DuckDB window
functions, which keeps it set-based and out of Python memory. Ingestion already
scales - it is async, paginated and watermarked. The warehouse scales as far as
DuckDB does, and past that the SQL moves to Snowflake or Synapse largely unchanged.

**How would you productionise this properly?**

Replace the committed DuckDB artifact with object storage (ADLS Gen2 or S3) and a
served warehouse. Move Dagster from local to Dagster Cloud or a Kubernetes
deployment. Add drift monitoring on feature distributions and prediction residuals,
with automated retraining triggers rather than a fixed weekly cron. Put the model
artifacts in a proper registry with staged promotion instead of a git directory.
Add alerting on the quality checks rather than only failing the run.

**Why flag bad data instead of dropping it?**

Because the flag is the finding. A meter reporting an identical value for six
straight hours is not stable, it is stuck - and that is an operational fault
somebody needs to know about. Dropping the row destroys the only evidence. The
warehouse keeps every reading with its flags; the modelling layer decides
separately what to exclude.

**What is the weakest part of this project?**

The deep models are unlikely to beat LightGBM at this data scale, and that is worth
saying out loud rather than hiding. With twelve series and a few years of history,
gradient boosting on well-designed features is genuinely the right tool; sequence
models start to win with many more series or richer exogenous inputs. They are
included because the architecture question is real, and the comparison is more
honest than only shipping the model that won.

**How is the LLM agent kept safe?**

Six layers, applied before execution: a read-only connection, a single-statement
rule, a SELECT/WITH-only prefix check, a DDL/DML keyword blocklist evaluated after
comment stripping, a table allowlist that blocks the internal catalogs, and an
enforced row cap. The generated SQL is always shown to the user, because an answer
nobody can audit is an answer nobody should trust. `tests/test_sql_guard.py` covers
the attack cases including stacked statements and comment-hidden payloads.
