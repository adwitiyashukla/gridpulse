# Interview notes

Questions I think an interviewer would ask about this project, and my honest
answers. I read through this before an interview.

---

**Why DuckDB instead of Postgres, Snowflake or Spark?**

The work here is all reading and aggregating: scan a few hundred million hourly
rows, group them up, join two small lookup tables. There is only ever one process
writing. That is exactly what DuckDB is built for, and it means there is no server
to run, no connection pool, and no container that has to stay alive. The SQL I
wrote is plain enough to move to Snowflake or Synapse without rewriting it. Spark
would make sense at a much bigger data size than this, and picking it here would
just be putting a buzzword on my CV.

**Why one model for all the regions instead of one per region?**

The regions behave in similar ways. How demand responds to temperature in Atlanta
really does tell you something about the same curve in Charlotte, so training on
all of them together lets the bigger regions help the smaller ones. It is also
simpler to run: one model file to version, deploy and monitor instead of twelve,
and adding a thirteenth region is just more data rather than more infrastructure.
The region code goes in as a categorical feature, so the model can still learn
that each one has its own size and shape.

**How do you know the model is not seeing the future?**

Three things. The train/test split is always by date, never random. Anything
calculated from past demand is shifted back by the full 24 hours, so a feature for
a given hour only uses data that existed 24 hours earlier. And
`tests/test_features.py` checks both of those directly, by rebuilding what the
window should be from the raw data and comparing.

The one thing the model does see from the future is the weather forecast and the
calendar. That is on purpose and I think it is fair: a real grid operator making a
day-ahead forecast also has tomorrow's weather forecast in front of them. Hiding
it would mean solving a harder problem than the real one.

**Why does beating the EIA forecast actually mean something?**

Because it is not a baseline I made up to be easy. EIA publishes each region's own
day-ahead forecast in the same dataset as the real values, and that is the forecast
grid operators actually published and actually used. Most portfolio projects invent
a simple baseline and beat that. This one is scored against what really happened in
production.

The caveat I would raise myself: EIA had to submit their forecast in advance, under
real deadlines, while my model is trained on the full history. So it is a fair
comparison of accuracy, but it does not prove my model would do better in real
operations.

**What breaks first if the data gets 100 times bigger?**

The feature engineering step, because right now it builds the whole table in
pandas memory. The fix would be to move the lag and rolling calculations into
DuckDB window functions so they stay in SQL and never load into Python. The
download step already scales, since it is async, paginated and remembers where it
got to. The warehouse scales as far as DuckDB does, and beyond that the SQL moves
to Snowflake or Synapse mostly unchanged.

**How would you make this properly production ready?**

Move the DuckDB file out of the repo and into object storage like S3 or ADLS, with
a real served warehouse behind it. Run Dagster on Dagster Cloud or Kubernetes
instead of locally. Add monitoring that watches whether the input data or the
prediction errors are drifting, and retrain when they do instead of just retraining
every Monday. Put the trained models in a proper model registry with staging and
promotion, instead of a folder in Git. And make the quality checks send an alert,
not just fail the run.

**Why flag bad data instead of deleting it?**

Because the flag is the useful part. A meter reporting the exact same value for six
hours in a row is not steady, it is stuck, and that is a real fault somebody should
know about. If I delete the row I also delete the only evidence it happened. So the
warehouse keeps every reading with its flags attached, and the modelling step
decides separately what to leave out.

**What is the weakest part of this project?**

The prediction bands. The P10-P90 range should contain 80% of the real values and
only contains about 58%, so the model is more confident than it should be. The fix
is conformal calibration and it is the next thing on my list.

After that, the deep learning models. They were never likely to beat LightGBM at
this data size, and I would rather say that than hide it. With twelve series and a
few years of data, gradient boosting on good features is genuinely the right tool.
LSTMs and Transformers start to win when you have many more series or a lot more
outside data. I kept them in because comparing the approaches was the point, and
showing only the winner would be less honest.

**How do you stop the LLM doing something dangerous?**

Six checks, all before anything runs: the database connection is read-only, only
one statement is allowed, the query has to start with SELECT or WITH, there is a
blocklist of dangerous keywords that runs after comments are stripped out, there
is a list of allowed tables that blocks the internal system tables, and there is a
row limit. The app always shows you the SQL it generated, because an answer you
cannot check is an answer you should not trust. `tests/test_sql_guard.py` covers
the attacks I tested, including stacking two statements together and hiding
keywords inside comments.
