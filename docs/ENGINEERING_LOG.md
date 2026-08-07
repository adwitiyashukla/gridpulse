# Bugs I hit while building this

Every real bug I ran into while building GridPulse, what was causing it, and how I
fixed it. I kept this in the repo on purpose. I learned more from these than from
the parts that worked first time, and a few of them were the kind that break
things without showing any error.

---

## 1. Timezone crash when building the hourly time series

**What happened.** Every warehouse build died with
`AssertionError: Inferred time zone not equal to passed time zone`, thrown from
somewhere inside pandas.

**Why.** When DuckDB hands data back to pandas, the timestamps already have a
timezone attached. I was passing those timestamps to `pd.date_range` and *also*
telling it `tz="UTC"`. Pandas will not accept a timezone on top of values that
already have one, so it refused.

**Fix.** I wrote a small `_as_utc()` helper to convert the timestamps once,
instead of declaring the timezone a second time. In `warehouse/build.py`.

---

## 2. The sample size I reported was not the sample I actually scored

**What happened.** A test failed with `assert 3 == 2`.

**Why.** My `evaluate_forecast` function counted every valid number when it
reported how many observations it used, but it calculated the actual metrics only
on rows that were valid **and** above zero. So every metric was describing a
smaller set of rows than the number printed next to it.

**Fix.** I made the row count come from the same filtering function the metrics
use, so they can never disagree. Worth saying: the test was right and my code was
wrong. In `models/metrics.py`.

---

## 3. A test that was wrong about its own data

**What happened.** `test_hourly_spine_is_continuous` said there were 12 gaps in a
series that clearly had no gaps.

**Why.** I was using `date_diff('hour', ...)`, which DuckDB works out using the
session timezone. So when clocks changed for daylight saving, the test counted
those as missing hours, even though the data was stored in UTC and had none.

**Fix.** I measured the difference in epoch seconds instead, which no calendar
rule can mess with. The lesson I took from this: when a test fails, check which
side is actually wrong before you change either one. In `tests/test_warehouse.py`.

---

## 4. A command line flag that ate its own subcommand

**What happened.** Running `gridpulse --bas PJM ingest` failed with
"the following arguments are required: command".

**Why.** I had set `nargs="*"` on the `--bas` flag, which makes argparse keep
taking values until it runs out. It read `ingest` as a second region code, and
then complained there was no command.

**Fix.** I switched to comma separated values (`--bas PJM,ERCO`), so there is no
question where the list ends. In `cli.py`.

---

## 5. I treated rate limiting like a normal error

**What happened.** Weather downloads finished for 8 of the 12 regions and then
died with `HTTP 429`, after five retries that together took about 14 seconds.

**Why.** I had one retry rule for every kind of failure. A server error usually
clears in a couple of seconds, but a rate limit does not. Waiting 2 seconds and
asking again is just a slower way of getting refused.

**Fix.** I gave 429 its own handling: use the `Retry-After` header if the server
sends one, otherwise wait between 20 and 90 seconds. I also stopped firing the
weather requests all at once and sent them one at a time with a 2 second gap,
which stopped the 429s happening at all. In `ingestion/http.py` and
`ingestion/weather.py`.

---

## 6. The deep learning models would have taken hours to train

**What happened.** The LSTM was taking 4 to 6 minutes per epoch, which works out
to well over an hour per model on my laptop's CPU.

**Why.** Two things at once. First, I was generating about 795,000 training
windows, and each window shared 167 of its 168 hours with the next one, so nearly
all of that was the same data over and over. Second, the model had to step through
all 168 hours one at a time, and an LSTM cannot do those steps in parallel because
each step needs the answer from the one before it.

**Fix.** I kept only every twelfth training window, and fed the model every third
hour instead of every hour, so 168 steps became 56. Electricity demand changes
slowly from hour to hour, so the hours I dropped were not telling the model much
that the remaining ones did not. Together this made training about 13 times
faster, and an epoch dropped to roughly 20 seconds. In `models/deep.py`.

---

## 7. Forty bad readings that wrecked the model

**What happened.** LightGBM stopped training after only **6 trees**. The final
error was 53.9% MAPE with an R² of -131, which is worse than guessing. It also
thought the day of year and cloud cover mattered more than yesterday's demand at
the same hour, which makes no sense for predicting electricity demand.

**How I found it.** Instead of guessing, I wrote a script to print the scaling
numbers the model had calculated:

```
PJM   mean 158,481 MW    std 10,739,790 MW    <- 10.7 million
TVA   mean  18,514 MW    std     56,844 MW    <- 3x the mean
```

PJM's demand is really somewhere between about 70,000 and 165,000 MW, so a
standard deviation of 10.7 million is impossible. Then I checked the error for each
region separately: **PJM was 583%, TVA was 20%, and every other region was between
2.9% and 7.3%.** Exactly the two regions with broken numbers. The median error was
4.2% while the mean was 53.9%, and the worst 1% of rows accounted for 47.5% of all
the error, which is the signature of a few extreme values rather than a bad model.

**Why.** Out of 797,677 hourly readings, **40** were physically impossible values.
I was scaling the demand for each region using the mean and standard deviation,
and both of those can be dragged anywhere by a single crazy value. Those 40 rows
broke the scaling, and the broken scaling broke every prediction for those regions.

**Fix.**

- Switched both scalers to **median and IQR**, which extreme values cannot drag around.
- Excluded impossible readings from training (anything outside 0.2x to 5x that region's median).
- Added two new **critical** quality checks: one for values that are physically impossible, and one that fails if any region's standard deviation is bigger than its mean.

**Result.** MAPE went from 53.9% to **3.68%**, and R² from -131 to **0.994**.

**What I actually learned.** My quality suite had 13 checks and all 13 passed. It
checked that demand was never *below* zero, and it never occurred to me to check
whether demand might be far too *big*. A set of quality checks can only catch the
problems whoever wrote them thought of.

---

## 8. The charts still had spikes after I cleaned the data

**What happened.** After I added spike removal, the website still drew tall
vertical spikes near the end of every chart.

**My first attempt, which did not work.** I flagged a point as a spike if it was
more than 25% away from *both* of its neighbours in the same direction. It caught
nothing at all.

**How I found the real problem.** I wrote a second script that printed the last 60
hours with every flag next to them:

```
2026-08-01 03:00   9,104 MW   +21.5% then -26.2%   flag_isolated_spike = False
2026-08-01 06:00   9,551 MW   +47.3%                flag_isolated_spike = False
```

Two separate problems. The first spike was only 21.5% away on one side, just under
my 25% threshold, so it slipped through. The second one was the **very last row**,
so it had no next neighbour at all and my condition could never be true. Comparing
against neighbours simply cannot work at the ends of a series, and the end is
exactly where the newest, least reliable data sits.

**Fix.** I compared each point against a 5 hour rolling median centred on it
instead. Total demand moves smoothly over five hours, so a normal day never
strays far from its local median, while a single bad reading stands out no matter
which side it falls on. It also still works at the ends, because a partial window
is fine. In `warehouse/build.py`.

---

## 9. Rebuilding the data quietly deleted my trained models

**What happened.** After running `gridpulse build --rebuild`, the file the app
uses dropped from 13.4 MB to 6.3 MB and five tables had disappeared.

**Why.** The rebuild was dropping `model_scores`, `model_predictions` and
`anomaly_scores` along with the tables it was actually supposed to rebuild. So
rebuilding the data threw away 24 minutes of training for no reason.

**Fix.** Rebuild now only drops the tables it owns, and prints a warning that the
model results are now out of date instead of deleting them. In
`warehouse/build.py`.

---

## 10. Version conflicts and a missing import in the app

Three smaller problems, one line each:

- **`statsmodels` 0.14.4 imports something that scipy 1.17 removed.** It was only being pulled in for one trendline on a chart. I replaced that with a binned median calculated in pandas, which needs no extra library and handles outliers better, and dropped `statsmodels` from the app requirements.
- **`SAMPLE_QUESTIONS` existed but was never exported** from the agent package's `__init__.py`, so importing it failed. I fixed it and then checked every other `__init__.py` for the same mistake.
- **The app export had a hand written list of columns** that was missing two weather columns the feature builder needed. Nothing failed until the model tried to make a prediction. The list is now built from `WEATHER_VARIABLES` so it cannot fall out of sync again.

---

## 11. A chart colour setting that broke the sort order

**What happened.** On the model leaderboard, the EIA benchmark bar was drawn at
the top of the chart instead of in its correct position. Every other bar was in
the right place.

**Why.** In Plotly Express, the `color=` argument splits your data into a separate
trace per colour. So highlighting one bar in a different colour pulled it out into
its own trace, and that trace was drawn without following the sort order.

**Fix.** I used a single `go.Bar` with a list of colours, one per bar, plus an
explicit category order on the axis. In `app.py`.

---

## 12. Fixing one problem quietly created another

**What happened.** The Hugging Face Space built without errors, the container
started, Streamlit served the page, and then the app showed:

```
IO Error: The file "/app/data/gold/gridpulse_app.duckdb" exists,
but it is not a valid DuckDB database file!
```

Every single step reported success. The only place the failure showed up was on
the page itself.

**What led to it.** Earlier the same day, `git status` was showing 910 changed
lines in `artifacts/` in a repo nobody had edited. The whole diff was line
endings: Git on Windows had rewritten them all on checkout. I fixed that by adding
a `.gitattributes` that pins everything to Unix line endings, which also protects
`deploy/entrypoint.sh`, because a shell script with Windows line endings fails
inside a Linux container with `bad interpreter`.

That fix was correct. It also broke the Space.

**Why.** Every Hugging Face Space comes with its own `.gitattributes`, and its job
is to tell Git which large files are stored in Git LFS. My sync workflow uploads
the whole repo, so my new `.gitattributes` replaced theirs.

Hugging Face automatically puts any file over about 10 MB into LFS, which in this
repo means:

| File | Size |
|---|---|
| `artifacts/gbm_hybrid/point.txt` | 35.8 MB |
| `artifacts/gbm/point.txt` | 35.6 MB |
| `data/gold/gridpulse_app.duckdb` | 13.4 MB |
| `artifacts/deep_*/weights.pt` | ~0.5 MB |

Git only swaps an LFS file back for the real thing **if `.gitattributes` says that
file is in LFS**. Once those lines were gone, the Space's Docker build checked out
133 byte placeholder files, and `COPY data/gold ./data/gold` copied a placeholder
into the image instead of my database.

On GitHub those same files are stored normally, not in LFS, because I committed
them without it and they are all under GitHub's 100 MB limit. So the exact same
repo was fine on one host and broken on the other, which is why my Streamlit
deployment carried on working and gave me no clue anything was wrong.

**Fix.** I added `deploy/gitattributes_space.txt` and made the sync workflow swap
it in, the same way it already swaps in a different README for the Space. The two
`.gitattributes` files cannot be combined, because the LFS lines are true on
Hugging Face and false on GitHub. Putting them in the main repo would push 85 MB
into GitHub LFS to fix a problem GitHub does not have.

The order of the rules inside that file matters. The general
`* text=auto eol=lf` line has to come **first**, because when two rules match the
same file the last one wins. If I had put it at the bottom it would have turned
text handling back on for every binary file listed above it, which is exactly what
corrupts a DuckDB file. I caught that by reading the file back before committing
it, not by testing.

I also added a step to the workflow that fails the build if any of the three big
files is still a placeholder, checked by file size and by looking for the
`git-lfs.github.com/spec` line that placeholders contain.

**What I actually learned.** To fix the line endings I had to overwrite a file
whose whole purpose was invisible from inside my own repo. I never read that
default `.gitattributes`, never looked at a diff of it, and never mentioned it in
a commit. Deleting it caused no error when I committed, none when the files
uploaded, none during the Docker build, and none when the container started. Every
check I had passed an image that could not possibly work.

The small lesson: when you deploy somewhere, that platform's conventions are part
of the deal, even when they show up as files you did not write. The bigger one:
if a fix works by replacing something completely instead of editing it, the
question to ask is not whether your new version is right, but what the old version
was doing that nobody bothered to write down.

---

## Things I keep coming back to

**Averages break on real data.** The mean and standard deviation can be dragged
anywhere by one bad value. The median and IQR cannot. Forty rows out of 797,677
were the difference between 53.9% error and 3.7%.

**Quality checks only catch what you thought of.** Thirteen checks passed while
the data was broken enough to destroy the model. Looking back it seems obvious,
which is exactly the problem.

**Find out what is wrong before you fix it.** Three of these I fixed on the first
try, because I wrote a script to print the actual numbers first. The one that took
two attempts was the one where I guessed.

**The ends of a series are the worst place for both data and logic.** The last few
rows hold the newest and least reliable data, and anything that compares against
neighbours cannot handle them at all. Both problems showed up in the same place.

**A green pipeline tells you about the pipeline, not about what it produced.** The
LFS bug passed the linter, the tests, the sync, the Docker build and the container
health check, and still shipped an image with a 133 byte text file where a 13 MB
database should have been. Checks only test what someone thought to test. The
verification step guarding that file now exists because the failure made it all
the way to a live page first.
