"""The public GridPulse website. Reads the committed database and model files, so
it starts instantly and never trains anything while someone is waiting."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from gridpulse.config import BALANCING_AUTHORITIES  # noqa: E402
from gridpulse.warehouse.duck import connect  # noqa: E402

st.set_page_config(
    page_title="GridPulse | US Electricity Demand Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#00C2A8"
ACCENT_2 = "#7C6BFF"
WARN = "#FF6B6B"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1400px; }}
      h1, h2, h3 {{ letter-spacing: -0.02em; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.9rem; color: {ACCENT}; }}
      div[data-testid="stMetricLabel"] {{ font-size: 0.8rem; text-transform: uppercase;
                                          letter-spacing: 0.06em; opacity: 0.75; }}
      .gp-hero {{ background: linear-gradient(120deg, rgba(0,194,168,0.14), rgba(124,107,255,0.14));
                  border: 1px solid rgba(0,194,168,0.3); border-radius: 14px;
                  padding: 1.4rem 1.8rem; margin-bottom: 1.4rem; }}
      .gp-hero h1 {{ margin: 0 0 0.3rem 0; font-size: 2.1rem; }}
      .gp-hero p {{ margin: 0; opacity: 0.85; font-size: 1.02rem; }}
      .gp-pill {{ display:inline-block; padding: 0.18rem 0.7rem; border-radius: 999px;
                  background: rgba(0,194,168,0.18); border: 1px solid rgba(0,194,168,0.35);
                  font-size: 0.78rem; margin-right: 0.4rem; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; }}
      .stTabs [data-baseweb="tab"] {{ padding: 0.5rem 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def database_path() -> Path:
    slim = ROOT / "data" / "gold" / "gridpulse_app.duckdb"
    return slim if slim.exists() else ROOT / "data" / "gold" / "gridpulse.duckdb"


def data_version() -> str:
    """Identify the currently deployed artifacts by size and modification time.

    Every cached function below carries its own 900 second expiry, and each of
    those clocks starts when that function is first called rather than when the
    data changed. The caches therefore expire at different moments, and in the
    window between them the page can report an hourly row count from one
    training run beside an accuracy figure from the previous one. Both numbers
    are individually correct and the pair is wrong, which is the least useful
    kind of error to put in front of someone.

    The weekly refresh rewrites the warehouse and the headline together, so
    comparing their size and mtime detects a new deployment directly instead of
    waiting for a timer to guess that one happened.
    """
    parts = []
    for path in (database_path(), ROOT / "artifacts" / "headline.json"):
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{path.name}:absent")
    return "|".join(parts)


@st.cache_resource
def _deployed_version() -> dict[str, str | None]:
    """Hold the last seen data version, shared across every user session.

    Deliberately `cache_resource` rather than `session_state`. Session state is
    per visitor, so the check below would fire once for each new arrival and
    every one of them would clear a cache that was already correct. A resource
    is shared process-wide, so the artifacts are detected as changed exactly
    once no matter how many people are on the page. `st.cache_data.clear()`
    does not touch `cache_resource`, so this record survives the clearing it
    triggers.
    """
    return {"version": None}


def invalidate_caches_if_data_changed() -> None:
    """Expire every cache at once when the artifacts change underneath us.

    Streamlit Cloud reruns the script when a new commit lands but does not
    always restart the process, so `st.cache_data` entries can outlive the files
    they were derived from. Clearing on a version change makes a refresh visible
    immediately and, more importantly, keeps the numbers on the page mutually
    consistent.
    """
    record = _deployed_version()
    current = data_version()
    if record["version"] != current:
        st.cache_data.clear()
        record["version"] = current


invalidate_caches_if_data_changed()


@st.cache_data(ttl=900, show_spinner=False)
def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    path = database_path()
    if not path.exists():
        return pd.DataFrame()
    try:
        with connect(path, read_only=True) as con:
            return con.execute(sql, list(params)).df()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Query failed: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def available_bas() -> list[str]:
    frame = run_query("SELECT DISTINCT ba_code FROM fact_demand_hourly ORDER BY ba_code")
    return frame["ba_code"].tolist() if not frame.empty else list(BALANCING_AUTHORITIES)


@st.cache_data(ttl=900, show_spinner=False)
def headline() -> dict:
    import json

    path = ROOT / "artifacts" / "headline.json"
    return json.loads(path.read_text()) if path.exists() else {}


def data_ready() -> bool:
    return database_path().exists() and not run_query(
        "SELECT 1 FROM fact_demand_hourly LIMIT 1"
    ).empty


head = headline()
skill = head.get("skill_vs_eia_pct")

st.markdown(
    """
    <div class="gp-hero">
      <h1>⚡ GridPulse</h1>
      <p>Day-ahead electricity demand forecasting for US balancing authorities,
         benchmarked against the EIA's own published forecast.</p>
      <div style="margin-top:0.8rem;">
        <span class="gp-pill">EIA-930 hourly telemetry</span>
        <span class="gp-pill">DuckDB lakehouse</span>
        <span class="gp-pill">LightGBM + PyTorch</span>
        <span class="gp-pill">Agentic SQL analytics</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not data_ready():
    st.warning(
        "**No warehouse found.** This deployment is missing its data artifact. "
        "Run `gridpulse all` locally and commit `data/gold/gridpulse_app.duckdb` "
        "plus the `artifacts/` directory."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    bas = available_bas()
    selected_ba = st.selectbox(
        "Balancing authority",
        bas,
        format_func=lambda c: f"{c} - {BALANCING_AUTHORITIES[c].name}" if c in BALANCING_AUTHORITIES else c,
    )
    ba_meta = BALANCING_AUTHORITIES.get(selected_ba)
    if ba_meta:
        st.caption(f"**Region:** {ba_meta.region}  \n**Load centre:** {ba_meta.load_centre}")

    lookback_days = st.slider("History window (days)", 7, 180, 30)
    st.divider()

    coverage = run_query(
        "SELECT min(period_utc) AS lo, max(period_utc) AS hi, count(*) AS n FROM fact_demand_hourly"
    )
    if not coverage.empty:
        st.caption(
            f"**Warehouse coverage**  \n{coverage.iloc[0]['lo']:%Y-%m-%d} → "
            f"{coverage.iloc[0]['hi']:%Y-%m-%d}  \n{int(coverage.iloc[0]['n']):,} hourly rows"
        )
    st.divider()
    st.caption(
        "Built by **Adwitiya Shukla**  \n"
        "[GitHub repository](https://github.com/adwitiyashukla/gridpulse) · Data: US EIA + Open-Meteo"
    )

c1, c2, c3, c4 = st.columns(4)
summary = run_query(
    """
    SELECT count(*) AS hours, count(DISTINCT ba_code) AS bas,
           round(avg(demand_clean_mwh)) AS avg_demand, max(demand_clean_mwh) AS peak
    FROM fact_demand_hourly WHERE demand_clean_mwh IS NOT NULL
    """
)
if not summary.empty:
    row = summary.iloc[0]
    c1.metric("Hourly observations", f"{int(row['hours']):,}")
    c2.metric("Balancing authorities", int(row["bas"]))
    c3.metric("Peak demand observed", f"{int(row['peak']):,} MW")
c4.metric(
    "Accuracy vs EIA forecast",
    f"{skill:+.1f}%" if isinstance(skill, int | float) else "-",
    help="Percentage improvement in MAPE over the EIA's own published day-ahead forecast.",
)

tabs = st.tabs([
    "Forecast", "Explorer", "Model Leaderboard",
    "Anomalies", "Data Quality", "Ask the Grid", "How it works",
])


with tabs[0]:
    st.subheader(f"24-hour demand forecast - {selected_ba}")
    st.caption(
        "Generated from a LightGBM global model using the last 336 hours of observed "
        "demand plus a live weather forecast for the load centre. Shaded band is the "
        "P10-P90 prediction interval."
    )

    left, right = st.columns([1, 3])
    with left:
        use_live = st.toggle("Fetch live weather", value=True,
                             help="Off replays the most recent 24 hours so you can see prediction against truth.")
        go_button = st.button("Generate forecast", type="primary", use_container_width=True)

    if go_button:
        with st.spinner("Building features and scoring the model…"):
            try:
                from gridpulse.models.inference import artifacts_available, forecast

                if not artifacts_available():
                    st.error("Model artifacts are missing. Run `gridpulse train` and commit `artifacts/`.")
                else:
                    result = forecast(selected_ba, allow_network=use_live)
                    frame = result.frame

                    st.session_state["forecast_result"] = (result.mode, result.notes or [], frame)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Forecast failed: {exc}")

    if "forecast_result" in st.session_state:
        mode, notes, frame = st.session_state["forecast_result"]

        badge = "Live forward forecast" if mode == "live" else "Replay of the last 24 hours"
        st.info(f"**{badge}** · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
        for note in notes:
            st.caption(f"↳ {note}")

        history = run_query(
            """
            SELECT period_utc, demand_clean_mwh AS demand_mwh
            FROM fact_demand_hourly
            WHERE ba_code = ? AND demand_clean_mwh IS NOT NULL
            ORDER BY period_utc DESC LIMIT 168
            """,
            (selected_ba,),
        ).sort_values("period_utc")

        figure = go.Figure()
        if not history.empty:
            figure.add_trace(go.Scatter(
                x=history["period_utc"], y=history["demand_mwh"],
                name="Observed history", line=dict(color="rgba(255,255,255,0.55)", width=1.6),
            ))
        if {"p10_mwh", "p90_mwh"} <= set(frame.columns):
            figure.add_trace(go.Scatter(
                x=pd.concat([frame["period_utc"], frame["period_utc"][::-1]]),
                y=pd.concat([frame["p90_mwh"], frame["p10_mwh"][::-1]]),
                fill="toself", fillcolor="rgba(0,194,168,0.18)",
                line=dict(color="rgba(0,0,0,0)"), name="P10-P90 interval", hoverinfo="skip",
            ))
        figure.add_trace(go.Scatter(
            x=frame["period_utc"], y=frame["forecast_mwh"],
            name="GridPulse forecast", line=dict(color=ACCENT, width=3),
        ))
        if "actual_mwh" in frame.columns and frame["actual_mwh"].notna().any():
            figure.add_trace(go.Scatter(
                x=frame["period_utc"], y=frame["actual_mwh"],
                name="Actual", line=dict(color=WARN, width=2.5, dash="dot"),
            ))

        figure.update_layout(
            height=470, hovermode="x unified", template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.1),
            yaxis_title="Demand (MW)", xaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast peak", f"{frame['forecast_mwh'].max():,.0f} MW")
        m2.metric("Forecast trough", f"{frame['forecast_mwh'].min():,.0f} MW")
        if "actual_mwh" in frame.columns and frame["actual_mwh"].notna().any():
            mape = ((frame["forecast_mwh"] - frame["actual_mwh"]).abs()
                    / frame["actual_mwh"]).mean() * 100
            m3.metric("MAPE on this window", f"{mape:.2f}%")

        with st.expander("Forecast table"):
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV", frame.to_csv(index=False).encode(),
                file_name=f"gridpulse_forecast_{selected_ba}.csv", mime="text/csv",
            )
    else:
        st.info("Choose a balancing authority in the sidebar and press **Generate forecast**.")


with tabs[1]:
    st.subheader(f"Historical explorer - {selected_ba}")

    history = run_query(
        f"""
        SELECT period_utc, hour_local, demand_clean_mwh AS demand_mwh,
               demand_forecast_mwh, temperature_2m, is_weekend, is_holiday, season
        FROM fact_demand_hourly
        WHERE ba_code = ?
          AND period_utc >= (SELECT max(period_utc) FROM fact_demand_hourly) - INTERVAL {lookback_days} DAY
        ORDER BY period_utc
        """,
        (selected_ba,),
    )

    if history.empty:
        st.info("No data in the selected window.")
    else:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=history["period_utc"], y=history["demand_mwh"],
                                    name="Actual demand", line=dict(color=ACCENT, width=1.8)))
        if history["demand_forecast_mwh"].notna().any():
            figure.add_trace(go.Scatter(x=history["period_utc"], y=history["demand_forecast_mwh"],
                                        name="EIA day-ahead forecast",
                                        line=dict(color=ACCENT_2, width=1.4, dash="dot")))
        figure.update_layout(height=380, template="plotly_dark", hovermode="x unified",
                             margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Demand (MW)",
                             legend=dict(orientation="h", y=1.12),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.markdown("**Demand response to temperature**")
            st.caption("The V-shape is the heating and cooling load split around the comfort balance point.")
            scatter = history.dropna(subset=["temperature_2m", "demand_mwh"])
            if not scatter.empty:
                figure = px.scatter(
                    scatter, x="temperature_2m", y="demand_mwh", color="season",
                    opacity=0.45,
                    labels={"temperature_2m": "Temperature (°C)", "demand_mwh": "Demand (MW)"},
                )

                binned = (
                    scatter.assign(bin=(scatter["temperature_2m"] / 2).round() * 2)
                    .groupby("bin")["demand_mwh"]
                    .agg(["median", "size"])
                    .query("size >= 5")
                    .reset_index()
                )
                if len(binned) > 2:
                    figure.add_trace(go.Scatter(
                        x=binned["bin"], y=binned["median"],
                        mode="lines+markers", name="Median response",
                        line=dict(color="#FFFFFF", width=2.5),
                        marker=dict(size=5),
                    ))

                figure.update_layout(height=360, template="plotly_dark",
                                     margin=dict(l=10, r=10, t=10, b=10),
                                     legend=dict(orientation="h", y=1.15),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(figure, use_container_width=True)

        with right:
            st.markdown("**Average daily load shape**")
            st.caption("Weekday and weekend profiles diverge sharply; the models encode this explicitly.")
            profile = (
                history.groupby(["hour_local", "is_weekend"])["demand_mwh"]
                .mean().reset_index()
            )
            profile["Day type"] = profile["is_weekend"].map({True: "Weekend", False: "Weekday"})
            figure = px.line(profile, x="hour_local", y="demand_mwh", color="Day type",
                             markers=True,
                             labels={"hour_local": "Hour (local)", "demand_mwh": "Mean demand (MW)"},
                             color_discrete_map={"Weekday": ACCENT, "Weekend": ACCENT_2})
            figure.update_layout(height=360, template="plotly_dark",
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, use_container_width=True)

        st.markdown("**Fleet comparison - mean demand by balancing authority**")
        fleet = run_query(
            f"""
            SELECT ba_code, round(avg(demand_clean_mwh)) AS mean_demand_mw,
                   round(max(demand_clean_mwh)) AS peak_demand_mw
            FROM fact_demand_hourly
            WHERE demand_clean_mwh IS NOT NULL
              AND period_utc >= (SELECT max(period_utc) FROM fact_demand_hourly) - INTERVAL {lookback_days} DAY
            GROUP BY ba_code ORDER BY mean_demand_mw DESC
            """
        )
        if not fleet.empty:
            figure = px.bar(fleet, x="ba_code", y=["mean_demand_mw", "peak_demand_mw"],
                            barmode="group", labels={"value": "MW", "ba_code": ""},
                            color_discrete_sequence=[ACCENT, ACCENT_2])
            figure.update_layout(height=330, template="plotly_dark",
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 legend=dict(orientation="h", y=1.15),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, use_container_width=True)


with tabs[2]:
    st.subheader("Model leaderboard")
    st.caption(
        "Every model is scored on the same out-of-sample window with the same metrics. "
        "`eia_official` is the forecast the US Energy Information Administration actually "
        "published and grid operators actually used - it is the benchmark, not a strawman."
    )

    board = run_query("""
        SELECT model, mape_pct, smape_pct, mae_mwh, rmse_mwh, r2,
               peak_hour_mape_pct, skill_vs_eia_pct, n_obs
        FROM model_scores
        WHERE trained_at_utc = (SELECT max(trained_at_utc) FROM model_scores)
        ORDER BY mape_pct
    """)

    if board.empty:
        st.info("No model scores yet. Run `gridpulse train`.")
    else:
        pretty = {
            "gbm": "LightGBM (global)", "gbm_hybrid": "LightGBM hybrid (+EIA input)",
            "lstm": "LSTM encoder", "transformer": "Transformer encoder",
            "ensemble": "Ensemble (GBM + LSTM)", "eia_official": "EIA official forecast",
            "seasonal_naive": "Seasonal naive (24h)", "weekly_naive": "Weekly naive (168h)",
        }
        board["Model"] = board["model"].map(lambda m: pretty.get(m, m))

        ordered = board.sort_values("mape_pct", ascending=False)
        bar_colours = [
            ACCENT_2 if model == "eia_official" else ACCENT
            for model in ordered["model"]
        ]

        figure = go.Figure(
            go.Bar(
                x=ordered["mape_pct"],
                y=ordered["Model"],
                orientation="h",
                text=[f"{v:.3f}%" for v in ordered["mape_pct"]],
                textposition="outside",
                marker_color=bar_colours,
                hovertemplate="%{y}<br>MAPE %{x:.3f}%<extra></extra>",
            )
        )
        figure.update_layout(
            height=420, template="plotly_dark", showlegend=False,
            margin=dict(l=10, r=70, t=20, b=10),
            xaxis_title="MAPE (%) - lower is better", yaxis_title=None,
            yaxis=dict(categoryorder="array", categoryarray=list(ordered["Model"])),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "The EIA benchmark is highlighted in purple. Anything to its left is "
            "more accurate than the forecast the US government actually published."
        )

        best = board.iloc[0]
        if best["model"] != "eia_official":
            st.success(
                f"**{pretty.get(best['model'], best['model'])}** achieves "
                f"**{best['mape_pct']:.3f}% MAPE**, which is "
                f"**{best['skill_vs_eia_pct']:.1f}% more accurate** than the EIA's own "
                f"published day-ahead forecast on the same {int(best['n_obs']):,} hours."
            )

        st.dataframe(
            board[["Model", "mape_pct", "smape_pct", "mae_mwh", "rmse_mwh", "r2",
                   "peak_hour_mape_pct", "skill_vs_eia_pct"]]
            .rename(columns={
                "mape_pct": "MAPE %", "smape_pct": "sMAPE %", "mae_mwh": "MAE (MW)",
                "rmse_mwh": "RMSE (MW)", "r2": "R²",
                "peak_hour_mape_pct": "Peak-hour MAPE %", "skill_vs_eia_pct": "Skill vs EIA %",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("**EIA forecast error by balancing authority**")
        accuracy = run_query("""
            SELECT ba_code, round(avg(abs_pct_error), 3) AS eia_mape_pct, count(*) AS hours
            FROM fact_forecast_accuracy GROUP BY ba_code ORDER BY eia_mape_pct
        """)
        if not accuracy.empty:
            figure = px.bar(accuracy, x="ba_code", y="eia_mape_pct",
                            labels={"eia_mape_pct": "EIA MAPE (%)", "ba_code": ""},
                            color_discrete_sequence=[ACCENT_2])
            figure.update_layout(height=300, template="plotly_dark",
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, use_container_width=True)


with tabs[3]:
    st.subheader("Anomaly monitor")
    st.caption(
        "Three independent detectors vote: a robust seasonal z-score, an Isolation "
        "Forest over the multivariate feature space, and an autoencoder over daily "
        "load shapes. Severity rises with the number of detectors that agree."
    )

    counts = run_query("""
        SELECT anomaly_type, severity, count(*) AS n
        FROM anomaly_scores WHERE is_anomaly GROUP BY 1, 2 ORDER BY n DESC
    """)

    if counts.empty:
        st.info("No anomaly scores yet. Run `gridpulse anomalies`.")
    else:
        left, right = st.columns([2, 1])
        with left:
            figure = px.bar(counts, x="anomaly_type", y="n", color="severity",
                            labels={"n": "Hours flagged", "anomaly_type": ""},
                            color_discrete_map={"high": WARN, "medium": "#FFA94D", "low": ACCENT})
            figure.update_layout(height=340, template="plotly_dark",
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, use_container_width=True)
        with right:
            by_ba = run_query("""
                SELECT ba_code, count(*) AS anomalies
                FROM anomaly_scores WHERE is_anomaly GROUP BY 1 ORDER BY anomalies DESC
            """)
            st.dataframe(by_ba, use_container_width=True, hide_index=True, height=340)

        st.markdown("**Most recent high-severity anomalies**")
        recent = run_query("""
            SELECT period_utc, ba_code, round(demand_mwh) AS demand_mw,
                   round(temperature_2m, 1) AS temp_c, anomaly_type, severity,
                   detector_votes, round(robust_z, 2) AS robust_z
            FROM anomaly_scores
            WHERE is_anomaly AND severity IN ('high', 'medium')
            ORDER BY period_utc DESC LIMIT 200
        """)
        st.dataframe(recent, use_container_width=True, hide_index=True, height=380)


with tabs[4]:
    st.subheader("Data quality scorecard")
    st.caption(
        "Utility interval data fails in domain-specific ways: daylight-saving "
        "duplicates, frozen telemetry, negative demand from sign-convention errors. "
        "Each check below targets one of those failure modes."
    )

    scorecard = run_query("SELECT * FROM dq_scorecard ORDER BY dimension")
    checks = run_query("""
        SELECT check_name, dimension, severity, failed_rows, total_rows,
               failure_rate_pct, threshold_pct, passed, description
        FROM dq_results
        WHERE run_at_utc = (SELECT max(run_at_utc) FROM dq_results)
        ORDER BY passed, severity, check_name
    """)

    if checks.empty:
        st.info("No quality results yet. Run `gridpulse quality`.")
    else:
        passed = int(checks["passed"].sum())
        total = len(checks)
        c1, c2, c3 = st.columns(3)
        c1.metric("Checks passed", f"{passed}/{total}")
        c2.metric("Pass rate", f"{100 * passed / total:.0f}%")
        c3.metric("Critical failures",
                  int(((~checks["passed"]) & (checks["severity"] == "critical")).sum()))

        if not scorecard.empty:
            figure = px.bar(scorecard, x="dimension", y="pass_pct",
                            labels={"pass_pct": "Pass rate (%)", "dimension": ""},
                            color_discrete_sequence=[ACCENT], range_y=[0, 105])
            figure.update_layout(height=300, template="plotly_dark",
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, use_container_width=True)

        display = checks.copy()
        display["Status"] = display["passed"].map({True: "PASS", False: "FAIL"})
        st.dataframe(
            display[["Status", "check_name", "dimension", "severity",
                     "failed_rows", "total_rows", "failure_rate_pct", "description"]],
            use_container_width=True, hide_index=True, height=460,
        )


with tabs[5]:
    st.subheader("Ask the Grid")
    st.caption(
        "Ask in plain English. The question is translated into DuckDB SQL, passed "
        "through a safety guard (read-only connection, SELECT-only, table allowlist, "
        "enforced row cap) and executed. The generated SQL is always shown, because "
        "an answer you cannot audit is an answer you cannot trust."
    )

    from gridpulse.agent import SAMPLE_QUESTIONS, GridAgent

    agent = GridAgent(database=database_path())

    if not agent.available:
        st.warning(
            "The AI agent needs a Groq API key. Locally, put `GROQ_API_KEY` in `.env`; "
            "when deployed, add it to your host's secrets. "
            "Free keys: https://console.groq.com/keys"
        )
    else:
        example = st.selectbox("Try an example", ["(write my own)"] + SAMPLE_QUESTIONS)
        default = "" if example.startswith("(") else example
        question = st.text_input("Your question", value=default,
                                 placeholder="e.g. Which BA has the worst forecast error in summer?")

        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Generating SQL and querying the warehouse…"):
                answer = agent.ask(question)

            if not answer.ok:
                st.error(answer.error)
                if answer.sql:
                    st.code(answer.sql, language="sql")
            else:
                if answer.summary:
                    st.success(answer.summary)
                for warning in answer.warnings:
                    st.caption(f"↳ {warning}")

                with st.expander("Generated SQL", expanded=True):
                    st.code(answer.sql, language="sql")

                st.dataframe(answer.data, use_container_width=True, hide_index=True, height=380)

                numeric = answer.data.select_dtypes("number").columns.tolist()
                if len(answer.data) > 1 and numeric:
                    label_columns = [c for c in answer.data.columns if c not in numeric]
                    if label_columns:
                        try:
                            figure = px.bar(answer.data.head(40), x=label_columns[0], y=numeric[0],
                                            color_discrete_sequence=[ACCENT])
                            figure.update_layout(height=340, template="plotly_dark",
                                                 margin=dict(l=10, r=10, t=10, b=10),
                                                 paper_bgcolor="rgba(0,0,0,0)",
                                                 plot_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(figure, use_container_width=True)
                        except Exception:  # noqa: BLE001
                            pass


with tabs[6]:
    st.subheader("How GridPulse works")

    st.markdown(
        """
### The problem

Electricity cannot really be stored at grid scale, so the companies that run the
grid have to decide today how much power to generate tomorrow. Guess too high and
they burn fuel making electricity nobody uses. Guess too low and they have to buy
the shortfall at emergency prices, or cut power to customers. On a large grid,
being off by one percent costs millions of dollars a year.

### What I compare against

The EIA publishes each region's own day-ahead forecast next to what actually
happened. So instead of making up an easy baseline, every model here is scored
against the forecast grid operators really published and really used, which means
anyone can check whether these results hold up.

### The pipeline
"""
    )

    st.code(
        """
EIA-930 API v2  ─┐
                 ├─► BRONZE (Parquet, partitioned, immutable, watermarked)
Open-Meteo      ─┘        │
                          ▼
                    SILVER (cleaned: measures become columns, weather joined,
                            every hour listed, local time, quality flags)
                          │
                          ▼
                    GOLD (DuckDB star schema)
                      dim_ba · dim_date
                      fact_demand_hourly
                      fact_forecast_accuracy   ← EIA benchmark scored here
                          │
        ┌─────────────────┼──────────────────┬───────────────────┐
        ▼                 ▼                  ▼                   ▼
  16 quality checks   Features         Anomaly detection    SQL agent
  (6 categories)      (40 of them)     (3 detectors vote)   (guarded LLM)
                          │
                          ▼
              LightGBM · LSTM · Transformer · Ensemble
                          │
                          ▼
              FastAPI  ·  This Streamlit app
""",
        language="text",
    )

    st.markdown(
        """
### Why I built it this way

Why DuckDB and not Postgres or Spark. It handles hundreds of millions of rows
inside a single file with no server to run. I built this on a laptop with 8 GB of
RAM, and that was the difference between a pipeline that finishes and one that runs
out of memory. The SQL would still run on Snowflake or BigQuery unchanged.

Why one model for all 12 regions. The regions behave similarly, so training
together lets the bigger ones help the smaller ones. It also means one model file
to deploy and monitor instead of twelve.

Why I flag bad data instead of deleting it. A meter stuck on the same value is
proof that the meter broke. Quietly dropping that row also drops the only record
that anything went wrong.

Why I only split by date. If you split time-series data randomly, rows from the
future sit next to rows from the past and the model effectively sees answers it
should not have. The scores look great and mean nothing.

Why using tomorrow's weather is not cheating. A real grid operator also has
tomorrow's weather forecast and knows what day of the week it is. Leaving it out
would mean solving a harder problem than the real one.

### What it is built with

| Part | Tools |
|---|---|
| Downloading data | Python, `httpx` async, only fetching what is new |
| Storage | Parquet in bronze/silver/gold, DuckDB warehouse |
| Transformations | SQL and dbt |
| Scheduling | Dagster assets, the same pipeline as an Airflow DAG, GitHub Actions |
| Quality | 16 checks across 6 categories |
| Machine learning | LightGBM with quantiles, PyTorch LSTM and Transformer |
| Experiment tracking | MLflow |
| Serving | FastAPI, Streamlit |
| AI features | Groq LLM writing SQL, with guardrails |
"""
    )

st.divider()
st.caption(
    "GridPulse · Data: US Energy Information Administration (EIA-930) and Open-Meteo · "
    "Built by Adwitiya Shukla"
)
