"""GridPulse command line interface.

Every stage of the platform is reachable from one entry point::

    gridpulse probe            # validate API credentials and response contracts
    gridpulse ingest           # extract EIA + weather into bronze
    gridpulse build            # bronze -> silver -> gold warehouse
    gridpulse quality          # run the data quality suite
    gridpulse train            # train and evaluate the full model suite
    gridpulse anomalies        # fit and score anomaly detectors
    gridpulse export           # write deployment artifacts for the public app
    gridpulse all              # the entire pipeline, in order
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "matplotlib", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _banner(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def _timed(label: str, fn: Callable, *args, **kwargs):
    _banner(label)
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"\n  -> {label} finished in {time.perf_counter() - started:,.1f}s")
    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_probe(args: argparse.Namespace) -> int:
    from gridpulse.config import SETTINGS, active_bas
    from gridpulse.ingestion import probe_eia

    _banner("Configuration")
    bas = active_bas()
    print(f"  Balancing authorities : {len(bas)} -> {', '.join(b.code for b in bas)}")
    print(f"  History from          : {SETTINGS.start_date}")
    print(f"  EIA API key           : {'SET' if SETTINGS.eia_api_key and not SETTINGS.eia_api_key.startswith('your_') else 'MISSING'}")
    print(f"  Groq API key          : {'SET' if SETTINGS.has_llm else 'missing (AI agent disabled)'}")

    _banner("EIA API v2 contract probe")
    try:
        info = probe_eia()
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 1

    print(f"  API version      : {info['api_version']}")
    print(f"  Rows matched     : {info['total_rows_matched']}")
    print(f"  Columns returned : {info['columns']}")
    print(f"  Sample row       : {info['sample_row']}")
    print("\n  Probe passed. Safe to run: gridpulse ingest")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from gridpulse.ingestion import ingest_eia, ingest_weather

    if not args.weather_only:
        _timed("EXTRACT  EIA-930 hourly grid telemetry", ingest_eia, args.bas, args.full_refresh)
    if not args.eia_only:
        _timed("EXTRACT  Open-Meteo hourly weather", ingest_weather, args.bas, args.full_refresh)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from gridpulse.warehouse import build_warehouse

    _timed("TRANSFORM  bronze -> silver -> gold", build_warehouse, rebuild=args.rebuild)
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    from gridpulse.quality import run_quality_suite

    report = _timed("VALIDATE  data quality suite", run_quality_suite)
    return 0 if report.passed else 2


def cmd_train(args: argparse.Namespace) -> int:
    from gridpulse.models.pipeline import train_all

    _timed("TRAIN  forecasting model suite", train_all, bas=args.bas, quick=args.quick)
    return 0


def cmd_anomalies(args: argparse.Namespace) -> int:
    from gridpulse.models.anomaly import run_anomaly_detection

    _timed("DETECT  grid anomalies", run_anomaly_detection)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from gridpulse.warehouse.export import export_for_app

    _timed("EXPORT  deployment artifacts", export_for_app)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_ingest, cmd_build, cmd_quality, cmd_train, cmd_anomalies, cmd_export):
        code = step(args)
        if code != 0 and step is not cmd_quality:  # quality warnings should not halt the run
            return code
    _banner("PIPELINE COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridpulse",
        description="GridPulse: US electricity grid demand intelligence platform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--bas", default=None, metavar="CODES",
        help="comma-separated balancing authority codes, e.g. --bas PJM,ERCO "
             "(default: every code in GRIDPULSE_BAS)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="validate API keys and response contracts").set_defaults(func=cmd_probe)

    p_ing = sub.add_parser("ingest", help="extract raw data into the bronze zone")
    p_ing.add_argument("--full-refresh", action="store_true", help="ignore watermarks and re-download everything")
    p_ing.add_argument("--eia-only", action="store_true")
    p_ing.add_argument("--weather-only", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    p_bld = sub.add_parser("build", help="build the silver and gold layers")
    p_bld.add_argument("--rebuild", action="store_true", help="drop and recreate the warehouse")
    p_bld.set_defaults(func=cmd_build)

    sub.add_parser("quality", help="run the data quality suite").set_defaults(func=cmd_quality)

    p_trn = sub.add_parser("train", help="train and evaluate all forecasting models")
    p_trn.add_argument("--quick", action="store_true", help="fewer epochs and trees; for smoke tests and CI")
    p_trn.set_defaults(func=cmd_train)

    sub.add_parser("anomalies", help="fit and score anomaly detectors").set_defaults(func=cmd_anomalies)
    sub.add_parser("export", help="write deployment artifacts for the public app").set_defaults(func=cmd_export)

    p_all = sub.add_parser("all", help="run the entire pipeline end to end")
    p_all.add_argument("--full-refresh", action="store_true")
    p_all.add_argument("--rebuild", action="store_true")
    p_all.add_argument("--quick", action="store_true")
    p_all.set_defaults(func=cmd_all, eia_only=False, weather_only=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    for flag in ("full_refresh", "rebuild", "quick", "eia_only", "weather_only"):
        setattr(args, flag, getattr(args, flag, False))

    # Normalise the comma-separated BA filter into a list once, here.
    args.bas = [c.strip().upper() for c in args.bas.split(",") if c.strip()] if args.bas else None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("gridpulse").error("%s: %s", type(exc).__name__, exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
