"""Settings, file paths and the list of 12 balancing authorities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class BalancingAuthority:
    """A region, with coordinates pointing at its biggest city rather than its
    geographic centre, since demand follows the weather where people live."""

    code: str
    name: str
    timezone: str
    latitude: float
    longitude: float
    region: str
    load_centre: str

    @property
    def slug(self) -> str:
        return self.code.lower()


BALANCING_AUTHORITIES: dict[str, BalancingAuthority] = {
    ba.code: ba
    for ba in [
        BalancingAuthority("PJM",  "PJM Interconnection",                  "America/New_York",    39.9526,  -75.1652, "Mid-Atlantic", "Philadelphia, PA"),
        BalancingAuthority("MISO", "Midcontinent ISO",                     "America/Chicago",     41.8781,  -87.6298, "Midwest",      "Chicago, IL"),
        BalancingAuthority("ERCO", "ERCOT",                                "America/Chicago",     29.7604,  -95.3698, "Texas",        "Houston, TX"),
        BalancingAuthority("CISO", "California ISO",                       "America/Los_Angeles", 34.0522, -118.2437, "California",   "Los Angeles, CA"),
        BalancingAuthority("NYIS", "New York ISO",                         "America/New_York",    40.7128,  -74.0060, "New York",     "New York, NY"),
        BalancingAuthority("ISNE", "ISO New England",                      "America/New_York",    42.3601,  -71.0589, "New England",  "Boston, MA"),
        BalancingAuthority("SWPP", "Southwest Power Pool",                 "America/Chicago",     39.0997,  -94.5786, "Central",      "Kansas City, MO"),
        BalancingAuthority("BPAT", "Bonneville Power Administration",      "America/Los_Angeles", 45.5152, -122.6784, "Northwest",    "Portland, OR"),
        BalancingAuthority("FPL",  "Florida Power & Light",                "America/New_York",    25.7617,  -80.1918, "Florida",      "Miami, FL"),
        BalancingAuthority("DUK",  "Duke Energy Carolinas",                "America/New_York",    35.2271,  -80.8431, "Carolinas",    "Charlotte, NC"),
        BalancingAuthority("SOCO", "Southern Company Services",            "America/New_York",    33.7490,  -84.3880, "Southeast",    "Atlanta, GA"),
        BalancingAuthority("TVA",  "Tennessee Valley Authority",           "America/Chicago",     36.1627,  -86.7816, "Tennessee",    "Nashville, TN"),
    ]
}


def active_bas() -> list[BalancingAuthority]:
    """Balancing authorities selected via ``GRIDPULSE_BAS``, in registry order."""
    requested = [c.strip().upper() for c in _env("GRIDPULSE_BAS", ",".join(BALANCING_AUTHORITIES)).split(",") if c.strip()]
    unknown = [c for c in requested if c not in BALANCING_AUTHORITIES]
    if unknown:
        raise ValueError(
            f"Unknown balancing authority code(s): {unknown}. "
            f"Valid codes: {sorted(BALANCING_AUTHORITIES)}"
        )
    return [BALANCING_AUTHORITIES[c] for c in requested]


EIA_MEASURES: dict[str, str] = {
    "D":  "demand_mwh",
    "DF": "demand_forecast_mwh",
    "NG": "net_generation_mwh",
    "TI": "total_interchange_mwh",
}

WEATHER_VARIABLES: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]


@dataclass(frozen=True)
class Paths:
    data: Path = field(default_factory=lambda: _resolve(_env("GRIDPULSE_DATA_DIR", "data")))

    @property
    def bronze(self) -> Path:
        return self.data / "bronze"

    @property
    def silver(self) -> Path:
        return self.data / "silver"

    @property
    def gold(self) -> Path:
        return self.data / "gold"

    @property
    def duckdb(self) -> Path:
        return _resolve(_env("GRIDPULSE_DUCKDB_PATH", "data/gold/gridpulse.duckdb"))

    @property
    def artifacts(self) -> Path:
        return _resolve(_env("GRIDPULSE_ARTIFACTS_DIR", "artifacts"))

    def ensure(self) -> None:
        for p in (self.bronze, self.silver, self.gold, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


@dataclass(frozen=True)
class Settings:
    eia_api_key: str = field(default_factory=lambda: _env("EIA_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    start_date: str = field(default_factory=lambda: _env("GRIDPULSE_START_DATE", "2019-01-01"))
    max_concurrency: int = field(default_factory=lambda: int(_env("GRIDPULSE_MAX_CONCURRENCY", "6")))
    page_size: int = field(default_factory=lambda: int(_env("GRIDPULSE_PAGE_SIZE", "5000")))
    request_timeout: int = field(default_factory=lambda: int(_env("GRIDPULSE_REQUEST_TIMEOUT", "60")))

    def require_eia_key(self) -> str:
        if not self.eia_api_key or self.eia_api_key.startswith("your_"):
            raise RuntimeError(
                "EIA_API_KEY is not set. Get a free key at "
                "https://www.eia.gov/opendata/register.php then put it in your .env file."
            )
        return self.eia_api_key

    @property
    def has_llm(self) -> bool:
        return bool(self.groq_api_key) and not self.groq_api_key.startswith("your_")


SETTINGS = Settings()

FORECAST_HORIZON = 24
LOOKBACK_HOURS = 168
QUANTILES = (0.1, 0.5, 0.9)
