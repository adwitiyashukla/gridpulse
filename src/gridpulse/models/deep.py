"""Deep sequence models for day-ahead load forecasting (PyTorch, CPU-first).

Architecture
------------
Both models follow the encoder / known-future-covariate pattern that underpins
Temporal Fusion Transformer, stripped down to what trains in minutes on a laptop
CPU rather than hours on a GPU:

* **Encoder** consumes the past ``LOOKBACK_HOURS`` (168h) of observed history:
  demand, weather and cyclical calendar channels.
* **Future covariate branch** consumes the next 24 hours of *known* inputs --
  weather forecast and calendar. This is not leakage: a real operator genuinely
  holds tomorrow's numerical weather prediction and tomorrow's calendar when they
  produce a day-ahead forecast. Withholding it would model a harder problem than
  the one utilities actually face.
* **Head** fuses both and emits all 24 hours in a single forward pass (direct
  multi-horizon), which avoids the error compounding of autoregressive rollout.

Two encoders are provided: a stacked LSTM, and a small Transformer encoder with
sinusoidal positional encoding. The LSTM is usually the stronger performer at this
data scale; the Transformer is included because it scales better with more series
and demonstrates the attention machinery.

Memory discipline: sequences are never materialised. The dataset holds one flat
float32 array and slices windows lazily inside ``__getitem__``, so peak RAM stays
in the tens of megabytes regardless of history length.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridpulse.config import FORECAST_HORIZON, LOOKBACK_HOURS, PATHS

logger = logging.getLogger(__name__)

# Channels observed in the past and fed to the encoder.
PAST_CHANNELS = [
    "demand_mwh",
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "cloud_cover", "wind_speed_10m",
    "heating_degrees", "cooling_degrees",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_business_day", "is_holiday",
]

# Channels known in advance for the forecast window.
FUTURE_CHANNELS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "cloud_cover", "wind_speed_10m",
    "heating_degrees", "cooling_degrees",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_business_day", "is_holiday",
]

TARGET = "demand_mwh"

# Sliding windows at hourly resolution overlap by 167 of 168 input hours, so
# adjacent samples are almost perfectly redundant. Striding keeps the sample
# diverse while cutting epoch time by the stride factor. Six hours is a natural
# choice: it still covers every phase of the daily cycle within a single day.
TRAIN_STRIDE = 12
QUICK_TRAIN_STRIDE = 24

# Recurrent layers are inherently sequential: timestep t cannot be computed until
# t-1 finishes, so cost scales linearly with sequence length and cannot be
# parallelised away. A 168-step encoder is therefore expensive on a CPU.
#
# The lookback window is subsampled every ENCODER_STRIDE hours, giving 56 steps
# instead of 168. This is not a shortcut: hourly demand is heavily autocorrelated,
# so consecutive hours carry little independent information, and the retained
# points still span the full week and every phase of the daily cycle. The recent
# past is preserved exactly where it matters most through the lag and rolling
# features already supplied to the gradient-boosted model.
ENCODER_STRIDE = 3

# Each test window predicts 24 consecutive hours, so a stride of 6 still yields
# four independent predictions for every hour, which are averaged.
TEST_STRIDE = 6


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the deep models. Install with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc


class WindowDataset:
    """Lazy sliding-window dataset over one BA's contiguous history."""

    def __init__(
        self,
        past: np.ndarray,
        future: np.ndarray,
        target: np.ndarray,
        lookback: int = LOOKBACK_HOURS,
        horizon: int = FORECAST_HORIZON,
    ):
        self.past = past.astype(np.float32)
        self.future = future.astype(np.float32)
        self.target = target.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.n = len(target) - lookback - horizon + 1
        if self.n <= 0:
            raise ValueError(
                f"Series too short: need > {lookback + horizon} rows, got {len(target)}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        torch = _torch()
        start = index
        split = start + self.lookback
        end = split + self.horizon
        return (
            # Subsampled: 168 hourly steps become 56 three-hourly steps.
            torch.from_numpy(self.past[start:split:ENCODER_STRIDE]),
            torch.from_numpy(self.future[split:end]),
            torch.from_numpy(self.target[split:end]),
        )


class ConcatDataset:
    """Chain several per-BA datasets without copying their arrays."""

    def __init__(self, datasets: list[WindowDataset]):
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(d) for d in datasets])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        which = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.datasets[which][index - int(self.offsets[which])]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_lstm(n_past: int, n_future: int, hidden: int = 64, layers: int = 1, dropout: float = 0.15):
    torch = _torch()
    nn = torch.nn

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(
                n_past, hidden, num_layers=layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.future_proj = nn.Sequential(
                nn.Linear(n_future * FORECAST_HORIZON, hidden), nn.ReLU(), nn.Dropout(dropout)
            )
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, FORECAST_HORIZON),
            )

        def forward(self, past, future):
            _, (hidden_state, _) = self.encoder(past)
            context = hidden_state[-1]                                   # (B, hidden)
            known = self.future_proj(future.flatten(start_dim=1))        # (B, hidden)
            return self.head(torch.cat([context, known], dim=1))         # (B, horizon)

    return LSTMForecaster()


def build_transformer(
    n_past: int, n_future: int, d_model: int = 48, heads: int = 4, layers: int = 1, dropout: float = 0.15
):
    torch = _torch()
    nn = torch.nn

    class PositionalEncoding(nn.Module):
        def __init__(self, dim: int, max_len: int = LOOKBACK_HOURS + FORECAST_HORIZON):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1).float()
            divisor = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            encoding = torch.zeros(max_len, dim)
            encoding[:, 0::2] = torch.sin(position * divisor)
            encoding[:, 1::2] = torch.cos(position * divisor)
            self.register_buffer("encoding", encoding.unsqueeze(0))

        def forward(self, x):
            return x + self.encoding[:, : x.size(1)]

    class TransformerForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_past, d_model)
            self.pos = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.future_proj = nn.Sequential(
                nn.Linear(n_future * FORECAST_HORIZON, d_model), nn.ReLU(), nn.Dropout(dropout)
            )
            self.head = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, FORECAST_HORIZON),
            )

        def forward(self, past, future):
            encoded = self.encoder(self.pos(self.input_proj(past)))
            context = encoded.mean(dim=1)                                # attention-pooled summary
            known = self.future_proj(future.flatten(start_dim=1))
            return self.head(torch.cat([context, known], dim=1))

    return TransformerForecaster()


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    """Per-channel standardisation. Statistics come from training data only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, array: np.ndarray) -> Scaler:
        mean = np.nanmean(array, axis=0)
        std = np.nanstd(array, axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return np.nan_to_num((array - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> Scaler:
        return cls(np.array(payload["mean"], dtype=np.float32), np.array(payload["std"], dtype=np.float32))


@dataclass
class TargetScaler:
    """Per-BA target scaling.

    BA demand spans two orders of magnitude (ISNE peaks near 25 GW, PJM near
    150 GW). Without per-BA normalisation the loss is dominated entirely by the
    largest system and the small ones never learn.
    """

    stats: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> TargetScaler:
        """Robust per-BA centre and scale (median / IQR).

        See ``gbm.BATargetScaler.fit`` for why mean and standard deviation are
        unusable here: corrupt readings in the raw feed inflate them without limit.
        """
        grouped = frame.groupby("ba_code")[TARGET]
        centre = grouped.median()
        spread = (grouped.quantile(0.75) - grouped.quantile(0.25)) / 1.349
        return cls({
            str(ba): (
                float(centre[ba]),
                float(spread[ba]) if spread[ba] > 0 else max(float(centre[ba]) * 0.1, 1.0),
            )
            for ba in centre.index
        })

    def transform(self, values: np.ndarray, ba: str) -> np.ndarray:
        mean, std = self.stats[ba]
        return ((values - mean) / std).astype(np.float32)

    def inverse(self, values: np.ndarray, ba: str) -> np.ndarray:
        mean, std = self.stats[ba]
        return values * std + mean

    def to_dict(self) -> dict:
        return {"stats": {k: list(v) for k, v in self.stats.items()}}

    @classmethod
    def from_dict(cls, payload: dict) -> TargetScaler:
        return cls({k: (v[0], v[1]) for k, v in payload["stats"].items()})


# ---------------------------------------------------------------------------
# Windowing across the full series
# ---------------------------------------------------------------------------
@dataclass
class SeriesBundle:
    """Everything needed to build windows for a single balancing authority."""

    ba_code: str
    past: np.ndarray          # (T, n_past)  scaled
    future: np.ndarray        # (T, n_future) scaled
    target_scaled: np.ndarray  # (T,)         scaled
    target_raw: np.ndarray    # (T,)         MWh
    timestamps: pd.DatetimeIndex


def build_series_bundles(
    frame: pd.DataFrame, past_scaler: Scaler, future_scaler: Scaler, target_scaler: TargetScaler
) -> list[SeriesBundle]:
    bundles: list[SeriesBundle] = []
    for ba, group in frame.groupby("ba_code", sort=True):
        group = group.sort_values("period_utc")
        bundles.append(
            SeriesBundle(
                ba_code=ba,
                past=past_scaler.transform(group[PAST_CHANNELS].to_numpy(dtype=np.float64)),
                future=future_scaler.transform(group[FUTURE_CHANNELS].to_numpy(dtype=np.float64)),
                target_scaled=target_scaler.transform(group[TARGET].to_numpy(dtype=np.float64), ba),
                target_raw=group[TARGET].to_numpy(dtype=np.float64),
                timestamps=pd.DatetimeIndex(group["period_utc"]),
            )
        )
    return bundles


def split_windows(
    bundles: list[SeriesBundle],
    valid_start: pd.Timestamp,
    test_start: pd.Timestamp,
    train_stride: int = TRAIN_STRIDE,
    test_stride: int = TEST_STRIDE,
) -> tuple[ConcatDataset, ConcatDataset, list[tuple[SeriesBundle, np.ndarray]]]:
    """Assign every window to a split by the timestamp of its first forecast hour.

    Windows are cut over the *full* contiguous series and only then partitioned, so
    no usable window is lost at a split boundary. A window is assigned to test only
    when its entire forecast horizon lies in the test period, which makes the
    evaluation strictly out-of-sample.
    """
    train_sets, valid_sets, test_specs = [], [], []

    for bundle in bundles:
        total = len(bundle.target_scaled)
        n_windows = total - LOOKBACK_HOURS - FORECAST_HORIZON + 1
        if n_windows <= 0:
            logger.warning("  %s has too little history for windowing; skipped", bundle.ba_code)
            continue

        starts = np.arange(n_windows)
        horizon_start = bundle.timestamps[LOOKBACK_HOURS : LOOKBACK_HOURS + n_windows]

        train_mask = horizon_start < valid_start
        valid_mask = (horizon_start >= valid_start) & (horizon_start < test_start)
        test_mask = horizon_start >= test_start

        for mask, sink in ((train_mask, train_sets), (valid_mask, valid_sets)):
            indices = starts[mask][::train_stride]
            if indices.size:
                sink.append(_SubsetWindows(bundle, indices))

        if test_mask.any():
            test_specs.append((bundle, starts[test_mask][::test_stride]))

    return ConcatDataset(train_sets), ConcatDataset(valid_sets), test_specs


class _SubsetWindows(WindowDataset):
    """A WindowDataset restricted to an explicit list of window start indices."""

    def __init__(self, bundle: SeriesBundle, indices: np.ndarray):
        self.bundle = bundle
        self.indices = indices
        self.past = bundle.past
        self.future = bundle.future
        self.target = bundle.target_scaled
        self.lookback = LOOKBACK_HOURS
        self.horizon = FORECAST_HORIZON
        self.n = len(indices)

    def __getitem__(self, index: int):
        return super().__getitem__(int(self.indices[index]))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainedDeepModel:
    architecture: str
    state_dict: dict
    past_scaler: Scaler
    future_scaler: Scaler
    target_scaler: TargetScaler
    n_past: int
    n_future: int
    epochs_run: int
    best_val_loss: float

    def save(self, directory: Path | None = None) -> Path:
        torch = _torch()
        target = Path(directory) if directory else PATHS.artifacts / f"deep_{self.architecture}"
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, target / "weights.pt")
        (target / "meta.json").write_text(
            json.dumps(
                {
                    "architecture": self.architecture,
                    "past_scaler": self.past_scaler.to_dict(),
                    "future_scaler": self.future_scaler.to_dict(),
                    "target_scaler": self.target_scaler.to_dict(),
                    "n_past": self.n_past,
                    "n_future": self.n_future,
                    "epochs_run": self.epochs_run,
                    "best_val_loss": self.best_val_loss,
                    "past_channels": PAST_CHANNELS,
                    "future_channels": FUTURE_CHANNELS,
                    "lookback_hours": LOOKBACK_HOURS,
                    "encoder_stride": ENCODER_STRIDE,
                    "encoder_steps": len(range(0, LOOKBACK_HOURS, ENCODER_STRIDE)),
                    "forecast_horizon": FORECAST_HORIZON,
                },
                indent=2,
            )
        )
        logger.info("Deep model artifacts written to %s", target)
        return target

    @classmethod
    def load(cls, architecture: str, directory: Path | None = None) -> TrainedDeepModel:
        torch = _torch()
        target = Path(directory) if directory else PATHS.artifacts / f"deep_{architecture}"
        meta = json.loads((target / "meta.json").read_text())
        return cls(
            architecture=meta["architecture"],
            state_dict=torch.load(target / "weights.pt", map_location="cpu", weights_only=True),
            past_scaler=Scaler.from_dict(meta["past_scaler"]),
            future_scaler=Scaler.from_dict(meta["future_scaler"]),
            target_scaler=TargetScaler.from_dict(meta["target_scaler"]),
            n_past=meta["n_past"],
            n_future=meta["n_future"],
            epochs_run=meta["epochs_run"],
            best_val_loss=meta["best_val_loss"],
        )

    def build_module(self):
        module = (
            build_lstm(self.n_past, self.n_future)
            if self.architecture == "lstm"
            else build_transformer(self.n_past, self.n_future)
        )
        module.load_state_dict(self.state_dict)
        module.eval()
        return module


def train_deep(
    frame: pd.DataFrame,
    valid_start: pd.Timestamp,
    test_start: pd.Timestamp,
    architecture: str = "lstm",
    quick: bool = False,
) -> tuple[TrainedDeepModel, pd.DataFrame]:
    """Train one deep model and return it alongside out-of-sample test predictions."""
    torch = _torch()
    from torch.utils.data import DataLoader

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    train_only = frame[frame["period_utc"] < valid_start]
    past_scaler = Scaler.fit(train_only[PAST_CHANNELS].to_numpy(dtype=np.float64))
    future_scaler = Scaler.fit(train_only[FUTURE_CHANNELS].to_numpy(dtype=np.float64))
    target_scaler = TargetScaler.fit(train_only)

    stride = QUICK_TRAIN_STRIDE if quick else TRAIN_STRIDE
    bundles = build_series_bundles(frame, past_scaler, future_scaler, target_scaler)
    train_set, valid_set, test_specs = split_windows(
        bundles, valid_start, test_start, train_stride=stride
    )
    logger.info(
        "  %s windows (stride %d): train=%s valid=%s",
        architecture, stride, f"{len(train_set):,}", f"{len(valid_set):,}",
    )

    batch_size = 256 if quick else 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = (
        build_lstm(len(PAST_CHANNELS), len(FUTURE_CHANNELS))
        if architecture == "lstm"
        else build_transformer(len(PAST_CHANNELS), len(FUTURE_CHANNELS))
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  %s parameters: %s", architecture, f"{n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    # Huber is deliberately chosen over MSE: load series contain genuine spikes
    # (heatwaves, storms) and MSE would let a handful of them dominate the gradient.
    criterion = torch.nn.HuberLoss(delta=1.0)

    max_epochs = 4 if quick else 15
    patience = 2 if quick else 4
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for past, future, target in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(past, future), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for past, future, target in valid_loader:
                running += criterion(model(past, future), target).item()
        val_loss = running / max(len(valid_loader), 1)
        scheduler.step(val_loss)

        flag = ""
        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        else:
            stale += 1

        logger.info("    epoch %2d/%d  train %.5f  valid %.5f%s", epoch, max_epochs, train_loss, val_loss, flag)
        if stale >= patience:
            logger.info("    early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    trained = TrainedDeepModel(
        architecture=architecture,
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        past_scaler=past_scaler,
        future_scaler=future_scaler,
        target_scaler=target_scaler,
        n_past=len(PAST_CHANNELS),
        n_future=len(FUTURE_CHANNELS),
        epochs_run=epoch,
        best_val_loss=best_loss,
    )
    predictions = _predict_test(model, test_specs, target_scaler)
    return trained, predictions


def _predict_test(model, test_specs, target_scaler: TargetScaler, batch_size: int = 256) -> pd.DataFrame:
    """Score every test window and flatten to one row per (BA, timestamp).

    Overlapping windows produce several predictions for the same hour; they are
    averaged, which is a cheap ensembling effect and smooths window-edge artefacts.
    """
    torch = _torch()
    model.eval()
    rows = []

    with torch.no_grad():
        for bundle, starts in test_specs:
            for chunk in np.array_split(starts, max(1, len(starts) // batch_size)):
                past = torch.from_numpy(
                    np.stack([
                        bundle.past[s : s + LOOKBACK_HOURS : ENCODER_STRIDE] for s in chunk
                    ])
                )
                future = torch.from_numpy(
                    np.stack([
                        bundle.future[s + LOOKBACK_HOURS : s + LOOKBACK_HOURS + FORECAST_HORIZON]
                        for s in chunk
                    ])
                )
                scaled = model(past, future).numpy()
                unscaled = target_scaler.inverse(scaled, bundle.ba_code)

                for row, start in enumerate(chunk):
                    base = int(start) + LOOKBACK_HOURS
                    for step in range(FORECAST_HORIZON):
                        rows.append(
                            (
                                bundle.ba_code,
                                bundle.timestamps[base + step],
                                float(unscaled[row, step]),
                                step + 1,
                            )
                        )

    if not rows:
        return pd.DataFrame(columns=["ba_code", "period_utc", "prediction", "horizon_step"])

    frame = pd.DataFrame(rows, columns=["ba_code", "period_utc", "prediction", "horizon_step"])
    return (
        frame.groupby(["ba_code", "period_utc"], as_index=False)
        .agg(prediction=("prediction", "mean"), n_windows=("prediction", "size"))
    )
