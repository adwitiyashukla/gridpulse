"""One LightGBM model across all 12 regions, plus P10/P50/P90 prediction bands."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridpulse.config import PATHS, QUANTILES
from gridpulse.features.build import FEATURE_COLUMNS, TARGET

logger = logging.getLogger(__name__)

CATEGORICAL = ["ba_code"]


@dataclass
class BATargetScaler:
    """Scales demand separately for each region so they can share one model."""

    stats: dict[str, tuple[float, float]]
    global_mean: float
    global_std: float

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> BATargetScaler:
        """Median and IQR per region, so one bad reading cannot skew the scaling."""
        grouped = frame.groupby("ba_code")[TARGET]
        centre = grouped.median()
        spread = (grouped.quantile(0.75) - grouped.quantile(0.25)) / 1.349

        stats = {
            str(ba): (
                float(centre[ba]),
                float(spread[ba]) if spread[ba] > 0 else max(float(centre[ba]) * 0.1, 1.0),
            )
            for ba in centre.index
        }

        global_centre = float(frame[TARGET].median())
        global_spread = float(
            (frame[TARGET].quantile(0.75) - frame[TARGET].quantile(0.25)) / 1.349
        )
        return cls(
            stats=stats,
            global_mean=global_centre,
            global_std=global_spread if global_spread > 0 else 1.0,
        )

    def _lookup(self, ba: str) -> tuple[float, float]:
        return self.stats.get(str(ba), (self.global_mean, self.global_std))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        params = frame["ba_code"].map(lambda b: self._lookup(b))
        mean = np.array([p[0] for p in params], dtype=float)
        std = np.array([p[1] for p in params], dtype=float)
        return ((frame[TARGET].to_numpy(dtype=float) - mean) / std)

    def inverse(self, values: np.ndarray, ba_codes: pd.Series) -> np.ndarray:
        params = ba_codes.map(lambda b: self._lookup(b))
        mean = np.array([p[0] for p in params], dtype=float)
        std = np.array([p[1] for p in params], dtype=float)
        return np.asarray(values, dtype=float) * std + mean

    def to_dict(self) -> dict:
        return {
            "stats": {k: list(v) for k, v in self.stats.items()},
            "global_mean": self.global_mean,
            "global_std": self.global_std,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BATargetScaler:
        return cls(
            stats={k: (v[0], v[1]) for k, v in payload["stats"].items()},
            global_mean=payload["global_mean"],
            global_std=payload["global_std"],
        )


def _base_params(quick: bool) -> dict:
    """Hyperparameters tuned for a constrained CPU rather than a GPU cluster.

    ``num_leaves`` is kept modest and ``feature_fraction`` low, which costs a little
    accuracy but keeps training to seconds on a laptop and keeps the model small
    enough to load inside a free-tier hosting container.
    """
    return {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 63 if quick else 127,
        "learning_rate": 0.08 if quick else 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 60,
        "lambda_l2": 1.0,
        "num_threads": 4,
        "verbose": -1,
        "seed": 42,
    }


@dataclass
class TrainedGBM:
    point_model: object
    quantile_models: dict[float, object]
    feature_names: list[str]
    ba_categories: list[str]
    best_iteration: int
    target_scaler: BATargetScaler

    def save(self, directory: Path | None = None) -> Path:
        target = Path(directory) if directory else PATHS.artifacts / "gbm"
        target.mkdir(parents=True, exist_ok=True)

        self.point_model.save_model(str(target / "point.txt"))
        for q, model in self.quantile_models.items():
            model.save_model(str(target / f"q{int(q * 100)}.txt"))

        (target / "meta.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "ba_categories": self.ba_categories,
                    "quantiles": list(self.quantile_models),
                    "best_iteration": self.best_iteration,
                    "target_scaler": self.target_scaler.to_dict(),
                },
                indent=2,
            )
        )
        logger.info("GBM artifacts written to %s", target)
        return target

    @classmethod
    def load(cls, directory: Path | None = None) -> TrainedGBM:
        import lightgbm as lgb

        target = Path(directory) if directory else PATHS.artifacts / "gbm"
        meta = json.loads((target / "meta.json").read_text())
        return cls(
            point_model=lgb.Booster(model_file=str(target / "point.txt")),
            quantile_models={
                float(q): lgb.Booster(model_file=str(target / f"q{int(float(q) * 100)}.txt"))
                for q in meta["quantiles"]
            },
            feature_names=meta["feature_names"],
            ba_categories=meta["ba_categories"],
            best_iteration=meta["best_iteration"],
            target_scaler=BATargetScaler.from_dict(meta["target_scaler"]),
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Point and quantile predictions, returned in megawatthours.

        The models operate on a per-BA z-score, so every output is inverted back
        to physical units here rather than leaving that to the caller.
        """
        matrix = prepare_matrix(frame, self.ba_categories, self.feature_names)
        ba_codes = frame["ba_code"]

        out = pd.DataFrame(index=frame.index)
        out["pred_gbm"] = self.target_scaler.inverse(self.point_model.predict(matrix), ba_codes)
        for q, model in sorted(self.quantile_models.items()):
            out[f"pred_gbm_p{int(q * 100)}"] = self.target_scaler.inverse(
                model.predict(matrix), ba_codes
            )
        quantile_columns = [c for c in out.columns if c.startswith("pred_gbm_p")]
        out[quantile_columns] = np.sort(out[quantile_columns].to_numpy(), axis=1)
        return out

    def importance(self, top_n: int = 25) -> pd.DataFrame:
        gains = self.point_model.feature_importance(importance_type="gain")
        names = list(self.point_model.feature_name())
        return (
            pd.DataFrame({"feature": names, "gain": gains})
            .sort_values("gain", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


def prepare_matrix(
    frame: pd.DataFrame, ba_categories: list[str], feature_names: list[str]
) -> pd.DataFrame:
    """Assemble the model matrix with a stable categorical encoding for ``ba_code``."""
    matrix = frame[feature_names].copy()
    matrix["ba_code"] = pd.Categorical(frame["ba_code"], categories=ba_categories)
    return matrix


def train_gbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    quick: bool = False,
    quantiles: tuple[float, ...] = QUANTILES,
    features: list[str] | None = None,
) -> TrainedGBM:
    """Fit the point model plus one model per quantile.

    Parameters
    ----------
    features
        Override the feature set. Used to train a hybrid variant that additionally
        consumes the EIA's published day-ahead forecast as an input.
    """
    import lightgbm as lgb

    ba_categories = sorted(pd.concat([train["ba_code"], valid["ba_code"]]).unique().tolist())
    features = list(features) if features else list(FEATURE_COLUMNS)

    target_scaler = BATargetScaler.fit(train)

    x_train = prepare_matrix(train, ba_categories, features)
    x_valid = prepare_matrix(valid, ba_categories, features)
    y_train = target_scaler.transform(train)
    y_valid = target_scaler.transform(valid)

    train_set = lgb.Dataset(x_train, y_train, categorical_feature=CATEGORICAL, free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set, categorical_feature=CATEGORICAL, free_raw_data=False)

    rounds = 400 if quick else 3000
    stopping = 50 if quick else 200

    logger.info(
        "Training GBM point model (%d features, %s rows, per-BA normalised target)",
        len(features), f"{len(train):,}",
    )
    point = lgb.train(
        _base_params(quick),
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping, verbose=False), lgb.log_evaluation(0)],
    )
    logger.info("  point model stopped at iteration %d", point.best_iteration)

    quantile_rounds = int(min(max(200, point.best_iteration), 250 if quick else 700))
    quantile_models: dict[float, object] = {}
    for q in quantiles or ():
        params = _base_params(quick) | {"objective": "quantile", "alpha": q, "metric": "quantile"}
        logger.info("  training quantile model P%d (max %d rounds)", int(q * 100), quantile_rounds)
        quantile_models[q] = lgb.train(
            params,
            train_set,
            num_boost_round=quantile_rounds,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(stopping, verbose=False), lgb.log_evaluation(0)],
        )

    return TrainedGBM(
        point_model=point,
        quantile_models=quantile_models,
        feature_names=features,
        ba_categories=ba_categories,
        best_iteration=point.best_iteration,
        target_scaler=target_scaler,
    )
