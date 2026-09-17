"""Reproduce la comparativa LightGBM vs XGBoost vs CatBoost de
notebooks/03-model.ipynb (mismo split 80/20 temporal, mismas métricas) y
guarda el resultado en output/metrics_comparison.csv.

Uso:
    python scripts/evaluate.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_metrics  # noqa: E402
from src.features.build_features import FEATURES, TARGET, build_temporal_features  # noqa: E402
from src.models.train_model import CATEGORICAL_FEATURES, get_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRAIN_FINAL_PATH = PROJECT_ROOT / "data" / "3.final" / "train_final.csv"
METRICS_OUT_PATH = PROJECT_ROOT / "output" / "metrics_comparison.csv"

VALID_FRAC = 0.2
EARLY_STOPPING_ROUNDS = 100
MAX_ROUNDS = 2000


def load_split():
    df = pd.read_csv(TRAIN_FINAL_PATH)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = build_temporal_features(df)
    df = df.dropna(subset=["lag_28", "rolling_mean_7_28"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * (1 - VALID_FRAC))
    train_df, valid_df = df.iloc[:split_idx], df.iloc[split_idx:]
    return train_df[FEATURES], train_df[TARGET], valid_df[FEATURES], valid_df[TARGET]


def eval_lgb(X_train, y_train, X_valid, y_valid) -> dict:
    import lightgbm as lgb

    params = get_model("lgb")
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    model = lgb.train(
        params, train_set, valid_sets=[valid_set], num_boost_round=MAX_ROUNDS,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    return evaluate_metrics(y_valid, model.predict(X_valid))


def eval_xgb(X_train, y_train, X_valid, y_valid) -> dict:
    model = get_model("xgb", n_estimators=MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return evaluate_metrics(y_valid, model.predict(X_valid))


def eval_catboost(X_train, y_train, X_valid, y_valid) -> dict:
    model = get_model("catboost", iterations=MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=CATEGORICAL_FEATURES)
    return evaluate_metrics(y_valid, model.predict(X_valid))


EVALUATORS = {"LightGBM": eval_lgb, "XGBoost": eval_xgb, "CatBoost": eval_catboost}


def main() -> None:
    logger.info("Cargando %s y generando features...", TRAIN_FINAL_PATH)
    X_train, y_train, X_valid, y_valid = load_split()
    logger.info("Train: %d filas | Valid: %d filas", len(X_train), len(X_valid))

    rows = {}
    for name, evaluator in EVALUATORS.items():
        logger.info("Entrenando %s...", name)
        rows[name] = evaluator(X_train, y_train, X_valid, y_valid)
        logger.info("  %s -> WAPE %% = %.3f | R2 = %.3f", name, rows[name]["WAPE %"], rows[name]["R2"])

    comparison = pd.DataFrame(rows).T.sort_values("WAPE %", ascending=True)

    METRICS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(METRICS_OUT_PATH)
    logger.info("Comparativa guardada en %s", METRICS_OUT_PATH)
    logger.info("Ganador (WAPE %% más bajo): %s", comparison.index[0])


if __name__ == "__main__":
    main()
