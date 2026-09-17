"""Entrena el modelo ganador (ver docs/model_selection.md) con el 100% de
data/3.final/train_final.csv y lo guarda en models/best_model.pkl.

`num_boost_round` / `n_estimators` / `iterations` NO se fija en un número
arbitrario: se separa un hold-out temporal del 12% más reciente del histórico
completo, se entrena con early stopping (100 rondas) sobre ese hold-out para
encontrar `best_iteration_full`, y luego se reentrena con el 100% de los datos
usando ese número de rondas ya validado (mismo criterio aplicado en
notebooks/03-model.ipynb, sección 7).

Uso:
    python scripts/train.py                # usa el modelo ganador (docs/model_selection.md)
    python scripts/train.py --model lgb     # fuerza un modelo puntual
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_metrics  # noqa: E402
from src.features.build_features import FEATURES, TARGET, build_temporal_features  # noqa: E402
from src.models.train_model import CATEGORICAL_FEATURES, get_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRAIN_FINAL_PATH = PROJECT_ROOT / "data" / "3.final" / "train_final.csv"
MODEL_OUT_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

# Ganador por defecto según docs/model_selection.md (criterio: WAPE % más bajo).
WINNER_MODEL = "catboost"

HOLDOUT_FRAC = 0.12
EARLY_STOPPING_ROUNDS = 100
MAX_ROUNDS = 2000


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not TRAIN_FINAL_PATH.exists():
        raise FileNotFoundError(
            f"No existe {TRAIN_FINAL_PATH}. Corre antes:\n"
            "  python -m src.data.make_dataset\n"
            "  python -m src.data.make_final_dataset"
        )
    df = pd.read_csv(TRAIN_FINAL_PATH)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = build_temporal_features(df)
    df = df.dropna(subset=["lag_28", "rolling_mean_7_14"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df[FEATURES], df[TARGET]


def _temporal_split(X: pd.DataFrame, y: pd.Series, holdout_frac: float):
    split_idx = int(len(X) * (1 - holdout_frac))
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


def train_lgb(X: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb

    params = get_model("lgb")
    X_fit, y_fit, X_hold, y_hold = _temporal_split(X, y, HOLDOUT_FRAC)

    fit_set = lgb.Dataset(X_fit, label=y_fit, categorical_feature=CATEGORICAL_FEATURES)
    hold_set = lgb.Dataset(X_hold, label=y_hold, reference=fit_set)

    probe = lgb.train(
        params, fit_set, valid_sets=[hold_set], num_boost_round=MAX_ROUNDS,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    best_iteration = probe.best_iteration
    logger.info("LightGBM best_iteration (hold-out %.0f%%): %d", HOLDOUT_FRAC * 100, best_iteration)

    full_set = lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(params, full_set, num_boost_round=best_iteration)


def train_xgb(X: pd.DataFrame, y: pd.Series):
    X_fit, y_fit, X_hold, y_hold = _temporal_split(X, y, HOLDOUT_FRAC)

    probe = get_model("xgb", n_estimators=MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    probe.fit(X_fit, y_fit, eval_set=[(X_hold, y_hold)], verbose=False)
    best_iteration = probe.best_iteration + 1  # best_iteration es 0-indexed
    logger.info("XGBoost best_iteration (hold-out %.0f%%): %d", HOLDOUT_FRAC * 100, best_iteration)

    final_model = get_model("xgb", n_estimators=best_iteration)
    final_model.fit(X, y, verbose=False)
    return final_model


def train_catboost(X: pd.DataFrame, y: pd.Series):
    X_fit, y_fit, X_hold, y_hold = _temporal_split(X, y, HOLDOUT_FRAC)

    probe = get_model("catboost", iterations=MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    probe.fit(X_fit, y_fit, eval_set=(X_hold, y_hold), cat_features=CATEGORICAL_FEATURES)
    best_iteration = probe.get_best_iteration()
    logger.info("CatBoost best_iteration (hold-out %.0f%%): %d", HOLDOUT_FRAC * 100, best_iteration)

    final_model = get_model("catboost", iterations=best_iteration, early_stopping_rounds=None)
    final_model.fit(X, y, cat_features=CATEGORICAL_FEATURES)
    return final_model


TRAINERS = {"lgb": train_lgb, "xgb": train_xgb, "catboost": train_catboost}


def main(model_name: str = WINNER_MODEL) -> None:
    if model_name not in TRAINERS:
        raise ValueError(f"Modelo desconocido: {model_name!r}. Usa uno de {list(TRAINERS)}")

    logger.info("Cargando %s y generando features...", TRAIN_FINAL_PATH)
    X, y = load_training_data()
    logger.info("Dataset de entrenamiento: %s filas, %d features", f"{len(X):,}", len(FEATURES))

    logger.info("Entrenando modelo final: %s", model_name)
    model = TRAINERS[model_name](X, y)

    # Métrica de sanity-check sobre el propio hold-out temporal de entrenamiento
    # (no reemplaza la comparativa oficial de docs/model_selection.md).
    X_fit, y_fit, X_hold, y_hold = _temporal_split(X, y, HOLDOUT_FRAC)
    sanity_metrics = evaluate_metrics(y_hold, model.predict(X_hold))
    logger.info("Sanity-check sobre hold-out: WAPE %% = %.3f | R2 = %.3f", sanity_metrics["WAPE %"], sanity_metrics["R2"])

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_name": model_name, "features": FEATURES}, MODEL_OUT_PATH)
    logger.info("Modelo guardado en %s", MODEL_OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(TRAINERS), default=WINNER_MODEL)
    args = parser.parse_args()
    main(args.model)
