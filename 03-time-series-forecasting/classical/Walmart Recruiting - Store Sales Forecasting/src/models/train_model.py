"""
Entrena el modelo de forecasting Store-Dept.

1. Carga data/3.final/{final_train,final_test}.csv
2. Calcula lags/rolling means (train+test concatenados, ver src/features/build_features.py)
3. Evalúa en un split temporal (últimas VALIDATION_WEEKS semanas) y reporta WMAE/MAE/RMSE/R2
4. Reentrena con TODOS los datos disponibles (igual que el experimento ganador) y
   guarda el modelo final en models/<algoritmo>/

Uso:
    python src/models/train_model.py                  # XGBoost (default, ganador actual)
    python src/models/train_model.py --model lgbm      # baseline LightGBM
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL,
    FEATURES,
    FINAL_TEST_PATH,
    FINAL_TRAIN_PATH,
    MODEL_REGISTRY,
    TARGET,
)
from src.evaluation.metrics import evaluate
from src.features.build_features import build_training_frame, time_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_estimator(model_name: str, params: dict):
    if model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(**params)
    if model_name == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(**params)
    raise ValueError(f"Modelo desconocido: {model_name}. Usa 'xgboost' o 'lgbm'.")


def fit(model_name: str, estimator, X, y, sample_weight, eval_set=None, eval_sample_weight=None):
    fit_kwargs = {"sample_weight": sample_weight}
    if model_name == "lgbm":
        fit_kwargs["categorical_feature"] = CATEGORICAL_FEATURES
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["eval_sample_weight"] = eval_sample_weight
            fit_kwargs["eval_metric"] = "mae"
    elif model_name == "xgboost":
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        fit_kwargs["verbose"] = False
    estimator.fit(X, y, **fit_kwargs)
    return estimator


def validate(model_name: str, params: dict, train_with_lags: pd.DataFrame) -> dict:
    """Entrena en train_split y evalúa en valid_split (últimas N semanas)."""
    train_split, valid_split = time_split(train_with_lags)

    X_train, y_train = train_split[FEATURES], train_split[TARGET]
    X_valid, y_valid = valid_split[FEATURES], valid_split[TARGET]

    weights_train = np.where(X_train["IsHoliday"], 5, 1)
    weights_valid = np.where(X_valid["IsHoliday"], 5, 1)

    estimator = get_estimator(model_name, params)
    fit(
        model_name, estimator, X_train, y_train, weights_train,
        eval_set=[(X_valid, y_valid)], eval_sample_weight=[weights_valid],
    )

    pred_valid = estimator.predict(X_valid)
    report = evaluate(y_valid, pred_valid, X_valid["IsHoliday"])
    logger.info(f"[{model_name}] Validación (últimas {len(valid_split)} filas): {report}")
    return report.as_dict()


def train_final(model_name: str, params: dict, train_with_lags: pd.DataFrame):
    """Reentrena con TODOS los datos de train disponibles (sin split), listo para producción."""
    X_full = train_with_lags[FEATURES]
    y_full = train_with_lags[TARGET]
    weights_full = np.where(X_full["IsHoliday"], 5, 1)

    estimator = get_estimator(model_name, params)
    fit(model_name, estimator, X_full, y_full, weights_full)
    return estimator


def save_model(model_name: str, estimator):
    registry = MODEL_REGISTRY[model_name]
    model_dir = registry["dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / registry["model_file"]
    joblib.dump(estimator, model_path)

    features_path = model_dir / registry["features_file"]
    with open(features_path, "w") as f:
        json.dump(FEATURES, f, indent=2)

    logger.info(f"Modelo guardado en: {model_path}")
    logger.info(f"Features guardadas en: {features_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo de forecasting Walmart Store-Dept")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default=DEFAULT_MODEL)
    args = parser.parse_args()

    logger.info(f"--- ENTRENANDO MODELO: {args.model} ---")
    params = MODEL_REGISTRY[args.model]["params"]

    df_train = pd.read_csv(FINAL_TRAIN_PATH)
    df_test = pd.read_csv(FINAL_TEST_PATH)
    train_with_lags, _ = build_training_frame(df_train, df_test)
    train_with_lags = train_with_lags.dropna(subset=FEATURES)

    metrics = validate(args.model, params, train_with_lags)

    estimator = train_final(args.model, params, train_with_lags)
    save_model(args.model, estimator)

    logger.info(f"--- FIN. Métricas de validación: {metrics} ---")


if __name__ == "__main__":
    main()
