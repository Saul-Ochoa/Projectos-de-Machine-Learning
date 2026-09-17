"""Genera data/3.final/submission_best.csv usando el modelo entrenado por
scripts/train.py (models/best_model.pkl) y predicción recursiva (walk-forward).

Uso:
    python scripts/train.py
    python scripts/predict.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_temporal_features, create_date_features  # noqa: E402
from src.models.predict import recursive_predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRAIN_FINAL_PATH = PROJECT_ROOT / "data" / "3.final" / "train_final.csv"
TEST_FINAL_PATH = PROJECT_ROOT / "data" / "3.final" / "test_final.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
SUBMISSION_OUT_PATH = PROJECT_ROOT / "data" / "3.final" / "submission_best.csv"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_PATH}. Corre antes: python scripts/train.py")
    bundle = joblib.load(MODEL_PATH)
    logger.info("Modelo cargado: %s", bundle["model_name"])
    return bundle["model"], bundle["features"]


def build_df_all(features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df_train = pd.read_csv(TRAIN_FINAL_PATH)
    df_train["date"] = pd.to_datetime(df_train["date"], format="%Y-%m-%d")
    df_train = build_temporal_features(df_train)
    df_train = df_train.dropna(subset=["lag_28", "rolling_mean_7_14"]).reset_index(drop=True)

    df_test = pd.read_csv(TEST_FINAL_PATH)
    df_test["date"] = pd.to_datetime(df_test["date"], format="%Y-%m-%d")
    df_test = create_date_features(df_test)

    # Concatenar train + test permite calcular lags de forma continua a través
    # de la frontera train/test (el lag_7 del primer día de test toma la venta
    # real de una semana antes, del propio train).
    df_all = pd.concat(
        [
            df_train[["store", "item", "date", "sales"]],
            df_test[["store", "item", "date"]].assign(sales=np.nan),
        ],
        ignore_index=True,
    ).sort_values(["store", "item", "date"]).reset_index(drop=True)
    df_all["sales"] = df_all["sales"].astype(float)

    return df_all, df_test["date"]


def main() -> None:
    model, features = load_model()

    logger.info("Reconstruyendo df_all (train + test) para predicción recursiva...")
    df_all, test_dates = build_df_all(features)

    logger.info("Prediciendo %d fechas de test (walk-forward)...", test_dates.nunique())
    df_all = recursive_predict(model, df_all, test_dates.unique(), features)

    submission = df_all[df_all["date"].isin(test_dates)][["store", "item", "date", "sales"]]

    SUBMISSION_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_OUT_PATH, index=False)
    logger.info("Submission guardada en %s (%d filas)", SUBMISSION_OUT_PATH, len(submission))


if __name__ == "__main__":
    main()
