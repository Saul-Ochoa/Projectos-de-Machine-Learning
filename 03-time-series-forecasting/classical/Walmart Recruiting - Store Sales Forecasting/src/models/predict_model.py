"""
Genera predicciones sobre data/3.final/final_test.csv usando un modelo ya entrenado
(ver src/models/train_model.py) y guarda la submission en output/.

Uso:
    python src/models/predict_model.py                 # XGBoost (default)
    python src/models/predict_model.py --model lgbm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DEFAULT_MODEL, FINAL_TEST_PATH, FINAL_TRAIN_PATH, MODEL_REGISTRY, OUTPUT_DIR
from src.features.build_features import build_training_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_name: str):
    registry = MODEL_REGISTRY[model_name]
    model_path = registry["dir"] / registry["model_file"]
    features_path = registry["dir"] / registry["features_file"]

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {model_path}. Corre src/models/train_model.py primero.")

    estimator = joblib.load(model_path)
    with open(features_path) as f:
        features = json.load(f)
    return estimator, features


def main():
    parser = argparse.ArgumentParser(description="Predice sobre final_test.csv con un modelo entrenado")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default=DEFAULT_MODEL)
    parser.add_argument("--output-name", default=None, help="Nombre del CSV de salida (default: submission_<model>.csv)")
    args = parser.parse_args()

    estimator, features = load_model(args.model)

    df_train = pd.read_csv(FINAL_TRAIN_PATH)
    df_test = pd.read_csv(FINAL_TEST_PATH)
    _, test_with_lags = build_training_frame(df_train, df_test)

    X_test = test_with_lags[features]
    pred_test = estimator.predict(X_test)

    submission = test_with_lags[["Store", "Dept", "Date"]].copy()
    submission["Weekly_Sales"] = pred_test

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"submission_{args.model}.csv"
    output_path = OUTPUT_DIR / output_name
    submission.to_csv(output_path, index=False)

    logger.info(f"Submission guardada: {output_path} ({submission.shape[0]} filas)")


if __name__ == "__main__":
    main()
