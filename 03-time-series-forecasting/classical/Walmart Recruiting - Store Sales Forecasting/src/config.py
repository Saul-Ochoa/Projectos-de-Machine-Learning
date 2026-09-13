"""
Configuración central del proyecto: rutas y parámetros de modelos.
Todas las rutas son relativas a la raíz del repo (nunca rutas absolutas D:\\...).
"""

from pathlib import Path

# --- RUTAS ---
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "1.raw"
DATA_PROCESSED_DIR = DATA_DIR / "2.processed"
DATA_FINAL_DIR = DATA_DIR / "3.final"

MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
DOCS_DIR = ROOT_DIR / "docs"

FINAL_TRAIN_PATH = DATA_FINAL_DIR / "final_train.csv"
FINAL_TEST_PATH = DATA_FINAL_DIR / "final_test.csv"

# --- FEATURES ---
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

BASE_FEATURES = ["Date", "Store", "Dept", "Size", "month", "IsHoliday", "MarkDown_Total", "MarkDown_Count"]

LAG_FEATURES = ["lag_1", "lag_4", "lag_52", "roll_mean_4", "roll_mean_52"]

FEATURES = ["Store", "Dept", "Size", "month", "IsHoliday", "MarkDown_Total", "MarkDown_Count",
            "lag_1", "lag_4", "lag_52", "roll_mean_4", "roll_mean_52"]

CATEGORICAL_FEATURES = ["Store", "Dept"]

TARGET = "Weekly_Sales"

# --- VALIDACIÓN ---
VALIDATION_WEEKS = 8  # últimas N semanas como set de validación (split temporal)

# --- MODELOS ---
MODEL_REGISTRY = {
    "lgbm": {
        "dir": MODELS_DIR / "lgbm",
        "model_file": "lgbm_wmae1323_final.pkl",
        "features_file": "features_wmae1323.json",
        "params": {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 1000,
        },
    },
    "xgboost": {
        "dir": MODELS_DIR / "XGBoost",
        "model_file": "xgb_wmae1277_final.pkl",
        "features_file": "features_xgb.json",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
}

DEFAULT_MODEL = "xgboost"
