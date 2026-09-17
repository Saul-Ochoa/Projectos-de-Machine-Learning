"""Factory de modelos con los hiperparámetros validados en notebooks/03-model.ipynb.

Uso:
    from src.models.train_model import get_model
    model = get_model("lgb")  # o "xgb" / "catboost"
"""
from __future__ import annotations

from typing import Any

CATEGORICAL_FEATURES = ["store", "item"]

# Hiperparámetros tal como quedaron documentados y comparados en la sección 3/6
# de notebooks/03-model.ipynb (ver docs/model_selection.md para las métricas).
LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

XGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "random_state": 42,
}

CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": False,
}


def get_model(name: str, **overrides: Any):
    """Devuelve una instancia sin entrenar del modelo pedido.

    `name`: 'lgb' | 'xgb' | 'catboost'. Los hiperparámetros se pueden
    sobreescribir puntualmente vía `overrides` (por ejemplo `n_estimators=500`).

    Nota: 'lgb' usa la API nativa `lightgbm.train` en vez de un estimador
    sklearn-like, porque así se validó en el notebook (permite `lgb.Dataset`
    con `categorical_feature` explícito). Para 'lgb' esta función devuelve el
    dict de parámetros, no un objeto; úsalo con `lightgbm.train(params, ...)`.
    """
    name = name.lower()

    if name == "lgb":
        params = {**LGB_PARAMS, **overrides}
        return params

    if name == "xgb":
        import xgboost as xgb

        params = {**XGB_PARAMS, **overrides}
        return xgb.XGBRegressor(**params)

    if name == "catboost":
        from catboost import CatBoostRegressor

        params = {**CATBOOST_PARAMS, **overrides}
        return CatBoostRegressor(**params)

    raise ValueError(f"Modelo desconocido: {name!r}. Usa 'lgb', 'xgb' o 'catboost'.")
