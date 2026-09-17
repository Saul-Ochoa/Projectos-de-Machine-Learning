"""Métricas de evaluación, movidas de notebooks/03-model.ipynb (`regression_report`)
para poder reutilizarlas desde scripts/evaluate.py y desde los notebooks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred) -> float:
    """SMAPE (Symmetric MAPE) — métrica oficial del Store Item Demand Forecasting Challenge."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100


def rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Log Error. Penaliza más los errores en ventas bajas."""
    y_pred = np.maximum(y_pred, 0)  # sales no puede ser negativo
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def evaluate_metrics(y_true, y_pred) -> dict:
    """Calcula RMSE, MAE, RMSLE, MAPE %, SMAPE %, WAPE %, R2 y BIAS para un set dado.

    Equivalente a `regression_report()` de notebooks/03-model.ipynb, sin el
    `print` interno (deja ese detalle a quien la llame).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
        "MAPE %": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100,
        "SMAPE %": smape(y_true, y_pred),
        "WAPE %": np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100,
        "R2": r2_score(y_true, y_pred),
        "BIAS": np.mean(y_pred - y_true),  # <0 subestima, >0 sobreestima
    }


def regression_report(y_true, y_pred, name: str = "Valid") -> dict:
    """Como `evaluate_metrics`, pero además imprime la tabla formateada
    (mismo comportamiento que la versión original del notebook)."""
    metrics = evaluate_metrics(y_true, y_pred)
    print(pd.DataFrame([metrics], index=[name]).T.round(3).to_string())
    return metrics
