"""
Métricas de evaluación para el forecasting de Walmart.

Métrica oficial de la competencia: WMAE (Weighted Mean Absolute Error),
que pondera x5 las semanas de feriado (IsHoliday=True).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_wmae(y_true, y_pred, is_holiday) -> float:
    """WMAE oficial de la competencia: peso 5 en semanas de feriado, 1 en el resto."""
    weights = np.where(is_holiday, 5, 1)
    return float(np.sum(weights * np.abs(np.asarray(y_true) - np.asarray(y_pred))) / np.sum(weights))


@dataclass
class RegressionReport:
    wmae: float
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict:
        return {"wmae": self.wmae, "mae": self.mae, "rmse": self.rmse, "r2": self.r2}

    def __str__(self) -> str:
        return (
            f"WMAE: {self.wmae:.2f} | MAE: {self.mae:.2f} | "
            f"RMSE: {self.rmse:.2f} | R2: {self.r2:.4f}"
        )


def evaluate(y_true, y_pred, is_holiday) -> RegressionReport:
    """Calcula el set completo de métricas de validación (WMAE, MAE, RMSE, R2)."""
    return RegressionReport(
        wmae=calculate_wmae(y_true, y_pred, is_holiday),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def wmae_by_group(df, y_true_col, y_pred_col, is_holiday_col, group_col, top_n=10):
    """WMAE agrupado (ej. por Dept) para diagnóstico de errores. Devuelve top_n descendente."""
    tmp = df.copy()
    tmp["_weight"] = np.where(tmp[is_holiday_col], 5, 1)
    tmp["_abs_err"] = np.abs(tmp[y_true_col] - tmp[y_pred_col]) * tmp["_weight"]
    result = tmp.groupby(group_col)["_abs_err"].sum() / tmp.groupby(group_col)["_weight"].sum()
    return result.nlargest(top_n)
