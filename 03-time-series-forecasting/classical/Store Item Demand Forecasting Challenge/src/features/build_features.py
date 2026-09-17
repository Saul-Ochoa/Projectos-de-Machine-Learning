"""Feature engineering compartida entre notebooks y scripts de entrenamiento/predicción.

La lógica replica exactamente la validada en notebooks/03-model.ipynb (comparativa
LightGBM/XGBoost/CatBoost), de modo que entrenar desde scripts/train.py produzca
las mismas features que se usaron para elegir el modelo ganador.
"""
from __future__ import annotations

import pandas as pd

# Ancla fija (no el mínimo de cada DataFrame) para que `days_since_start` sea
# comparable entre train y test: si se usara `df['date'].min()` por DataFrame,
# train y test tendrían escalas de tiempo distintas y el feature sería inútil
# para el modelo en inferencia.
EPOCH_START = pd.Timestamp("2013-01-01")

DEFAULT_LAGS = [7, 14, 21, 28]
DEFAULT_ROLLING_WINDOWS = [7, 14, 28]
DEFAULT_ROLLING_SHIFT = 7


def create_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Agrega variables de calendario derivadas de `date_col`.

    Columnas creadas: day, month, week, is_weekend, is_month_start,
    is_month_end, days_since_start, is_year_end.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")

    df["day"] = df[date_col].dt.day.astype("int16")
    df["month"] = df[date_col].dt.month.astype("int8")
    df["week"] = df[date_col].dt.isocalendar().week.astype("int8")
    df["is_weekend"] = (df[date_col].dt.weekday >= 5).astype("int8")
    df["is_month_start"] = df[date_col].dt.is_month_start.astype("int8")
    df["is_month_end"] = df[date_col].dt.is_month_end.astype("int8")
    df["days_since_start"] = (df[date_col] - EPOCH_START).dt.days.astype("int32")
    df["is_year_end"] = (
        (df[date_col].dt.month == 12) & (df[date_col].dt.day >= 20)
    ).astype("int8")
    return df


def create_lag_features(
    df: pd.DataFrame,
    lags: list[int] = None,
    group_cols: list[str] = ("store", "item"),
    target_col: str = "sales",
    date_col: str = "date",
) -> pd.DataFrame:
    """Agrega lags de `target_col` por cada combinación de `group_cols`.

    Ordena por `group_cols + [date_col]` antes de calcular los `shift`, porque
    `groupby().shift()` requiere que las filas estén en orden cronológico
    dentro de cada grupo para que el lag sea correcto.
    """
    lags = lags if lags is not None else DEFAULT_LAGS
    group_cols = list(group_cols)

    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    grouped = df.groupby(group_cols)[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = None,
    shift: int = DEFAULT_ROLLING_SHIFT,
    group_cols: list[str] = ("store", "item"),
    target_col: str = "sales",
    date_col: str = "date",
    include_std_window: int = 28,
) -> pd.DataFrame:
    """Agrega medias móviles (y una desviación estándar móvil) de `target_col`.

    Cada ventana se desplaza `shift` días hacia atrás antes de aplicar el
    `rolling`, para no filtrar información del día que se quiere predecir
    (data leakage). Genera columnas `rolling_mean_{shift}_{window}` y,
    para `include_std_window`, también `rolling_std_{shift}_{include_std_window}`.
    """
    windows = windows if windows is not None else DEFAULT_ROLLING_WINDOWS
    group_cols = list(group_cols)

    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    grouped = df.groupby(group_cols)[target_col]

    for window in windows:
        min_periods = 1 if window == shift else shift
        df[f"rolling_mean_{shift}_{window}"] = grouped.transform(
            lambda x, w=window, mp=min_periods: x.shift(shift).rolling(w, min_periods=mp).mean()
        )

    if include_std_window is not None:
        df[f"rolling_std_{shift}_{include_std_window}"] = grouped.transform(
            lambda x: x.shift(shift).rolling(include_std_window, min_periods=shift).std()
        ).fillna(0)  # con una sola observación el std da NaN -> "sin variabilidad" (0)

    return df


def build_temporal_features(
    df: pd.DataFrame,
    lags: list[int] = None,
    rolling_windows: list[int] = None,
    rolling_shift: int = DEFAULT_ROLLING_SHIFT,
    group_cols: list[str] = ("store", "item"),
    target_col: str = "sales",
    date_col: str = "date",
) -> pd.DataFrame:
    """Aplica, en orden, fechas + lags + rolling stats. Atajo usado por
    scripts/train.py y scripts/predict.py para no repetir la secuencia de 3 llamadas."""
    df = create_date_features(df, date_col=date_col)
    df = create_lag_features(df, lags=lags, group_cols=group_cols, target_col=target_col, date_col=date_col)
    df = create_rolling_features(
        df,
        windows=rolling_windows,
        shift=rolling_shift,
        group_cols=group_cols,
        target_col=target_col,
        date_col=date_col,
    )
    return df


# Set de features validado en notebooks/03-model.ipynb (WAPE 10.89% LightGBM en validación).
FEATURES = [
    "store", "item", "day", "month", "week", "is_weekend", "is_month_start", "is_month_end",
    "days_since_start", "is_year_end",
    "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7_7", "rolling_mean_7_14",
]
TARGET = "sales"
