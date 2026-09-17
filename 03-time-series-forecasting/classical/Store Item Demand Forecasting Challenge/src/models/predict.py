"""Predicción recursiva (walk-forward) para el set de test.

Corrige un bug del prototipo en notebooks/03-model.ipynb: allí los lags/rolling
stats de `df_all` se calculaban UNA sola vez, antes del loop, sobre una columna
`sales` que todavía tenía NaN en todas las fechas de test. Como consecuencia,
cualquier lag que cayera en una fecha de test (p. ej. `lag_7` para el día 8 de
test) quedaba en NaN y el `fillna(0)` del loop lo convertía silenciosamente en
0 en vez de usar la predicción real de días anteriores. Eso rompe el walk-forward
para todo el test excepto sus primeros ~28 días.

`recursive_predict` recalcula los lags/rolling en cada iteración a partir del
historial acumulado en `df_all['sales']` (train real + predicciones ya hechas),
así que el día 29 de test sí puede depender de la predicción del día 1.

También garantiza que existan las variables de calendario (`day`, `month`,
`week`, ...) que pide `FEATURES`: si `df_all` se construyó concatenando solo
`['store', 'item', 'date', 'sales']` (el caso típico al unir train + test para
tener un historial continuo), esas columnas no existen todavía y el `.loc[...,
FEATURES]` de más abajo fallaría con un `KeyError`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from src.features.build_features import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_ROLLING_SHIFT,
    create_date_features,
    create_lag_features,
    create_rolling_features,
)


def recursive_predict(
    model,
    df_all: pd.DataFrame,
    test_dates: Iterable[pd.Timestamp],
    FEATURES: Sequence[str],
    date_col: str = "date",
    target_col: str = "sales",
    group_cols: Sequence[str] = ("store", "item"),
    lags: list[int] = None,
    rolling_windows: list[int] = None,
    rolling_shift: int = DEFAULT_ROLLING_SHIFT,
) -> pd.DataFrame:
    """Predice `target_col` día por día sobre `test_dates`, escribiendo cada
    predicción de vuelta en `df_all` antes de calcular los features del día
    siguiente (walk-forward real).

    `model` puede ser cualquier objeto con `.predict(X) -> array-like`
    (Booster de `lightgbm.train`, `XGBRegressor`, `CatBoostRegressor`, ...).

    Devuelve una copia de `df_all` con `target_col` completo para `test_dates`.
    Nota de rendimiento: por simplicidad y corrección, cada iteración recalcula
    lags/rolling sobre TODO `df_all` (no solo sobre las filas nuevas). Para este
    dataset (~1M filas, ~90 días de test) toma segundos por iteración; si el
    volumen crece, vale la pena limitar el recálculo a una ventana reciente por
    grupo en vez de al DataFrame completo.
    """
    lags = lags if lags is not None else DEFAULT_LAGS
    rolling_windows = rolling_windows if rolling_windows is not None else DEFAULT_ROLLING_WINDOWS
    group_cols = list(group_cols)

    df_all = df_all.sort_values(group_cols + [date_col]).reset_index(drop=True)

    # Las variables de calendario no dependen de `sales`, así que se calculan
    # una sola vez aquí (a diferencia de lags/rolling, no hace falta repetirlo
    # en cada iteración del loop).
    df_all = create_date_features(df_all, date_col=date_col)

    for d in sorted(test_dates):
        df_all = create_lag_features(
            df_all, lags=lags, group_cols=group_cols, target_col=target_col, date_col=date_col
        )
        df_all = create_rolling_features(
            df_all,
            windows=rolling_windows,
            shift=rolling_shift,
            group_cols=group_cols,
            target_col=target_col,
            date_col=date_col,
            include_std_window=None,
        )

        mask = df_all[date_col] == d
        X_d = df_all.loc[mask, list(FEATURES)].fillna(0)  # fallback si algún lag aún no tiene historia
        df_all.loc[mask, target_col] = model.predict(X_d)

    return df_all
