"""
Feature engineering para el forecasting semanal Store-Dept.

Agrega lags y medias móviles de Weekly_Sales por serie (Store, Dept), y provee
el split temporal usado para validación (últimas N semanas).

Uso:
    from src.features.build_features import add_lags, time_split
"""

import logging

import pandas as pd

from src.config import FEATURES, TARGET, VALIDATION_WEEKS

logger = logging.getLogger(__name__)


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega lag_1, lag_4, lag_52, roll_mean_4 y roll_mean_52 de Weekly_Sales
    por serie (Store, Dept). Requiere que df tenga la columna Date como datetime.
    """
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    g = df.groupby(["Store", "Dept"])[TARGET]
    df["lag_1"] = g.shift(1)
    df["lag_4"] = g.shift(4)
    df["lag_52"] = g.shift(52)
    df["roll_mean_4"] = g.transform(lambda x: x.shift(1).rolling(4).mean())
    df["roll_mean_52"] = g.transform(lambda x: x.shift(1).rolling(52).mean())
    return df


def build_training_frame(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatena train+test antes de calcular lags (necesario para que las filas de
    test tengan lag_52/roll_mean_52 completos), y separa de vuelta al final.
    Devuelve (train_con_lags, test_con_lags).
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["Date"] = pd.to_datetime(df_train["Date"])
    df_test["Date"] = pd.to_datetime(df_test["Date"])

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all = add_lags(df_all)

    train_with_lags = df_all[df_all[TARGET].notna()].copy()
    test_with_lags = df_all[df_all[TARGET].isna()].copy()
    return train_with_lags, test_with_lags


def time_split(df: pd.DataFrame, weeks: int = VALIDATION_WEEKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal: las últimas `weeks` semanas del set de train van a validación.
    Descarta filas con NaN en FEATURES (generados por los lags al inicio de cada serie).
    """
    split_date = df["Date"].max() - pd.Timedelta(weeks=weeks)
    train_split = df[df["Date"] <= split_date].dropna(subset=FEATURES)
    valid_split = df[df["Date"] > split_date].dropna(subset=FEATURES)
    logger.info(f"Split temporal en {split_date.date()} -> train={len(train_split)} valid={len(valid_split)}")
    return train_split, valid_split
