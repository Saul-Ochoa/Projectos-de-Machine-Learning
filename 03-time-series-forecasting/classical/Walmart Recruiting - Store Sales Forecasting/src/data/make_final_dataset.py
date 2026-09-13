"""
Script de producción: genera datasets finales para modelado (sin lags, esos
se calculan en src/features/build_features.py sobre train+test concatenados).

Entrada: data/2.processed/{processed_train,processed_test}.csv
Salida:  data/3.final/{final_train,final_test}.csv

Uso:
    python src/data/make_final_dataset.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import BASE_FEATURES, DATA_FINAL_DIR, DATA_PROCESSED_DIR, MARKDOWN_COLS, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_TRAIN = DATA_PROCESSED_DIR / "processed_train.csv"
INPUT_TEST = DATA_PROCESSED_DIR / "processed_test.csv"
OUTPUT_TRAIN = DATA_FINAL_DIR / "final_train.csv"
OUTPUT_TEST = DATA_FINAL_DIR / "final_test.csv"


def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Cargando {path}")
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    return pd.read_csv(path)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creando features de fecha")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        logger.warning("Algunas fechas no se pudieron parsear")

    df["month"] = df["Date"].dt.month.astype("Int16")
    df["year"] = df["Date"].dt.year.astype("Int16")
    # Solo dejamos month para producción (year/quarter colinean); para análisis, agrega quarter.
    return df


def clean_markdowns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia MarkDown1-5: NA = 0 (no hubo promo) y crea Total/Count."""
    existing_md = [c for c in MARKDOWN_COLS if c in df.columns]

    if not existing_md:
        logger.warning("No se encontraron columnas MarkDown, creando totales en 0")
        df["MarkDown_Total"] = 0
        df["MarkDown_Count"] = 0
        return df

    logger.info(f"Limpiando markdowns: {existing_md} - NaNs antes: {df[existing_md].isna().sum().sum()}")

    for col in existing_md:
        df[f"{col}_active"] = df[col].notna().astype("int8")

    df[existing_md] = df[existing_md].fillna(0)
    df["MarkDown_Total"] = df[existing_md].sum(axis=1)
    df["MarkDown_Count"] = (df[existing_md] > 0).sum(axis=1)

    logger.info(f"NaNs después: {df[existing_md].isna().sum().sum()}")
    return df


def validate_and_select(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    required = BASE_FEATURES + ([TARGET] if is_train else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {'train' if is_train else 'test'}: {missing}")

    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(int)

    cols = BASE_FEATURES + ([TARGET] if is_train else [])
    return df[cols].copy()


def main():
    logger.info("--- INICIO PIPELINE FINAL ---")
    DATA_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    df_train = load_data(INPUT_TRAIN)
    df_test = load_data(INPUT_TEST)

    df_train = add_date_features(df_train)
    df_test = add_date_features(df_test)

    df_train = clean_markdowns(df_train)
    df_test = clean_markdowns(df_test)

    df_train_final = validate_and_select(df_train, is_train=True)
    df_test_final = validate_and_select(df_test, is_train=False)

    logger.info(f"Train final shape: {df_train_final.shape} | Test final shape: {df_test_final.shape}")

    df_train_final.to_csv(OUTPUT_TRAIN, index=False)
    df_test_final.to_csv(OUTPUT_TEST, index=False)

    logger.info(f"Guardado: {OUTPUT_TRAIN}")
    logger.info(f"Guardado: {OUTPUT_TEST}")
    logger.info("--- FIN PIPELINE ---")


if __name__ == "__main__":
    main()
