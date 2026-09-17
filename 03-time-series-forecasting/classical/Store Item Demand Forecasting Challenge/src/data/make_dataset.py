"""Carga los CSV crudos del challenge y aplica limpieza básica -> data/2.processed.

Limpieza básica = parseo de fechas y verificación de nulos/duplicados; el
feature engineering (lags, rolling stats, variables de calendario) vive en
`src/features/build_features.py` y se aplica después, en `make_final_dataset.py`.

Uso:
    python -m src.data.make_dataset
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "1.raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "2.processed"

INPUT_FILES = {"train": RAW_DIR / "train.csv", "test": RAW_DIR / "test.csv"}
OUTPUT_FILES = {"train": PROCESSED_DIR / "train_processed.csv", "test": PROCESSED_DIR / "test_processed.csv"}


def clean_dataset(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Limpieza básica: parseo de fechas y remoción de duplicados exactos."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")

    n_null_dates = df[date_col].isna().sum()
    if n_null_dates:
        logger.warning("%d filas con fecha inválida serán descartadas", n_null_dates)
        df = df.dropna(subset=[date_col])

    n_dupes = df.duplicated().sum()
    if n_dupes:
        logger.warning("%d filas duplicadas serán descartadas", n_dupes)
        df = df.drop_duplicates()

    return df.reset_index(drop=True)


def main() -> None:
    for name, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada '{name}': {path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for key, in_path in INPUT_FILES.items():
        logger.info("Procesando %s (%s)...", key, in_path)
        df = pd.read_csv(in_path)
        df = clean_dataset(df)

        out_path = OUTPUT_FILES[key]
        df.to_csv(out_path, index=False)
        logger.info("  -> %s (%d filas, %d nulos totales)", out_path, len(df), int(df.isnull().sum().sum()))

    logger.info("make_dataset: completado.")


if __name__ == "__main__":
    main()
