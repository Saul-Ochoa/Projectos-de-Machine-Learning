"""
Script de producción: junta train/test con stores.csv y features.csv.
Entrada: data/1.raw/{train,test,features,stores}.csv
Salida: data/2.processed/{processed_train,processed_test}.csv

Uso:
    python src/data/make_dataset.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FEATURES_COLS = [
    "Store", "Date", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment",
]


def load_raw_data():
    logger.info(f"Cargando data desde: {DATA_RAW_DIR}")
    try:
        df_train = pd.read_csv(DATA_RAW_DIR / "train.csv")
        df_test = pd.read_csv(DATA_RAW_DIR / "test.csv")
        df_features = pd.read_csv(DATA_RAW_DIR / "features.csv")
        df_stores = pd.read_csv(DATA_RAW_DIR / "stores.csv")
        return df_train, df_test, df_features, df_stores
    except FileNotFoundError as e:
        logger.error(f"Falta un archivo en {DATA_RAW_DIR}: {e}")
        raise


def transform_data(df_train, df_test, df_features, df_stores):
    logger.info("Transformando...")
    df_features_filtered = df_features[FEATURES_COLS]
    df_inner_train = df_train.merge(df_stores, how="left").merge(df_features_filtered, how="left")
    df_inner_test = df_test.merge(df_stores, how="left").merge(df_features_filtered, how="left")
    logger.info(f"Train: {df_inner_train.shape} | Test: {df_inner_test.shape}")
    return df_inner_train, df_inner_test


def save_processed_data(df_train_proc, df_test_proc):
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_train_proc.to_csv(DATA_PROCESSED_DIR / "processed_train.csv", index=False)
    df_test_proc.to_csv(DATA_PROCESSED_DIR / "processed_test.csv", index=False)
    logger.info(f"Guardado en: {DATA_PROCESSED_DIR}")


def main():
    logger.info("Iniciando pipeline make_dataset")
    df_train, df_test, df_features, df_stores = load_raw_data()
    df_inner_train, df_inner_test = transform_data(df_train, df_test, df_features, df_stores)
    save_processed_data(df_inner_train, df_inner_test)
    logger.info("Pipeline OK")


if __name__ == "__main__":
    main()
