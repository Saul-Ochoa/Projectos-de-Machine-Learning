"""Genera los datasets finales (data/3.final) agregando las variables de
calendario a los CSV ya limpios en data/2.processed.

Los lags y rolling stats NO se calculan aquí: dependen de si se está
construyendo un set de train (con dropna directo) o un set de test para
predicción recursiva (walk-forward, ver src/models/predict.py), así que se
aplican más adelante con src/features/build_features.py.

Uso:
    python -m src.data.make_final_dataset
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Bootstrap necesario para que `from src...` funcione tanto si el script se
# corre como módulo (`python -m src.data.make_final_dataset`) como si se
# corre directo (`python src/data/make_final_dataset.py`, p. ej. desde un IDE).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import create_date_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "2.processed"
FINAL_DIR = PROJECT_ROOT / "data" / "3.final"

INPUT_FILES = {"train": PROCESSED_DIR / "train_processed.csv", "test": PROCESSED_DIR / "test_processed.csv"}
OUTPUT_FILES = {"train": FINAL_DIR / "train_final.csv", "test": FINAL_DIR / "test_final.csv"}


def main() -> None:
    for name, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No existe '{name}': {path}. Corre antes: python -m src.data.make_dataset"
            )

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    for key, in_path in INPUT_FILES.items():
        logger.info("Procesando %s (%s)...", key, in_path)
        df = pd.read_csv(in_path)
        df = create_date_features(df, date_col="date")

        out_path = OUTPUT_FILES[key]
        df.to_csv(out_path, index=False)
        logger.info("  -> %s (%s)", out_path, df.shape)

    logger.info("make_final_dataset: completado.")


if __name__ == "__main__":
    main()
