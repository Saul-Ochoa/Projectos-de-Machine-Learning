"""
Genera los gráficos de diagnóstico (4 paneles) para cada modelo entrenado y
un resumen JSON con las métricas comparativas. Pensado para regenerar docs/images/
después de reentrenar.

Uso:
    python scripts/generate_report.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DOCS_DIR, FEATURES, FINAL_TEST_PATH, FINAL_TRAIN_PATH, MODEL_REGISTRY
from src.evaluation.metrics import evaluate
from src.features.build_features import build_training_frame, time_split
from src.models.train_model import fit, get_estimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGES_DIR = DOCS_DIR / "images"


def four_panel_plot(model_name, model, valid_split, pred_valid, y_valid, wmae):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    df_res = valid_split.copy()
    df_res["pred"] = pred_valid
    sample = df_res[(df_res["Store"] == 1) & (df_res["Dept"] == 1)].sort_values("Date")

    axes[0, 0].plot(sample["Date"], sample["Weekly_Sales"], label="Real", marker="o")
    axes[0, 0].plot(sample["Date"], sample["pred"], label="Predicción", linestyle="--", marker="x")
    axes[0, 0].set_title("Ventas Reales vs Predichas (Store 1, Dept 1)")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(alpha=0.3)

    if model_name == "lgbm":
        importances = model.booster_.feature_importance(importance_type="gain")
    else:
        importances = model.feature_importances_
    importance = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    sns.barplot(data=importance, x="importance", y="feature", hue="feature", ax=axes[0, 1], palette="viridis", legend=False)
    axes[0, 1].set_title("Importancia de Features")

    residuals = y_valid - pred_valid
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1, 0], color="purple")
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_title(f"Distribución de Residuos | WMAE: {wmae:.1f}")

    df_res["weight"] = np.where(df_res["IsHoliday"], 5, 1)
    df_res["abs_err"] = np.abs(df_res["Weekly_Sales"] - df_res["pred"]) * df_res["weight"]
    wmae_dept = (df_res.groupby("Dept")["abs_err"].sum() / df_res.groupby("Dept")["weight"].sum()).nlargest(10)
    wmae_dept.plot(kind="barh", ax=axes[1, 1], color="crimson")
    axes[1, 1].set_title("Top 10 Departamentos con mayor WMAE")
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGES_DIR / f"{model_name}_diagnostics.png"
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info(f"Guardado: {out_path}")
    return out_path


def main():
    df_train = pd.read_csv(FINAL_TRAIN_PATH)
    df_test = pd.read_csv(FINAL_TEST_PATH)
    train_with_lags, _ = build_training_frame(df_train, df_test)
    train_with_lags = train_with_lags.dropna(subset=FEATURES)

    train_split, valid_split = time_split(train_with_lags)
    X_train, y_train = train_split[FEATURES], train_split["Weekly_Sales"]
    X_valid, y_valid = valid_split[FEATURES], valid_split["Weekly_Sales"]
    weights_train = np.where(X_train["IsHoliday"], 5, 1)
    weights_valid = np.where(X_valid["IsHoliday"], 5, 1)

    summary = {}
    for model_name, registry in MODEL_REGISTRY.items():
        logger.info(f"Generando reporte para: {model_name}")
        estimator = get_estimator(model_name, registry["params"])
        fit(
            model_name, estimator, X_train, y_train, weights_train,
            eval_set=[(X_valid, y_valid)], eval_sample_weight=[weights_valid],
        )
        pred_valid = estimator.predict(X_valid)
        report = evaluate(y_valid, pred_valid, X_valid["IsHoliday"])
        four_panel_plot(model_name, estimator, valid_split, pred_valid, y_valid, report.wmae)
        summary[model_name] = report.as_dict()

    summary_path = DOCS_DIR / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Resumen de métricas guardado en: {summary_path}")
    for model_name, metrics in summary.items():
        logger.info(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()
