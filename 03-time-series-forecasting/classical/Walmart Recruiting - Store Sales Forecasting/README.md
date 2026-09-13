# Walmart Recruiting — Store Sales Forecasting

![WMAE](https://img.shields.io/badge/WMAE-1277.90-brightgreen) ![Modelo](https://img.shields.io/badge/modelo-XGBoost-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Forecasting semanal de ventas por combinación Store-Dept, usando gradient boosting
(LightGBM / XGBoost) con features de lags y medias móviles. Basado en la competencia
de Kaggle [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
([descripción completa del dataset](docs/DATASET.md)).

## Resultado actual

| Modelo | WMAE | MAE | RMSE | R² |
|---|---|---|---|---|
| LightGBM (baseline) | 1323.06 | 1259.98 | 2621.34 | 0.9856 |
| **XGBoost (ganador)** | **1277.90** | **1218.12** | **2545.89** | **0.9865** |

```bash
python src/models/train_model.py --model xgboost
```

WMAE (Weighted MAE) es la métrica oficial de la competencia: pondera x5 las semanas
de feriado (`IsHoliday=True`) frente al resto. Ver [`docs/PROGRESO.md`](docs/PROGRESO.md)
para el detalle de la evolución del modelo, gráficos de diagnóstico y próximos pasos.

> Nota de reproducibilidad: al reentrenar con `src/models/train_model.py` en este
> entorno (XGBoost 3.3.0, Python 3.13) el WMAE reproducido es 1283.72 — a 0.4% del
> 1277.90 original, diferencia esperable entre versiones de librería. Detalle en
> [`docs/PROGRESO.md`](docs/PROGRESO.md#2-comparativa-de-modelos).

## Estructura del proyecto

```
.
├── data/
│   ├── 1.raw/            # train.csv, test.csv, features.csv, stores.csv (Kaggle)
│   ├── 2.processed/      # merge de train/test con stores + features
│   └── 3.final/          # dataset final (sin lags) listo para feature engineering
├── docs/
│   ├── PROGRESO.md        # bitácora de experimentos y próximos pasos
│   ├── metrics_summary.json
│   └── images/             # gráficos de diagnóstico (4 paneles) por modelo
├── models/
│   ├── lgbm/               # modelo LightGBM + lista de features usada
│   └── XGBoost/            # modelo XGBoost (ganador) + lista de features usada
├── notebooks/               # solo EDA / prototipado, sin lógica de producción
├── output/                  # submissions generadas por predict_model.py
├── scripts/
│   └── generate_report.py   # regenera docs/images/*.png y docs/metrics_summary.json
├── src/
│   ├── config.py             # rutas, features y params de modelos (única fuente de verdad)
│   ├── data/
│   │   ├── make_dataset.py         # 1.raw -> 2.processed
│   │   └── make_final_dataset.py   # 2.processed -> 3.final
│   ├── features/
│   │   └── build_features.py       # lags, rolling means, split temporal
│   ├── models/
│   │   ├── train_model.py          # entrena + valida + guarda modelo final
│   │   └── predict_model.py        # predice sobre final_test.csv -> output/
│   └── evaluation/
│       └── metrics.py              # WMAE oficial + MAE/RMSE/R2
├── tests/                    # pytest para features y métricas
├── Makefile
├── Dockerfile
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Coloca los CSV originales de Kaggle en `data/1.raw/` (`train.csv`, `test.csv`,
`features.csv`, `stores.csv`) y luego ejecuta el pipeline completo:

```bash
python src/data/make_dataset.py         # 1.raw -> 2.processed
python src/data/make_final_dataset.py   # 2.processed -> 3.final
python src/models/train_model.py        # entrena XGBoost, valida y guarda el modelo
python src/models/predict_model.py      # genera output/submission_xgboost.csv
```

Para entrenar el baseline LightGBM en vez del ganador actual:

```bash
python src/models/train_model.py --model lgbm
python src/models/predict_model.py --model lgbm
```

Para regenerar los gráficos de diagnóstico y el resumen de métricas en `docs/`:

```bash
python scripts/generate_report.py
```

Con `make` (Linux/Mac/WSL): `make install`, `make data`, `make train`, `make predict`, `make test`, `make report`.

### Tests

```bash
pytest tests/ -v
```

## Metodología

- **Split de validación**: temporal, últimas 8 semanas de `train.csv` (no random split,
  para evitar fuga de información de series temporales).
- **Feature engineering ganador** (`src/features/build_features.py`): lags de
  `Weekly_Sales` (1, 4 y 52 semanas) y medias móviles (4 y 52 semanas) por serie
  Store-Dept, calculados sobre train+test concatenados para que las filas de test
  tengan historial completo.
- **Features usadas**: `Store, Dept, Size, month, IsHoliday, MarkDown_Total,
  MarkDown_Count, lag_1, lag_4, lag_52, roll_mean_4, roll_mean_52`.
- **Entrenamiento final**: una vez validado, el modelo se reentrena con TODOS los
  datos disponibles (mismo criterio que el experimento ganador) antes de guardarse
  en `models/`.

## Notas de diseño

- Todas las rutas se resuelven vía `pathlib.Path` relativas a la raíz del repo
  (`src/config.py`), nunca rutas absolutas de disco.
- `config.py` centraliza rutas, features y grillas de hiperparámetros por modelo
  en vez de un `config.yaml` separado: al ser Python puro, evita duplicar lógica
  de parseo y mantiene los `Path`/dicts tipados. Si en el futuro se necesita
  configuración editable sin tocar código (ej. para CI o para no-programadores),
  es sencillo migrar este archivo a YAML.
- Los notebooks (`notebooks/`) se dejan solo para EDA y prototipado exploratorio;
  la lógica reproducible vive en `src/`.

## Próximos pasos

Ver [`docs/PROGRESO.md`](docs/PROGRESO.md#next-steps): ensemble LGBM+XGBoost, CatBoost,
tracking de experimentos (MLflow), y CI con GitHub Actions.
