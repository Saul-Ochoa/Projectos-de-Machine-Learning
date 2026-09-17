# Store Item Demand Forecasting Challenge

Forecasting de demanda diaria (`sales`) por combinación `store` x `item`, usando
LightGBM, XGBoost y CatBoost sobre features de calendario y lags de ventas.

**Modelo ganador: CatBoost** — WAPE 10.82% / R² 0.931 en validación.
Ver [`docs/model_selection.md`](docs/model_selection.md) para la comparativa
completa de los 3 modelos y el criterio de decisión.

## Estructura del proyecto

```
.
├── data/
│   ├── 1.raw/              # train.csv, test.csv originales del challenge
│   ├── 2.processed/        # limpieza básica (fechas parseadas, sin duplicados)
│   └── 3.final/            # + features de calendario; aquí caen las submissions
├── docs/
│   └── model_selection.md  # comparativa LightGBM/XGBoost/CatBoost y ganador
├── models/
│   └── best_model.pkl      # generado por scripts/train.py
├── notebooks/
│   ├── 01-data_transformation.ipynb
│   ├── 02-eda.ipynb
│   ├── 03-model.ipynb                       # entrena y compara los 3 modelos
│   └── 04-final_training_and_submission.ipynb  # corre scripts/train.py + predict.py
├── output/
│   └── metrics_comparison.csv  # generado por scripts/evaluate.py
├── scripts/
│   ├── train.py    # entrena el modelo ganador con 100% de los datos -> models/best_model.pkl
│   ├── predict.py  # predicción recursiva (walk-forward) -> data/3.final/submission_best.csv
│   └── evaluate.py # reproduce la comparativa de los 3 modelos -> output/metrics_comparison.csv
├── src/
│   ├── data/
│   │   ├── make_dataset.py        # 1.raw -> 2.processed (limpieza básica)
│   │   └── make_final_dataset.py  # 2.processed -> 3.final (features de calendario)
│   ├── features/
│   │   └── build_features.py      # create_date_features, create_lag_features, create_rolling_features
│   ├── models/
│   │   ├── train_model.py         # get_model('lgb' | 'xgb' | 'catboost')
│   │   └── predict.py             # recursive_predict (walk-forward)
│   └── evaluation/
│       └── metrics.py             # evaluate_metrics(y_true, y_pred) -> dict
├── tests/
└── requirements.txt
```

## Cómo reproducir

```bash
pip install -r requirements.txt

python -m src.data.make_dataset          # data/1.raw -> data/2.processed
python -m src.data.make_final_dataset    # data/2.processed -> data/3.final

python scripts/train.py                  # entrena el ganador -> models/best_model.pkl
python scripts/predict.py                # -> data/3.final/submission_best.csv
```

Opcional, para regenerar la comparativa de modelos:

```bash
python scripts/evaluate.py               # -> output/metrics_comparison.csv
```

Para entrenar un modelo distinto al ganador (por ejemplo, si el sesgo sistemático
importa más que el WAPE agregado — ver la nota en `docs/model_selection.md`):

```bash
python scripts/train.py --model lgb      # o --model xgb
```

## Métrica final (modelo ganador: CatBoost)

| Métrica | Validación |
|---|---|
| WAPE % | 10.824 |
| SMAPE % (oficial del challenge) | 12.563 |
| RMSE | 8.310 |
| MAE | 6.398 |
| R² | 0.9309 |
| BIAS | -0.417 |

## Notas de diseño

- **`recursive_predict`** (`src/models/predict.py`) recalcula lags/rolling stats en
  cada paso del walk-forward usando el historial acumulado (train real + predicciones
  previas), en vez de calcularlos una sola vez antes del loop — así las predicciones
  de test más allá de los primeros ~28 días no dependen de lags rellenados con 0.
- **`num_boost_round` / `n_estimators` / `iterations`** del modelo final no se fija
  en un número arbitrario: `scripts/train.py` separa un hold-out temporal del 12%
  para encontrar el número de rondas óptimo con early stopping, y luego reentrena
  con el 100% de los datos usando ese valor.
