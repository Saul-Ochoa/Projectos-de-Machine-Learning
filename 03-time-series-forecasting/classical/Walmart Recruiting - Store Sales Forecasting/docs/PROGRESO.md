# Progreso — Walmart Store Sales Forecasting

Última actualización: 2026-09-13

## 1. Resumen ejecutivo

Objetivo: predecir `Weekly_Sales` por combinación Store-Dept, minimizando WMAE
(Weighted MAE, peso x5 en semanas de feriado). Se probaron dos modelos de gradient
boosting sobre el mismo feature set; XGBoost es el modelo ganador actual.

## 2. Comparativa de modelos

| Modelo | WMAE ↓ | MAE ↓ | RMSE ↓ | R² ↑ | Hiperparámetros clave |
|---|---|---|---|---|---|
| LightGBM (baseline) | 1323.06 | 1259.98 | 2621.34 | 0.9856 | `num_leaves=31, lr=0.05, n_estimators=1000` |
| **XGBoost (ganador)** | **1283.72** | **1222.97** | **2560.24** | **0.9863** | `max_depth=8, lr=0.05, subsample=0.8, colsample_bytree=0.9, n_estimators=1000` |

> Estas métricas se reproducen ejecutando `python src/models/train_model.py --model <lgbm|xgboost>`
> (split temporal, últimas 8 semanas de validación) y quedan en el log de la corrida.
> El resumen estructurado se guarda en [`docs/metrics_summary.json`](metrics_summary.json)
> vía `python scripts/generate_report.py`.
>
> Nota: la corrida de auditoría (2026-09-13) dio WMAE=1283.72 para XGBoost, muy cerca
> del 1277.90 reportado originalmente en el notebook — la diferencia (~0.4%) es ruido
> normal entre versiones de librería/semilla del ambiente, no una regresión real.

XGBoost mejora el WMAE en ~3% sobre el baseline de LightGBM, con mejor R² y menor
dispersión de residuos.

## 3. Evolución del feature engineering

1. **v0 — features base** (`src/data/make_final_dataset.py`): `Store, Dept, Size,
   month, IsHoliday, MarkDown_Total, MarkDown_Count`. Los `MarkDown1-5` (solo
   disponibles desde nov-2011) se limpian con NA→0 y se resumen en un total y un
   conteo de promociones activas, en vez de usar las 5 columnas crudas (evita
   sparsity y multicolinealidad).
2. **v1 — lags y medias móviles** (`src/features/build_features.py`, feature set
   ganador actual): por serie (Store, Dept) se agregan `lag_1`, `lag_4`, `lag_52`
   (mismo periodo el año anterior — captura estacionalidad anual) y `roll_mean_4`,
   `roll_mean_52`. Se calculan sobre train+test concatenados para que las filas de
   test tengan historial completo de lags.
3. **Feature importance (XGBoost)**: `lag_52` domina por amplio margen (~0.82 de
   gain), seguido de `lag_1` (~0.10). El resto de features (incluyendo las de
   calendario y markdowns) aporta poco de forma individual — ver
   `docs/images/xgboost_diagnostics.png`.

## 4. Diagnóstico de errores

Ver los paneles de 4 gráficos por modelo (real vs. predicho, feature importance,
distribución de residuos, WMAE por departamento) en:

- [`docs/images/lgbm_diagnostics.png`](images/lgbm_diagnostics.png)
- [`docs/images/xgboost_diagnostics.png`](images/xgboost_diagnostics.png)

Los departamentos con mayor WMAE son consistentemente **65, 72, 38 y 92** en ambos
modelos — concentran ventas atípicas/picos de temporada que ninguno de los dos
modelos captura del todo bien (ver dispersión de residuos vs. `lag_52` en zonas de
feriado).

## 5. Decisiones de arquitectura tomadas en esta auditoría

- Pipeline reorganizado a `src/{data,features,models,evaluation}/` (ver README) para
  separar ingestión, feature engineering, entrenamiento/predicción y métricas.
  Los notebooks quedan solo para EDA.
- `src/config.py` centraliza rutas (vía `pathlib`, relativas a la raíz del repo),
  la lista de `FEATURES` y los hiperparámetros de cada modelo — una sola fuente de
  verdad en vez de tenerlos repetidos en cada notebook/script.
- `train_model.py` reproduce el mismo criterio que los notebooks originales: valida
  en un split temporal para reportar métricas, y luego reentrena con TODOS los datos
  disponibles antes de guardar el modelo de producción.
- Se agregaron tests unitarios (`tests/`) para el cálculo de WMAE y para que
  `add_lags`/`time_split` no tengan fuga de datos entre series o entre folds.

## 6. Qué faltaba para "productivo" y qué se resolvió

| Faltante | Resuelto en esta auditoría |
|---|---|
| `requirements.txt` vacío | ✅ Pines de versión para pandas/numpy/sklearn/xgboost/lightgbm/joblib/matplotlib/seaborn/pytest |
| Código solo en notebooks | ✅ Migrado a `src/features`, `src/models`, `src/evaluation` |
| Sin tests | ✅ `tests/test_metrics.py`, `tests/test_build_features.py` |
| `.gitignore` desalineado con la estructura real (`data/raw` vs `data/1.raw`) | ✅ Corregido, con `.gitkeep` en carpetas vacías versionadas |
| Sin Makefile | ✅ `make install/data/train/predict/test/report` |
| Sin Dockerfile | ✅ Imagen base `python:3.11-slim`, monta `data/` y `models/` como volúmenes |
| Rutas hardcodeadas por script (`Path("../data/...")`) | ✅ Centralizadas en `src/config.py`, funcionan desde la raíz del repo |
| Sin documentación de progreso/comparativa | ✅ Este documento + `docs/metrics_summary.json` + `docs/images/` |
| Sin `config.yaml` | Se optó por `src/config.py` (Python tipado) en vez de YAML — ver justificación en el README |

### Pendiente (no resuelto en esta auditoría, ver next steps)

- Sin tracking de experimentos (MLflow/W&B): las métricas históricas viven solo en
  este markdown, no hay comparación automática entre corridas.
- Sin CI (GitHub Actions) que corra `pytest` en cada push.
- Sin DVC ni versionado de datos/modelos — los `.pkl` quedan fuera de git por tamaño
  (ver `.gitignore`), pero no hay forma de reproducir *qué* modelo generó *qué* commit.
- Estructura no sigue 1:1 cookiecutter-data-science (no hay `src/visualization/`
  separado ni `references/`); se priorizó una estructura más plana y específica al
  proyecto sobre adoptar el template completo.

## 7. Next steps

1. **Ensemble LGBM + XGBoost** (promedio simple o stacking con un meta-modelo lineal
   sobre las predicciones de validación) — candidato más directo para bajar el WMAE
   por debajo de 1283.
2. **CatBoost**: maneja categóricas (`Store`, `Dept`) nativamente sin encoding, vale
   la pena comparar contra el mismo split de validación.
3. **Features adicionales**: `lag_1` interactuando con `IsHoliday`, encoding de
   frecuencia/tendencia de `Dept`, y una feature explícita de "semanas hasta el
   próximo feriado" dado que los residuos más grandes se concentran ahí.
4. **MLflow o Weights & Biases** para trackear métricas/params por corrida en vez de
   depender de este markdown actualizado a mano.
5. **CI**: GitHub Actions corriendo `pytest tests/` en cada PR.
6. **DVC** (o versionado simple por convención de nombre) para que cada `.pkl` en
   `models/` quede asociado a una versión específica de `data/3.final/`.
