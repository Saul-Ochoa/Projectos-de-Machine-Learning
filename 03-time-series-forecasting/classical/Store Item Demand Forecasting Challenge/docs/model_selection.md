# Selección de modelo — Store Item Demand Forecasting Challenge

Fuente: comparativa final de `notebooks/03-model.ipynb` (sección "Comparativa final
LightGBM vs XGBoost vs CatBoost"), ejecutada sobre el mismo split temporal 80/20
(`X_valid`/`y_valid`) y las mismas `FEATURES` para los tres modelos.

## Resultados (set de validación)

| Modelo | RMSE | MAE | RMSLE | MAPE % | SMAPE % | **WAPE %** | R2 | BIAS |
|---|---|---|---|---|---|---|---|---|
| **CatBoost** | 8.310 | 6.398 | 0.162 | 13.086 | 12.563 | **10.824** | 0.9309 | -0.417 |
| LightGBM | 8.359 | 6.440 | 0.162 | 13.202 | 12.628 | 10.894 | 0.9300 | **-0.052** |
| XGBoost | 8.392 | 6.459 | 0.163 | 13.191 | 12.663 | 10.926 | 0.9295 | -0.207 |

(Ordenado por WAPE % ascendente — criterio principal.)

## Criterio de decisión

1. **Principal: WAPE % más bajo.**
2. Secundario: R² más alto.
3. Terciario: BIAS más cercano a 0.

Aplicando el criterio tal como está definido: **CatBoost gana por el criterio
principal** (10.824% vs. 10.894% de LightGBM) y también tiene el R² más alto
(0.9309), así que no hace falta llegar al criterio terciario para desempatar.

## Ganador: **CatBoost**

## ⚠️ Nota — esto contradice la expectativa inicial de "LightGBM ganador"

La corrida anterior del proyecto asumía LightGBM (~10.89% WAPE) como ganador,
pero al correr la comparativa completa con los tres modelos bajo el mismo split,
**CatBoost lo superó por ~0.07 puntos de WAPE** y tiene mejor R². La diferencia
es pequeña (ambos modelos están dentro de ~1% relativo uno del otro en WAPE), así
que en la práctica son estadísticamente muy similares.

Sin embargo, hay un matiz relevante que el criterio principal no captura:

- **BIAS de CatBoost es ~8x mayor que el de LightGBM** (-0.417 vs. -0.052): CatBoost
  subestima la demanda de forma sistemáticamente más marcada. Para un caso de uso
  de *reposición de inventario* (donde subestimar tiene un costo de quiebre de
  stock), este sesgo puede pesar más que 0.07 puntos de WAPE.
- La diferencia de WAPE entre los tres modelos es pequeña frente a la variabilidad
  esperable entre corridas (distinto `random_state`/semilla, distinto hardware).

**Decisión tomada para este refactor:** se sigue el criterio principal tal como
fue definido (WAPE % más bajo) y se usa **CatBoost** como modelo de producción en
`scripts/train.py` (`WINNER_MODEL = "catboost"`). Si el objetivo de negocio
prioriza minimizar el sesgo sistemático sobre el error relativo agregado, LightGBM
es la alternativa recomendada — basta con correr `python scripts/train.py --model lgb`.

## Reproducibilidad

- Métricas generadas por `scripts/evaluate.py` → `output/metrics_comparison.csv`.
- Modelo final entrenado por `scripts/train.py` → `models/best_model.pkl`.
- Hiperparámetros de cada modelo: `src/models/train_model.py`.
