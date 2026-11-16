"""
Reentrena el modelo final y regenera los archivos de la carpeta modelo/.

Uso:
    python entrenar_modelo.py

Cuándo usarlo:
  - Cuando se actualice data/tabla_modelado.csv con nuevos meses.
  - Cuando cambie la versión de scikit-learn y el .pkl no cargue.
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import huella_utils as hu


def main() -> None:
    df = hu.cargar_datos()
    print(f"Datos: {df.shape[0]} filas, {df['CODIGO_VEHICULO'].nunique()} vehículos")

    # Métricas de referencia: split aleatorio 80/20 (igual que el Notebook 1)
    X, y = df[hu.FEATURES], df[hu.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    prueba = hu.construir_pipeline().fit(X_tr, y_tr)
    pred_te, pred_tr = prueba.predict(X_te), prueba.predict(X_tr)

    # Modelo final: 100 % de los datos
    modelo = hu.entrenar_modelo(df)
    hu.RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, hu.RUTA_MODELO)

    ranking, _ = hu.ranking_flota(df)
    metadatos = {
        "modelo": "GradientBoostingRegressor",
        "hiperparametros": hu.HIPERPARAMETROS,
        "variables_categoricas": hu.CATEGORICAS,
        "variables_numericas": hu.NUMERICAS,
        "variable_objetivo": hu.TARGET,
        "metricas_split_80_20": {
            "mae_test": float(mean_absolute_error(y_te, pred_te)),
            "rmse_test": float(np.sqrt(mean_squared_error(y_te, pred_te))),
            "r2_test": float(r2_score(y_te, pred_te)),
            "r2_train": float(r2_score(y_tr, pred_tr)),
        },
        "vehiculo_mayor_huella": str(ranking.iloc[0]["CODIGO_VEHICULO"]),
        "n_filas_entrenamiento": int(len(df)),
        "version_scikit_learn": sklearn.__version__,
        "fecha_entrenamiento": date.today().isoformat(),
    }
    hu.RUTA_META.write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modelo guardado en {hu.RUTA_MODELO}")
    print(json.dumps(metadatos["metricas_split_80_20"], indent=2))


if __name__ == "__main__":
    main()
