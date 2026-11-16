"""
Lógica del modelo de huella de carbono de la flota de Positiva.

Este módulo NO depende de Streamlit: contiene la carga/entrenamiento del
modelo, las predicciones, el ranking de vehículos y las validaciones.
Así se puede probar por separado (tests/test_utils.py) y reutilizar.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:  # scikit-learn >= 1.3
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover
    InconsistentVersionWarning = UserWarning

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent


def _ruta(subcarpeta: str, nombre: str) -> Path:
    """Busca el archivo en la subcarpeta; si no existe, lo busca suelto en la raíz."""
    en_carpeta = BASE / subcarpeta / nombre
    return en_carpeta if en_carpeta.exists() else BASE / nombre


RUTA_MODELO = _ruta("modelo", "modelo_huella_co2.pkl")
RUTA_META = _ruta("modelo", "metadatos.json")
RUTA_DATOS = _ruta("data", "tabla_modelado.csv")

# --------------------------------------------------------------------------
# Variables del modelo (las mismas del Notebook de Modelamiento)
# --------------------------------------------------------------------------
CATEGORICAS = ["Clase", "Marca", "Linea", "TIPO_CAJA"]
NUMERICAS = [
    "CILINDRAJE_CM3", "POTENCIA_HP", "PESO_KG",
    "anio", "mes", "gasto_mant_rep", "n_eventos_mant_rep",
    "km_mes", "antiguedad", "temp_prom", "humedad_prom",
]
FEATURES = CATEGORICAS + NUMERICAS
TARGET = "kg_co2e"

# Hiperparámetros finales (RandomizedSearchCV del Notebook 1)
HIPERPARAMETROS = dict(
    n_estimators=100, learning_rate=0.05, max_depth=2,
    min_samples_leaf=2, random_state=42,
)

# Perfiles técnicos Fasecolda de los 4 tipos de vehículo de la flota activa
PERFILES = {
    "VERSA": {
        "Clase": "AUTOMOVIL", "Marca": "NISSAN", "TIPO_CAJA": "AUTOMATICA",
        "CILINDRAJE_CM3": 1598, "POTENCIA_HP": 106, "PESO_KG": 1044,
        "fasecolda": "06433094",
    },
    "TIIDA": {
        "Clase": "AUTOMOVIL", "Marca": "NISSAN", "TIPO_CAJA": "MECANICA",
        "CILINDRAJE_CM3": 1598, "POTENCIA_HP": 106, "PESO_KG": 1102,
        "fasecolda": "06433111",
    },
    "D22/NP300": {
        "Clase": "CAMIONETA", "Marca": "NISSAN", "TIPO_CAJA": "MECANICA",
        "CILINDRAJE_CM3": 2389, "POTENCIA_HP": 143, "PESO_KG": 1435,
        "fasecolda": "06442038",
    },
    "OPTIMA": {
        "Clase": "AUTOMOVIL", "Marca": "KIA", "TIPO_CAJA": "TIPTRONICA",
        "CILINDRAJE_CM3": 1998, "POTENCIA_HP": 162, "PESO_KG": 1470,
        "fasecolda": "04633056",
    },
}

# Imágenes opcionales: se suben al repositorio con estos nombres (cualquier extensión)
LOGO = "logo_positiva"
FOTOS = {
    "VERSA": "foto_versa",
    "TIIDA": "foto_tiida",
    "D22/NP300": "foto_d22_np300",
    "OPTIMA": "foto_optima",
}
EXTENSIONES_IMG = (".png", ".jpg", ".jpeg", ".webp", ".svg")
MIME_IMG = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".svg": "image/svg+xml"}

# Columnas que debe traer un archivo para cálculo por lotes
COLUMNAS_USUARIO = [
    "anio", "mes", "km_mes", "antiguedad",
    "gasto_mant_rep", "n_eventos_mant_rep", "temp_prom", "humedad_prom",
]
COLUMNAS_LOTE = ["Linea"] + COLUMNAS_USUARIO


# --------------------------------------------------------------------------
# Datos y modelo
# --------------------------------------------------------------------------
def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Tabla de modelado (840 filas: 14 vehículos x 60 meses)."""
    return pd.read_csv(ruta)


def construir_pipeline() -> Pipeline:
    """Pipeline idéntico al del Notebook 1: OneHot + Gradient Boosting."""
    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("num", "passthrough", NUMERICAS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", GradientBoostingRegressor(**HIPERPARAMETROS)),
        ]
    )


def entrenar_modelo(df: pd.DataFrame | None = None) -> Pipeline:
    """Entrena el modelo final con el 100 % de los datos."""
    if df is None:
        df = cargar_datos()
    pipe = construir_pipeline()
    pipe.fit(df[FEATURES], df[TARGET])
    return pipe


def cargar_modelo() -> tuple[Pipeline, str]:
    """
    Carga el .pkl. Si no existe, o fue guardado con una versión distinta de
    scikit-learn (o no se puede leer), reentrena desde data/tabla_modelado.csv.
    Devuelve (modelo, origen) con origen 'pkl' o 'reentrenado'.
    """
    if RUTA_MODELO.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                return joblib.load(RUTA_MODELO), "pkl"
        except Exception:
            pass  # versión incompatible o archivo ilegible -> reentrenar
    return entrenar_modelo(), "reentrenado"


def cargar_metadatos() -> dict:
    if RUTA_META.exists():
        return json.loads(RUTA_META.read_text(encoding="utf-8"))
    return {}


# --------------------------------------------------------------------------
# Predicción
# --------------------------------------------------------------------------
def completar_perfil(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega Clase/Marca/TIPO_CAJA/CILINDRAJE/POTENCIA/PESO según 'Linea'."""
    out = df.copy()
    out["Linea"] = out["Linea"].astype(str).str.strip().str.upper()
    for col in ["Clase", "Marca", "TIPO_CAJA", "CILINDRAJE_CM3", "POTENCIA_HP", "PESO_KG"]:
        out[col] = out["Linea"].map(lambda l, c=col: PERFILES.get(l, {}).get(c, np.nan))
    return out


def predecir(modelo: Pipeline, registros: pd.DataFrame) -> np.ndarray:
    """Predice kg CO2e/mes. La huella no puede ser negativa -> se recorta en 0."""
    pred = modelo.predict(registros[FEATURES])
    return np.clip(pred, 0, None)


def validar_lote(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Valida un archivo cargado por el usuario. Devuelve (df_listo, mensajes).
    df_listo es None si hay errores que impiden calcular.
    """
    errores: list[str] = []
    faltan = [c for c in COLUMNAS_LOTE if c not in df.columns]
    if faltan:
        return None, [f"Faltan columnas obligatorias: {', '.join(faltan)}."]

    lote = completar_perfil(df)
    desconocidas = sorted(set(lote.loc[lote["Clase"].isna(), "Linea"].astype(str)))
    if desconocidas:
        errores.append(
            "Líneas no reconocidas (el modelo solo conoce "
            f"{', '.join(PERFILES)}): {', '.join(desconocidas)}."
        )
        return None, errores

    for col in COLUMNAS_USUARIO:
        lote[col] = pd.to_numeric(lote[col], errors="coerce")
    nulos = [c for c in COLUMNAS_USUARIO if lote[c].isna().any()]
    if nulos:
        return None, [f"Hay valores vacíos o no numéricos en: {', '.join(nulos)}."]

    if not lote["mes"].between(1, 12).all():
        return None, ["La columna 'mes' debe tener valores entre 1 y 12."]
    return lote, errores


# --------------------------------------------------------------------------
# Interpretación
# --------------------------------------------------------------------------
def nivel_huella(valor: float, historico: pd.Series) -> str:
    """Clasifica un valor mensual frente a la distribución histórica de la flota."""
    p33, p66 = np.percentile(historico, [33, 66])
    if valor <= p33:
        return "Bajo"
    if valor <= p66:
        return "Medio"
    return "Alto"


def importancias_por_variable(modelo: Pipeline) -> pd.DataFrame:
    """Importancia del modelo agregada por variable original (suma de one-hot)."""
    nombres = modelo.named_steps["preprocesador"].get_feature_names_out()
    imp = modelo.named_steps["modelo"].feature_importances_
    filas: dict[str, float] = {}
    for nombre, valor in zip(nombres, imp):
        limpio = nombre.split("__", 1)[1]
        original = next(
            (c for c in CATEGORICAS if limpio == c or limpio.startswith(c + "_")),
            limpio,
        )
        filas[original] = filas.get(original, 0.0) + float(valor)
    return (
        pd.DataFrame({"variable": list(filas), "importancia": list(filas.values())})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def ranking_flota(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ranking de vehículos por huella acumulada + residuo promedio.
    El 'esperado' se obtiene con predicciones out-of-fold (GroupKFold por
    vehículo): el modelo predice cada vehículo sin haberlo visto al entrenar.
    residuo > 0  => emite MÁS de lo esperado para su perfil y su uso.
    Devuelve (ranking, df con columna 'esperado_oof').
    """
    pipe = construir_pipeline()
    oof = cross_val_predict(
        pipe, df[FEATURES], df[TARGET],
        groups=df["CODIGO_VEHICULO"], cv=GroupKFold(n_splits=5),
    )
    d = df.copy()
    d["esperado_oof"] = oof
    d["residuo"] = d[TARGET] - d["esperado_oof"]
    ranking = (
        d.groupby(["CODIGO_VEHICULO", "Marca", "Linea"])
        .agg(
            huella_total=(TARGET, "sum"),
            huella_promedio_mensual=(TARGET, "mean"),
            residuo_promedio=("residuo", "mean"),
        )
        .reset_index()
        .sort_values("huella_total", ascending=False)
        .reset_index(drop=True)
    )
    return ranking, d


# --------------------------------------------------------------------------
# Imágenes (logo y fotos de vehículos) - opcionales
# --------------------------------------------------------------------------
def buscar_imagen(nombre_base: str) -> Path | None:
    """
    Busca <nombre_base>.(png|jpg|jpeg|webp|svg) en la raíz o en 'imagenes/',
    sin distinguir mayúsculas. Devuelve None si no existe (la app sigue funcionando).
    """
    for carpeta in (BASE, BASE / "imagenes"):
        if not carpeta.is_dir():
            continue
        for archivo in sorted(carpeta.iterdir()):
            if archivo.stem.lower() == nombre_base and archivo.suffix.lower() in EXTENSIONES_IMG:
                return archivo
    return None
