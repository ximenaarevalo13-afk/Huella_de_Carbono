"""
Huella de carbono de la flota de vehículos de Positiva - App Streamlit.

Proyecto de Machine Learning Probabilístico (CRISP-DM, fase Despliegue).
Ejecutar:  streamlit run app.py
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import huella_utils as hu

VERDE, NARANJA, GRIS = "#1F4529", "#FF7A00", "#8A948E"

st.set_page_config(
    page_title="Huella de carbono - Flota Positiva",
    page_icon="🚗",
    layout="wide",
)


# --------------------------------------------------------------------------
# Recursos en caché
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo...")
def get_modelo():
    return hu.cargar_modelo()


@st.cache_data(show_spinner="Cargando datos de la flota...")
def get_datos():
    return hu.cargar_datos()


@st.cache_data(show_spinner="Calculando ranking de la flota...")
def get_ranking():
    return hu.ranking_flota(hu.cargar_datos())


modelo, origen_modelo = get_modelo()
datos = get_datos()
meta = hu.cargar_metadatos()


def fig_a_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Encabezado y barra lateral
# --------------------------------------------------------------------------
st.title("Huella de carbono de la flota de Positiva")
st.caption(
    "Estimación mensual de kg CO2e por vehículo a partir de sus características "
    "técnicas y de uso, con un modelo Gradient Boosting."
)

with st.sidebar:
    st.header("Sobre esta app")
    st.write(
        "Calcula la huella de carbono (kg CO2e/mes) de un vehículo de la flota, "
        "muestra qué vehículo tiene la huella más alta y qué variables la explican."
    )
    st.markdown("**Modelo:** Gradient Boosting")
    st.markdown(f"**Variables predictoras:** {len(hu.FEATURES)}")
    m = meta.get("metricas_split_80_20", {})
    if m:
        st.markdown(f"**R² (prueba 80/20):** {m['r2_test']:.2f}")
        st.markdown(f"**MAE (prueba):** {m['mae_test']:.0f} kg CO2e/mes")
    if origen_modelo == "reentrenado":
        st.info(
            "El modelo se reentrenó al iniciar porque el .pkl no es compatible con "
            "la versión de scikit-learn instalada."
        )
    st.divider()
    st.caption("Grupo: Ximena Coronel y Camilo Macias")

tab_calc, tab_flota, tab_lote, tab_modelo = st.tabs(
    ["Calculadora", "Flota y causas", "Cálculo por lotes", "Sobre el modelo"]
)

# ==========================================================================
# TAB 1 - CALCULADORA
# ==========================================================================
with tab_calc:
    st.subheader("Calcular la huella de un vehículo")
    col_in, col_out = st.columns([1, 1])

    with col_in:
        linea = st.selectbox("Línea del vehículo", list(hu.PERFILES), index=0)
        perfil = hu.PERFILES[linea]
        st.caption(
            f"Ficha técnica Fasecolda {perfil['fasecolda']}: {perfil['Marca']} {linea} · "
            f"{perfil['Clase'].title()} · caja {perfil['TIPO_CAJA'].lower()} · "
            f"{perfil['CILINDRAJE_CM3']} cm³ · {perfil['POTENCIA_HP']} HP · {perfil['PESO_KG']} kg"
        )

        hist_linea = datos[datos["Linea"] == linea]
        km_med = int(hist_linea["km_mes"].median())
        ant_med = int(round(hist_linea["antiguedad"].median()))

        c1, c2 = st.columns(2)
        anio = c1.number_input("Año", min_value=2021, max_value=2035, value=2025, step=1)
        mes = c2.number_input("Mes", min_value=1, max_value=12, value=6, step=1)
        km = st.number_input(
            "Kilometraje del vehículo (odómetro, km)",
            min_value=0, max_value=600_000, value=km_med, step=1000,
            key=f"km_{linea}",
            help="En la base de datos la variable 'km_mes' es la lectura del odómetro "
                 "del vehículo en ese mes (kilometraje acumulado).",
        )
        antig = st.number_input(
            "Antigüedad (años)", min_value=0, max_value=30, value=ant_med, step=1,
            key=f"ant_{linea}",
        )
        c3, c4 = st.columns(2)
        gasto = c3.number_input(
            "Gasto en mantenimiento y repuestos del mes (COP)",
            min_value=0, max_value=50_000_000, value=0, step=50_000,
        )
        n_ev = c4.number_input(
            "Eventos de mantenimiento/reparación en el mes",
            min_value=0, max_value=60, value=0, step=1,
        )
        c5, c6 = st.columns(2)
        temp = c5.slider("Temperatura promedio (°C)", 5, 35, 14)
        hum = c6.slider("Humedad promedio (%)", 40, 95, 75)
        st.caption("Referencia de la flota actual: temperatura de 14, 22 y 28 °C según la ciudad.")

    base = {
        "Linea": linea, "anio": anio, "mes": mes, "km_mes": km, "antiguedad": antig,
        "gasto_mant_rep": gasto, "n_eventos_mant_rep": n_ev,
        "temp_prom": temp, "humedad_prom": hum,
    }
    registro = hu.completar_perfil(pd.DataFrame([base]))
    pred = float(hu.predecir(modelo, registro)[0])
    nivel = hu.nivel_huella(pred, datos[hu.TARGET])
    prom_linea = float(hist_linea[hu.TARGET].mean())

    with col_out:
        st.markdown("#### Resultado")
        m1, m2 = st.columns(2)
        m1.metric("Huella estimada del mes", f"{pred:,.0f} kg CO2e")
        m2.metric(
            f"Frente al promedio de {linea}",
            f"{prom_linea:,.0f} kg CO2e",
            delta=f"{pred - prom_linea:+,.0f} kg",
            delta_color="inverse",
        )
        m3, m4 = st.columns(2)
        m3.metric("Nivel frente a la flota", nivel)
        m4.metric("Si se mantiene todo el año", f"{pred * 12 / 1000:,.2f} t CO2e")
        st.caption(
            "El nivel compara contra la distribución histórica de la flota "
            "(Bajo: tercio inferior · Medio · Alto: tercio superior)."
        )

        # Sensibilidad al kilometraje
        rango = np.linspace(
            float(datos["km_mes"].min()), float(datos["km_mes"].max()), 40
        )
        barrido = pd.concat([registro] * len(rango), ignore_index=True)
        barrido["km_mes"] = rango
        y_barrido = hu.predecir(modelo, barrido)

        fig1, ax1 = plt.subplots(figsize=(6, 3.2))
        ax1.plot(rango / 1000, y_barrido, color=VERDE, linewidth=2)
        ax1.scatter([km / 1000], [pred], color=NARANJA, zorder=3, s=60, label="Vehículo actual")
        ax1.set_xlabel("Kilometraje del odómetro (miles de km)")
        ax1.set_ylabel("kg CO2e / mes")
        ax1.set_title("¿Cómo cambia la huella con el kilometraje?")
        ax1.legend(frameon=False)
        ax1.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig1)
        plt.close(fig1)

        # Mismo uso, distintos perfiles
        filas = []
        for lin in hu.PERFILES:
            r = hu.completar_perfil(pd.DataFrame([{**base, "Linea": lin}]))
            filas.append((lin, float(hu.predecir(modelo, r)[0])))
        comp = pd.DataFrame(filas, columns=["Linea", "kg_co2e"])
        fig2, ax2 = plt.subplots(figsize=(6, 2.6))
        ax2.bar(comp["Linea"], comp["kg_co2e"],
                color=[NARANJA if l == linea else VERDE for l in comp["Linea"]])
        ax2.set_ylabel("kg CO2e / mes")
        ax2.set_title("Mismo uso con distintos perfiles de vehículo")
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2)
        plt.close(fig2)

# ==========================================================================
# TAB 2 - FLOTA Y CAUSAS
# ==========================================================================
with tab_flota:
    ranking, datos_oof = get_ranking()
    top = ranking.iloc[0]

    st.subheader("¿Qué vehículo tiene la huella más alta?")
    k1, k2, k3 = st.columns(3)
    k1.metric("Vehículo con mayor huella acumulada", top["CODIGO_VEHICULO"])
    k2.metric("Huella acumulada 2021-2025", f"{top['huella_total'] / 1000:,.1f} t CO2e")
    k3.metric("Promedio mensual", f"{top['huella_promedio_mensual']:,.0f} kg CO2e")
    st.write(
        f"El **{top['Marca']} {top['Linea']}** `{top['CODIGO_VEHICULO']}` encabeza el ranking. "
        f"Su residuo promedio es **{top['residuo_promedio']:+.0f} kg CO2e/mes**: "
        + ("emite más de lo esperado para su perfil y su uso."
           if top["residuo_promedio"] > 0 else
           "emite menos de lo esperado para su perfil y su uso.")
    )

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    orden = ranking.sort_values("huella_total")
    colores = [NARANJA if v == top["CODIGO_VEHICULO"] else VERDE for v in orden["CODIGO_VEHICULO"]]
    ax3.barh(orden["CODIGO_VEHICULO"] + "  (" + orden["Linea"] + ")",
             orden["huella_total"] / 1000, color=colores)
    ax3.set_xlabel("Huella acumulada 2021-2025 (t CO2e)")
    ax3.set_title("Ranking de huella de carbono por vehículo")
    ax3.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig3)
    st.download_button("Descargar gráfico (PNG)", fig_a_png(fig3),
                       file_name="ranking_huella_flota.png", mime="image/png")
    plt.close(fig3)

    tabla = ranking.rename(columns={
        "CODIGO_VEHICULO": "Vehículo", "huella_total": "Huella total (kg CO2e)",
        "huella_promedio_mensual": "Promedio mensual (kg CO2e)",
        "residuo_promedio": "Residuo vs esperado (kg/mes)",
    }).round(1)
    st.dataframe(tabla)
    st.caption(
        "Residuo = huella real − huella esperada por el modelo para su perfil y uso "
        "(predicción fuera de muestra, validación por vehículo). Positivo = emite más de lo esperado."
    )

    st.divider()
    st.subheader("¿Qué está incidiendo en la huella?")
    imp = hu.importancias_por_variable(modelo)
    etiquetas = {
        "km_mes": "Kilometraje (odómetro)", "anio": "Año", "gasto_mant_rep": "Gasto mant./repuestos",
        "temp_prom": "Temperatura", "n_eventos_mant_rep": "Eventos de mantenimiento",
        "antiguedad": "Antigüedad", "Clase": "Clase de vehículo", "mes": "Mes",
        "PESO_KG": "Peso", "Marca": "Marca", "CILINDRAJE_CM3": "Cilindraje", "Linea": "Línea",
        "POTENCIA_HP": "Potencia", "TIPO_CAJA": "Tipo de caja", "humedad_prom": "Humedad",
    }
    top_imp = imp.head(8).iloc[::-1]
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.barh(top_imp["variable"].map(lambda v: etiquetas.get(v, v)),
             top_imp["importancia"] * 100, color=VERDE)
    ax4.set_xlabel("Importancia (%)")
    ax4.set_title("Variables que más explican la huella (modelo final)")
    ax4.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig4)
    plt.close(fig4)
    st.write(
        f"La variable dominante es el **{etiquetas.get(imp.iloc[0]['variable'], imp.iloc[0]['variable']).lower()}** "
        f"({imp.iloc[0]['importancia'] * 100:.0f} % de la importancia). "
        "Las características técnicas (cilindraje, potencia, peso) pesan poco porque la flota "
        "tiene solo 4 perfiles técnicos distintos."
    )

    st.divider()
    st.subheader("Detalle de un vehículo: real vs esperado")
    veh = st.selectbox("Vehículo", list(ranking["CODIGO_VEHICULO"]))
    dv = datos_oof[datos_oof["CODIGO_VEHICULO"] == veh].sort_values(["anio", "mes"])
    fechas = pd.to_datetime(dict(year=dv["anio"], month=dv["mes"], day=1))
    fig5, ax5 = plt.subplots(figsize=(9, 3.6))
    ax5.plot(fechas, dv[hu.TARGET], color=NARANJA, linewidth=1.8, label="Huella real")
    ax5.plot(fechas, dv["esperado_oof"], color=VERDE, linewidth=1.8, linestyle="--",
             label="Esperada por el modelo")
    ax5.set_ylabel("kg CO2e / mes")
    ax5.set_title(f"{veh}: huella mensual real vs esperada")
    ax5.legend(frameon=False)
    ax5.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig5)
    plt.close(fig5)

# ==========================================================================
# TAB 3 - CÁLCULO POR LOTES
# ==========================================================================
with tab_lote:
    st.subheader("Calcular la huella de varios registros a la vez")
    st.write(
        "Sube un archivo CSV o Excel con una fila por vehículo y mes. "
        "El perfil técnico (clase, marca, caja, cilindraje, potencia y peso) se completa "
        "automáticamente a partir de la **Línea**."
    )
    st.markdown("**Columnas obligatorias:** " + ", ".join(f"`{c}`" for c in hu.COLUMNAS_LOTE))
    st.markdown(f"**Líneas válidas:** {', '.join(hu.PERFILES)}  ·  `CODIGO_VEHICULO` es opcional.")

    plantilla = pd.DataFrame([
        {"CODIGO_VEHICULO": "VEH_EJEMPLO_1", "Linea": "VERSA", "anio": 2025, "mes": 6,
         "km_mes": 90000, "antiguedad": 6, "gasto_mant_rep": 0, "n_eventos_mant_rep": 0,
         "temp_prom": 14, "humedad_prom": 75},
        {"CODIGO_VEHICULO": "VEH_EJEMPLO_2", "Linea": "D22/NP300", "anio": 2025, "mes": 6,
         "km_mes": 200000, "antiguedad": 12, "gasto_mant_rep": 350000, "n_eventos_mant_rep": 2,
         "temp_prom": 28, "humedad_prom": 78},
    ])
    st.download_button("Descargar plantilla (CSV)", plantilla.to_csv(index=False).encode("utf-8"),
                       file_name="plantilla_lote.csv", mime="text/csv")

    archivo = st.file_uploader("Archivo con los registros", type=["csv", "xlsx"])
    if archivo is not None:
        try:
            df_in = pd.read_csv(archivo) if archivo.name.lower().endswith(".csv") else pd.read_excel(archivo)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            df_in = None

        if df_in is not None:
            lote, mensajes = hu.validar_lote(df_in)
            if lote is None:
                for msg in mensajes:
                    st.error(msg)
            else:
                lote["huella_kg_co2e_pred"] = hu.predecir(modelo, lote).round(1)
                salida = lote[[c for c in df_in.columns if c in lote.columns] + ["huella_kg_co2e_pred"]]
                st.success(f"Se calcularon {len(salida)} registros.")
                st.dataframe(salida)
                st.download_button("Descargar resultados (CSV)",
                                   salida.to_csv(index=False).encode("utf-8"),
                                   file_name="huella_calculada.csv", mime="text/csv")

                if "CODIGO_VEHICULO" in salida.columns:
                    resumen = (
                        salida.groupby("CODIGO_VEHICULO")["huella_kg_co2e_pred"]
                        .agg(["sum", "mean", "count"]).round(1)
                        .rename(columns={"sum": "Total (kg CO2e)", "mean": "Promedio mensual",
                                         "count": "Meses"})
                        .sort_values("Total (kg CO2e)", ascending=False)
                    )
                    st.markdown("**Resumen por vehículo** (de mayor a menor huella)")
                    st.dataframe(resumen)

# ==========================================================================
# TAB 4 - SOBRE EL MODELO
# ==========================================================================
with tab_modelo:
    st.subheader("Cómo funciona el modelo")
    st.markdown(
        """
- **Objetivo:** estimar la huella de carbono mensual (kg CO2e) de cada vehículo y
  explicar qué la impulsa. No incluye escenarios de continuidad, eléctricos ni alquiler.
- **Datos:** 14 vehículos activos × 60 meses (2021-2025) = 840 registros.
- **Modelo:** Gradient Boosting, elegido entre Regresión Lineal, Random Forest y
  Gradient Boosting, con ajuste de hiperparámetros (`RandomizedSearchCV`).
- **Variables predictoras:** línea, clase, marca, tipo de caja, cilindraje, potencia, peso,
  año, mes, kilometraje (odómetro), antigüedad, gasto y eventos de mantenimiento,
  temperatura y humedad.
- **Variables excluidas para evitar fuga de datos:** galones, gasto de combustible y número de
  facturas de combustible, porque la huella se calcula directamente a partir del consumo.
        """
    )
    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² prueba (80/20)", f"{m['r2_test']:.2f}")
        c2.metric("R² entrenamiento", f"{m['r2_train']:.2f}")
        c3.metric("MAE prueba", f"{m['mae_test']:.0f} kg")
        c4.metric("RMSE prueba", f"{m['rmse_test']:.0f} kg")

    st.warning(
        "**Limitaciones.** Con solo 14 vehículos y 4 perfiles técnicos, el modelo explica bien "
        "la flota actual pero **no es un predictor confiable para vehículos de líneas nuevas**: "
        "al validar dejando vehículos completos fuera (GroupKFold) el R² cae a ≈ 0.09. "
        "Úsalo para comparar y priorizar vehículos de la flota, no como medición oficial de emisiones."
    )
    st.caption(
        f"Modelo cargado desde: {'archivo .pkl' if origen_modelo == 'pkl' else 'reentrenamiento automático'} · "
        f"scikit-learn del entrenamiento: {meta.get('version_scikit_learn', 'n/d')} · "
        f"fecha: {meta.get('fecha_entrenamiento', 'n/d')}"
    )
