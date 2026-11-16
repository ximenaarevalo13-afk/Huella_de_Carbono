# Huella de carbono de la flota de Positiva

App web (Streamlit) que estima la **huella de carbono mensual (kg CO2e) de cada vehículo** de la flota de Positiva, identifica **qué vehículo tiene la huella más alta** y muestra **qué variables la explican**.

Es la fase de **Despliegue** del proyecto de *Machine Learning Probabilístico* (metodología CRISP-DM).

**Equipo:** Yury Ximena Coronel Arévalo · José Camilo Macias Sandoval

## Qué hace la app

| Pestaña | Contenido |
|---|---|
| **Calculadora** | Eliges la línea del vehículo, ingresas su uso (odómetro, antigüedad, mantenimiento, clima) y obtienes la huella estimada, su nivel (Bajo/Medio/Alto) y cómo cambia con el kilometraje. |
| **Flota y causas** | Ranking de los 14 vehículos por huella acumulada, variables que más inciden, y detalle real vs. esperado por vehículo. |
| **Cálculo por lotes** | Sube un CSV/Excel con muchos registros y descarga los resultados. |
| **Sobre el modelo** | Metodología, métricas y limitaciones. |

## Estructura del repositorio

Todos los archivos están en la raíz (sin carpetas):

```
app.py                     # Interfaz Streamlit (archivo principal)
huella_utils.py            # Lógica: carga del modelo, predicción, ranking, validaciones
entrenar_modelo.py         # Reentrena y regenera el modelo (ver más abajo)
modelo_huella_co2.pkl      # Modelo entrenado (Pipeline completo)
metadatos.json             # Hiperparámetros, métricas y versión de scikit-learn
tabla_modelado.csv         # Tabla de modelado (14 vehículos x 60 meses)
plantilla_lote.csv         # Ejemplo de archivo para el cálculo por lotes
requirements.txt           # Dependencias
README.md                  # Este archivo
```

## Ejecutar en tu computador

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en <http://localhost:8501>.

## Subir a GitHub

```bash
git init
git add .
git commit -m "App de huella de carbono de la flota de Positiva"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<nombre-del-repo>.git
git push -u origin main
```

> **Privacidad de los datos.** `tabla_modelado.csv` contiene información de la flota de la empresa (con códigos de vehículo anonimizados). Si el repositorio es **público**, cualquiera puede verla. Recomendamos crear el repositorio como **privado**, o pedir autorización a Positiva antes de publicarlo. Streamlit Community Cloud puede desplegar repositorios privados.

## Desplegar en Streamlit Community Cloud

1. Entra a <https://share.streamlit.io> e inicia sesión con tu cuenta de GitHub.
2. Pulsa **Create app** (o **New app**) y elige tu repositorio.
3. Rama: `main`. Archivo principal: `app.py`.
4. En **Advanced settings** elige Python 3.11 o 3.12.
5. Pulsa **Deploy**. La primera vez tarda unos minutos mientras instala `requirements.txt`.

## El modelo

- **Algoritmo:** Gradient Boosting (`GradientBoostingRegressor`) dentro de un `Pipeline` con `OneHotEncoder` para las variables categóricas. Fue el mejor entre Regresión Lineal, Random Forest y Gradient Boosting, y sus hiperparámetros se ajustaron con `RandomizedSearchCV`.
- **Variable objetivo:** `kg_co2e` por vehículo y mes.
- **Variables predictoras (15):**
  - Categóricas: `Clase`, `Marca`, `Linea`, `TIPO_CAJA`.
  - Numéricas: `CILINDRAJE_CM3`, `POTENCIA_HP`, `PESO_KG`, `anio`, `mes`, `gasto_mant_rep`, `n_eventos_mant_rep`, `km_mes`, `antiguedad`, `temp_prom`, `humedad_prom`.
- **Excluidas para evitar fuga de datos:** `galones`, `gasto_combustible` y `n_facturas_combustible`. La huella se calcula directamente del consumo de combustible, así que incluirlas haría trampa.
- **Nota sobre `km_mes`:** en la base de datos esta variable es la **lectura del odómetro** del vehículo en ese mes (kilometraje acumulado), no los kilómetros recorridos en el mes. La app la presenta así.

### Resultados

| Métrica (prueba 80/20) | Valor |
|---|---|
| R² | 0.34 |
| MAE | 118 kg CO2e/mes |
| RMSE | 154 kg CO2e/mes |

- **Vehículo con mayor huella:** `VEH_2D5T43TE` (Nissan D22/NP300), ≈ 26,4 t CO2e acumuladas en 2021-2025.
- **Variable que más incide:** el kilometraje (odómetro), con ≈ 64 % de la importancia.

### Limitaciones

La flota solo tiene **14 vehículos y 4 perfiles técnicos**. Al validar dejando vehículos completos fuera (`GroupKFold`), el R² baja a ≈ 0.09. Por eso el modelo sirve para **comparar y priorizar vehículos de la flota actual**, no como predictor confiable para líneas de vehículos nuevas ni como medición oficial de emisiones.

## Actualizar o regenerar el modelo

Si agregas meses nuevos a `tabla_modelado.csv`:

```bash
python entrenar_modelo.py
```

Esto reescribe `modelo_huella_co2.pkl` y `metadatos.json`.

**Compatibilidad de versiones:** un `.pkl` de scikit-learn solo se garantiza en la misma versión con la que se guardó. Si el servidor tiene otra versión, la app lo detecta y **reentrena el modelo automáticamente** desde `tabla_modelado.csv` (el resultado es idéntico), y lo avisa en la barra lateral.
