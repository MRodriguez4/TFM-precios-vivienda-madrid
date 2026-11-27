# Precios de la vivienda en Madrid: aportación de la información contextual

Código del Trabajo Fin de Máster (UNIR) de Miguel Rodríguez Marcos.

El objetivo del trabajo es evaluar qué aporta la incorporación progresiva de información contextual sobre las viviendas de Madrid a los modelos de aprendizaje automático que predicen su precio. Para ello se comparan cuatro escenarios de información anidados:

| Escenario | Información | Variables explicativas |
| --- | --- | --- |
| D0 | Características de la vivienda | 30 |
| D1 | D0 + localización | 35 |
| D2 | D1 + contexto socioeconómico del distrito | 39 |
| D3 | D2 + entorno urbano del distrito | 43 |

Todos los escenarios usan las mismas 75.799 viviendas y la misma variable objetivo: el precio de oferta (`PRICE`, en euros).

## Fuentes de datos

- **[idealista18](https://paezha.github.io/idealista18/)**: anuncios de venta de viviendas en Madrid durante 2018, con sus características, datos catastrales del edificio, coordenadas y distancias al centro, al metro y a la Castellana. Rey-Blanco, D., Arbués, P., López, F. y Páez, A. (2024). *A geo-referenced micro-data set of real estate listings for Spain's three largest cities*. Environment and Planning B. https://doi.org/10.1177/23998083241242844. Licencia ODbL 1.0.
- **Panel de Indicadores de Distritos y Barrios** del Ayuntamiento de Madrid, ediciones 2019 a 2022: demografía, renta, empleo, educación, salud, vulnerabilidad, equipamientos, zonas verdes y seguridad de los 21 distritos.

Todos los indicadores se refieren a 2018, el año de los anuncios. La población, los hogares y el nivel de estudios se toman a 1 de enero de 2018. Los equipamientos municipales (inventario de 2019) y las zonas verdes (2016 y 2019) son las únicas excepciones, porque no hay dato de 2018. El año de cada indicador figura en `datos/procesados/distritos/diccionario_distritos_2018.csv`.

## Estructura del repositorio

```
├── idealista18_madrid.py     1. Descarga los anuncios y les asigna zona y distrito
├── distritos_2018.py         2. Extrae los indicadores de 2018 de cada distrito
├── calidad_datos.py          3. Valida la calidad de los datos integrados, antes de depurar
├── construir_conjunto.py     4. Integra, depura y define los escenarios D0-D3
├── analisis_exploratorio.py  5. Análisis exploratorio del conjunto final, con sus figuras
├── modelizacion.py           6. Entrena y evalúa los modelos en cada escenario
├── interpretabilidad.py      7. Contribución de cada variable y bloque en D3 (SHAP)
├── rutas.py                  Rutas de la carpeta datos/
├── requirements.txt
└── datos/
    ├── originales/           Ficheros tal como se descargan (no se modifican)
    │   ├── ayuntamiento/     Paneles de indicadores 2018-2022
    │   └── idealista18/      Ficheros .rda del paquete idealista18
    ├── procesados/           Salidas intermedias (no se suben al repositorio)
    └── final/                Conjunto de estudio (no se sube al repositorio)
```

`datos/procesados/` y `datos/final/` no se versionan porque se regeneran con los scripts, y porque `conjunto_estudio.csv` (unos 115 MB) supera el límite de tamaño de GitHub.

## Instalación

Requiere Python 3.9 o superior. Desde la carpeta del proyecto:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

En Linux o macOS, el intérprete del entorno es `.venv/bin/python`.

## Uso

Los scripts se ejecutan en este orden:

```bash
.venv\Scripts\python idealista18_madrid.py
.venv\Scripts\python distritos_2018.py
.venv\Scripts\python calidad_datos.py
.venv\Scripts\python construir_conjunto.py
.venv\Scripts\python analisis_exploratorio.py
.venv\Scripts\python modelizacion.py
.venv\Scripts\python interpretabilidad.py
```

1. **`idealista18_madrid.py`** descarga los ficheros `.rda` de idealista18 (si no están ya en `datos/originales/idealista18/`) y los convierte a CSV. Asigna a cada anuncio la zona de idealista y el distrito en el que se encuentra, a partir de sus coordenadas; los que no caen dentro de ninguna zona se asignan a la más cercana si está a menos de 500 m. Opciones: `--forzar-descarga` y `--sin-barrios`.
2. **`distritos_2018.py`** lee las hojas de cada distrito de los paneles del Ayuntamiento y genera una tabla de 21 distritos por 119 indicadores (109 extraídos del panel y 10 tasas calculadas a partir de ellos), con su diccionario. Para cada indicador comprueba que el año que figura en el Excel es el esperado.
3. **`calidad_datos.py`** une todos los anuncios con los indicadores de su distrito, sin eliminar nada, y aplica diez validaciones a todas las variables: estructura, completitud, unicidad, validez, variabilidad, distribución, consistencia, integración, temporalidad de los indicadores que no se refieren a 2018 y redundancia (de los indicadores de distrito y de las variables de la vivienda y la localización). Cada una guarda una tabla de resultados; las decisiones de depuración se toman a partir de ellas.
4. **`construir_conjunto.py`** une los anuncios con los indicadores de su distrito y aplica las decisiones de depuración tomadas a partir de la validación: elimina el anuncio sin distrito y los estudios con dos o más dormitorios, y conserva de cada inmueble la fila con el precio más cercano a su mediana (los empates se deciden al azar con semilla fija). Después representa cada dimensión conceptual seleccionada del contexto (cuatro por bloque, respaldadas en el estado del arte del TFM: nivel económico, nivel educativo, población extranjera y estructura por edad; densidad urbana, zonas verdes, seguridad y comercio de proximidad) con un solo indicador, sin usar el precio: el primero de sus representantes que no sea casi constante, no mida solo el tamaño del distrito y no tenga una correlación de 0,9 o más con otro ya elegido. Las 21 dimensiones y la asignación de los 119 indicadores están en `distritos_2018.py`. Guarda el conjunto de estudio, la lista de variables de cada escenario y los pliegues de la validación cruzada: cinco aleatorios (semilla fija) y cinco espaciales formados por distritos completos, iguales para todos los escenarios y modelos.
5. **`analisis_exploratorio.py`** describe el conjunto final una vez definidos los escenarios: la distribución del precio, el precio por distrito y por zona (con la parte de su varianza que se da entre distritos), la relación del precio con las variables de D0 y D1, la de los 8 indicadores de contexto con el precio de su distrito, la dependencia espacial del precio (índice de Moran entre zonas vecinas), el reparto de los distritos en los cinco pliegues de la validación espacial, el precio de las viviendas según conste o no su planta, su posición y su calidad catastral, los atípicos del precio en euros y en logaritmo, y las viviendas con más habitaciones. Usa el precio, pero no modifica los escenarios: solo apoya decisiones de modelización que afectan por igual a todos ellos.
6. **`modelizacion.py`** entrena Ridge, Random Forest y XGBoost en cada escenario con el logaritmo del precio y los evalúa con las validaciones aleatoria y espacial, usando los pliegues de `pliegues.csv`. La preparación (imputación con indicador de dato ausente, superficie en logaritmo, trimestre en one-hot y, en Ridge, estandarización) se ajusta dentro de cada pliegue. Los hiperparámetros se eligen con validación cruzada anidada: tres pliegues internos del mismo tipo que la validación, una rejilla de 17 valores para Ridge y 15 combinaciones al azar para cada modelo de árboles. Calcula MAE, RMSE y R² en euros y la mejora de cada escenario pliegue a pliegue, y dibuja el MAE de cada pliegue por escenario y modelo. Con `--prueba` ejecuta un solo pliegue de D3 para medir tiempos; el experimento completo tarda varias horas y, si se interrumpe, continúa donde se quedó.
7. **`interpretabilidad.py`** explica con SHAP las predicciones de los modelos de D3. Para cada modelo, validación y pliegue vuelve a entrenar el modelo con los hiperparámetros elegidos en ese pliegue, comprueba que reproduce el MAE de `metricas_pliegues.csv` y calcula los valores SHAP (en logaritmo del precio) de las viviendas del pliegue de evaluación: todas en Ridge y XGBoost, y una muestra de 100 por pliegue en Random Forest, cuyo cálculo es mucho más lento. Resume la importancia de cada variable y de cada bloque de información con la media y la desviación entre pliegues. Con `--modelos` se ejecutan solo los modelos indicados y con `--prueba`, un pliegue con 20 viviendas; si se interrumpe, continúa donde se quedó.

## Resultados

| Fichero | Contenido |
| --- | --- |
| `datos/final/conjunto_estudio.csv` | 75.799 viviendas con el precio, los identificadores y las 43 variables explicativas |
| `datos/final/seleccion_indicadores.csv` | Decisión sobre cada representante de las dimensiones seleccionadas, con su motivo |
| `datos/final/escenarios.json` | Variable objetivo, columna que agrupa la validación espacial (distrito) y lista de variables de cada escenario (D0 a D3) |
| `datos/final/pliegues.csv` | Pliegue de cada vivienda en la validación aleatoria y en la espacial |
| `datos/procesados/distritos/distritos_2018.csv` | Indicadores de 2018 de los 21 distritos (también en `.xlsx`) |
| `datos/procesados/distritos/diccionario_distritos_2018.csv` | Descripción, fuente, fecha y bloque (socioeconómico o entorno urbano) de cada indicador |
| `datos/procesados/calidad/perfil_variables.csv` | Resultados de las validaciones para cada una de las 167 variables |
| `datos/procesados/calidad/calidad_datos.xlsx` | Todas las tablas de validación, una hoja por validación |
| `datos/procesados/exploratorio/` | Tablas y figuras del análisis exploratorio del conjunto final |
| `datos/procesados/modelizacion/` | Métricas por pliegue, resumen, mejoras entre escenarios, hiperparámetros elegidos, predicciones, figura del MAE por escenario, versiones de las librerías y equipo de cálculo |
| `datos/procesados/interpretabilidad/` | Comprobación de los modelos reproducidos, importancia SHAP de cada variable y bloque, valores SHAP de cada vivienda y figuras |
| `datos/procesados/idealista18/madrid_sale_2018.csv` | Los 94.815 anuncios originales con su barrio y distrito |

Para cargar un escenario desde Python:

```python
import json
import pandas as pd

conjunto = pd.read_csv("datos/final/conjunto_estudio.csv")
escenarios = json.load(open("datos/final/escenarios.json", encoding="utf-8"))

X = conjunto[escenarios["escenarios"]["D2"]]
y = conjunto[escenarios["objetivo"]]
grupos = conjunto[escenarios["grupo_validacion_espacial"]]  # para GroupKFold: distritos completos
```

El distrito solo se usa para repartir los datos en la validación espacial; no es una variable explicativa de ningún escenario.
