"""
Rutas de los datos del proyecto, en un único sitio.

    datos/
    ├── originales/      ficheros tal como se descargan (no se modifican)
    │   ├── ayuntamiento/    paneles de indicadores de distritos 2018-2022
    │   └── idealista18/     .rda del paquete idealista18
    ├── procesados/      salidas intermedias de los scripts
    │   ├── idealista18/     anuncios, barrios y puntos de interés en CSV
    │   ├── distritos/       indicadores de distrito de 2018 y su diccionario
    │   ├── calidad/         perfil de las variables y cifras del capítulo de datos
    │   ├── exploratorio/    análisis exploratorio del conjunto final y sus figuras
    │   └── modelizacion/    métricas, mejoras, hiperparámetros y predicciones de los modelos
    └── final/           conjunto de estudio, definición de los escenarios y pliegues
"""

from pathlib import Path

DIR_DATOS = Path(__file__).resolve().parent / "datos"

DIR_ORIGINALES = DIR_DATOS / "originales"
DIR_AYUNTAMIENTO = DIR_ORIGINALES / "ayuntamiento"
DIR_IDEALISTA_RDA = DIR_ORIGINALES / "idealista18"

DIR_PROCESADOS = DIR_DATOS / "procesados"
DIR_IDEALISTA = DIR_PROCESADOS / "idealista18"
DIR_DISTRITOS = DIR_PROCESADOS / "distritos"
DIR_CALIDAD = DIR_PROCESADOS / "calidad"
DIR_EXPLORATORIO = DIR_PROCESADOS / "exploratorio"
DIR_MODELIZACION = DIR_PROCESADOS / "modelizacion"
DIR_INTERPRETABILIDAD = DIR_PROCESADOS / "interpretabilidad"

DIR_FINAL = DIR_DATOS / "final"

# Ficheros que pasan de un script a otro
VENTAS_CSV = DIR_IDEALISTA / "madrid_sale_2018.csv"
DISTRITOS_CSV = DIR_DISTRITOS / "distritos_2018.csv"
CONJUNTO_CSV = DIR_FINAL / "conjunto_estudio.csv"
ESCENARIOS_JSON = DIR_FINAL / "escenarios.json"
SELECCION_CSV = DIR_FINAL / "seleccion_indicadores.csv"
PLIEGUES_CSV = DIR_FINAL / "pliegues.csv"
