"""
Construcción del conjunto de estudio y de los escenarios de información D0-D3.

Une los anuncios de venta de idealista18 (datos/procesados/idealista18/madrid_sale_2018.csv)
con los indicadores de distrito de 2018 (datos/procesados/distritos/distritos_2018.csv) mediante el
código de distrito, depura el resultado y define qué variables forma cada
escenario:

    D0: características de la vivienda
    D1: D0 + localización
    D2: D1 + contexto socioeconómico del distrito
    D3: D2 + entorno urbano del distrito

Requiere haber ejecutado antes idealista18_madrid.py y distritos_2018.py.

Salida (en datos/final/):
    - conjunto_estudio.csv: un anuncio por inmueble con todas las variables.
    - escenarios.json: lista de variables explicativas de cada escenario.

Uso:
    python construir_conjunto.py
"""

import json

import pandas as pd

from distritos_2018 import bloque
from rutas import CONJUNTO_CSV, DIR_FINAL, DISTRITOS_CSV, ESCENARIOS_JSON, VENTAS_CSV

OBJETIVO = "PRICE"

# Identificadores y variables de referencia: se guardan en el conjunto pero no
# entran en ningún escenario como variables explicativas.
REFERENCIA = ["ASSETID", "PERIOD", "COD_DISTRITO", "DISTRITO", "LOCATIONID", "LOCATIONNAME"]

# Variables descartadas, con el motivo
EXCLUIDAS = {
    "UNITPRICE": "es el precio dividido por la superficie (fuga de la variable objetivo)",
    "PARKINGSPACEPRICE": "vale 1 (valor de relleno) en el 97,7 % de los anuncios",
    "CONSTRUCTIONYEAR": "falta en el 59 % y tiene valores imposibles; se usa CADCONSTRUCTIONYEAR",
}

CARACTERISTICAS = [
    # Tamaño y distribución
    "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "ISDUPLEX", "ISSTUDIO",
    # Equipamiento de la vivienda y del edificio
    "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "AMENITYID", "HASPARKINGSPACE",
    "ISPARKINGSPACEINCLUDEDINPRICE", "HASBOXROOM", "HASWARDROBE", "HASSWIMMINGPOOL",
    "HASDOORMAN", "HASGARDEN",
    # Orientación, planta y vistas
    "HASNORTHORIENTATION", "HASSOUTHORIENTATION", "HASEASTORIENTATION", "HASWESTORIENTATION",
    "FLOORCLEAN", "ISINTOPFLOOR", "FLATLOCATIONID",
    # Estado y antigüedad
    "BUILTTYPEID_1", "BUILTTYPEID_2", "BUILTTYPEID_3",
    # Datos catastrales del edificio
    "CADCONSTRUCTIONYEAR", "CADMAXBUILDINGFLOOR", "CADDWELLINGCOUNT", "CADASTRALQUALITYID",
    # Control temporal: trimestre del anuncio
    "TRIMESTRE",
]

LOCALIZACION = [
    "LONGITUDE", "LATITUDE",
    "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA",
]


def cargar_y_depurar():
    """Carga los anuncios, los depura y les añade los indicadores de distrito."""
    ventas = pd.read_csv(VENTAS_CSV, dtype={"COD_DISTRITO": str})
    distritos = pd.read_csv(DISTRITOS_CSV, dtype={"COD_DISTRITO": str})
    pasos = [("Anuncios de partida", len(ventas))]

    # Anuncios con coordenadas fuera de Madrid: no caen en ningún distrito
    ventas = ventas.dropna(subset=["COD_DISTRITO"])
    pasos.append(("Tras eliminar coordenadas fuera de Madrid", len(ventas)))

    # Un mismo inmueble puede anunciarse en varios trimestres: se conserva el más reciente
    ventas = ventas.sort_values("PERIOD").drop_duplicates("ASSETID", keep="last")
    pasos.append(("Tras conservar un anuncio por inmueble", len(ventas)))

    conjunto = ventas.drop(columns=list(EXCLUIDAS)).merge(
        distritos.drop(columns="DISTRITO"), on="COD_DISTRITO", how="left", validate="many_to_one"
    )
    return conjunto.reset_index(drop=True), distritos, pasos


def definir_escenarios(distritos):
    indicadores = [c for c in distritos.columns if c not in ("COD_DISTRITO", "DISTRITO")]
    socio = [c for c in indicadores if bloque(c) == "socioeconomico"]
    entorno = [c for c in indicadores if bloque(c) == "entorno_urbano"]

    d0 = CARACTERISTICAS
    d1 = d0 + LOCALIZACION
    d2 = d1 + socio
    d3 = d2 + entorno
    return {"D0": d0, "D1": d1, "D2": d2, "D3": d3}


def main():
    conjunto, distritos, pasos = cargar_y_depurar()
    escenarios = definir_escenarios(distritos)

    columnas = REFERENCIA + [OBJETIVO] + escenarios["D3"]
    faltan = set(columnas) - set(conjunto.columns)
    if faltan:
        raise KeyError(f"Columnas que no están en el conjunto: {sorted(faltan)}")

    DIR_FINAL.mkdir(parents=True, exist_ok=True)
    conjunto[columnas].to_csv(CONJUNTO_CSV, index=False)
    ESCENARIOS_JSON.write_text(
        json.dumps({"objetivo": OBJETIVO, "escenarios": escenarios}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for paso, n in pasos:
        print(f"{paso}: {n:,}")
    print(f"\nConjunto de estudio: {len(conjunto):,} inmuebles x {len(columnas)} columnas")
    for nombre, variables in escenarios.items():
        print(f"  {nombre}: {len(variables)} variables explicativas")
    print(f"\nFicheros guardados en {DIR_FINAL}")


if __name__ == "__main__":
    main()
