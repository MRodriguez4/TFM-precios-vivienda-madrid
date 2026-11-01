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
    - escenarios.json: variable objetivo, columna que agrupa la validación espacial y lista
      de variables explicativas de cada escenario.
    - seleccion_indicadores.csv: decisión sobre cada indicador evaluado para D2 y D3, con su motivo.

Uso:
    python construir_conjunto.py
"""

import json

import numpy as np
import pandas as pd

from distritos_2018 import DIMENSIONES
from rutas import CONJUNTO_CSV, DIR_FINAL, DISTRITOS_CSV, ESCENARIOS_JSON, SELECCION_CSV, VENTAS_CSV

OBJETIVO = "PRICE"
# La validación espacial reserva distritos completos para la evaluación. El distrito se usa
# solo para repartir los datos: no es una variable explicativa de ningún escenario.
GRUPO_VALIDACION_ESPACIAL = "COD_DISTRITO"
SEMILLA = 2018  # para elegir al azar entre filas empatadas del mismo inmueble

# Identificadores y variables de referencia: se guardan en el conjunto pero no
# entran en ningún escenario como variables explicativas.
REFERENCIA = ["ASSETID", "PERIOD", "COD_DISTRITO", "DISTRITO", "LOCATIONID", "LOCATIONNAME"]

# Variables descartadas, con el motivo
EXCLUIDAS = {
    "UNITPRICE": "es el precio dividido por la superficie (fuga de la variable objetivo)",
    "PARKINGSPACEPRICE": "vale 1 en casi todos los anuncios, que no es un precio de garaje posible",
    "CONSTRUCTIONYEAR": "muchos ausentes y años fuera de dominio; idealista18 la sustituye por CADCONSTRUCTIONYEAR",
}

# Variables de la vivienda descartadas por redundancia exacta (ver la validación 10b de
# calidad_datos.py, que las sigue analizando junto a las de D0 y D1)
EXCLUIDAS_POR_REDUNDANCIA = {
    "ISPARKINGSPACEINCLUDEDINPRICE": "idéntica a HASPARKINGSPACE en todos los anuncios; la documentación "
                                     "de idealista18 las describe con la misma definición",
}
EXCLUIDAS.update(EXCLUIDAS_POR_REDUNDANCIA)

CARACTERISTICAS = [
    # Tamaño y distribución
    "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "ISDUPLEX", "ISSTUDIO",
    # Equipamiento de la vivienda y del edificio
    "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "AMENITYID", "HASPARKINGSPACE",
    "HASBOXROOM", "HASWARDROBE", "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN",
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

# Selección de indicadores de distrito para D2 y D3: una variable por dimensión.
# Los 119 indicadores aportan como máximo 20 dimensiones independientes con 21
# distritos (ver la validación 10a de calidad_datos.py). Cada dimensión
# conceptual seleccionada (definidas en distritos_2018.DIMENSIONES) se representa
# con un solo indicador, el primero de sus representantes que cumple la regla:
#   - no es casi constante (más de MIN_VALORES_DISTINTOS valores en los 21 distritos),
#   - no mide el tamaño del distrito (|r| con la población <= MAX_R_POBLACION),
#   - no repite a uno ya elegido (|r| < MAX_R_ENTRE_ELEGIDAS con todos ellos).
# Si ningún representante la cumple, la dimensión queda sin representar porque
# su información ya la aporta otro indicador elegido. El precio no interviene.
MIN_VALORES_DISTINTOS = 5
MAX_R_POBLACION = 0.8
MAX_R_ENTRE_ELEGIDAS = 0.9
# Cuatro dimensiones por bloque, todas respaldadas en el estado del arte del TFM
# (capítulo 2.2). La accesibilidad al transporte ya está en D1 (distancia al metro).
DIMENSIONES_SELECCIONADAS = {
    "socioeconomico": [
        "Nivel económico",             # Bayer, Ferreira y McMillan (2007)
        "Nivel educativo",             # Marmolejo-Duarte (2017)
        "Población extranjera",        # Saiz y Wachter (2011)
        "Estructura por edad y sexo",  # Mankiw y Weil (1989)
    ],
    "entorno_urbano": [
        "Densidad urbana",             # Marmolejo-Duarte (2017): densidad edificada
        "Zonas verdes",                # Panduro y Veie (2013)
        "Seguridad",                   # Gibbons y Machin (2008)
        "Comercio de proximidad",      # Gibbons y Machin (2008)
    ],
}


def una_fila_por_inmueble(ventas):
    """De cada inmueble repetido, conserva la fila con el precio más cercano a su mediana.

    Las repeticiones de un ASSETID están todas en el mismo trimestre y con precios
    distintos (ver la validación de unicidad de calidad_datos.py). Se conserva un
    registro real, sin combinar valores de filas distintas. Si varias filas están
    igual de cerca de la mediana (siempre ocurre cuando el inmueble aparece dos
    veces), se elige una al azar con semilla fija: es reproducible y no sesga el
    precio en ningún sentido.
    """
    mediana = ventas.groupby("ASSETID")["PRICE"].transform("median")
    azar = np.random.default_rng(SEMILLA).random(len(ventas))
    orden = ventas.assign(_distancia=(ventas["PRICE"] - mediana).abs(), _azar=azar)
    orden = orden.sort_values(["ASSETID", "_distancia", "_azar"])

    # Cuántas veces decide el azar: se comparan las dos mejores filas de cada inmueble repetido
    dos_mejores = orden[orden.duplicated("ASSETID", keep=False)].groupby("ASSETID").head(2)
    primera, segunda = dos_mejores.iloc[::2], dos_mejores.iloc[1::2]
    empate = primera["_distancia"].to_numpy() == segunda["_distancia"].to_numpy()
    desempates = {"Inmuebles repetidos": len(primera),
                  "Inmuebles repetidos decididos al azar por empate": int(empate.sum())}
    return orden.drop_duplicates("ASSETID", keep="first").drop(columns=["_distancia", "_azar"]), desempates


def depurar_anuncios(ventas):
    """Aplica las decisiones de depuración tomadas a partir de calidad_datos.py."""
    pasos = [("Anuncios de partida", len(ventas))]

    # Anuncios a más de MARGEN_ZONA_M metros de cualquier zona de Madrid (ver idealista18_madrid.py)
    ventas = ventas.dropna(subset=["COD_DISTRITO"])
    pasos.append(("Tras eliminar los anuncios sin distrito", len(ventas)))

    # Contradicción en los datos del anunciante: estudio con dos o más dormitorios
    ventas = ventas[~((ventas["ISSTUDIO"] == 1) & (ventas["ROOMNUMBER"] > 1))]
    pasos.append(("Tras eliminar los estudios con dos o más dormitorios", len(ventas)))

    ventas, desempates = una_fila_por_inmueble(ventas)
    pasos.append(("Tras conservar una fila por inmueble", len(ventas)))
    pasos += [(f"  {k}", v) for k, v in desempates.items()]
    return ventas, pasos


def integrar():
    """Une todos los anuncios con los indicadores de su distrito, sin filtrar nada.

    Los anuncios que no caen en ningún distrito se conservan con los indicadores vacíos.
    """
    ventas = pd.read_csv(VENTAS_CSV, dtype={"COD_DISTRITO": str})
    distritos = pd.read_csv(DISTRITOS_CSV, dtype={"COD_DISTRITO": str})
    integrado = ventas.merge(distritos.drop(columns="DISTRITO"), on="COD_DISTRITO", how="left",
                             validate="many_to_one")
    return integrado, distritos


def cargar_y_depurar():
    """Integra los anuncios con los indicadores de distrito y depura el resultado."""
    integrado, distritos = integrar()
    conjunto, pasos = depurar_anuncios(integrado)
    conjunto = conjunto.drop(columns=list(EXCLUIDAS))
    return conjunto.reset_index(drop=True), distritos, pasos


def criterios_indicadores(distritos):
    """Criterios fijos de la regla de selección para cada indicador de distrito.

    Valores distintos en los 21 distritos y correlación con la población. Los usan la
    regla de selección y la validación 10a de calidad_datos.py, para que no haya dos
    cálculos distintos de lo mismo. El precio no interviene.
    """
    t = distritos.set_index("COD_DISTRITO").drop(columns="DISTRITO")
    return pd.DataFrame({"valores_distintos": t.nunique(), "r_poblacion": t.corrwith(t["poblacion"])})


def seleccionar_indicadores(distritos):
    """Elige un indicador por cada dimensión seleccionada, sin usar el precio.

    Devuelve la selección de cada bloque y una tabla con la decisión sobre cada
    representante evaluado.
    """
    corr = distritos.set_index("COD_DISTRITO").drop(columns="DISTRITO").corr().abs()
    criterios = criterios_indicadores(distritos)
    elegidas, decisiones = [], []
    for nombre_bloque, dimensiones in DIMENSIONES_SELECCIONADAS.items():
        for dim in dimensiones:
            representada = False
            for orden, col in enumerate(DIMENSIONES[dim]["representantes"], start=1):
                valores, r_poblacion = criterios.loc[col, "valores_distintos"], abs(criterios.loc[col, "r_poblacion"])
                r_max = corr.loc[col, elegidas].max() if elegidas else 0.0
                if representada:
                    motivo = "la dimensión ya está representada"
                elif valores <= MIN_VALORES_DISTINTOS:
                    motivo = f"casi constante: {valores} valores distintos en 21 distritos"
                elif r_poblacion > MAX_R_POBLACION:
                    motivo = f"mide el tamaño del distrito: |r| con la población = {r_poblacion:.2f}"
                elif r_max >= MAX_R_ENTRE_ELEGIDAS:
                    motivo = f"redundante con {corr.loc[col, elegidas].idxmax()}: |r| = {r_max:.2f}"
                else:
                    motivo = None
                    elegidas.append(col)
                    representada = True
                decisiones.append({"bloque": nombre_bloque, "dimension": dim, "orden": orden, "indicador": col,
                                   "seleccionado": motivo is None, "motivo_descarte": motivo,
                                   "max_abs_r_con_seleccionados_previos": round(float(r_max), 4)})
    decisiones = pd.DataFrame(decisiones)
    seleccion = {b: list(decisiones.query("bloque == @b and seleccionado").indicador)
                 for b in DIMENSIONES_SELECCIONADAS}
    return seleccion, decisiones


def definir_escenarios(seleccion):
    d0 = CARACTERISTICAS
    d1 = d0 + LOCALIZACION
    d2 = d1 + seleccion["socioeconomico"]
    d3 = d2 + seleccion["entorno_urbano"]
    return {"D0": d0, "D1": d1, "D2": d2, "D3": d3}


def main():
    conjunto, distritos, pasos = cargar_y_depurar()
    seleccion, decisiones = seleccionar_indicadores(distritos)
    escenarios = definir_escenarios(seleccion)

    columnas = REFERENCIA + [OBJETIVO] + escenarios["D3"]
    faltan = set(columnas) - set(conjunto.columns)
    if faltan:
        raise KeyError(f"Columnas que no están en el conjunto: {sorted(faltan)}")

    DIR_FINAL.mkdir(parents=True, exist_ok=True)
    conjunto[columnas].to_csv(CONJUNTO_CSV, index=False)
    decisiones.to_csv(SELECCION_CSV, index=False, encoding="utf-8-sig")
    ESCENARIOS_JSON.write_text(
        json.dumps({"objetivo": OBJETIVO, "grupo_validacion_espacial": GRUPO_VALIDACION_ESPACIAL,
                    "escenarios": escenarios}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for paso, n in pasos:
        print(f"{paso}: {n:,}")
    print(f"\nConjunto de estudio: {len(conjunto):,} inmuebles x {len(columnas)} columnas")
    for nombre, variables in escenarios.items():
        print(f"  {nombre}: {len(variables)} variables explicativas")

    print("\nSelección de indicadores de distrito (sin usar el precio):")
    for _, d in decisiones.iterrows():
        estado = "SÍ" if d.seleccionado else "no"
        print(f"  [{estado}] {d.dimension}: {d.indicador}" + ("" if d.seleccionado else f"  ({d.motivo_descarte})"))
    elegidos = seleccion["socioeconomico"] + seleccion["entorno_urbano"]
    z = distritos[elegidos].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    pares = distritos[elegidos].corr().abs().where(lambda m: ~np.eye(len(m), dtype=bool))
    print(f"  Indicadores seleccionados: {len(elegidos)}; rango de su matriz estandarizada: "
          f"{np.linalg.matrix_rank(z.to_numpy())}; |r| máxima entre ellos: {pares.max().max():.2f}")

    final = conjunto[columnas]
    precio = final[OBJETIVO]
    print(f"\nPrecio: mínimo {precio.min():,.0f} €, mediana {precio.median():,.0f} €, máximo {precio.max():,.0f} €")
    print("Viviendas por trimestre de su anuncio:")
    for periodo, k in final["PERIOD"].value_counts().sort_index().items():
        print(f"  {periodo}: {k:,} ({100 * k / len(final):.1f} %)")
    print("Valores ausentes:")
    for col, k in final.isna().sum()[lambda x: x > 0].items():
        print(f"  {col}: {k:,} ({100 * k / len(final):.2f} %)")
    print(f"\nFicheros guardados en {DIR_FINAL}")


if __name__ == "__main__":
    main()
