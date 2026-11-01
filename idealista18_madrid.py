"""
Descarga y preparación del dataset de viviendas en venta de Madrid (idealista18).

Fuente: paquete R `idealista18` (https://paezha.github.io/idealista18/).
Los datos se publican como ficheros .rda (objetos `sf` de R) en el repositorio
https://github.com/paezha/idealista18, así que se descargan y se convierten a
pandas con la librería `rdata`, sin necesidad de tener R instalado.

Ficheros que se obtienen:
    - Madrid_Sale:     anuncios de venta de 2018 (94.815 filas, 42 columnas).
    - Madrid_Polygons: 135 zonas/barrios de idealista con su geometría.
    - Madrid_POIS:     puntos de interés (centro, estaciones de metro, Castellana).

Además, a cada anuncio se le asigna la zona (barrio) y el distrito en el que
cae, para poder cruzarlo con los indicadores por distrito del Ayuntamiento.

Uso:
    python idealista18_madrid.py                 # descarga y genera los CSV
    python idealista18_madrid.py --sin-barrios   # sin el cruce espacial
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import requests
import rdata

from rutas import DIR_IDEALISTA, DIR_IDEALISTA_RDA, VENTAS_CSV

BASE_URL = "https://github.com/paezha/idealista18/raw/master/data/{}.rda"
FICHEROS = ["Madrid_Sale", "Madrid_Polygons", "Madrid_POIS"]

# Códigos oficiales de distrito del Ayuntamiento de Madrid. Coinciden con el
# octavo segmento de LOCATIONID (p. ej. 0-EU-ES-28-07-001-079-16-002 -> 16).
DISTRITOS = {
    "01": "Centro", "02": "Arganzuela", "03": "Retiro", "04": "Salamanca",
    "05": "Chamartín", "06": "Tetuán", "07": "Chamberí",
    "08": "Fuencarral-El Pardo", "09": "Moncloa-Aravaca", "10": "Latina",
    "11": "Carabanchel", "12": "Usera", "13": "Puente de Vallecas",
    "14": "Moratalaz", "15": "Ciudad Lineal", "16": "Hortaleza",
    "17": "Villaverde", "18": "Villa de Vallecas", "19": "Vicálvaro",
    "20": "San Blas-Canillejas", "21": "Barajas",
}


def descargar(nombre, destino=DIR_IDEALISTA_RDA, forzar=False):
    """Descarga un .rda del repositorio de idealista18 (con caché local)."""
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{nombre}.rda"
    if ruta.exists() and not forzar:
        return ruta

    print(f"Descargando {nombre}.rda ...")
    resp = requests.get(BASE_URL.format(nombre), timeout=120)
    resp.raise_for_status()
    ruta.write_bytes(resp.content)
    return ruta


def leer_rda(ruta):
    """Lee un .rda y devuelve el objeto que contiene."""
    # rdata avisa de que no conoce las clases de `sf` (sfc, crs, bbox...),
    # pero las convierte a estructuras básicas, que es lo que necesitamos.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        objetos = rdata.read_rda(ruta, default_encoding="utf8")
    return next(iter(objetos.values()))


def cargar_ventas(ruta):
    """Anuncios de venta. Se descarta la geometría (ya hay LONGITUDE/LATITUDE)."""
    df = leer_rda(ruta).drop(columns="geometry").reset_index(drop=True)
    df["ASSETID"] = df["ASSETID"].astype(str)
    # PERIOD es YYYYMM con el mes final del trimestre (03, 06, 09, 12)
    df["TRIMESTRE"] = (df["PERIOD"] % 100 // 3).astype("Int8")
    return df


def cargar_barrios(ruta):
    """Zonas de idealista con su geometría como objetos shapely."""
    from shapely.geometry import MultiPolygon, Polygon

    df = leer_rda(ruta).reset_index(drop=True)
    for col in ["LOCATIONID", "LOCATIONNAME"]:
        df[col] = df[col].astype(str)

    # Cada geometría es un MULTIPOLYGON de sf: lista de polígonos, y cada
    # polígono una lista de anillos (exterior + huecos) como arrays (n, 2).
    df["geometry"] = [
        MultiPolygon([Polygon(p[0], p[1:]) for p in mp]) for mp in df["geometry"]
    ]
    df["COD_DISTRITO"] = df["LOCATIONID"].str.split("-").str[7]
    df["DISTRITO"] = df["COD_DISTRITO"].map(DISTRITOS)
    return df


def cargar_pois(ruta):
    """Puntos de interés en un único DataFrame con columna TIPO."""
    pois = leer_rda(ruta)
    return pd.concat(
        [d.assign(TIPO=str(tipo)) for tipo, d in pois.items()], ignore_index=True
    )[["TIPO", "Lon", "Lat"]]


# Un anuncio que no cae dentro de ninguna zona de idealista se asigna a la zona más cercana
# si está a menos de esta distancia; si no, queda sin zona.
MARGEN_ZONA_M = 500


def a_metros(geometrias, lat_referencia):
    """Proyección local aproximada de longitud/latitud a metros (válida a escala de ciudad)."""
    import shapely

    escala = np.array([111_320 * np.cos(np.radians(lat_referencia)), 110_540])
    return shapely.transform(geometrias, lambda c: c * escala)


def asignar_barrio(ventas, barrios):
    """Añade a cada anuncio la zona y el distrito en el que se encuentra.

    Primero busca la zona que contiene el anuncio. Si no hay ninguna, lo asigna a la zona
    más cercana cuando está a menos de MARGEN_ZONA_M metros. ASIGNACION_ZONA indica cómo se
    asignó ("dentro" o "zona más cercana") y DISTANCIA_ZONA_M la distancia a la zona asignada
    (0 si está dentro o justo en su borde). Ambas quedan vacías si el anuncio no tiene zona.
    """
    import shapely

    puntos = shapely.points(ventas["LONGITUDE"].to_numpy(), ventas["LATITUDE"].to_numpy())
    zonas = barrios["geometry"].to_numpy()
    idx_punto, idx_barrio = shapely.STRtree(zonas).query(puntos, predicate="within")

    # Un punto en el borde exacto entre dos zonas podría aparecer dos veces
    asignacion = (
        pd.DataFrame({"punto": idx_punto, "barrio": idx_barrio, "distancia": 0.0, "metodo": "dentro"})
        .drop_duplicates("punto")
        .set_index("punto")
    )

    # Anuncios fuera de toda zona: zona más cercana, si está a menos de MARGEN_ZONA_M
    sin_zona = np.setdiff1d(np.arange(len(ventas)), asignacion.index)
    if len(sin_zona):
        lat = float(ventas["LATITUDE"].median())
        zonas_m = a_metros(zonas, lat)
        puntos_m = a_metros(puntos[sin_zona], lat)
        (i_punto, i_barrio), distancia = shapely.STRtree(zonas_m).query_nearest(
            puntos_m, max_distance=MARGEN_ZONA_M, return_distance=True, all_matches=False)
        cercanos = pd.DataFrame({"punto": sin_zona[i_punto], "barrio": i_barrio, "distancia": distancia,
                                 "metodo": "zona más cercana"})
        asignacion = pd.concat([asignacion, cercanos.set_index("punto")])

    cols = ["LOCATIONID", "LOCATIONNAME", "COD_DISTRITO", "DISTRITO"]
    info = barrios[cols].iloc[asignacion["barrio"].to_numpy()].set_index(asignacion.index)
    info["ASIGNACION_ZONA"] = asignacion["metodo"]
    info["DISTANCIA_ZONA_M"] = asignacion["distancia"].round(1)
    return ventas.join(info)


def obtener_dataset(con_barrios=True, forzar_descarga=False):
    """Descarga (si hace falta) y devuelve (ventas, barrios, pois)."""
    rutas = {n: descargar(n, forzar=forzar_descarga) for n in FICHEROS}

    ventas = cargar_ventas(rutas["Madrid_Sale"])
    pois = cargar_pois(rutas["Madrid_POIS"])
    barrios = None
    if con_barrios:
        barrios = cargar_barrios(rutas["Madrid_Polygons"])
        ventas = asignar_barrio(ventas, barrios)
    return ventas, barrios, pois


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--forzar-descarga", action="store_true",
                        help="volver a descargar los .rda aunque existan")
    parser.add_argument("--sin-barrios", action="store_true",
                        help="no asignar barrio/distrito (no requiere shapely)")
    args = parser.parse_args()

    ventas, barrios, pois = obtener_dataset(
        con_barrios=not args.sin_barrios, forzar_descarga=args.forzar_descarga
    )

    DIR_IDEALISTA.mkdir(parents=True, exist_ok=True)
    ventas.to_csv(VENTAS_CSV, index=False)
    pois.to_csv(DIR_IDEALISTA / "madrid_pois.csv", index=False)
    if barrios is not None:
        barrios.assign(geometry=barrios["geometry"].map(lambda g: g.wkt)).to_csv(
            DIR_IDEALISTA / "madrid_barrios.csv", index=False
        )

    print(f"\nAnuncios: {len(ventas):,} filas x {ventas.shape[1]} columnas")
    print(f"Inmuebles únicos (ASSETID): {ventas['ASSETID'].nunique():,}")
    print("Anuncios por periodo:")
    print(ventas["PERIOD"].value_counts().sort_index().to_string())
    if barrios is not None:
        sin_barrio = ventas["LOCATIONID"].isna().sum()
        cercanos = (ventas["ASIGNACION_ZONA"] == "zona más cercana").sum()
        print(f"Anuncios asignados a la zona más cercana (a menos de {MARGEN_ZONA_M} m): {cercanos:,}")
        print(f"Anuncios sin barrio asignado: {sin_barrio:,}")
    print(f"\nFicheros guardados en {DIR_IDEALISTA}")


if __name__ == "__main__":
    main()
