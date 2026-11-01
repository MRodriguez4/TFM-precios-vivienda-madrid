"""
Validación de la calidad de los datos integrados, antes de cualquier depuración.

Parte de todos los anuncios unidos con los indicadores de su distrito, sin
eliminar nada, y aplica una serie de validaciones. Cada validación corresponde
a una dimensión de calidad, se aplica igual a todas las variables y guarda una
tabla con sus resultados, sin sacar conclusiones. Las decisiones de depuración
se toman después, a partir de estas tablas, y se aplican en construir_conjunto.py.

    1. Estructura        tipo y número de valores de cada variable
    2. Completitud       valores ausentes por variable
    3. Unicidad          filas, inmuebles y anuncios repetidos
    4. Validez           valores fuera del dominio documentado de cada variable
    5. Variabilidad      valores distintos y peso del valor más frecuente
    6. Distribución      estadísticos y valores atípicos (criterio del rango intercuartílico)
    7. Consistencia      relaciones entre variables que se deducen de su definición
    8. Integración       cruce con los distritos y extracción de los indicadores del panel
    9. Temporalidad      estabilidad, entre ediciones del panel, de los indicadores que no se
                         refieren a 2018 y se usan como aproximación
   10. Redundancia       información repetida entre variables, sin usar el precio, en dos partes:
                         10a. indicadores de distrito: rango, correlaciones y, por indicador, los
                              criterios de la regla de selección
                         10b. variables de la vivienda y la localización (D0 y D1): combinaciones
                              exactas, multicolinealidad (VIF) y correlaciones

Las variables de los anuncios se analizan sobre los anuncios y las de distrito
sobre los 21 distritos, que son su unidad de origen.

Requiere haber ejecutado antes idealista18_madrid.py y distritos_2018.py.

Salida (en datos/procesados/calidad/):
    - NN_<validacion>.csv: una tabla por validación.
    - perfil_variables.csv: todos los resultados por variable en una sola tabla.
    - calidad_datos.xlsx: las mismas tablas, una hoja por validación.

Uso:
    python calidad_datos.py
"""

import warnings

import numpy as np
import pandas as pd

from construir_conjunto import (CARACTERISTICAS, EXCLUIDAS_POR_REDUNDANCIA, LOCALIZACION, criterios_indicadores,
                                integrar)
from idealista18_madrid import MARGEN_ZONA_M
from distritos_2018 import HOJAS, INDICADORES, PANELES, bloque, buscar_fila, extraer, leer_hoja
from rutas import DIR_CALIDAD, DIR_DISTRITOS, DIR_IDEALISTA

COL_CIUDAD = 2  # en las hojas del panel, valor absoluto de la ciudad
AÑO_DATOS = 2018
# Año mínimo de construcción admitido: los autores de idealista18 consideran erróneos los
# años anteriores a 1500 (Rey-Blanco et al., 2024, apartado sobre CADCONSTRUCTIONYEAR)
AÑO_MINIMO = 1500

# Dominio de las variables de los anuncios según la documentación de idealista18
# (https://paezha.github.io/idealista18/reference/Madrid_Sale.html)
DUMMIES = ["HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "HASPARKINGSPACE",
           "ISPARKINGSPACEINCLUDEDINPRICE", "HASNORTHORIENTATION", "HASSOUTHORIENTATION",
           "HASEASTORIENTATION", "HASWESTORIENTATION", "HASBOXROOM", "HASWARDROBE",
           "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN", "ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR",
           "BUILTTYPEID_1", "BUILTTYPEID_2", "BUILTTYPEID_3"]
# Variables cuyo dominio documentado es una lista cerrada de códigos (no cantidades)
DOMINIOS_CODIGOS = {
    "AMENITYID": ("1, 2 o 3", [1, 2, 3]),
    "FLATLOCATIONID": ("1 (exterior) o 2 (interior)", [1, 2]),
    "PERIOD": ("201803, 201806, 201809 o 201812", [201803, 201806, 201809, 201812]),
    "TRIMESTRE": ("1 a 4", [1, 2, 3, 4]),
}
DOMINIO_ANUNCIOS = {
    **{col: ("0 o 1 (variable dummy)", lambda s: s.isin([0, 1])) for col in DUMMIES},
    **{col: (desc, lambda s, codigos=codigos: s.isin(codigos)) for col, (desc, codigos) in DOMINIOS_CODIGOS.items()},
    "PRICE": ("mayor que 0", lambda s: s > 0),
    "UNITPRICE": ("mayor que 0", lambda s: s > 0),
    "CONSTRUCTEDAREA": ("mayor que 0", lambda s: s > 0),
    "ROOMNUMBER": ("entero mayor o igual que 0", lambda s: (s >= 0) & (s == s.round())),
    "BATHNUMBER": ("entero mayor o igual que 0", lambda s: (s >= 0) & (s == s.round())),
    "FLOORCLEAN": ("entero mayor o igual que 0 (0 = planta baja)", lambda s: (s >= 0) & (s == s.round())),
    "PARKINGSPACEPRICE": ("mayor o igual que 0", lambda s: s >= 0),
    "CADMAXBUILDINGFLOOR": ("entero mayor o igual que 0", lambda s: (s >= 0) & (s == s.round())),
    "CADDWELLINGCOUNT": ("entero mayor o igual que 0", lambda s: (s >= 0) & (s == s.round())),
    "CONSTRUCTIONYEAR": (f"año entre {AÑO_MINIMO} y {AÑO_DATOS}", lambda s: s.between(AÑO_MINIMO, AÑO_DATOS)),
    "CADCONSTRUCTIONYEAR": (f"año entre {AÑO_MINIMO} y {AÑO_DATOS}", lambda s: s.between(AÑO_MINIMO, AÑO_DATOS)),
    "DISTANCE_TO_CITY_CENTER": ("mayor o igual que 0", lambda s: s >= 0),
    "DISTANCE_TO_METRO": ("mayor o igual que 0", lambda s: s >= 0),
    "DISTANCE_TO_CASTELLANA": ("mayor o igual que 0", lambda s: s >= 0),
}

# Relaciones entre variables que se deducen de su definición:
# (nombre, descripción, variables que necesita, condición que se cumple).
# Las distancias calculadas (_distancia_*) se obtienen en v7_consistencia a partir de las
# coordenadas del anuncio y de los puntos de interés de idealista18.
TOLERANCIA_DISTANCIA = 0.01


def coincide(d, publicada, calculada):
    return (d[publicada] - d[calculada]).abs() <= TOLERANCIA_DISTANCIA * d[calculada]


CONSISTENCIA = [
    ("precio_unitario", "UNITPRICE = PRICE / CONSTRUCTEDAREA (tolerancia 1 %)",
     ["UNITPRICE", "PRICE", "CONSTRUCTEDAREA"],
     lambda d: (d.UNITPRICE - d.PRICE / d.CONSTRUCTEDAREA).abs() <= 0.01 * d.UNITPRICE),
    ("estudio_dormitorios", "ISSTUDIO = 1 implica ROOMNUMBER <= 1",
     ["ISSTUDIO", "ROOMNUMBER"],
     lambda d: (d.ISSTUDIO == 0) | (d.ROOMNUMBER <= 1)),
    ("planta_edificio", "FLOORCLEAN <= CADMAXBUILDINGFLOOR",
     ["FLOORCLEAN", "CADMAXBUILDINGFLOOR"],
     lambda d: d.FLOORCLEAN <= d.CADMAXBUILDINGFLOOR),
    ("ultima_planta", "ISINTOPFLOOR = 1 implica FLOORCLEAN = CADMAXBUILDINGFLOOR",
     ["ISINTOPFLOOR", "FLOORCLEAN", "CADMAXBUILDINGFLOOR"],
     lambda d: (d.ISINTOPFLOOR == 0) | (d.FLOORCLEAN == d.CADMAXBUILDINGFLOOR)),
    ("distancia_centro", "DISTANCE_TO_CITY_CENTER = distancia desde las coordenadas al centro (tolerancia 1 %)",
     ["DISTANCE_TO_CITY_CENTER", "LONGITUDE", "LATITUDE"],
     lambda d: coincide(d, "DISTANCE_TO_CITY_CENTER", "_distancia_centro")),
    ("distancia_metro", "DISTANCE_TO_METRO = distancia desde las coordenadas a la estación de metro más "
     "cercana (tolerancia 1 %)",
     ["DISTANCE_TO_METRO", "LONGITUDE", "LATITUDE"],
     lambda d: coincide(d, "DISTANCE_TO_METRO", "_distancia_metro")),
    ("distancia_castellana", "DISTANCE_TO_CASTELLANA = distancia desde las coordenadas al punto más cercano "
     "del eje de la Castellana (tolerancia 1 %)",
     ["DISTANCE_TO_CASTELLANA", "LONGITUDE", "LATITUDE"],
     lambda d: coincide(d, "DISTANCE_TO_CASTELLANA", "_distancia_castellana")),
]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def tipo_variable(col, s):
    """Tipo según la definición de la variable, no según los valores observados.

    Binarias son solo las variables que la documentación de idealista18 define como dummy;
    un recuento que en los datos solo toma 0 y 1 sigue siendo numérico.
    """
    if not pd.api.types.is_numeric_dtype(s):
        return "texto"
    return "binaria" if col in DUMMIES else "numérica"


# --- 1. Estructura --------------------------------------------------------------------------

def v1_estructura(anuncios, distritos, cols_distrito):
    filas = []
    for col in anuncios.columns:
        s = distritos[col] if col in cols_distrito else anuncios[col]
        filas.append({"variable": col, "origen": "distrito" if col in cols_distrito else "anuncio",
                      "tipo_pandas": str(s.dtype), "tipo": tipo_variable(col, s), "valores": len(s)})
    return pd.DataFrame(filas)


# --- 2. Completitud -------------------------------------------------------------------------

def v2_completitud(anuncios, distritos, cols_distrito):
    filas = []
    for col in anuncios.columns:
        s = distritos[col] if col in cols_distrito else anuncios[col]
        filas.append({"variable": col, "ausentes": int(s.isna().sum()),
                      "pct_ausentes": round(100 * s.isna().mean(), 3)})
    return pd.DataFrame(filas)


# --- 3. Unicidad ----------------------------------------------------------------------------

def v3_unicidad(anuncios):
    por_inmueble = anuncios.groupby("ASSETID")
    apariciones = por_inmueble.size()
    repetidos = anuncios[anuncios.ASSETID.duplicated(keep=False)]
    cambian = repetidos.groupby("ASSETID").agg(
        precio=("PRICE", "nunique"), superficie=("CONSTRUCTEDAREA", "nunique"),
        dormitorios=("ROOMNUMBER", "nunique"), coordenadas=("LONGITUDE", "nunique"))
    trimestres = por_inmueble.PERIOD.nunique()
    precios = repetidos.groupby("ASSETID").PRICE
    ratio_precio = precios.max() / precios.min()
    # Separación máxima entre cualquier pareja de filas de cada inmueble repetido
    posiciones = repetidos[["ASSETID", "LONGITUDE", "LATITUDE"]]
    parejas = posiciones.merge(posiciones, on="ASSETID", suffixes=("_1", "_2"))
    distancia = haversine_km(parejas.LONGITUDE_1, parejas.LATITUDE_1, parejas.LONGITUDE_2, parejas.LATITUDE_2)
    separacion = 1000 * distancia.groupby(parejas.ASSETID).max()
    filas = [
        ("Filas", len(anuncios)),
        ("Filas idénticas en todas las columnas", int(anuncios.duplicated().sum())),
        ("Inmuebles distintos (ASSETID)", anuncios.ASSETID.nunique()),
        ("Inmuebles repetidos (aparecen en más de una fila)", int((apariciones > 1).sum())),
        ("Inmuebles sobrantes (filas que habría que quitar para dejar una por inmueble; "
         "son más que los repetidos porque algunos se repiten más de una vez)",
         int(anuncios.ASSETID.duplicated().sum())),
        *[(f"Inmuebles que aparecen en {k} trimestre(s) distinto(s)", int((trimestres == k).sum()))
          for k in sorted(trimestres.unique())],
        ("Inmuebles repetidos cuyo precio cambia entre apariciones", int((cambian.precio > 1).sum())),
        ("Inmuebles repetidos cuya superficie cambia entre apariciones", int((cambian.superficie > 1).sum())),
        ("Inmuebles repetidos cuyos dormitorios cambian entre apariciones", int((cambian.dormitorios > 1).sum())),
        ("Inmuebles repetidos cuyas coordenadas cambian entre apariciones", int((cambian.coordenadas > 1).sum())),
        # La anonimización de idealista18 cambia el precio un ±2,5 % y desplaza las coordenadas
        # entre 30 y 60 m en cada fila: dos copias del mismo anuncio diferirían como máximo
        # un 5,1 % en precio y unos 120 m en posición
        *[(f"Inmuebles repetidos: cociente precio máximo / mínimo, percentil {q}",
           round(float(ratio_precio.quantile(q / 100)), 4)) for q in (50, 75, 90, 95, 99)],
        ("Inmuebles repetidos con cociente de precios mayor que 1,0513 (fuera del margen del ruido)",
         int((ratio_precio > 1.0513).sum())),
        *[(f"Inmuebles repetidos: separación máxima entre dos de sus filas en metros, percentil {q}",
           round(float(separacion.quantile(q / 100)), 1)) for q in (50, 75, 90, 95, 99)],
        ("Inmuebles repetidos con filas separadas más de 120 m (fuera del margen del ruido)",
         int((separacion > 120).sum())),
    ]
    return pd.DataFrame(filas, columns=["comprobación", "resultado"], dtype=object)


# --- 4. Validez -----------------------------------------------------------------------------

def v4_validez(anuncios, distritos, cols_distrito):
    filas = []
    for col in anuncios.columns:
        if col in cols_distrito:
            s = distritos[col].dropna()
            if col.startswith(("pct_", "prop_", "tasa_paro")):
                dominio, ok = "proporción entre 0 y 1", s.between(0, 1)
            elif col == "tasa_crecimiento_demografico":
                dominio, ok = "cualquier número (puede ser negativo)", pd.Series(True, index=s.index)
            else:
                dominio, ok = "mayor o igual que 0", s >= 0
        elif col in DOMINIO_ANUNCIOS:
            s = anuncios[col].dropna()
            dominio, cond = DOMINIO_ANUNCIOS[col]
            ok = cond(s)
        else:
            filas.append({"variable": col, "dominio": "sin dominio documentado"})
            continue
        fuera = s[~ok]
        filas.append({"variable": col, "dominio": dominio, "valores_comprobados": len(s),
                      "fuera_de_dominio": len(fuera), "pct_fuera": round(100 * len(fuera) / max(len(s), 1), 3),
                      "ejemplos_fuera": ", ".join(map(str, fuera.value_counts().index[:5]))})
    return pd.DataFrame(filas)


# --- 5. Variabilidad ------------------------------------------------------------------------

def v5_variabilidad(anuncios, distritos, cols_distrito):
    filas = []
    for col in anuncios.columns:
        s = (distritos[col] if col in cols_distrito else anuncios[col]).dropna()
        frec = s.value_counts()
        filas.append({"variable": col, "valores_distintos": s.nunique(),
                      "valor_mas_frecuente": frec.index[0] if len(frec) else None,
                      "pct_valor_mas_frecuente": round(100 * frec.iloc[0] / len(s), 3) if len(frec) else None})
    return pd.DataFrame(filas)


# --- 6. Distribución ------------------------------------------------------------------------

def v6_distribucion(anuncios, distritos, cols_distrito):
    filas = []
    for col in anuncios.columns:
        s = (distritos[col] if col in cols_distrito else anuncios[col]).dropna()
        if tipo_variable(col, s) != "numérica":
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        riq = q3 - q1
        # El rango intercuartílico solo tiene sentido en cantidades, no en códigos de categorías
        codigos = col in DOMINIOS_CODIGOS
        filas.append({
            "variable": col, "n": len(s), "media": s.mean(), "desviacion": s.std(), "asimetria": s.skew(),
            "minimo": s.min(), "p1": s.quantile(0.01), "p5": s.quantile(0.05), "q1": q1, "mediana": s.median(),
            "q3": q3, "p95": s.quantile(0.95), "p99": s.quantile(0.99), "maximo": s.max(),
            "ceros": int((s == 0).sum()), "negativos": int((s < 0).sum()),
            "criterio_atipicos": ("no aplica: dominio documentado de códigos" if codigos
                                  else "rango intercuartílico"),
            "atipicos_1_5_riq": None if codigos else int(((s < q1 - 1.5 * riq) | (s > q3 + 1.5 * riq)).sum()),
            "atipicos_3_riq": None if codigos else int(((s < q1 - 3 * riq) | (s > q3 + 3 * riq)).sum()),
        })
    return pd.DataFrame(filas)


# --- 7. Consistencia ------------------------------------------------------------------------

def distancia_al_mas_cercano(lon, lat, puntos):
    """Distancia en km desde cada coordenada al punto más cercano de `puntos` (columnas Lon, Lat)."""
    minimo = np.full(len(lon), np.inf)
    for p in puntos.itertuples():
        minimo = np.minimum(minimo, haversine_km(lon, lat, p.Lon, p.Lat))
    return minimo


def v7_consistencia(anuncios):
    pois = pd.read_csv(DIR_IDEALISTA / "madrid_pois.csv")
    lon, lat = anuncios.LONGITUDE.to_numpy(), anuncios.LATITUDE.to_numpy()
    d = anuncios.assign(
        _distancia_centro=distancia_al_mas_cercano(lon, lat, pois[pois.TIPO == "City_Center"]),
        _distancia_metro=distancia_al_mas_cercano(lon, lat, pois[pois.TIPO == "Metro"]),
        _distancia_castellana=distancia_al_mas_cercano(lon, lat, pois[pois.TIPO == "Castellana"]),
    )
    filas = []
    for nombre, desc, variables, cond in CONSISTENCIA:
        evaluables = d[variables].notna().all(axis=1)
        incumple = evaluables & ~cond(d)
        filas.append({"comprobación": nombre, "relación": desc, "anuncios_evaluables": int(evaluables.sum()),
                      "incumplimientos": int(incumple.sum()),
                      "pct_incumplimientos": round(100 * incumple.sum() / max(evaluables.sum(), 1), 3)})
    return pd.DataFrame(filas)


# --- 8. Integración -------------------------------------------------------------------------

def v8_integracion(anuncios, distritos):
    sin_distrito = anuncios[anuncios.COD_DISTRITO.isna()]
    cercanos = anuncios[anuncios.ASIGNACION_ZONA == "zona más cercana"]
    zonas = pd.read_csv(DIR_IDEALISTA / "madrid_barrios.csv", dtype={"COD_DISTRITO": str})
    pois = pd.read_csv(DIR_IDEALISTA / "madrid_pois.csv")
    filas = [
        ("Anuncios", len(anuncios)),
        ("Anuncios dentro de una zona", int((anuncios.ASIGNACION_ZONA == "dentro").sum())),
        ("Anuncios fuera de toda zona", int((anuncios.ASIGNACION_ZONA != "dentro").sum())),
        (f"Anuncios fuera de toda zona, asignados a la más cercana (a menos de {MARGEN_ZONA_M} m)",
         int((anuncios.ASIGNACION_ZONA == "zona más cercana").sum())),
        ("Distancia a su zona de los asignados a la más cercana, en m (mediana / máximo)",
         f"{cercanos.DISTANCIA_ZONA_M.median():.1f} / {cercanos.DISTANCIA_ZONA_M.max():.1f}"),
        ("Anuncios sin zona ni distrito asignado", len(sin_distrito)),
        ("Distancia al centro de los anuncios sin distrito, en km (mínimo / máximo)",
         f"{sin_distrito.DISTANCE_TO_CITY_CENTER.min():.1f} / {sin_distrito.DISTANCE_TO_CITY_CENTER.max():.1f}"),
        ("Distancia al centro de los anuncios con distrito, en km (máximo)",
         f"{anuncios.DISTANCE_TO_CITY_CENTER[anuncios.COD_DISTRITO.notna()].max():.1f}"),
        ("Distritos con al menos un anuncio", anuncios.COD_DISTRITO.nunique()),
        ("Distritos en la tabla de indicadores", len(distritos)),
        ("Zonas de idealista", len(zonas)),
        ("Distritos que contienen alguna zona", zonas.COD_DISTRITO.nunique()),
        *[(f"Puntos de interés: {tipo}", n) for tipo, n in pois.TIPO.value_counts().items()],
    ]
    resumen = pd.DataFrame(filas, columns=["comprobación", "resultado"], dtype=object)
    por_distrito = (anuncios.groupby("COD_DISTRITO").size().rename("anuncios").reset_index()
                    .merge(distritos[["COD_DISTRITO", "DISTRITO"]], on="COD_DISTRITO", how="outer"))
    return resumen, por_distrito


# Indicadores que no son aditivos: su suma entre distritos no tiene sentido (porcentajes,
# proporciones, tasas, medias, índices, rankings y valores por habitante). Los demás son
# recuentos o superficies, cuya suma en los 21 distritos debe igualar el total de la ciudad.
PREFIJOS_NO_ADITIVOS = ("pct_", "prop_", "tasa_", "edad_", "esperanza_", "pension_", "renta_", "indice_",
                        "ranking_", "valor_catastral_", "densidad_", "tamano_", "residuos_")


def es_aditivo(col):
    return not col.startswith(PREFIJOS_NO_ADITIVOS) and "hab" not in col


def v8_panel(distritos):
    """Suma de cada indicador en los 21 distritos frente al valor de la ciudad en el panel.

    La comparación solo es válida en los indicadores aditivos (recuentos y superficies);
    el resumen cuenta las coincidencias únicamente entre ellos.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centro = {panel: leer_hoja(panel, "CENTRO") for panel in PANELES}
    filas = []
    for col, panel, etiqueta, tipo, despues, fecha, _ in INDICADORES:
        hoja = centro[panel]
        fila = buscar_fila(hoja, etiqueta, despues)
        ciudad_abs = pd.to_numeric(hoja.iat[fila, COL_CIUDAD], errors="coerce")
        ciudad_pct = pd.to_numeric(hoja.iat[fila, COL_CIUDAD - 1], errors="coerce")
        s = distritos[col]
        aditivo = es_aditivo(col)
        filas.append({"indicador": col, "panel": panel, "fecha": fecha, "aditivo": aditivo,
                      "grupo": ("Equipamientos y zonas verdes" if despues and str(despues).startswith("1.14")
                                else "Resto de indicadores"),
                      "suma_distritos": s.sum(), "ciudad_valor": ciudad_abs, "ciudad_proporcion": ciudad_pct,
                      "suma_entre_ciudad": (s.sum() / ciudad_abs if aditivo and ciudad_abs else np.nan),
                      "coincide": (bool(abs(s.sum() - ciudad_abs) <= 1e-6 * abs(ciudad_abs)) if aditivo else None)})
    detalle = pd.DataFrame(filas)
    aditivos = detalle[detalle.aditivo]
    resumen = (aditivos.groupby("grupo")
               .agg(indicadores_aditivos=("indicador", "count"),
                    suma_igual_al_total_de_la_ciudad=("coincide", "sum"))
               .reset_index())
    resumen.loc[len(resumen)] = ["Total", len(aditivos), int(aditivos.coincide.sum())]
    return detalle, resumen


def v8_proporciones(distritos):
    """Suma por distrito de las proporciones que reparten la población en categorías."""
    grupos = {
        "edades": ["pct_0_14", "pct_15_29", "pct_30_44", "pct_45_64", "pct_65_79", "pct_80_mas"],
        "nivel de estudios": ["pct_sin_estudios", "pct_primaria_incompleta", "pct_estudios_obligatorios",
                              "pct_bachiller_fp2", "pct_titulados_medios", "pct_estudios_superiores",
                              "pct_estudios_desconocido"],
        "tipo de centro educativo": ["pct_alumnos_publica", "pct_alumnos_concertada", "pct_alumnos_privada"],
    }
    filas = [{"grupo": g, "suma_minima": distritos[c].sum(axis=1).min(), "suma_maxima": distritos[c].sum(axis=1).max()}
             for g, c in grupos.items()]
    return pd.DataFrame(filas)


# --- 9. Temporalidad ------------------------------------------------------------------------

# Fechas que se refieren a 2018: los indicadores con cualquier otra fecha son aproximaciones
FECHAS_2018 = {None, "2018", "01/01/2018", "01-01-2018", "31/12/2018"}


def grupo_temporal(col, panel, despues):
    if panel == 2019 and despues == "1.14.":
        return "Equipamientos municipales (inventario de 2019)"
    if col.startswith(("zona", "parques")):
        return "Zonas verdes"
    if col.startswith("pct_alumnos_"):
        return "Alumnado por tipo de centro (curso 2017/18)"
    return "Otros"


def v9_temporalidad(distritos):
    """Estabilidad de los indicadores que no se refieren a 2018.

    Para cada indicador usado como aproximación de 2018 (otra fecha), compara su valor
    en cada distrito con el de la edición siguiente del panel, buscando la misma etiqueta.
    Si apenas cambia, la aproximación es razonable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hojas = {(p, cod): leer_hoja(p, hoja) for p in PANELES for cod, hoja in HOJAS.items()}
    filas = []
    for col, panel, etiqueta, tipo, despues, fecha, _ in INDICADORES:
        if fecha in FECHAS_2018:
            continue
        siguiente = panel + 1
        fila = {"indicador": col, "grupo": grupo_temporal(col, panel, despues), "fecha": fecha,
                "panel": panel, "panel_siguiente": siguiente}
        if siguiente not in PANELES:
            filas.append({**fila, "resultado": "no hay edición siguiente"})
            continue
        try:
            nuevos = pd.Series({cod: extraer(hojas[(siguiente, cod)],
                                             buscar_fila(hojas[(siguiente, cod)], etiqueta, despues), tipo)
                                for cod in HOJAS})
        except KeyError:
            filas.append({**fila, "resultado": "la etiqueta no aparece en la edición siguiente"})
            continue
        actuales = distritos.set_index("COD_DISTRITO")[col]
        cambio = (nuevos - actuales).abs()
        filas.append({**fila, "resultado": "comparado",
                      "distritos_que_cambian": int((cambio > 1e-9).sum()),
                      "cambio_relativo_medio_pct": round(100 * (cambio / actuales.abs().replace(0, np.nan)).mean(), 3),
                      "correlacion": round(actuales.corr(nuevos), 4)})
    detalle = pd.DataFrame(filas)
    comparados = detalle[detalle.resultado == "comparado"]
    resumen = (comparados.groupby("grupo")
               .agg(indicadores_comparados=("indicador", "count"),
                    sin_cambios_en_ningun_distrito=("distritos_que_cambian", lambda d: int((d == 0).sum())))
               .reset_index())
    return detalle, resumen


# --- 10. Redundancia -----------------------------------------------------------------------
# 10a: indicadores de distrito, sobre los 21 distritos.
# 10b: variables de la vivienda y la localización, sobre los anuncios.

def v10a_redundancia_distrito(distritos, cols_distrito):
    """Información independiente y pares muy correlacionados entre los indicadores de distrito."""
    t = distritos[sorted(cols_distrito)]
    z = (t - t.mean()) / t.std(ddof=0)
    valores = np.linalg.svd(z.to_numpy(), compute_uv=False)
    varianza = valores ** 2 / np.sum(valores ** 2)
    corr = t.corr()
    i, j = np.triu_indices_from(corr, 1)
    pares = pd.DataFrame({"indicador_1": corr.columns[i], "indicador_2": corr.columns[j],
                          "correlacion": corr.to_numpy()[i, j]})
    resumen = pd.DataFrame([
        ("Distritos x indicadores", f"{t.shape[0]} x {t.shape[1]}"),
        ("Rango de la matriz de indicadores estandarizada", int(np.linalg.matrix_rank(z.to_numpy()))),
        *[(f"Varianza explicada por el componente principal {k + 1}", round(float(varianza[k]), 4)) for k in range(5)],
        ("Varianza explicada por los 3 primeros componentes", round(float(varianza[:3].sum()), 4)),
        ("Pares de indicadores", len(pares)),
        ("Pares con |correlación| > 0,9", int((pares.correlacion.abs() > 0.9).sum())),
    ], columns=["comprobación", "resultado"], dtype=object)
    pares = pares.reindex(pares.correlacion.abs().sort_values(ascending=False).index)
    return resumen, pares


def v10a_variables_distrito(distritos, cols_distrito):
    """Por cada indicador de distrito, los criterios que aplica la regla de selección.

    Valores distintos, correlación con la población y correlación máxima con otro indicador
    de su bloque y del otro bloque. Solo usa los propios indicadores en los 21 distritos; el
    precio de la vivienda no interviene.
    """
    dicc = pd.read_csv(DIR_DISTRITOS / "diccionario_distritos_2018.csv").set_index("columna")
    t = distritos[sorted(cols_distrito)]
    corr_abs = t.corr().abs().fillna(0)
    criterios = criterios_indicadores(distritos)
    filas = []
    for c in t.columns:
        mismo_bloque = [o for o in t.columns if o != c and bloque(o) == bloque(c)]
        otro_bloque = [o for o in t.columns if bloque(o) != bloque(c)]
        mismo, otro = corr_abs.loc[c, mismo_bloque], corr_abs.loc[c, otro_bloque]
        filas.append({
            "variable": c, "bloque": bloque(c), "dimension": dicc.loc[c, "dimension"],
            "descripcion": dicc.loc[c, "descripcion"], "fecha": dicc.loc[c, "fecha"],
            "valores_distintos": criterios.loc[c, "valores_distintos"], "distritos_con_cero": int((t[c] == 0).sum()),
            "r_poblacion": criterios.loc[c, "r_poblacion"],
            "max_abs_r_mismo_bloque": mismo.max(), "con_variable_mismo_bloque": mismo.idxmax(),
            "max_abs_r_otro_bloque": otro.max(), "con_variable_otro_bloque": otro.idxmax(),
        })
    return pd.DataFrame(filas).sort_values(["bloque", "dimension", "variable"])


def v10b_redundancia_anuncio(anuncios):
    """Redundancia entre las variables de D0 y D1, sobre los anuncios sin valores ausentes.

    No usa el precio. Mide el rango de la matriz de variables estandarizadas, las
    combinaciones lineales exactas (valores singulares casi nulos), el factor de
    inflación de la varianza (VIF) de cada variable y las correlaciones por parejas.
    """
    # Incluye las variables ya descartadas por redundancia, para que la evidencia siga visible
    variables = CARACTERISTICAS + list(EXCLUIDAS_POR_REDUNDANCIA) + LOCALIZACION
    completos = anuncios[variables].dropna().astype(float)
    z = (completos - completos.mean()) / completos.std(ddof=0)
    x = z.to_numpy()

    # Combinaciones lineales exactas: direcciones con valor singular casi nulo
    _, valores, vt = np.linalg.svd(x, full_matrices=False)
    tolerancia = valores.max() * max(x.shape) * np.finfo(float).eps
    dependencias = []
    for k in np.where(valores <= tolerancia)[0]:
        pesos = pd.Series(vt[k], index=variables)
        implicadas = pesos[pesos.abs() > 1e-6].sort_values(key=abs, ascending=False)
        dependencias.append({"combinacion": len(dependencias) + 1, "valor_singular": valores[k],
                             "variables": ", ".join(implicadas.index),
                             "pesos": ", ".join(f"{v:.3f}" for v in implicadas)})

    # VIF: 1 / (1 - R²) de cada variable explicada por las demás
    filas = []
    for j, col in enumerate(variables):
        otras = np.delete(x, j, axis=1)
        diseno = np.column_stack([np.ones(len(x)), otras])
        coef, *_ = np.linalg.lstsq(diseno, x[:, j], rcond=None)
        residuo = x[:, j] - diseno @ coef
        r2 = 1 - residuo.var() / x[:, j].var()
        filas.append({"variable": col, "escenario": "D1" if col in LOCALIZACION else "D0",
                      "r2_con_las_demas": round(float(r2), 6),
                      "vif": float("inf") if r2 >= 1 - 1e-10 else round(float(1 / (1 - r2)), 3)})
    vif = pd.DataFrame(filas)

    corr = completos.corr()
    i, j = np.triu_indices_from(corr, 1)
    pares = pd.DataFrame({"variable_1": corr.columns[i], "variable_2": corr.columns[j],
                          "correlacion": corr.to_numpy()[i, j]})
    # Columnas idénticas en todos los anuncios (incluidos los que tienen ausentes)
    todos = anuncios[variables]
    pares["identicas_en_todos_los_anuncios"] = [
        bool(todos[a].equals(todos[b])) for a, b in zip(pares.variable_1, pares.variable_2)]
    pares = pares.reindex(pares.correlacion.abs().sort_values(ascending=False).index)

    resumen = pd.DataFrame([
        ("Variables de D0 y D1", len(variables)),
        ("Anuncios sin valores ausentes en esas variables", len(completos)),
        ("Rango de la matriz de variables estandarizadas", int(np.linalg.matrix_rank(x))),
        ("Combinaciones lineales exactas", len(dependencias)),
        ("Variables con VIF infinito (forman parte de una combinación exacta)", int(np.isinf(vif.vif).sum())),
        ("Variables con VIF mayor que 10", int((vif.vif > 10).sum())),
        ("Variables con VIF mayor que 5", int((vif.vif > 5).sum())),
        ("Pares de variables", len(pares)),
        ("Pares de variables idénticas en todos los anuncios", int(pares.identicas_en_todos_los_anuncios.sum())),
        ("Pares con |correlación| > 0,9", int((pares.correlacion.abs() > 0.9).sum())),
        ("Pares con |correlación| > 0,8", int((pares.correlacion.abs() > 0.8).sum())),
        ("Pares con |correlación| > 0,7", int((pares.correlacion.abs() > 0.7).sum())),
    ], columns=["comprobación", "resultado"], dtype=object)
    return resumen, pd.DataFrame(dependencias), vif.sort_values("vif", ascending=False), pares


def main():
    anuncios, distritos = integrar()
    cols_distrito = set(distritos.columns) - {"COD_DISTRITO", "DISTRITO"}

    print("Calculando validaciones...")
    resultados = {
        "01_estructura": v1_estructura(anuncios, distritos, cols_distrito),
        "02_completitud": v2_completitud(anuncios, distritos, cols_distrito),
        "03_unicidad": v3_unicidad(anuncios),
        "04_validez": v4_validez(anuncios, distritos, cols_distrito),
        "05_variabilidad": v5_variabilidad(anuncios, distritos, cols_distrito),
        "06_distribucion": v6_distribucion(anuncios, distritos, cols_distrito),
        "07_consistencia": v7_consistencia(anuncios),
    }
    resultados["08_integracion"], resultados["08_anuncios_por_distrito"] = v8_integracion(anuncios, distritos)
    resultados["08_panel_ciudad"], resultados["08_panel_ciudad_resumen"] = v8_panel(distritos)
    resultados["08_proporciones"] = v8_proporciones(distritos)
    resultados["09_temporalidad"], resultados["09_temporalidad_resumen"] = v9_temporalidad(distritos)
    resultados["10a_redundancia_distrito"], resultados["10a_correlaciones_distrito"] = \
        v10a_redundancia_distrito(distritos, cols_distrito)
    resultados["10a_variables_distrito"] = v10a_variables_distrito(distritos, cols_distrito)
    (resultados["10b_redundancia_anuncio"], resultados["10b_combinaciones_exactas"],
     resultados["10b_vif"], resultados["10b_correlaciones_anuncio"]) = v10b_redundancia_anuncio(anuncios)

    # Todos los resultados por variable en una sola tabla
    perfil = resultados["01_estructura"]
    for clave in ["02_completitud", "04_validez", "05_variabilidad", "06_distribucion"]:
        perfil = perfil.merge(resultados[clave], on="variable", how="left")

    DIR_CALIDAD.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in resultados.items():
        tabla.to_csv(DIR_CALIDAD / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    perfil.to_csv(DIR_CALIDAD / "perfil_variables.csv", index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(DIR_CALIDAD / "calidad_datos.xlsx") as excel:
            perfil.to_excel(excel, sheet_name="perfil_variables", index=False)
            for nombre, tabla in resultados.items():
                tabla.to_excel(excel, sheet_name=nombre[:31], index=False)
    except PermissionError:
        print("AVISO: calidad_datos.xlsx está abierto en otro programa; no se ha actualizado")

    for nombre, tabla in resultados.items():
        print(f"  {nombre}: {len(tabla)} filas")
    print(f"  perfil_variables: {len(perfil)} variables x {perfil.shape[1]} columnas")
    print(f"Ficheros guardados en {DIR_CALIDAD}")


if __name__ == "__main__":
    main()
