"""
Análisis exploratorio del conjunto de estudio.

Se ejecuta sobre el conjunto final (75.799 viviendas), una vez definidos los escenarios.
La selección de variables de construir_conjunto.py no utiliza el precio; este análisis sí lo
usa, pero es descriptivo y no modifica la composición de D0 a D3. Solo apoya decisiones de
modelización que afectan por igual a todos los escenarios, como la transformación de la
variable objetivo.

Análisis:
    1. Distribución del precio de oferta, en euros, en logaritmo y por m².
    2. Precio por distrito y por zona, y parte de la variación del precio que se da entre
       distritos y entre zonas.
    3. Relación del precio con las variables de la vivienda y la localización (D0 y D1).
    4. Relación de los 8 indicadores de contexto (D2 y D3) con el precio de su distrito.
    5. Dependencia espacial del precio: índice de Moran del precio por m² entre zonas vecinas.
    6. Tamaño de los pliegues de la validación espacial por distritos.
    7. Precio de las viviendas según conste o no su planta, su posición exterior o interior y
       su calidad catastral (justifica el indicador de dato ausente de la modelización).
    8. Atípicos del precio en euros y en logaritmo, las viviendas más baratas y más caras, y
       las viviendas con más habitaciones.

Requiere haber ejecutado antes construir_conjunto.py, que también genera los pliegues.

Salida (en datos/procesados/exploratorio/):
    - 01_precio.csv: estadísticos del precio, de su logaritmo y del precio por m².
    - 02_distritos.csv: viviendas y precio mediano de cada distrito.
    - 02_varianza_entre_grupos.csv: parte de la varianza del precio que se da entre distritos y entre zonas.
    - 03_correlaciones_vivienda.csv: correlación de Spearman de cada variable de D0 y D1 con el precio.
    - 04_contexto_distritos.csv: los 8 indicadores y el precio mediano por m² de cada distrito.
    - 04_correlaciones_contexto.csv: correlación de cada indicador con el precio del distrito (21 distritos).
    - 05_dependencia_espacial.csv: índice de Moran del precio por m² entre zonas vecinas.
    - 06_pliegues_espaciales.csv: distritos y viviendas de cada pliegue de la validación espacial.
    - 07_ausentes_precio.csv: viviendas sin dato de planta, posición o calidad catastral y su
      precio frente al resto.
    - 08_atipicos_precio.csv: atípicos del precio en euros y en logaritmo.
    - 08_precios_extremos.csv: las cinco viviendas más baratas y las cinco más caras.
    - 08_habitaciones_extremas.csv: las cinco viviendas con más habitaciones.
    - figuras/: las figuras del apartado.

Uso:
    python analisis_exploratorio.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as ParcheMatplotlib
from shapely import wkt

from idealista18_madrid import a_metros
from rutas import CONJUNTO_CSV, DIR_EXPLORATORIO, DIR_IDEALISTA, ESCENARIOS_JSON, PLIEGUES_CSV

DIR_FIGURAS = DIR_EXPLORATORIO / "figuras"
GRIS = "#7f7f7f"
SEMILLA = 2018
# Dos zonas son vecinas si sus límites están a menos de esta distancia: las zonas de idealista
# no cubren la ciudad sin huecos, así que exigir que se toquen dejaría vecinas sin conectar
DISTANCIA_VECINAS_M = 100
PERMUTACIONES_MORAN = 999
AZUL = "#1f5fa8"

# Nombre, unidad del eje y factor de escala de cada indicador en la figura de contexto
ETIQUETAS_CONTEXTO = {
    "renta_media_hogar": ("Renta media de los hogares", "€ al año", 1),
    "pct_estudios_superiores": ("Población con estudios superiores", "% de la población de 25 años o más", 100),
    "pct_extranjeros": ("Población extranjera", "% de la población", 100),
    "edad_media": ("Edad media", "años", 1),
    "densidad_hab_ha": ("Densidad de población", "habitantes por hectárea", 1),
    "zonas_verdes_parques_distrito_ha_10000hab_2019": ("Zonas verdes y parques de distrito",
                                                       "ha por 10.000 habitantes", 1),
    "detenidos_por_1000_hab": ("Detenidos", "por cada 1.000 habitantes", 1),
    "mercados_municipales_por_km2": ("Mercados municipales", "por km²", 1),
}


def numero_es(valor, decimales=0):
    """Número con punto de miles y coma decimal."""
    return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formato_eje_es(valor, _):
    """Marcas de eje en notación española, con los decimales justos."""
    decimales = next((d for d in range(5) if abs(valor * 10 ** d - round(valor * 10 ** d)) < 1e-9), 4)
    return numero_es(valor, decimales)


def cargar():
    """Conjunto final con el logaritmo del precio y el precio por m², y la definición de escenarios."""
    conjunto = pd.read_csv(CONJUNTO_CSV, dtype={"COD_DISTRITO": str, "ASSETID": str})
    escenarios = json.loads(ESCENARIOS_JSON.read_text(encoding="utf-8"))
    objetivo = escenarios["objetivo"]
    conjunto["LOG_PRECIO"] = np.log(conjunto[objetivo])
    conjunto["PRECIO_M2"] = conjunto[objetivo] / conjunto["CONSTRUCTEDAREA"]
    return conjunto, escenarios


def guardar_figura(fig, nombre):
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 1. Distribución del precio

def analizar_precio(conjunto, objetivo):
    columnas = {"precio (€)": conjunto[objetivo], "logaritmo del precio": conjunto["LOG_PRECIO"],
                "precio por m² (€)": conjunto["PRECIO_M2"]}
    filas = []
    for nombre, s in columnas.items():
        filas.append({"variable": nombre, "n": len(s), "media": s.mean(), "desviacion": s.std(),
                      "asimetria": s.skew(), "minimo": s.min(), "p1": s.quantile(0.01),
                      "q1": s.quantile(0.25), "mediana": s.median(), "q3": s.quantile(0.75),
                      "p99": s.quantile(0.99), "maximo": s.max()})
    tabla = pd.DataFrame(filas)

    # El segundo panel muestra el mismo precio en escala logarítmica, con valores en euros
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    ejes[0].hist(conjunto[objetivo] / 1e6, bins=100, color=GRIS)
    ejes[0].set(title="Precio de oferta", xlabel="Precio (millones de €)", ylabel="Viviendas")
    limites = np.log10([conjunto[objetivo].min(), conjunto[objetivo].max()])
    ejes[1].hist(conjunto[objetivo], bins=np.logspace(*limites, 100), color=AZUL)
    ejes[1].set_xscale("log")
    marcas = [25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000]
    ejes[1].set_xticks(marcas, [numero_es(m) for m in marcas], rotation=30)
    ejes[1].minorticks_off()
    ejes[1].set(title="Precio de oferta (escala logarítmica)", xlabel="Precio (€, escala logarítmica)",
                ylabel="Viviendas")
    for eje in ejes:
        eje.yaxis.set_major_formatter(formato_eje_es)
    ejes[0].xaxis.set_major_formatter(formato_eje_es)
    guardar_figura(fig, "01_distribucion_precio.png")
    return tabla


# 2. Precio por distrito y por zona

def parte_entre_grupos(valores, grupos):
    """Parte de la varianza que se da entre grupos: 1 - suma de cuadrados dentro / total."""
    total = ((valores - valores.mean()) ** 2).sum()
    dentro = ((valores - valores.groupby(grupos).transform("mean")) ** 2).sum()
    return 1 - dentro / total


def analizar_distritos(conjunto, objetivo):
    distritos = (conjunto.groupby(["COD_DISTRITO", "DISTRITO"])
                 .agg(viviendas=(objetivo, "size"), precio_mediano=(objetivo, "median"),
                      precio_m2_mediano=("PRECIO_M2", "median"))
                 .reset_index().sort_values("precio_m2_mediano", ascending=False))

    log_m2 = np.log(conjunto["PRECIO_M2"])
    varianza = pd.DataFrame([
        {"variable": variable, "agrupacion": agrupacion, "grupos": conjunto[col].nunique(),
         "parte_entre_grupos": round(parte_entre_grupos(valores, conjunto[col]), 4)}
        for variable, valores in [("logaritmo del precio", conjunto["LOG_PRECIO"]),
                                  ("logaritmo del precio por m²", log_m2)]
        for agrupacion, col in [("distrito", "COD_DISTRITO"), ("zona de idealista", "LOCATIONID")]
    ])

    # Diagrama de cajas del precio por m² por distrito, ordenado por la mediana
    orden = distritos.sort_values("precio_m2_mediano")["COD_DISTRITO"].tolist()
    nombres = distritos.set_index("COD_DISTRITO").loc[orden, "DISTRITO"].tolist()
    datos = [conjunto.loc[conjunto.COD_DISTRITO == c, "PRECIO_M2"] for c in orden]
    fig, eje = plt.subplots(figsize=(9, 6.5))
    eje.boxplot(datos, vert=False, showfliers=False, patch_artist=True,
                boxprops={"facecolor": "#dfe7f2", "color": GRIS}, medianprops={"color": AZUL},
                whiskerprops={"color": GRIS}, capprops={"color": GRIS})
    eje.set_yticks(range(1, len(nombres) + 1), nombres)
    eje.set(title="Precio de oferta por m² en cada distrito", xlabel="€ por m² (valores atípicos no representados)")
    eje.xaxis.set_major_formatter(formato_eje_es)
    guardar_figura(fig, "02_precio_m2_distritos.png")

    mapa_por_zona(conjunto)
    return distritos, varianza


def mapa_por_zona(conjunto):
    """Mapa del precio mediano por m² en cada zona de idealista."""
    zonas = pd.read_csv(DIR_IDEALISTA / "madrid_barrios.csv")
    mediana = conjunto.groupby("LOCATIONID")["PRECIO_M2"].median()
    zonas["precio_m2"] = zonas["LOCATIONID"].map(mediana)

    parches, valores = [], []
    for geometria, valor in zip(zonas["geometry"].map(wkt.loads), zonas["precio_m2"]):
        for poligono in geometria.geoms:
            parches.append(ParcheMatplotlib(np.asarray(poligono.exterior.coords), closed=True))
            valores.append(valor)
    coleccion = PatchCollection(parches, cmap="Blues", edgecolor="white", linewidth=0.3)
    coleccion.set_array(np.array(valores))

    # Contorno de los distritos, como unión de sus zonas, para situarlos en el mapa. Las zonas
    # de idealista dejan pequeños huecos entre sí: se amplían ligeramente antes de unirlas y se
    # reducen después, para que el contorno no dibuje esos huecos como límites
    geometrias = zonas["geometry"].map(wkt.loads)
    margen = 0.0006  # grados, unos 50-65 m
    contornos = [shapely.union_all(shapely.buffer(geometrias[zonas["COD_DISTRITO"] == cod].to_numpy(), margen))
                 .buffer(-margen) for cod in zonas["COD_DISTRITO"].unique()]

    fig, eje = plt.subplots(figsize=(7, 6.5))
    eje.add_collection(coleccion)
    for contorno in contornos:
        for poligono in getattr(contorno, "geoms", [contorno]):
            eje.plot(*poligono.exterior.xy, color="#333333", linewidth=0.9)
    eje.autoscale_view()
    eje.set_aspect(1 / np.cos(np.radians(40.42)))
    eje.set_axis_off()
    eje.set_title("Precio mediano de oferta por m² en cada zona de idealista")
    barra = fig.colorbar(coleccion, ax=eje, shrink=0.6, label="€ por m²")
    barra.ax.yaxis.set_major_formatter(formato_eje_es)
    eje.legend(handles=[plt.Line2D([], [], color="#333333", linewidth=0.9, label="Límite de distrito")],
               loc="lower left", fontsize=8, frameon=False)
    guardar_figura(fig, "02_mapa_precio_m2_zonas.png")


# 3. Precio y variables de la vivienda y la localización

def analizar_vivienda(conjunto, escenarios, objetivo):
    variables = escenarios["escenarios"]["D1"]
    d0 = set(escenarios["escenarios"]["D0"])
    filas = []
    for col in variables:
        pares = conjunto[[col, objetivo]].dropna()
        filas.append({"variable": col, "escenario": "D0" if col in d0 else "D1", "n": len(pares),
                      "spearman_con_precio": round(pares[col].corr(pares[objetivo], method="spearman"), 4)})
    tabla = pd.DataFrame(filas).sort_values("spearman_con_precio", key=abs, ascending=False)

    # Superficie y precio en escala logarítmica: se leen los valores reales, no sus logaritmos
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    superficie = ejes[0].hexbin(conjunto["CONSTRUCTEDAREA"], conjunto[objetivo] / 1e3, gridsize=60,
                                xscale="log", yscale="log", cmap="Blues", mincnt=1)
    ejes[0].set(title="Precio y superficie", xlabel="Superficie construida (m², escala logarítmica)",
                ylabel="Precio (miles de €, escala logarítmica)")
    ejes[0].set_xticks([25, 50, 100, 200, 500, 1000], ["25", "50", "100", "200", "500", "1.000"])
    ejes[0].set_yticks([50, 100, 200, 500, 1000, 2000, 5000],
                       ["50", "100", "200", "500", "1.000", "2.000", "5.000"])
    ejes[0].minorticks_off()
    distancia = ejes[1].hexbin(conjunto["DISTANCE_TO_CITY_CENTER"], conjunto["PRECIO_M2"], gridsize=60,
                               cmap="Blues", mincnt=1)
    ejes[1].set(title="Precio por m² y distancia al centro", xlabel="Distancia al centro (km)",
                ylabel="€ por m²")
    ejes[1].yaxis.set_major_formatter(lambda valor, _: f"{valor:,.0f}".replace(",", "."))

    # Una sola escala de color para los dos paneles, para que la leyenda sirva para ambos
    escala = LogNorm(vmin=1, vmax=max(superficie.get_array().max(), distancia.get_array().max()))
    superficie.set_norm(escala)
    distancia.set_norm(escala)
    fig.colorbar(distancia, ax=list(ejes), label="Viviendas por hexágono (escala logarítmica)",
                 format="%.0f", aspect=30)
    fig.savefig(DIR_FIGURAS / "03_precio_vivienda_localizacion.png", dpi=200)
    plt.close(fig)
    return tabla


# 4. Indicadores de contexto y precio del distrito

def analizar_contexto(conjunto, escenarios, distritos):
    d1 = set(escenarios["escenarios"]["D1"])
    d2 = set(escenarios["escenarios"]["D2"])
    indicadores = [c for c in escenarios["escenarios"]["D3"] if c not in d1]

    # Los indicadores son constantes dentro de cada distrito: se toma su valor una vez
    contexto = (conjunto.groupby("COD_DISTRITO")[indicadores].first().reset_index()
                .merge(distritos, on="COD_DISTRITO"))
    correlaciones = pd.DataFrame([
        {"indicador": c, "escenario": "D2" if c in d2 else "D3", "distritos": len(contexto),
         "pearson_con_precio_m2_mediano": round(contexto[c].corr(contexto["precio_m2_mediano"]), 4),
         "spearman_con_precio_m2_mediano": round(contexto[c].corr(contexto["precio_m2_mediano"],
                                                                    method="spearman"), 4)}
        for c in indicadores
    ])

    fig, ejes = plt.subplots(2, 4, figsize=(13, 7), sharey=True, layout="constrained")
    for eje, fila in zip(ejes.flat, correlaciones.itertuples()):
        titulo, unidad, factor = ETIQUETAS_CONTEXTO[fila.indicador]
        x = contexto[fila.indicador] * factor
        eje.scatter(x, contexto["precio_m2_mediano"], color=AZUL if fila.escenario == "D2" else GRIS, s=22)
        eje.set_title(f"{titulo}\nPearson {numero_es(fila.pearson_con_precio_m2_mediano, 2)} · "
                      f"Spearman {numero_es(fila.spearman_con_precio_m2_mediano, 2)}", fontsize=9)
        eje.set_xlabel(unidad, fontsize=8)
        eje.xaxis.set_major_formatter(formato_eje_es)
        eje.yaxis.set_major_formatter(formato_eje_es)
        eje.tick_params(labelsize=7)
        if fila.indicador == "detenidos_por_1000_hab":
            centro = contexto["DISTRITO"] == "Centro"
            eje.annotate("Centro", (x[centro].iloc[0], contexto.loc[centro, "precio_m2_mediano"].iloc[0]),
                         xytext=(-8, -12), textcoords="offset points", ha="right", fontsize=8)
    for eje in ejes[:, 0]:
        eje.set_ylabel("€ por m² (mediana del distrito)", fontsize=8)
    fig.suptitle("Indicadores de contexto y precio por m² de los 21 distritos", fontsize=11)
    fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=AZUL, label="D2: contexto socioeconómico"),
                        plt.Line2D([], [], marker="o", linestyle="", color=GRIS, label="D3: entorno urbano")],
               loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.savefig(DIR_FIGURAS / "04_contexto_precio_distritos.png", dpi=200)
    plt.close(fig)
    return contexto, correlaciones


# 5. Dependencia espacial del precio

def indice_moran(valores, pesos):
    """Índice de Moran con una matriz de pesos estandarizada por filas."""
    z = valores - valores.mean()
    return len(z) / pesos.sum() * (z @ pesos @ z) / (z @ z)


def analizar_dependencia_espacial(conjunto):
    """Índice de Moran del precio mediano por m² de las zonas, con zonas vecinas como referencia."""
    zonas = pd.read_csv(DIR_IDEALISTA / "madrid_barrios.csv")
    zonas = zonas[zonas["LOCATIONID"].isin(conjunto["LOCATIONID"])].reset_index(drop=True)
    valores = zonas["LOCATIONID"].map(conjunto.groupby("LOCATIONID")["PRECIO_M2"].median()).to_numpy()

    geometrias = a_metros(zonas["geometry"].map(wkt.loads).to_numpy(), float(conjunto["LATITUDE"].median()))
    n = len(zonas)
    vecinas = np.array([[i != j and geometrias[i].distance(geometrias[j]) < DISTANCIA_VECINAS_M
                         for j in range(n)] for i in range(n)], dtype=float)
    pesos = vecinas / vecinas.sum(axis=1, keepdims=True)

    observado = indice_moran(valores, pesos)
    azar = np.random.default_rng(SEMILLA)
    permutados = np.array([indice_moran(azar.permutation(valores), pesos) for _ in range(PERMUTACIONES_MORAN)])
    p_valor = (1 + (permutados >= observado).sum()) / (1 + PERMUTACIONES_MORAN)
    return pd.DataFrame([
        ("Zonas", n),
        (f"Criterio de vecindad: límites a menos de {DISTANCIA_VECINAS_M} m", ""),
        ("Vecinas por zona (mínimo / mediana / máximo)",
         f"{int(vecinas.sum(1).min())} / {int(np.median(vecinas.sum(1)))} / {int(vecinas.sum(1).max())}"),
        ("Índice de Moran del precio mediano por m²", round(observado, 4)),
        ("Valor esperado sin dependencia espacial", round(-1 / (n - 1), 4)),
        (f"Índice de Moran máximo en {PERMUTACIONES_MORAN} permutaciones al azar", round(permutados.max(), 4)),
        ("p-valor (permutaciones)", round(p_valor, 4)),
    ], columns=["comprobacion", "resultado"])


# 6. Pliegues de la validación espacial

def analizar_pliegues(conjunto):
    """Distritos y viviendas de cada pliegue de la validación espacial (los de construir_conjunto.py)."""
    pliegues = pd.read_csv(PLIEGUES_CSV, dtype={"ASSETID": str, "COD_DISTRITO": str})
    datos = conjunto[["ASSETID", "DISTRITO"]].merge(pliegues[["ASSETID", "pliegue_espacial"]], on="ASSETID",
                                                     validate="one_to_one")
    # Distritos de cada pliegue, de mayor a menor número de viviendas
    por_distrito = datos.groupby(["pliegue_espacial", "DISTRITO"]).size().rename("viviendas").reset_index()
    por_distrito = por_distrito.sort_values(["pliegue_espacial", "viviendas"], ascending=[True, False])
    tabla = por_distrito.groupby("pliegue_espacial").agg(
        distritos=("DISTRITO", "size"), viviendas=("viviendas", "sum"), nombres=("DISTRITO", ", ".join))
    tabla["pct_viviendas"] = (100 * tabla["viviendas"] / len(datos)).round(1)
    return tabla.reset_index().rename(columns={"pliegue_espacial": "pliegue"})[
        ["pliegue", "distritos", "viviendas", "pct_viviendas", "nombres"]]


# 7. Precio según conste o no la planta y la posición

def analizar_ausentes(conjunto, objetivo):
    """Precio de las viviendas sin dato de planta, posición o calidad catastral frente al resto.

    Si las viviendas sin dato tienen precios distintos, que falte el dato es informativo: por eso
    la modelización añade un indicador de dato ausente además de imputar el valor. La calidad
    catastral solo falta en una vivienda, así que no lleva indicador.
    """
    filas = []
    for col, nombre in [("FLOORCLEAN", "planta"), ("FLATLOCATIONID", "posición exterior o interior"),
                        ("CADASTRALQUALITYID", "calidad catastral de la construcción")]:
        for consta, grupo in [("sí", conjunto[conjunto[col].notna()]), ("no", conjunto[conjunto[col].isna()])]:
            filas.append({"variable": nombre, "consta_el_dato": consta, "viviendas": len(grupo),
                          "pct_viviendas": round(100 * len(grupo) / len(conjunto), 1),
                          "precio_mediano": grupo[objetivo].median(),
                          "precio_m2_mediano": round(grupo["PRECIO_M2"].median()),
                          "superficie_mediana": grupo["CONSTRUCTEDAREA"].median()})
    return pd.DataFrame(filas)


# 8. Atípicos del precio

def analizar_atipicos(conjunto, objetivo):
    """Atípicos del precio según el rango intercuartílico, en euros y en logaritmo.

    En logaritmo, que es la escala en la que se entrenan los modelos, los atípicos casi
    desaparecen: por eso se conservan todas las viviendas. También devuelve las viviendas con
    más habitaciones, donde hay valores improbables que se conservan porque no pueden comprobarse.
    """
    filas = []
    for escala, valores in [("euros", conjunto[objetivo]), ("logaritmo", conjunto["LOG_PRECIO"])]:
        q1, q3 = valores.quantile([0.25, 0.75])
        riq = q3 - q1
        for veces in (1.5, 3):
            fuera = (valores < q1 - veces * riq) | (valores > q3 + veces * riq)
            filas.append({"escala": escala, "criterio": f"más de {veces} rangos intercuartílicos",
                          "atipicos": int(fuera.sum()), "pct_viviendas": round(100 * fuera.mean(), 2)})
    columnas = [objetivo, "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "PRECIO_M2", "DISTRITO"]
    extremos = pd.concat([conjunto.nsmallest(5, objetivo)[columnas].assign(grupo="más baratas"),
                          conjunto.nlargest(5, objetivo)[columnas].assign(grupo="más caras")])
    extremos["PRECIO_M2"] = extremos["PRECIO_M2"].round()
    habitaciones = conjunto.nlargest(5, "ROOMNUMBER")[columnas].round({"PRECIO_M2": 0})
    return pd.DataFrame(filas), extremos[["grupo"] + columnas], habitaciones


def main():
    conjunto, escenarios = cargar()
    objetivo = escenarios["objetivo"]
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    precio = analizar_precio(conjunto, objetivo)
    distritos, varianza = analizar_distritos(conjunto, objetivo)
    vivienda = analizar_vivienda(conjunto, escenarios, objetivo)
    contexto, correlaciones = analizar_contexto(conjunto, escenarios, distritos)
    dependencia = analizar_dependencia_espacial(conjunto)
    pliegues = analizar_pliegues(conjunto)
    ausentes = analizar_ausentes(conjunto, objetivo)
    atipicos, extremos, habitaciones = analizar_atipicos(conjunto, objetivo)

    salidas = {"01_precio": precio, "02_distritos": distritos, "02_varianza_entre_grupos": varianza,
               "03_correlaciones_vivienda": vivienda, "04_contexto_distritos": contexto,
               "04_correlaciones_contexto": correlaciones, "05_dependencia_espacial": dependencia,
               "06_pliegues_espaciales": pliegues, "07_ausentes_precio": ausentes,
               "08_atipicos_precio": atipicos, "08_precios_extremos": extremos,
               "08_habitaciones_extremas": habitaciones}
    for nombre, tabla in salidas.items():
        tabla.to_csv(DIR_EXPLORATORIO / f"{nombre}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(f"Viviendas analizadas: {len(conjunto):,}\n")
    print("Precio:")
    print(precio.round(3).to_string(index=False))
    print("\nParte de la varianza entre grupos:")
    print(varianza.to_string(index=False))
    print("\nDistritos (precio por m² mediano):")
    print(distritos.round(0).to_string(index=False))
    print("\nCorrelación de Spearman con el precio (D0 y D1, 10 primeras):")
    print(vivienda.head(10).to_string(index=False))
    print("\nIndicadores de contexto y precio por m² del distrito (21 distritos):")
    print(correlaciones.to_string(index=False))
    print("\nDependencia espacial del precio:")
    print(dependencia.to_string(index=False))
    print(f"\nPliegues de la validación espacial ({len(pliegues)} grupos de distritos):")
    print(pliegues.to_string(index=False))
    print("\nPrecio según conste o no el dato:")
    print(ausentes.to_string(index=False))
    print("\nAtípicos del precio:")
    print(atipicos.to_string(index=False))
    print(extremos.to_string(index=False))
    print("\nViviendas con más habitaciones:")
    print(habitaciones.to_string(index=False))
    print(f"\nFicheros guardados en {DIR_EXPLORATORIO}")


if __name__ == "__main__":
    main()
