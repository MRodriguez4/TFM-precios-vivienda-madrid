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

Requiere haber ejecutado antes construir_conjunto.py.

Salida (en datos/procesados/exploratorio/):
    - 01_precio.csv: estadísticos del precio, de su logaritmo y del precio por m².
    - 02_distritos.csv: viviendas y precio mediano de cada distrito.
    - 02_varianza_entre_grupos.csv: parte de la varianza del precio que se da entre distritos y entre zonas.
    - 03_correlaciones_vivienda.csv: correlación de Spearman de cada variable de D0 y D1 con el precio.
    - 04_contexto_distritos.csv: los 8 indicadores y el precio mediano por m² de cada distrito.
    - 04_correlaciones_contexto.csv: correlación de cada indicador con el precio del distrito (21 distritos).
    - 05_dependencia_espacial.csv: índice de Moran del precio por m² entre zonas vecinas.
    - 06_pliegues_espaciales.csv: distritos y viviendas de cada pliegue de la validación espacial.
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
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as ParcheMatplotlib
from shapely import wkt

from idealista18_madrid import a_metros
from rutas import CONJUNTO_CSV, DIR_EXPLORATORIO, DIR_IDEALISTA, ESCENARIOS_JSON

DIR_FIGURAS = DIR_EXPLORATORIO / "figuras"
GRIS = "#7f7f7f"
SEMILLA = 2018
# Dos zonas son vecinas si sus límites están a menos de esta distancia: las zonas de idealista
# no cubren la ciudad sin huecos, así que exigir que se toquen dejaría vecinas sin conectar
DISTANCIA_VECINAS_M = 100
PERMUTACIONES_MORAN = 999
N_PLIEGUES = 5
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
    decimales = next((d for d in range(3) if abs(valor * 10 ** d - round(valor * 10 ** d)) < 1e-9), 2)
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
    fig.savefig(DIR_FIGURAS / nombre, dpi=200)
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

    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    ejes[0].hist(conjunto[objetivo] / 1e6, bins=100, color=GRIS)
    ejes[0].set(title="Precio de oferta", xlabel="Millones de €", ylabel="Viviendas")
    ejes[1].hist(conjunto["LOG_PRECIO"], bins=100, color=AZUL)
    ejes[1].set(title="Logaritmo del precio de oferta", xlabel="ln(precio)", ylabel="Viviendas")
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

    fig, eje = plt.subplots(figsize=(7, 7.5))
    eje.add_collection(coleccion)
    eje.autoscale_view()
    eje.set_aspect(1 / np.cos(np.radians(40.42)))
    eje.set_axis_off()
    eje.set_title("Precio mediano de oferta por m² en cada zona de idealista")
    fig.colorbar(coleccion, ax=eje, shrink=0.6, label="€ por m²")
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

def analizar_pliegues(distritos):
    """Reparto de los distritos en N_PLIEGUES grupos de tamaño parecido.

    Sigue el criterio de GroupKFold de scikit-learn: los distritos, de mayor a menor número de
    viviendas, se asignan uno a uno al pliegue que tiene menos viviendas en ese momento.
    """
    viviendas = [0] * N_PLIEGUES
    asignacion = {}
    for fila in distritos.sort_values("viviendas", ascending=False).itertuples():
        pliegue = int(np.argmin(viviendas))
        viviendas[pliegue] += fila.viviendas
        asignacion.setdefault(pliegue, []).append(fila.DISTRITO)

    total = sum(viviendas)
    return pd.DataFrame([
        {"pliegue": k + 1, "distritos": len(asignacion[k]), "viviendas": viviendas[k],
         "pct_viviendas": round(100 * viviendas[k] / total, 1), "nombres": ", ".join(asignacion[k])}
        for k in range(N_PLIEGUES)
    ])


def main():
    conjunto, escenarios = cargar()
    objetivo = escenarios["objetivo"]
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    precio = analizar_precio(conjunto, objetivo)
    distritos, varianza = analizar_distritos(conjunto, objetivo)
    vivienda = analizar_vivienda(conjunto, escenarios, objetivo)
    contexto, correlaciones = analizar_contexto(conjunto, escenarios, distritos)
    dependencia = analizar_dependencia_espacial(conjunto)
    pliegues = analizar_pliegues(distritos)

    salidas = {"01_precio": precio, "02_distritos": distritos, "02_varianza_entre_grupos": varianza,
               "03_correlaciones_vivienda": vivienda, "04_contexto_distritos": contexto,
               "04_correlaciones_contexto": correlaciones, "05_dependencia_espacial": dependencia,
               "06_pliegues_espaciales": pliegues}
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
    print(f"\nPliegues de la validación espacial ({N_PLIEGUES} grupos de distritos):")
    print(pliegues.to_string(index=False))
    print(f"\nFicheros guardados en {DIR_EXPLORATORIO}")


if __name__ == "__main__":
    main()
