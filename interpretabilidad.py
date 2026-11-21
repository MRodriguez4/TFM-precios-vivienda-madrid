"""
Interpretabilidad de los modelos de D3 mediante SHAP.

Explica las predicciones de los mismos modelos que se evaluaron en modelizacion.py: para cada
modelo, validación y pliegue de D3, vuelve a entrenar el modelo con los hiperparámetros que se
eligieron en ese pliegue y calcula los valores SHAP de las viviendas del pliegue de evaluación,
que el modelo no ha visto. Antes de calcularlos comprueba que el modelo reproduce el MAE guardado
en metricas_pliegues.csv, es decir, que es exactamente el modelo evaluado.

Decisiones:
    - Escenario D3, que incluye las variables de los cuatro bloques de información.
    - Valores SHAP en la escala del modelo (logaritmo del precio), en la que la predicción es la
      suma del valor base y las contribuciones de todas las variables.
    - XGBoost y Random Forest: TreeSHAP (versión path-dependent). Ridge: el valor SHAP de una
      variable es su coeficiente por su valor estandarizado, que es la diferencia respecto a la
      media del entrenamiento; el valor base es la constante del modelo.
    - Ridge y XGBoost se explican con todas las viviendas de cada pliegue. Random Forest, con una
      muestra al azar de cada pliegue, porque su TreeSHAP tarda unos 12 segundos por vivienda
      (árboles sin límite de profundidad).
    - Las columnas que salen de una misma variable (el trimestre, y la planta y la posición con su
      indicador de dato ausente) se suman en cada vivienda antes de tomar el valor absoluto; lo
      mismo con todas las variables de un bloque.
    - Importancia de una variable o bloque: media del valor SHAP absoluto en las viviendas
      explicadas de cada pliegue; se resume con la media y la desviación entre los cinco pliegues.

Requiere haber ejecutado antes modelizacion.py.

Salida (en datos/procesados/interpretabilidad/):
    - comprobacion.csv: MAE guardado y reproducido, viviendas explicadas, error de aditividad y
      tiempo de cada modelo, validación y pliegue.
    - importancia_variables.csv: importancia de cada variable por modelo y validación (media y
      desviación entre pliegues), con su bloque y su posición.
    - importancia_bloques.csv: importancia de cada bloque y su peso sobre el total de bloques.
    - importancia_pliegues.csv: importancia de cada variable y bloque en cada pliegue.
    - shap_<modelo>_<validacion>.csv.gz: valor SHAP de cada variable para cada vivienda explicada.
    - figuras/: importancia por bloque, variables más importantes y efecto de los indicadores de
      contexto por distrito.
    Los resultados de cada pliegue se guardan en parciales/ a medida que se calculan, así que una
    ejecución interrumpida continúa donde se quedó.

Uso:
    python interpretabilidad.py                       # los tres modelos
    python interpretabilidad.py --modelos xgboost     # solo los indicados
    python interpretabilidad.py --prueba              # un pliegue con pocas viviendas, para medir
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from analisis_exploratorio import AZUL, ETIQUETAS_CONTEXTO, GRIS, formato_eje_es, numero_es
from modelizacion import COLUMNA_PLIEGUE, METRICAS_CSV, MODELOS, SEMILLA, VALIDACIONES, cargar, crear_pipeline
from rutas import DIR_INTERPRETABILIDAD

ESCENARIO = "D3"
# Viviendas de cada pliegue que se explican; None explica todas
VIVIENDAS_POR_PLIEGUE = {"ridge": None, "xgboost": None, "random_forest": 100}
BLOQUES = {"D0": "Vivienda", "D1": "Localización", "D2": "Contexto socioeconómico", "D3": "Entorno urbano"}
# Diferencia máxima admitida entre el MAE reproducido y el guardado, en euros
TOLERANCIA_MAE = 0.01
VARIABLES_EN_FIGURA = 15

DIR_PARCIALES = DIR_INTERPRETABILIDAD / "parciales"
DIR_FIGURAS = DIR_INTERPRETABILIDAD / "figuras"


def bloque_de_variables(escenarios):
    """Bloque al que pertenece cada variable: el primer escenario en el que aparece."""
    bloque = {}
    for escenario in BLOQUES:
        for variable in escenarios[escenario]:
            bloque.setdefault(variable, BLOQUES[escenario])
    return bloque


def variable_original(columna):
    """Variable del conjunto de la que sale cada columna que ve el modelo."""
    if columna.startswith("missingindicator_"):
        return columna.removeprefix("missingindicator_")
    if columna.startswith("TRIMESTRE_"):
        return "TRIMESTRE"
    return columna


def valores_shap(modelo, pipeline, X):
    """Valores SHAP (en logaritmo del precio) de cada columna del modelo y valor base."""
    estimador = pipeline[-1]
    if modelo == "ridge":
        # Variables estandarizadas con la media del entrenamiento: la contribución de cada una es
        # su coeficiente por su diferencia respecto a esa media
        return X * estimador.coef_, np.full(len(X), estimador.intercept_)
    if modelo == "xgboost":
        contribuciones = estimador.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contribuciones[:, :-1], contribuciones[:, -1]
    explicador = shap.TreeExplainer(estimador)
    return explicador.shap_values(X), np.full(len(X), float(np.ravel(explicador.expected_value)[0]))


def explicar_pliegue(conjunto, variables, objetivo, modelo, validacion, pliegue, fila, n_viviendas):
    """Reentrena el modelo del pliegue, comprueba su MAE y calcula los SHAP de su evaluación."""
    columna = COLUMNA_PLIEGUE[validacion]
    entrenamiento = conjunto[conjunto[columna] != pliegue]
    evaluacion = conjunto[conjunto[columna] == pliegue]

    inicio = time.perf_counter()
    hiperparametros = {f"modelo__{k}": v for k, v in json.loads(fila["hiperparametros"]).items()}
    pipeline = crear_pipeline(modelo, variables).set_params(**hiperparametros)
    pipeline.fit(entrenamiento[variables], np.log(entrenamiento[objetivo]))
    mae = mean_absolute_error(evaluacion[objetivo], np.exp(pipeline.predict(evaluacion[variables])))
    if abs(mae - fila["mae"]) > TOLERANCIA_MAE:
        raise RuntimeError(f"{modelo}, {validacion}, pliegue {pliegue}: el MAE reproducido ({mae:.2f}) "
                           f"no coincide con el guardado ({fila['mae']:.2f})")

    if n_viviendas and len(evaluacion) > n_viviendas:
        evaluacion = evaluacion.sample(n_viviendas, random_state=SEMILLA)
    X = pipeline[:-1].transform(evaluacion[variables])
    contribuciones, base = valores_shap(modelo, pipeline, X)
    prediccion = pipeline.predict(evaluacion[variables])
    error_aditividad = float(np.abs(base + contribuciones.sum(axis=1) - prediccion).max())

    columnas = pipeline[:-1].get_feature_names_out()
    por_variable = (pd.DataFrame(contribuciones, columns=columnas)
                    .T.groupby(lambda c: variable_original(c)).sum().T)
    por_variable.insert(0, "ASSETID", evaluacion["ASSETID"].to_numpy())
    por_variable.insert(1, "pliegue", pliegue)
    por_variable.insert(2, "valor_base", base)
    por_variable.insert(3, "prediccion_log", prediccion)
    comprobacion = {"modelo": modelo, "validacion": validacion, "pliegue": pliegue,
                    "mae_guardado": round(fila["mae"], 2), "mae_reproducido": round(mae, 2),
                    "viviendas_explicadas": len(evaluacion), "error_aditividad_max": error_aditividad,
                    "segundos": round(time.perf_counter() - inicio, 1)}
    return por_variable, comprobacion


def ejecutar(conjunto, escenarios, objetivo, modelos, prueba=False):
    """Calcula los pliegues que faltan y guarda cada uno en parciales/ en cuanto termina."""
    variables = escenarios[ESCENARIO]
    metricas = pd.read_csv(METRICAS_CSV)
    metricas = metricas[metricas["escenario"] == ESCENARIO]
    destino = DIR_PARCIALES / "prueba" if prueba else DIR_PARCIALES
    destino.mkdir(parents=True, exist_ok=True)
    for modelo in modelos:
        for validacion in VALIDACIONES:
            for pliegue in sorted(metricas["pliegue"].unique())[:1 if prueba else None]:
                nombre = f"{modelo}_{validacion}_{pliegue}"
                if (destino / f"{nombre}.csv.gz").exists():
                    continue
                fila = metricas[(metricas["modelo"] == modelo) & (metricas["validacion"] == validacion)
                                & (metricas["pliegue"] == pliegue)].iloc[0]
                n_viviendas = 20 if prueba else VIVIENDAS_POR_PLIEGUE[modelo]
                shap_pliegue, comprobacion = explicar_pliegue(conjunto, variables, objetivo, modelo, validacion,
                                                              pliegue, fila, n_viviendas)
                pd.DataFrame([comprobacion]).to_csv(destino / f"{nombre}_comprobacion.csv", index=False)
                shap_pliegue.to_csv(destino / f"{nombre}.csv.gz", index=False)
                print(f"{nombre}: MAE {comprobacion['mae_reproducido']:,.2f} (guardado "
                      f"{comprobacion['mae_guardado']:,.2f}), {comprobacion['viviendas_explicadas']:,} viviendas, "
                      f"{comprobacion['segundos']:,.0f} s", flush=True)


def importancias(shap_viviendas, bloque):
    """Importancia de cada variable y bloque en cada pliegue: media del valor SHAP absoluto."""
    variables = [c for c in shap_viviendas.columns if c in bloque]
    filas = []
    for pliegue, grupo in shap_viviendas.groupby("pliegue"):
        for variable in variables:
            filas.append({"pliegue": pliegue, "nivel": "variable", "nombre": variable, "bloque": bloque[variable],
                          "importancia": grupo[variable].abs().mean()})
        for nombre in BLOQUES.values():
            suma = grupo[[v for v in variables if bloque[v] == nombre]].sum(axis=1)
            filas.append({"pliegue": pliegue, "nivel": "bloque", "nombre": nombre, "bloque": nombre,
                          "importancia": suma.abs().mean()})
    return pd.DataFrame(filas)


def resumir(escenarios, modelos):
    """Une los parciales y resume la importancia de variables y bloques entre pliegues."""
    bloque = bloque_de_variables(escenarios)
    comprobaciones, pliegues = [], []
    for modelo in modelos:
        for validacion in VALIDACIONES:
            ficheros = sorted(DIR_PARCIALES.glob(f"{modelo}_{validacion}_*.csv.gz"))
            if not ficheros:
                continue
            shap_viviendas = pd.concat([pd.read_csv(f, dtype={"ASSETID": str}) for f in ficheros])
            shap_viviendas.to_csv(DIR_INTERPRETABILIDAD / f"shap_{modelo}_{validacion}.csv.gz", index=False)
            tabla = importancias(shap_viviendas, bloque)
            tabla.insert(0, "modelo", modelo)
            tabla.insert(1, "validacion", validacion)
            pliegues.append(tabla)
            comprobaciones += [pd.read_csv(f) for f in sorted(DIR_PARCIALES.glob(f"{modelo}_{validacion}_*_comprobacion.csv"))]
    if not pliegues:
        return None
    pliegues = pd.concat(pliegues, ignore_index=True)
    claves = ["modelo", "validacion", "nivel", "nombre", "bloque"]
    resumen = (pliegues.groupby(claves)["importancia"].agg(importancia_media="mean", importancia_desviacion="std",
                                                            pliegues="count").reset_index())
    resumen["posicion"] = (resumen.groupby(["modelo", "validacion", "nivel"])["importancia_media"]
                           .rank(ascending=False, method="first").astype(int))
    bloques = resumen[resumen["nivel"] == "bloque"].drop(columns=["nivel", "bloque"]).rename(columns={"nombre": "bloque"})
    bloques["peso_pct"] = 100 * bloques["importancia_media"] / bloques.groupby(
        ["modelo", "validacion"])["importancia_media"].transform("sum")
    variables = resumen[resumen["nivel"] == "variable"].drop(columns="nivel").rename(columns={"nombre": "variable"})

    pliegues.to_csv(DIR_INTERPRETABILIDAD / "importancia_pliegues.csv", index=False, encoding="utf-8-sig")
    variables.sort_values(["modelo", "validacion", "posicion"]).to_csv(
        DIR_INTERPRETABILIDAD / "importancia_variables.csv", index=False, encoding="utf-8-sig")
    bloques.sort_values(["modelo", "validacion", "posicion"]).to_csv(
        DIR_INTERPRETABILIDAD / "importancia_bloques.csv", index=False, encoding="utf-8-sig")
    pd.concat(comprobaciones).to_csv(DIR_INTERPRETABILIDAD / "comprobacion.csv", index=False)
    return variables, bloques


NOMBRES_MODELOS = {"ridge": "Ridge", "random_forest": "Random Forest", "xgboost": "XGBoost"}
NOMBRES_VALIDACIONES = {"aleatoria": "Validación aleatoria", "espacial": "Validación espacial"}


def figura_bloques(bloques):
    """Importancia de cada bloque por modelo, con una barra por validación."""
    modelos = [m for m in MODELOS if m in set(bloques["modelo"])]
    fig, ejes = plt.subplots(1, len(modelos), figsize=(4.2 * len(modelos), 3.6), sharey=True, squeeze=False)
    posiciones = np.arange(len(BLOQUES))
    for eje, modelo in zip(ejes[0], modelos):
        for desplazamiento, (validacion, color) in zip((-0.2, 0.2), (("aleatoria", GRIS), ("espacial", AZUL))):
            datos = (bloques[(bloques["modelo"] == modelo) & (bloques["validacion"] == validacion)]
                     .set_index("bloque").reindex(BLOQUES.values()))
            eje.barh(posiciones + desplazamiento, datos["importancia_media"], height=0.4, color=color,
                     xerr=datos["importancia_desviacion"], error_kw={"elinewidth": 0.8, "capsize": 2},
                     label=NOMBRES_VALIDACIONES[validacion])
        eje.set_title(NOMBRES_MODELOS[modelo], fontsize=10)
        eje.set_yticks(posiciones, BLOQUES.values(), fontsize=8)
        eje.invert_yaxis()
        eje.set_xlabel("Media del valor SHAP absoluto (log del precio)", fontsize=8)
        eje.xaxis.set_major_formatter(formato_eje_es)
        eje.tick_params(labelsize=7)
    ejes[0][0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "01_importancia_bloques.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_variables(variables, modelo="xgboost"):
    """Variables más importantes de un modelo en las dos validaciones."""
    datos = variables[variables["modelo"] == modelo]
    if datos.empty:
        return
    colores = dict(zip(BLOQUES.values(), ("#9ecae1", "#4292c6", "#e8a33d", "#a65628")))
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.5), squeeze=False)
    for eje, validacion in zip(ejes[0], VALIDACIONES):
        seleccion = datos[datos["validacion"] == validacion].nsmallest(VARIABLES_EN_FIGURA, "posicion")
        eje.barh(seleccion["variable"], seleccion["importancia_media"], xerr=seleccion["importancia_desviacion"],
                 color=seleccion["bloque"].map(colores), error_kw={"elinewidth": 0.8, "capsize": 2})
        eje.invert_yaxis()
        eje.set_title(f"{NOMBRES_MODELOS[modelo]}: {NOMBRES_VALIDACIONES[validacion].lower()}", fontsize=10)
        eje.set_xlabel("Media del valor SHAP absoluto (log del precio)", fontsize=8)
        eje.xaxis.set_major_formatter(formato_eje_es)
        eje.tick_params(labelsize=7)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=b) for b, c in colores.items()],
               loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"02_variables_{modelo}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_contexto(conjunto, escenarios, modelo="xgboost"):
    """Contribución media de cada indicador de contexto en cada distrito frente a su valor."""
    ficheros = {v: DIR_INTERPRETABILIDAD / f"shap_{modelo}_{v}.csv.gz" for v in VALIDACIONES}
    if not all(f.exists() for f in ficheros.values()):
        return
    indicadores = [c for c in escenarios["D3"] if c not in escenarios["D1"]]
    valores = conjunto.groupby("COD_DISTRITO")[indicadores].first()
    fig, ejes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    for validacion, color in (("aleatoria", GRIS), ("espacial", AZUL)):
        shap_viviendas = pd.read_csv(ficheros[validacion], dtype={"ASSETID": str}).merge(
            conjunto[["ASSETID", "COD_DISTRITO"]], on="ASSETID")
        medias = shap_viviendas.groupby("COD_DISTRITO")[indicadores].mean()
        for eje, indicador in zip(ejes.flat, indicadores):
            titulo, unidad, factor = ETIQUETAS_CONTEXTO[indicador]
            eje.scatter(valores.loc[medias.index, indicador] * factor, medias[indicador], color=color, s=20,
                        label=NOMBRES_VALIDACIONES[validacion])
            eje.set_title(titulo, fontsize=9)
            eje.set_xlabel(unidad, fontsize=8)
    for eje in ejes.flat:
        eje.axhline(0, color="#333333", linewidth=0.6)
        eje.xaxis.set_major_formatter(formato_eje_es)
        eje.yaxis.set_major_formatter(formato_eje_es)
        eje.tick_params(labelsize=7)
    for eje in ejes[:, 0]:
        eje.set_ylabel("Valor SHAP medio del distrito (log del precio)", fontsize=8)
    fig.suptitle(f"Contribución de los indicadores de contexto en cada distrito ({NOMBRES_MODELOS[modelo]}, D3)",
                 fontsize=11)
    manejadores, etiquetas = ejes.flat[0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.savefig(DIR_FIGURAS / f"03_contexto_distritos_{modelo}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modelos", nargs="+", choices=MODELOS, default=MODELOS)
    parser.add_argument("--prueba", action="store_true", help="un pliegue por modelo y validación, 20 viviendas")
    args = parser.parse_args()

    conjunto, definicion = cargar()
    escenarios, objetivo = definicion["escenarios"], definicion["objetivo"]
    ejecutar(conjunto, escenarios, objetivo, args.modelos, prueba=args.prueba)
    if args.prueba:
        print(pd.concat([pd.read_csv(f) for f in sorted((DIR_PARCIALES / "prueba").glob("*_comprobacion.csv"))])
              .to_string(index=False))
        return

    resultado = resumir(escenarios, MODELOS)
    if resultado is None:
        return
    variables, bloques = resultado
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_bloques(bloques)
    figura_variables(variables)
    figura_contexto(conjunto, escenarios)

    pd.set_option("display.width", 200)
    print("\nImportancia de cada bloque (media del valor SHAP absoluto entre pliegues):")
    print(bloques[["modelo", "validacion", "bloque", "importancia_media", "importancia_desviacion", "peso_pct"]]
          .round(4).to_string(index=False))
    print(f"\nFicheros guardados en {DIR_INTERPRETABILIDAD}")


if __name__ == "__main__":
    main()
