"""
Modelización y evaluación de la predicción del precio en los escenarios D0-D3.

Entrena Ridge, Random Forest y XGBoost en cada escenario y los evalúa con dos validaciones
cruzadas de cinco pliegues: aleatoria y espacial (distritos completos). Los pliegues son los
de datos/final/pliegues.csv, iguales para todos los escenarios y modelos, de modo que cada
comparación entre escenarios se hace con exactamente las mismas viviendas.

Decisiones:
    - Los modelos se entrenan con el logaritmo del precio y las métricas se calculan en euros,
      deshaciendo el logaritmo (sin corrección: la predicción se acerca al precio mediano).
    - La preparación se ajusta dentro de cada pliegue, solo con las viviendas de entrenamiento:
      imputación de la planta, la posición y la calidad catastral (con un indicador de dato
      ausente para la planta y la posición), superficie en logaritmo, trimestre en one-hot y,
      solo en Ridge, una variable del estado de conservación como referencia y estandarización.
    - Los hiperparámetros se eligen con validación cruzada anidada: dentro del entrenamiento de
      cada pliegue, GridSearchCV (Ridge) o RandomizedSearchCV con el mismo número de combinaciones
      (Random Forest y XGBoost), con tres pliegues internos del mismo tipo que la validación (por
      distritos en la espacial). La
      búsqueda usa una muestra del entrenamiento para acotar el tiempo de cálculo, y el modelo
      final se entrena con todo el entrenamiento del pliegue.
    - Métricas: MAE, RMSE y R² en euros por pliegue; la mejora entre escenarios se calcula
      pliegue a pliegue.

Requiere haber ejecutado antes construir_conjunto.py.

Salida (en datos/procesados/modelizacion/):
    - metricas_pliegues.csv: métricas de cada validación, escenario, modelo y pliegue, con los
      hiperparámetros elegidos y el tiempo de cálculo. Se guarda tras cada pliegue, así que una
      ejecución interrumpida continúa donde se quedó.
    - resumen.csv: media y desviación de las métricas entre pliegues.
    - mejoras.csv: mejora de cada escenario respecto al anterior (y de D3 respecto a D0) en
      euros y en porcentaje, y número de pliegues en que mejora.
    - hiperparametros.csv: hiperparámetros elegidos en cada pliegue.
    - predicciones.csv: predicción de cada vivienda cuando estaba en el pliegue de evaluación.
    - figuras/02_mae_escenarios.png: MAE de cada pliegue y su media por escenario y modelo, con
      validación aleatoria y espacial.
    - versiones.csv: versiones de Python y de las librerías utilizadas en todo el trabajo, sistema
      operativo, procesador, hilos de ejecución y memoria del equipo.
    - prueba_tiempos.csv (solo con --prueba): tiempo de un pliegue de D3 por modelo y validación.

Uso:
    python modelizacion.py             experimento completo (continúa si se interrumpió)
    python modelizacion.py --prueba    un pliegue de D3 por modelo y validación, para medir tiempos
    python modelizacion.py --reiniciar borra los resultados anteriores y empieza de cero
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from analisis_exploratorio import AZUL, GRIS, formato_eje_es
from rutas import CONJUNTO_CSV, DIR_MODELIZACION, ESCENARIOS_JSON, PLIEGUES_CSV

SEMILLA = 2018
VALIDACIONES = ["aleatoria", "espacial"]
# Columna de datos/final/pliegues.csv con el pliegue de cada vivienda en cada validación
COLUMNA_PLIEGUE = {"aleatoria": "pliegue_aleatorio", "espacial": "pliegue_espacial"}
MODELOS = ["ridge", "random_forest", "xgboost"]
N_PLIEGUES_INTERNOS = 3
# La búsqueda de hiperparámetros usa como mucho estas viviendas del entrenamiento de cada pliegue;
# None usa todas. El modelo final siempre se entrena con todo el entrenamiento del pliegue.
MUESTRA_BUSQUEDA = 20_000
# Métrica con la que se eligen los hiperparámetros, en la escala de entrenamiento (logaritmo)
METRICA_BUSQUEDA = "neg_mean_absolute_error"
# Los dos modelos de árboles tienen el mismo presupuesto de búsqueda: este número de combinaciones
# elegidas al azar de su espacio, para que la comparación no dependa del esfuerzo de ajuste.
# Ridge, con un solo hiperparámetro, prueba todos los valores de su rejilla.
COMBINACIONES_ARBOLES = 15

# Variables con valores ausentes y cómo se imputan. Las que llevan indicador añaden una columna
# 0/1 "dato ausente", porque que falte el dato se asocia a precios distintos (ver
# 07_ausentes_precio.csv del análisis exploratorio).
IMPUTACION = {
    "FLOORCLEAN": ("median", True),
    "FLATLOCATIONID": ("most_frequent", True),
    "CADASTRALQUALITYID": ("median", False),
}
TRIMESTRES = [1, 2, 3, 4]
# La superficie entra en logaritmo, como el precio: así Ridge relaciona variaciones relativas de
# ambos, como un modelo hedónico log-log. Con la superficie tal cual, el precio crecería de forma
# exponencial con los metros y Ridge daría precios desorbitados en las viviendas más grandes. Los
# árboles no cambian, porque un corte en la superficie equivale a un corte en su logaritmo.
EN_LOGARITMO = ["CONSTRUCTEDAREA"]
# En Ridge, las tres variables del estado de conservación suman siempre 1: una queda como
# referencia (la más frecuente, segunda mano en buen estado)
REFERENCIA_ESTADO = "BUILTTYPEID_3"

REJILLAS = {
    "ridge": {"modelo__alpha": np.logspace(-3, 5, 17)},
    # El número de árboles se fija en 300 (crear_pipeline): más árboles nunca empeoran el modelo
    "random_forest": {
        "modelo__max_features": ["sqrt", 0.2, 0.33, 0.5, 0.7],
        "modelo__min_samples_leaf": [1, 2, 5, 10],
        "modelo__max_depth": [None, 20, 30],
        "modelo__max_samples": [None, 0.5, 0.8],
    },
    "xgboost": {
        "modelo__n_estimators": [300, 600, 900, 1200, 1500],
        "modelo__max_depth": [4, 6, 8, 10],
        "modelo__min_child_weight": [1, 5, 10],
        "modelo__subsample": [0.7, 0.85, 1.0],
        "modelo__colsample_bytree": [0.5, 0.7, 1.0],
        "modelo__reg_lambda": [1, 5, 10],
    },
}

COMPARACIONES = [("D0", "D1"), ("D1", "D2"), ("D2", "D3"), ("D0", "D3")]

METRICAS_CSV = DIR_MODELIZACION / "metricas_pliegues.csv"
DIR_PREDICCIONES_PARCIALES = DIR_MODELIZACION / "predicciones_parciales"
DIR_FIGURAS = DIR_MODELIZACION / "figuras"

# Nombre, color y desplazamiento horizontal de cada modelo en la figura del MAE
ESTILO_MODELOS = {"ridge": ("Ridge", GRIS, -0.1), "random_forest": ("Random Forest", AZUL, 0.0),
                  "xgboost": ("XGBoost", "#e08214", 0.1)}
ETIQUETAS_ESCENARIOS = {"D0": "D0\nVivienda", "D1": "D1\n+ Localización", "D2": "D2\n+ Socioeconómico",
                        "D3": "D3\n+ Entorno urbano"}


def cargar():
    """Conjunto final con sus pliegues y la definición de los escenarios."""
    conjunto = pd.read_csv(CONJUNTO_CSV, dtype={"ASSETID": str, "COD_DISTRITO": str})
    pliegues = pd.read_csv(PLIEGUES_CSV, dtype={"ASSETID": str, "COD_DISTRITO": str})
    conjunto = conjunto.merge(pliegues.drop(columns="COD_DISTRITO"), on="ASSETID", validate="one_to_one")
    escenarios = json.loads(ESCENARIOS_JSON.read_text(encoding="utf-8"))
    # Posición como binaria: 1 si la vivienda es interior (en idealista18, 1 exterior y 2 interior).
    # Es una recodificación fija, sin nada que ajustar, así que puede hacerse antes de los pliegues.
    conjunto["FLATLOCATIONID"] = conjunto["FLATLOCATIONID"].map({1: 0, 2: 1})
    return conjunto, escenarios


def preparacion(variables, lineal):
    """Transformaciones de un escenario, que el pipeline ajusta con el entrenamiento de cada pliegue."""
    imputadas = [v for v in IMPUTACION if v in variables]
    logaritmo = [v for v in EN_LOGARITMO if v in variables]
    resto = [v for v in variables if v not in imputadas + logaritmo and v != "TRIMESTRE"
             and not (lineal and v == REFERENCIA_ESTADO)]
    pasos = [(f"imputar_{v}", SimpleImputer(strategy=IMPUTACION[v][0], add_indicator=IMPUTACION[v][1]), [v])
             for v in imputadas]
    pasos.append(("logaritmo", FunctionTransformer(np.log, feature_names_out="one-to-one"), logaritmo))
    pasos.append(("trimestre", OneHotEncoder(categories=[TRIMESTRES], drop="first" if lineal else None,
                                             sparse_output=False), ["TRIMESTRE"]))
    pasos.append(("resto", "passthrough", resto))
    return ColumnTransformer(pasos, verbose_feature_names_out=False)


def crear_pipeline(modelo, variables):
    """Preparación y modelo en un solo pipeline; Ridge además estandariza las variables."""
    if modelo == "ridge":
        return Pipeline([("preparacion", preparacion(variables, lineal=True)),
                         ("escalado", StandardScaler()), ("modelo", Ridge())])
    if modelo == "random_forest":
        estimador = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEMILLA)
    else:
        estimador = XGBRegressor(learning_rate=0.05, tree_method="hist", n_jobs=-1, random_state=SEMILLA)
    return Pipeline([("preparacion", preparacion(variables, lineal=False)), ("modelo", estimador)])


def crear_busqueda(modelo, variables, validacion):
    """Búsqueda de hiperparámetros con pliegues internos del mismo tipo que la validación."""
    interna = (KFold(N_PLIEGUES_INTERNOS, shuffle=True, random_state=SEMILLA) if validacion == "aleatoria"
               else GroupKFold(N_PLIEGUES_INTERNOS))
    pipeline = crear_pipeline(modelo, variables)
    if modelo == "ridge":
        return GridSearchCV(pipeline, REJILLAS[modelo], cv=interna, scoring=METRICA_BUSQUEDA, refit=False)
    return RandomizedSearchCV(pipeline, REJILLAS[modelo], n_iter=COMBINACIONES_ARBOLES, cv=interna,
                              scoring=METRICA_BUSQUEDA, refit=False, random_state=SEMILLA)


def evaluar_pliegue(conjunto, variables, modelo, validacion, pliegue, objetivo, muestra):
    """Busca hiperparámetros, entrena con el resto de pliegues y evalúa en el pliegue indicado."""
    columna = COLUMNA_PLIEGUE[validacion]
    entrenamiento = conjunto[conjunto[columna] != pliegue]
    evaluacion = conjunto[conjunto[columna] == pliegue]

    inicio = time.perf_counter()
    busqueda_datos = entrenamiento
    if muestra and len(entrenamiento) > muestra:
        busqueda_datos = entrenamiento.sample(muestra, random_state=SEMILLA)
    busqueda = crear_busqueda(modelo, variables, validacion)
    busqueda.fit(busqueda_datos[variables], np.log(busqueda_datos[objetivo]),
                 groups=busqueda_datos["COD_DISTRITO"] if validacion == "espacial" else None)
    mejores = busqueda.best_params_

    final = crear_pipeline(modelo, variables).set_params(**mejores)
    final.fit(entrenamiento[variables], np.log(entrenamiento[objetivo]))
    prediccion = np.exp(final.predict(evaluacion[variables]))
    real = evaluacion[objetivo].to_numpy()

    metricas = {"n_entrenamiento": len(entrenamiento), "n_evaluacion": len(evaluacion),
                "mae": mean_absolute_error(real, prediccion),
                "rmse": float(np.sqrt(mean_squared_error(real, prediccion))),
                "r2": r2_score(real, prediccion),
                "segundos": round(time.perf_counter() - inicio, 1),
                "hiperparametros": json.dumps({k.replace("modelo__", ""): (v.item() if hasattr(v, "item") else v)
                                               for k, v in mejores.items()})}
    predicciones = pd.DataFrame({"ASSETID": evaluacion["ASSETID"].to_numpy(), "pliegue": pliegue,
                                 "prediccion": prediccion.round(0)})
    return metricas, predicciones


def tareas_pendientes(escenarios, n_pliegues):
    """Combinaciones de validación, escenario, modelo y pliegue que aún no tienen resultado."""
    hechas = set()
    if METRICAS_CSV.exists():
        previas = pd.read_csv(METRICAS_CSV)
        hechas = set(zip(previas.validacion, previas.escenario, previas.modelo, previas.pliegue))
    return [(v, e, m, p) for v in VALIDACIONES for e in escenarios for m in MODELOS
            for p in range(1, n_pliegues + 1) if (v, e, m, p) not in hechas]


def ejecutar_experimento(conjunto, escenarios, objetivo, muestra):
    """Recorre todas las combinaciones pendientes y guarda cada resultado al terminarlo."""
    n_pliegues = int(conjunto["pliegue_aleatorio"].max())
    pendientes = tareas_pendientes(escenarios, n_pliegues)
    total = len(VALIDACIONES) * len(escenarios) * len(MODELOS) * n_pliegues
    print(f"Combinaciones pendientes: {len(pendientes)} de {total}")
    DIR_PREDICCIONES_PARCIALES.mkdir(parents=True, exist_ok=True)

    for i, (validacion, escenario, modelo, pliegue) in enumerate(pendientes, start=1):
        metricas, predicciones = evaluar_pliegue(conjunto, escenarios[escenario], modelo, validacion,
                                                 pliegue, objetivo, muestra)
        fila = pd.DataFrame([{"validacion": validacion, "escenario": escenario, "modelo": modelo,
                              "pliegue": pliegue, **metricas}])
        fila.to_csv(METRICAS_CSV, mode="a", header=not METRICAS_CSV.exists(), index=False)
        parcial = DIR_PREDICCIONES_PARCIALES / f"{validacion}_{escenario}_{modelo}.csv"
        predicciones.to_csv(parcial, mode="a", header=not parcial.exists(), index=False)
        print(f"[{i}/{len(pendientes)}] {validacion} {escenario} {modelo} pliegue {pliegue}: "
              f"MAE {metricas['mae']:,.0f} €, R² {metricas['r2']:.3f} ({metricas['segundos']} s)")


def resumir(metricas):
    """Media y desviación de las métricas entre pliegues."""
    resumen = (metricas.groupby(["validacion", "escenario", "modelo"])[["mae", "rmse", "r2"]]
               .agg(["mean", "std"]))
    resumen.columns = [f"{m}_{'media' if e == 'mean' else 'desviacion'}" for m, e in resumen.columns]
    return resumen.reset_index()


def calcular_mejoras(metricas):
    """Mejora de un escenario sobre otro, pliegue a pliegue, en euros, en porcentaje y en R².

    Una mejora positiva significa menor error (MAE, RMSE) o mayor R² en el escenario con más
    información. La mejora en porcentaje se calcula sobre el error medio del escenario base, para
    que tenga siempre el mismo signo que la mejora en euros. Se cuenta en cuántos pliegues mejora
    el MAE.
    """
    tabla = metricas.pivot_table(index=["validacion", "modelo", "pliegue"], columns="escenario",
                                 values=["mae", "rmse", "r2"])
    filas = []
    for (validacion, modelo), grupo in tabla.groupby(level=["validacion", "modelo"]):
        for base, nuevo in COMPARACIONES:
            fila = {"validacion": validacion, "modelo": modelo, "comparacion": f"{base} → {nuevo}"}
            for m in ["mae", "rmse"]:
                diferencia = grupo[(m, base)] - grupo[(m, nuevo)]
                fila[f"mejora_{m}_euros"] = diferencia.mean()
                fila[f"mejora_{m}_pct"] = 100 * diferencia.mean() / grupo[(m, base)].mean()
            fila["mejora_r2"] = (grupo[("r2", nuevo)] - grupo[("r2", base)]).mean()
            fila["pliegues_con_mejora_mae"] = int((grupo[("mae", base)] > grupo[("mae", nuevo)]).sum())
            fila["pliegues"] = len(grupo)
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_hiperparametros(metricas):
    parametros = pd.json_normalize(metricas["hiperparametros"].map(json.loads).tolist())
    return pd.concat([metricas[["validacion", "escenario", "modelo", "pliegue"]].reset_index(drop=True),
                      parametros], axis=1)


def unir_predicciones():
    """Una columna de predicción por validación, escenario y modelo, una fila por vivienda."""
    tabla = None
    for parcial in sorted(DIR_PREDICCIONES_PARCIALES.glob("*.csv")):
        p = pd.read_csv(parcial, dtype={"ASSETID": str})[["ASSETID", "prediccion"]]
        p = p.rename(columns={"prediccion": parcial.stem})
        tabla = p if tabla is None else tabla.merge(p, on="ASSETID", how="outer", validate="one_to_one")
    return tabla


def figura_mae(metricas):
    """MAE de cada pliegue (puntos) y su media (línea) por escenario y modelo, en las dos validaciones."""
    escenarios = list(ETIQUETAS_ESCENARIOS)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for eje, validacion in zip(ejes, VALIDACIONES):
        for modelo, (nombre, color, desplazamiento) in ESTILO_MODELOS.items():
            datos = metricas[(metricas["validacion"] == validacion) & (metricas["modelo"] == modelo)]
            x = np.arange(len(escenarios)) + desplazamiento
            for posicion, escenario in zip(x, escenarios):
                mae = datos.loc[datos["escenario"] == escenario, "mae"]
                eje.scatter(np.full(len(mae), posicion), mae, s=12, color=color, alpha=0.35, linewidths=0)
            medias = datos.groupby("escenario")["mae"].mean().reindex(escenarios)
            eje.plot(x, medias, "-o", color=color, ms=6, label=nombre)
        eje.set_xticks(range(len(escenarios)), list(ETIQUETAS_ESCENARIOS.values()), fontsize=8)
        eje.set_title(f"Validación {validacion}")
        eje.yaxis.set_major_formatter(formato_eje_es)
        eje.grid(axis="y", alpha=0.3)
    ejes[0].set_ylabel("MAE (€)")
    ejes[0].legend(frameon=False)
    fig.text(0.5, -0.02, "Línea: media de los cinco pliegues. Puntos: MAE de cada pliegue.",
             ha="center", fontsize=9)
    fig.tight_layout()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / "02_mae_escenarios.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def procesador_y_memoria():
    """Nombre comercial del procesador y memoria RAM en GB (None si no se puede obtener)."""
    procesador, memoria = platform.processor() or None, None
    if platform.system() == "Windows":
        import ctypes
        import winreg

        clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        procesador = winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()

        class EstadoMemoria(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (campo, ctypes.c_ulonglong) for campo in ["ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                          "ullAvailPageFile", "ullTotalVirtual",
                                                          "ullAvailVirtual", "ullAvailExtendedVirtual"]]

        estado = EstadoMemoria(dwLength=ctypes.sizeof(EstadoMemoria))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
        memoria = estado.ullTotalPhys
    elif platform.system() == "Linux":
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            procesador = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), procesador)
        memoria = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return procesador, None if memoria is None else round(memoria / 1024 ** 3, 1)


def versiones():
    """Versiones del software utilizado en todo el trabajo y equipo en que se ejecuta la modelización."""
    procesador, memoria = procesador_y_memoria()
    return pd.DataFrame([("Python", platform.python_version()), ("pandas", pd.__version__),
                         ("NumPy", np.__version__), ("rdata", importlib.metadata.version("rdata")),
                         ("Shapely", importlib.metadata.version("shapely")),
                         ("Matplotlib", importlib.metadata.version("matplotlib")),
                         ("scikit-learn", sklearn.__version__), ("XGBoost", xgboost.__version__),
                         ("SHAP", shap.__version__), ("Sistema operativo", platform.platform()),
                         ("Procesador", procesador), ("Hilos de ejecución", os.cpu_count()),
                         ("Memoria RAM (GB)", memoria)],
                        columns=["elemento", "valor"])


def prueba_tiempos(conjunto, escenarios, objetivo, muestra):
    """Un pliegue de D3, el escenario más costoso, por modelo y validación, y estimación del total."""
    filas = []
    for validacion in VALIDACIONES:
        for modelo in MODELOS:
            metricas, _ = evaluar_pliegue(conjunto, escenarios["D3"], modelo, validacion, 1, objetivo, muestra)
            filas.append({"validacion": validacion, "modelo": modelo, "segundos": metricas["segundos"],
                          "mae": round(metricas["mae"]), "r2": round(metricas["r2"], 3),
                          "hiperparametros": metricas["hiperparametros"]})
            print(f"{validacion} {modelo}: {metricas['segundos']} s, MAE {metricas['mae']:,.0f} €, "
                  f"R² {metricas['r2']:.3f}")
    tabla = pd.DataFrame(filas)
    n_pliegues = int(conjunto["pliegue_aleatorio"].max())
    # Cota superior: todos los escenarios tardarían como D3, que es el que más variables tiene
    estimacion = tabla["segundos"].sum() * len(escenarios) * n_pliegues
    print(f"\nEstimación del experimento completo (cota superior): {estimacion / 3600:.1f} horas")
    return tabla


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prueba", action="store_true",
                        help="un pliegue de D3 por modelo y validación, para medir tiempos")
    parser.add_argument("--reiniciar", action="store_true", help="borra los resultados anteriores")
    parser.add_argument("--sin-muestra", action="store_true",
                        help="buscar hiperparámetros con todo el entrenamiento de cada pliegue")
    args = parser.parse_args()

    conjunto, definicion = cargar()
    objetivo, escenarios = definicion["objetivo"], definicion["escenarios"]
    muestra = None if args.sin_muestra else MUESTRA_BUSQUEDA
    DIR_MODELIZACION.mkdir(parents=True, exist_ok=True)
    versiones().to_csv(DIR_MODELIZACION / "versiones.csv", index=False)

    if args.prueba:
        prueba_tiempos(conjunto, escenarios, objetivo, muestra).to_csv(
            DIR_MODELIZACION / "prueba_tiempos.csv", index=False)
        return

    if args.reiniciar:
        METRICAS_CSV.unlink(missing_ok=True)
        shutil.rmtree(DIR_PREDICCIONES_PARCIALES, ignore_errors=True)

    ejecutar_experimento(conjunto, escenarios, objetivo, muestra)

    metricas = pd.read_csv(METRICAS_CSV)
    resumir(metricas).to_csv(DIR_MODELIZACION / "resumen.csv", index=False)
    calcular_mejoras(metricas).to_csv(DIR_MODELIZACION / "mejoras.csv", index=False, encoding="utf-8-sig")
    tabla_hiperparametros(metricas).to_csv(DIR_MODELIZACION / "hiperparametros.csv", index=False)
    unir_predicciones().to_csv(DIR_MODELIZACION / "predicciones.csv", index=False)
    figura_mae(metricas)

    pd.set_option("display.width", 200)
    print("\nResumen (media de los pliegues):")
    print(resumir(metricas)[["validacion", "escenario", "modelo", "mae_media", "rmse_media", "r2_media"]]
          .round({"mae_media": 0, "rmse_media": 0, "r2_media": 3}).to_string(index=False))
    print(f"\nFicheros guardados en {DIR_MODELIZACION}")


if __name__ == "__main__":
    main()
