"""
Indicadores socioeconómicos y del entorno de los 21 distritos de Madrid (2018).

Fuente: "Panel de indicadores de distritos y barrios" del Ayuntamiento de Madrid
(ficheros datos/originales/ayuntamiento/20XX-indicadores-distritos). Cada fichero tiene una hoja por
distrito con filas = indicadores y columnas = ciudad, distrito y barrios.
En todas las hojas el distrito ocupa las columnas 3 (porcentaje o ratio) y
4 (valor absoluto o nivel); los indicadores con un único valor están en la 4
o, si esta está vacía, en la 3.

Como los paneles se publican con retraso, la información de 2018 está repartida
entre varios ficheros. Para cada indicador se indica de qué panel sale y qué
año debe figurar en su etiqueta (o en la del apartado), y el script avisa si
no coincide.

Población, hogares y estudios se toman a 01/01/2018 (Padrón): es la cifra
oficial de población "de 2018" y describe el distrito al empezar el periodo
de los anuncios, sin recoger cambios posteriores a los precios observados.

Salida (en datos/procesados/distritos/):
    - distritos_2018.csv / distritos_2018.xlsx: una fila por distrito.
    - diccionario_distritos_2018.csv: descripción, fuente y fecha de cada columna.

Uso:
    python distritos_2018.py
"""

import re
import warnings

import pandas as pd

from idealista18_madrid import DISTRITOS
from rutas import DIR_AYUNTAMIENTO, DIR_DISTRITOS, DISTRITOS_CSV

PANELES = {
    2019: DIR_AYUNTAMIENTO / "2019-indicadores-distritos-xls.xls",
    2020: DIR_AYUNTAMIENTO / "2020-indicadores-distritos-xls.xls",
    2021: DIR_AYUNTAMIENTO / "2021-indicadores-distritos-xlsx.xlsx",
    2022: DIR_AYUNTAMIENTO / "2022-indicadores-distritos-xlsx.xlsx",
}

# Nombre de la hoja de cada distrito (es igual en todos los paneles)
HOJAS = {cod: nombre.upper() for cod, nombre in DISTRITOS.items()}

COL_PCT, COL_VALOR = 3, 4

# (columna, panel, inicio de la etiqueta, tipo, buscar después de, año esperado, descripción)
#   tipo "valor": columna 4 o, si está vacía, la 3.  tipo "pct": columna 3.
#   "buscar después de" sirve para etiquetas repetidas (Hombres, Mujeres...).
INDICADORES = [
    # --- Territorio y población (Padrón 01/01/2018) ---
    ("superficie_ha", 2019, "Superficie (Ha.)", "valor", None, None, "Superficie del distrito (ha)"),
    ("densidad_hab_ha", 2019, "Densidad (hab./Ha.)", "valor", None, None, "Densidad de población (hab/ha), 01/01/2018"),
    ("poblacion", 2019, "Sexo de la población", "valor", None, "01/01/2018", "Población empadronada, 01/01/2018"),
    ("pct_mujeres", 2019, "Mujeres", "pct", "Sexo de la población", "01/01/2018", "Proporción de mujeres"),
    ("edad_media", 2019, "Edad media de la población", "valor", None, "01/01/2018", "Edad media de la población (años)"),
    ("pct_0_14", 2019, "De 0 a 14 años", "pct", None, "01/01/2018", "Proporción de población de 0 a 14 años"),
    ("pct_15_29", 2019, "De 15 a 29", "pct", None, "01/01/2018", "Proporción de población de 15 a 29 años"),
    ("pct_30_44", 2019, "De 30 a 44", "pct", None, "01/01/2018", "Proporción de población de 30 a 44 años"),
    ("pct_45_64", 2019, "De 45 a 64", "pct", None, "01/01/2018", "Proporción de población de 45 a 64 años"),
    ("pct_65_79", 2019, "De 65 a 79", "pct", None, "01/01/2018", "Proporción de población de 65 a 79 años"),
    ("pct_80_mas", 2019, "De 80 y +", "pct", None, "01/01/2018", "Proporción de población de 80 años o más"),
    ("prop_juventud", 2019, "Proporción de juventud", "valor", None, "01/01/2018", "Población de 0-15 años / población total"),
    ("prop_envejecimiento", 2019, "Proporción de envejecimiento", "valor", None, "01/01/2018", "Población de más de 65 años / población total"),
    ("prop_sobreenvejecimiento", 2019, "Proporción de sobreenvejecimiento", "valor", None, "01/01/2018", "Población de más de 80 años / población de más de 65"),
    ("prop_dependencia", 2019, "Proporción de dependencia", "valor", None, "01/01/2018", "(Población 0-15 + 65 y más) / población 16-64"),
    ("pct_extranjeros", 2019, "Personas con nacionalidad extranjera", "pct", None, "01/01/2018", "Proporción de población con nacionalidad extranjera"),
    ("prop_inmigrantes", 2019, "Proporción de inmigrantes", "valor", None, "01/01/2018", "Extranjeros de fuera de la UE y la OCDE / población total"),
    # --- Hogares (01/01/2018) ---
    ("hogares", 2019, "Nº total de hogares", "valor", None, "01-01-2018", "Número de hogares"),
    ("tamano_medio_hogar", 2019, "Tamaño medio del hogar", "valor", None, "01-01-2018", "Personas por hogar"),
    ("pct_hogares_mujer_sola_65", 2019, "Hogares con una mujer sola", "pct", None, "01-01-2018", "Hogares de una mujer sola mayor de 65 años / hogares"),
    ("pct_hogares_hombre_solo_65", 2019, "Hogares con un hombre solo", "pct", None, "01-01-2018", "Hogares de un hombre solo mayor de 65 años / hogares"),
    ("pct_hogares_monoparentales_mujer", 2019, "Hogares monoparentales: una mujer", "pct", None, "01-01-2018", "Hogares de una mujer adulta con menores / hogares"),
    ("pct_hogares_monoparentales_hombre", 2019, "Hogares monoparentales: un hombre", "pct", None, "01-01-2018", "Hogares de un hombre adulto con menores / hogares"),
    # --- Dinámica demográfica y esperanza de vida (2018) ---
    ("tasa_natalidad", 2020, "Tasa bruta de natalidad 2018", "valor", None, "2018", "Nacimientos por 1.000 habitantes, 2018"),
    ("tasa_crecimiento_demografico", 2020, "Tasa de crecimiento demográfico 2018", "valor", None, "2018", "Crecimiento de la población en 2018 (%)"),
    ("esperanza_vida_mujeres", 2020, "Esperanza de vida al nacer MUJERES (2018)", "valor", None, "2018", "Esperanza de vida al nacer de las mujeres (años), 2018"),
    ("esperanza_vida_hombres", 2020, "Esperanza de vida al nacer HOMBRES (2018)", "valor", None, "2018", "Esperanza de vida al nacer de los hombres (años), 2018"),
    # --- Renta y pensiones (2018) ---
    ("renta_media_hogar", 2022, "Renta neta media anual de los hogares", "valor", None, "2018", "Renta neta media anual de los hogares (€, Urban Audit), 2018"),
    ("pension_media_hombres", 2020, "Pensión media mensual del Distrito HOMBRES (2018)", "valor", None, "2018", "Pensión media mensual de los hombres (€), 2018"),
    ("pension_media_mujeres", 2020, "Pensión media mensual del Distrito MUJERES (2018)", "valor", None, "2018", "Pensión media mensual de las mujeres (€), 2018"),
    # --- Desempleo (SEPE, agosto 2018) ---
    ("parados_registrados", 2019, "Paro registrado", "valor", None, "2018", "Personas en paro registrado, agosto 2018"),
    ("tasa_paro", 2019, "Tasa absoluta de paro registrado (", "valor", None, "2018", "Paro registrado / población de 16 a 64 años, agosto 2018"),
    ("tasa_paro_mujeres", 2019, "Tasa absoluta de paro registrado MUJERES", "valor", None, None, "Tasa absoluta de paro registrado de las mujeres, agosto 2018"),
    ("tasa_paro_hombres", 2019, "Tasa absoluta de paro registrado HOMBRES", "valor", None, None, "Tasa absoluta de paro registrado de los hombres, agosto 2018"),
    ("pct_parados_larga_duracion", 2019, "Parados de larga duración", "pct", None, "2018", "Parados de larga duración / parados, agosto 2018"),
    ("pct_parados_con_prestacion", 2019, "Parados que SÍ perciben prestaciones", "pct", None, "2018", "Parados con prestación / parados, agosto 2018"),
    # --- Educación ---
    ("pct_sin_estudios", 2019, "No sabe leer ni escribir o sin estudios", "pct", "1.5.3.", "01/01/2018", "Población de 25 años o más sin estudios"),
    ("pct_primaria_incompleta", 2019, "Primaria incompleta", "pct", "1.5.3.", "01/01/2018", "Población de 25 años o más con primaria incompleta"),
    ("pct_estudios_obligatorios", 2019, "Bachiller Elemental", "pct", "1.5.3.", "01/01/2018", "Población de 25 años o más con Graduado Escolar/ESO/FP1"),
    ("pct_bachiller_fp2", 2019, "Formación profesional 2º grado", "pct", "1.5.3.", "01/01/2018", "Población de 25 años o más con Bachillerato/BUP/FP2"),
    ("pct_titulados_medios", 2019, "Titulados medios", "pct", "1.5.3.", "01/01/2018", "Población de 25 años o más con diplomatura o ingeniería técnica"),
    ("pct_estudios_superiores", 2019, "Estudios superiores", "pct", "1.5.3.", "01/01/2018", "Población de 25 años o más con licenciatura o estudios superiores"),
    ("pct_alumnos_publica", 2020, "En centros PÚBLICOS", "pct", "1.5.2.", "2017/18", "Alumnos en centros públicos / alumnos, curso 2017/18"),
    ("pct_alumnos_concertada", 2020, "En centros PRIVADOS CONCERTADOS", "pct", "1.5.2.", "2017/18", "Alumnos en centros concertados / alumnos, curso 2017/18"),
    ("pct_alumnos_privada", 2020, "En centros PRIVADOS SIN CONCIERTO", "pct", "1.5.2.", "2017/18", "Alumnos en centros privados / alumnos, curso 2017/18"),
    ("pct_alumnos_extranjeros", 2020, "Total alumnos extranjeros", "pct", "1.5.2.", "2017/18", "Alumnos extranjeros / alumnos, curso 2017/18"),
    # --- Salud (Estudio de Salud de la Ciudad de Madrid 2018) ---
    ("pct_sedentarismo", 2021, "Sedentarismo", "valor", None, "2018", "Población sedentaria"),
    ("pct_tabaco_diario", 2021, "Consumo de tabaco diario", "valor", None, "2018", "Población que fuma a diario"),
    ("pct_buena_salud", 2021, "Autopercepción de buen estado de salud", "valor", None, "2018", "Población que declara salud buena o muy buena"),
    ("pct_enfermedad_cronica", 2021, "Presencia de enfermedad crónica", "valor", None, "2018", "Población con alguna enfermedad crónica"),
    ("pct_hipertension", 2021, "Hipertensión arterial", "valor", None, "2018", "Población con hipertensión arterial"),
    ("pct_diabetes", 2021, "Diabetes Mellitus", "valor", None, "2018", "Población con diabetes"),
    ("pct_obesidad", 2021, "Personas con obesidad", "valor", None, "2018", "Población con obesidad"),
    ("pct_sobrepeso", 2021, "Personas con sobrepeso", "valor", None, "2018", "Población con sobrepeso"),
    ("pct_riesgo_salud_mental", 2021, "Probabilidad de padecer enfermedad mental", "valor", None, "2018", "Población con riesgo de mala salud mental (GHQ-12)"),
    ("personas_discapacidad", 2020, "Nº de personas con grado de discapacidad", "valor", None, "31/12/2018", "Personas con discapacidad reconocida, 31/12/2018"),
    # --- Vulnerabilidad y servicios sociales (2018) ---
    ("indice_vulnerabilidad", 2019, "Índice de vulnerabilidad territorial", "valor", None, "2018", "Índice de vulnerabilidad territorial (a mayor valor, más vulnerable), 2018"),
    ("ranking_vulnerabilidad", 2019, "Posición en el ranking", "valor", None, "2018", "Puesto en el ranking de vulnerabilidad (1 = más vulnerable), 2018"),
    ("solicitudes_rmi", 2021, "Solicitudes tramitadas de Renta Mínima de Inserción", "valor", None, "2018", "Solicitudes de Renta Mínima de Inserción tramitadas, 2018"),
    ("beneficiarios_prestaciones", 2021, ("Personas beneficiarias de prestaciones sociales", "Beneficiarios de prestaciones sociales"), "valor", None, "2018", "Beneficiarios de prestaciones sociales económicas, 2018"),
    # --- Vivienda (2018) ---
    ("valor_catastral_medio_pf", 2019, "Valor catastral medio de los bienes inmuebles: personas físicas", "valor", None, "2018", "Valor catastral medio de los inmuebles de personas físicas (€), 2018"),
    ("valor_catastral_medio_pj", 2019, "Valor catastral medio de los bienes inmuebles: personas jurídicas", "valor", None, "2018", "Valor catastral medio de los inmuebles de personas jurídicas (€), 2018"),
    # --- Entorno: residuos y deportes (2018) ---
    ("residuos_kg_hab_ano", 2020, "kg/habitante/año", "valor", None, "2018", "Recogida domiciliaria de residuos (kg por habitante y año), 2018"),
    ("centros_deportivos_municipales", 2020, "Centros deportivos Municipales", "valor", "1.14.5.", "2018", "Número de centros deportivos municipales, 2018"),
    ("instalaciones_deportivas_basicas", 2020, "Instalaciones deportivas básicas", "valor", "1.14.5.", "2018", "Número de instalaciones deportivas básicas, 2018"),
    ("superficie_deportiva_m2", 2020, "Superficie deportiva m2", "valor", "1.14.5.", "2018", "Superficie deportiva municipal (m2), 2018"),
    ("instalaciones_acceso_controlado", 2020, "Acceso controlado", "valor", "1.14.5.", "2018", "Instalaciones deportivas básicas de acceso controlado, 2018"),
    ("campos_futbol_11", 2020, "Campos de fútbol 11", "valor", "1.14.5.", "2018", "Campos de fútbol 11, 2018"),
    ("pistas_atletismo", 2020, "Pista de atletismo", "valor", "1.14.5.", "2018", "Pistas de atletismo, 2018"),
    ("piscinas_cubiertas", 2020, "Piscinas cubiertas", "valor", "1.14.5.", "2018", "Piscinas cubiertas, 2018"),
    ("piscinas_verano", 2020, "Piscinas de verano", "valor", "1.14.5.", "2018", "Piscinas de verano, 2018"),
    # --- Equipamientos municipales (inventario 2019) ---
    # No hay inventario de 2018: se usa el de 2019 como aproximación al stock de
    # 2018, porque apenas cambia (entre los paneles 2019 y 2020 la mayoría de
    # recuentos son idénticos en todos los distritos). Recuentos sin normalizar.
    ("centros_servicios_sociales", 2019, "Centros de Servicios Sociales", "valor", "1.14.", "2019", "Centros de Servicios Sociales, 2019"),
    ("centros_mayores", 2019, "Centros Municipales de Mayores", "valor", "1.14.", "2019", "Centros Municipales de Mayores, 2019"),
    ("centros_dia_alzheimer_fisicos", 2019, "Centros de Día de Alzheimer", "valor", "1.14.", "2019", "Centros de Día de Alzheimer y Físicos, 2019"),
    ("apartamentos_mayores", 2019, "Apartamentos Municipales para Mayores", "valor", "1.14.", "2019", "Apartamentos Municipales para Mayores, 2019"),
    ("residencias_mayores", 2019, "Residencias para personas Mayores", "valor", "1.14.", "2019", "Residencias municipales para personas mayores, 2019"),
    ("centros_apoyo_familias", 2019, "Centros de Apoyo a las Familias", "valor", "1.14.", "2019", "Centros de Apoyo a las Familias (CAF), 2019"),
    ("centros_atencion_infancia", 2019, "Centros de Atención a la Infancia", "valor", "1.14.", "2019", "Centros de Atención a la Infancia (CAI), 2019"),
    ("centros_dia_infancia", 2019, "Centro de Día de Atención a Niños", "valor", "1.14.", "2019", "Centros de Día de Atención a Niños y Niñas (3 a 12 años), 2019"),
    ("espacios_ocio_adolescentes", 2019, "Espacios de Ocio para Adolescentes", "valor", "1.14.", "2019", "Espacios de Ocio para Adolescentes (El Enredadero), 2019"),
    ("centros_adolescentes_jovenes", 2019, "Centros de Adolescentes y Jóvenes", "valor", "1.14.", "2019", "Centros de Adolescentes y Jóvenes (ASPA), 2019"),
    ("centros_personas_sin_hogar", 2019, "Centros para personas sin hogar", "valor", "1.14.", "2019", "Centros para personas sin hogar, 2019"),
    ("espacios_igualdad", 2019, "Espacios de Igualdad", "valor", "1.14.", "2019", "Espacios de Igualdad, 2019"),
    ("centros_salud_comunitaria", 2019, "Centros Municipales de Salud Comunitaria", "valor", "1.14.", "2019", "Centros Municipales de Salud Comunitaria (CMSC), 2019"),
    ("centros_atencion_adicciones", 2019, "Centros de Atención a las Adicciones", "valor", "1.14.", "2019", "Centros de Atención a las Adicciones (CAD y CCAD), 2019"),
    ("bibliotecas_municipales", 2019, "Nº de Bibliotecas Municipales", "valor", "1.14.", "2019", "Bibliotecas municipales, 2019"),
    ("bibliotecas_comunidad", 2019, "Nº de Bibliotecas Comunidad Madrid", "valor", "1.14.", "2019", "Bibliotecas de la Comunidad de Madrid, 2019"),
    ("centros_culturales", 2019, "Nº de Centros Culturales", "valor", "1.14.", "2019", "Centros culturales, 2019"),
    ("escuelas_infantiles_municipales", 2019, "Escuelas Infantiles Municipales", "valor", "1.14.", "2019", "Escuelas infantiles municipales, 2019"),
    ("escuelas_infantiles_publicas_cam", 2019, "Escuelas Infantiles Públicas CAM", "valor", "1.14.", "2018-2019", "Escuelas infantiles públicas de la Comunidad de Madrid, curso 2018-2019"),
    ("escuelas_infantiles_privadas", 2019, "Escuelas Infantiles Privadas", "valor", "1.14.", "2018-2019", "Escuelas infantiles privadas, curso 2018-2019"),
    ("colegios_publicos", 2019, "Colegios Públicos Infantil y Primaria", "valor", "1.14.", "2018-2019", "Colegios públicos de Infantil y Primaria, curso 2018-2019"),
    ("institutos_publicos", 2019, "Institutos Públicos de Educación Secundaria", "valor", "1.14.", "2018-2019", "Institutos públicos de Secundaria, curso 2018-2019"),
    ("colegios_privados_inf_pri", 2019, "Colegios Privados Inf. o Pri.", "valor", "1.14.", "2018-2019", "Colegios privados de Infantil y/o Primaria, curso 2018-2019"),
    ("colegios_privados_con_secundaria", 2019, "Colegios Privados Inf., Pri. y Sec.", "valor", "1.14.", "2018-2019", "Colegios privados con Secundaria, curso 2018-2019"),
    ("mercados_municipales", 2019, "Mercados Municipales", "valor", "1.14.", "2019", "Mercados municipales, 2019"),
    # --- Zonas verdes (no hay dato de 2018) ---
    # Se incluyen las dos mediciones disponibles, con metodologías distintas, y
    # el año en el nombre: 2016 (superficie total) y 2019 (parques de distrito
    # separados de los parques históricos, singulares y forestales).
    ("zonas_verdes_ha_2016", 2019, "Zonas verdes (Ha)", "valor", "1.14.6.", "2016", "Superficie de zonas verdes (ha), 2016"),
    ("zona_verde_m2_hab_2016", 2019, "Relación de zona verde (m2)/habitante", "valor", "1.14.6.", "2016", "Zona verde por habitante (m2/hab), 2016"),
    ("zonas_verdes_parques_distrito_ha_2019", 2020, "Superficie de zonas verdes y Parques de distrito (Ha)", "valor", "1.14.6.", "2019", "Superficie de zonas verdes y parques de distrito (ha), 2019"),
    ("zonas_verdes_parques_distrito_ha_10000hab_2019", 2020, "Superficie de zonas verdes y Parques de distrito (Ha) /", "valor", "1.14.6.", "2019", "Zonas verdes y parques de distrito (ha por 10.000 habitantes), 2019"),
    ("parques_historicos_forestales_ha_2019", 2020, "Superficie Parques Históricos, Singulares y Forestales por distrito", "valor", "1.14.6.", "2019", "Superficie de parques históricos, singulares y forestales (ha), 2019"),
    ("parques_historicos_forestales_ha_10000hab_2019", 2020, "Superficie Parques Históricos, Singulares y Forestales distrito /", "valor", "1.14.6.", "2019", "Parques históricos, singulares y forestales (ha por 10.000 habitantes), 2019"),
    # --- Seguridad (Policía Municipal, 2018) ---
    ("interv_policia_personas", 2019, "Relacionadas con las personas", "valor", "Intervenciones de la Policía Municipal", "2018", "Intervenciones de la Policía Municipal relacionadas con las personas, 2018"),
    ("interv_policia_armas", 2019, "Relacionadas con la tenencia de armas", "valor", "Intervenciones de la Policía Municipal", "2018", "Intervenciones por tenencia de armas, 2018"),
    ("interv_policia_patrimonio", 2019, "Relacionadas con el patrimonio", "valor", "Intervenciones de la Policía Municipal", "2018", "Intervenciones relacionadas con el patrimonio, 2018"),
    ("interv_policia_drogas", 2019, "Relacionadas con la tenencia y consumo de drogas", "valor", "Intervenciones de la Policía Municipal", "2018", "Intervenciones por tenencia y consumo de drogas, 2018"),
    ("accidentes_trafico", 2019, "Atestados/partes de accidentes de tráfico", "valor", "Intervenciones de la Policía Municipal", "2018", "Partes de accidentes de tráfico, 2018"),
    ("detenidos", 2019, "Total detenidos e investigados", "valor", "Detenidos e investigados", "2018", "Detenidos e investigados por la Policía Municipal, 2018"),
]

# Recuentos que además se expresan por cada 1.000 habitantes (población 01/01/2018)
POR_1000_HAB = [
    "personas_discapacidad", "solicitudes_rmi", "beneficiarios_prestaciones",
    "interv_policia_personas", "interv_policia_armas", "interv_policia_patrimonio",
    "interv_policia_drogas", "accidentes_trafico", "detenidos",
]

# Bloque de información de cada indicador, para los escenarios del TFM:
# "entorno_urbano" (forma urbana, dotaciones, zonas verdes, seguridad) y
# "socioeconomico" (todo lo demás: población, renta, empleo, educación, salud...).
ENTORNO_URBANO = {
    "superficie_ha", "densidad_hab_ha", "residuos_kg_hab_ano",
    "centros_deportivos_municipales", "instalaciones_deportivas_basicas", "superficie_deportiva_m2",
    "instalaciones_acceso_controlado", "campos_futbol_11", "pistas_atletismo",
    "piscinas_cubiertas", "piscinas_verano",
    "centros_servicios_sociales", "centros_mayores", "centros_dia_alzheimer_fisicos",
    "apartamentos_mayores", "residencias_mayores", "centros_apoyo_familias",
    "centros_atencion_infancia", "centros_dia_infancia", "espacios_ocio_adolescentes",
    "centros_adolescentes_jovenes", "centros_personas_sin_hogar", "espacios_igualdad",
    "centros_salud_comunitaria", "centros_atencion_adicciones", "bibliotecas_municipales",
    "bibliotecas_comunidad", "centros_culturales", "escuelas_infantiles_municipales",
    "escuelas_infantiles_publicas_cam", "escuelas_infantiles_privadas", "colegios_publicos",
    "institutos_publicos", "colegios_privados_inf_pri", "colegios_privados_con_secundaria",
    "mercados_municipales",
    "zonas_verdes_ha_2016", "zona_verde_m2_hab_2016", "zonas_verdes_parques_distrito_ha_2019",
    "zonas_verdes_parques_distrito_ha_10000hab_2019", "parques_historicos_forestales_ha_2019",
    "parques_historicos_forestales_ha_10000hab_2019",
    "interv_policia_personas", "interv_policia_armas", "interv_policia_patrimonio",
    "interv_policia_drogas", "accidentes_trafico", "detenidos",
}


def bloque(col):
    """Bloque (socioeconomico / entorno_urbano) de una columna de la tabla."""
    return "entorno_urbano" if col.removesuffix("_por_1000_hab") in ENTORNO_URBANO else "socioeconomico"


PATRON_APARTADO = re.compile(r"^\d+\.\d+")


def normalizar(texto):
    return re.sub(r"\s+", " ", str(texto)).strip().lower()


def leer_hoja(panel, hoja):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(PANELES[panel], sheet_name=hoja, header=None)


def buscar_fila(hoja, etiqueta, despues=None):
    """Primera fila cuya etiqueta empieza por `etiqueta` (tras la fila `despues`).

    `etiqueta` puede ser una tupla de alternativas, porque la redacción cambia
    en algunas hojas.
    """
    alternativas = tuple(map(normalizar, etiqueta if isinstance(etiqueta, tuple) else (etiqueta,)))
    etiquetas = hoja[0].map(normalizar)
    inicio = 0
    if despues:
        previas = etiquetas.index[etiquetas.str.startswith(normalizar(despues))]
        if previas.empty:
            raise KeyError(f"No se encuentra la fila de referencia '{despues}'")
        inicio = previas[0] + 1
    candidatas = etiquetas.index[(etiquetas.index >= inicio) & etiquetas.str.startswith(alternativas)]
    if candidatas.empty:
        raise KeyError(f"No se encuentra el indicador '{etiqueta}'")
    return candidatas[0]


def apartado(hoja, fila):
    """Título del apartado (1.2.1., 1.12., ...) en el que está una fila."""
    for i in range(fila, -1, -1):
        texto = str(hoja.iat[i, 0]).strip()
        if PATRON_APARTADO.match(texto):
            return texto
    return ""


def extraer(hoja, fila, tipo):
    pct, valor = (pd.to_numeric(hoja.iat[fila, c], errors="coerce") for c in (COL_PCT, COL_VALOR))
    if tipo == "pct":
        return pct
    return valor if pd.notna(valor) else pct


def construir_tabla():
    filas, avisos = [], []
    # Las hojas se leen una sola vez por panel y distrito
    hojas = {
        (panel, cod): leer_hoja(panel, HOJAS[cod])
        for panel in sorted({ind[1] for ind in INDICADORES})
        for cod in DISTRITOS
    }

    for cod, nombre in DISTRITOS.items():
        fila = {"COD_DISTRITO": cod, "DISTRITO": nombre}
        for col, panel, etiqueta, tipo, despues, fecha, _ in INDICADORES:
            hoja = hojas[(panel, cod)]
            cabecera = normalizar(hoja.iat[1, COL_PCT])
            if cabecera != normalizar(HOJAS[cod]):
                raise ValueError(f"La hoja {HOJAS[cod]} del panel {panel} es de {cabecera}")

            try:
                n = buscar_fila(hoja, etiqueta, despues)
            except KeyError as e:
                raise KeyError(f"{e.args[0]} en la hoja {HOJAS[cod]} del panel {panel}") from None
            # El año puede estar en la etiqueta, en el apartado o en la fila de referencia
            contexto = f"{hoja.iat[n, 0]} {apartado(hoja, n)}"
            if despues:
                contexto += " " + str(hoja.iat[buscar_fila(hoja, despues), 0])
            if fecha and fecha not in contexto:
                avisos.append(f"{col} ({nombre}, panel {panel}): no aparece '{fecha}' en '{hoja.iat[n, 0]}'")
            fila[col] = extraer(hoja, n, tipo)
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    for col in POR_1000_HAB:
        tabla[f"{col}_por_1000_hab"] = tabla[col] / tabla["poblacion"] * 1000
    return tabla, avisos


def diccionario():
    filas = [
        {"columna": "COD_DISTRITO", "descripcion": "Código oficial del distrito (01-21)", "panel": "", "fecha": ""},
        {"columna": "DISTRITO", "descripcion": "Nombre del distrito", "panel": "", "fecha": ""},
    ]
    for col, panel, _, _, _, fecha, desc in INDICADORES:
        filas.append({"columna": col, "descripcion": desc, "panel": PANELES[panel].name, "fecha": fecha or "2018"})
    for col in POR_1000_HAB:
        filas.append({"columna": f"{col}_por_1000_hab", "descripcion": f"{col} por cada 1.000 habitantes",
                      "panel": "calculada", "fecha": "2018"})
    dicc = pd.DataFrame(filas)
    dicc["bloque"] = [bloque(c) if i >= 2 else "" for i, c in enumerate(dicc["columna"])]
    return dicc


def main():
    tabla, avisos = construir_tabla()

    DIR_DISTRITOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DISTRITOS_CSV, index=False, encoding="utf-8-sig")
    diccionario().to_csv(DIR_DISTRITOS / "diccionario_distritos_2018.csv", index=False, encoding="utf-8-sig")
    try:
        tabla.to_excel(DIR_DISTRITOS / "distritos_2018.xlsx", sheet_name="distritos", index=False)
    except PermissionError:
        print("AVISO: distritos_2018.xlsx está abierto en otro programa; no se ha actualizado")

    print(f"Tabla de distritos: {tabla.shape[0]} filas x {tabla.shape[1]} columnas")
    vacios = tabla.columns[tabla.isna().any()].tolist()
    print(f"Columnas con valores vacíos: {vacios or 'ninguna'}")
    for aviso in avisos:
        print("AVISO:", aviso)
    print(f"Ficheros guardados en {DIR_DISTRITOS}")


if __name__ == "__main__":
    main()
