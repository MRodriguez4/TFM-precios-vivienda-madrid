# Precios de la vivienda en Madrid: aportación de la información contextual

Código del Trabajo Fin de Máster (UNIR) de Miguel Rodríguez Marcos.

El objetivo del trabajo es evaluar qué aporta la incorporación progresiva de información contextual sobre las viviendas de Madrid a los modelos de aprendizaje automático que predicen su precio. Para ello se comparan cuatro escenarios de información anidados:

| Escenario | Información | Variables explicativas |
| --- | --- | --- |
| D0 | Características de la vivienda | 31 |
| D1 | D0 + localización | 36 |
| D2 | D1 + contexto socioeconómico del distrito | 99 |
| D3 | D2 + entorno urbano del distrito | 153 |

Todos los escenarios usan las mismas 75.739 viviendas y la misma variable objetivo: el precio de oferta (`PRICE`, en euros).

## Fuentes de datos

- **[idealista18](https://paezha.github.io/idealista18/)**: anuncios de venta de viviendas en Madrid durante 2018, con sus características, datos catastrales del edificio, coordenadas y distancias al centro, al metro y a la Castellana. Rey-Blanco, D., Arbués, P., López, F. y Páez, A. (2024). *A geo-referenced micro-data set of real estate listings for Spain's three largest cities*. Environment and Planning B. https://doi.org/10.1177/23998083241242844. Licencia ODbL 1.0.
- **Panel de Indicadores de Distritos y Barrios** del Ayuntamiento de Madrid, ediciones 2019 a 2022: demografía, renta, empleo, educación, salud, vulnerabilidad, equipamientos, zonas verdes y seguridad de los 21 distritos.

Todos los indicadores se refieren a 2018, el año de los anuncios. La población, los hogares y el nivel de estudios se toman a 1 de enero de 2018. Los equipamientos municipales (inventario de 2019) y las zonas verdes (2016 y 2019) son las únicas excepciones, porque no hay dato de 2018. El año de cada indicador figura en `datos/procesados/distritos/diccionario_distritos_2018.csv`.

## Estructura del repositorio

```
├── idealista18_madrid.py     1. Descarga los anuncios y les asigna barrio y distrito
├── distritos_2018.py         2. Extrae los indicadores de 2018 de cada distrito
├── construir_conjunto.py     3. Integra, depura y define los escenarios D0-D3
├── rutas.py                  Rutas de la carpeta datos/
├── requirements.txt
└── datos/
    ├── originales/           Ficheros tal como se descargan (no se modifican)
    │   ├── ayuntamiento/     Paneles de indicadores 2018-2022
    │   └── idealista18/      Ficheros .rda del paquete idealista18
    ├── procesados/           Salidas intermedias (no se suben al repositorio)
    └── final/                Conjunto de estudio (no se sube al repositorio)
```

`datos/procesados/` y `datos/final/` no se versionan porque se regeneran con los scripts, y porque `conjunto_estudio.csv` (unos 115 MB) supera el límite de tamaño de GitHub.

## Instalación

Requiere Python 3.9 o superior. Desde la carpeta del proyecto:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

En Linux o macOS, el intérprete del entorno es `.venv/bin/python`.

## Uso

Los scripts se ejecutan en este orden:

```bash
.venv\Scripts\python idealista18_madrid.py
.venv\Scripts\python distritos_2018.py
.venv\Scripts\python construir_conjunto.py
```

1. **`idealista18_madrid.py`** descarga los ficheros `.rda` de idealista18 (si no están ya en `datos/originales/idealista18/`) y los convierte a CSV. Asigna a cada anuncio la zona de idealista y el distrito en el que se encuentra, a partir de sus coordenadas. Opciones: `--forzar-descarga` y `--sin-barrios`.
2. **`distritos_2018.py`** lee las hojas de cada distrito de los paneles del Ayuntamiento y genera una tabla de 21 distritos por 117 indicadores, con su diccionario. Para cada indicador comprueba que el año que figura en el Excel es el esperado.
3. **`construir_conjunto.py`** une los anuncios con los indicadores de su distrito, elimina las coordenadas erróneas y deja un único anuncio por inmueble (el más reciente). Guarda el conjunto de estudio y la lista de variables de cada escenario.

## Resultados

| Fichero | Contenido |
| --- | --- |
| `datos/final/conjunto_estudio.csv` | 75.739 viviendas con el precio, los identificadores y las 153 variables explicativas |
| `datos/final/escenarios.json` | Variable objetivo y lista de variables de cada escenario (D0 a D3) |
| `datos/procesados/distritos/distritos_2018.csv` | Indicadores de 2018 de los 21 distritos (también en `.xlsx`) |
| `datos/procesados/distritos/diccionario_distritos_2018.csv` | Descripción, fuente, fecha y bloque (socioeconómico o entorno urbano) de cada indicador |
| `datos/procesados/idealista18/madrid_sale_2018.csv` | Los 94.815 anuncios originales con su barrio y distrito |

Para cargar un escenario desde Python:

```python
import json
import pandas as pd

conjunto = pd.read_csv("datos/final/conjunto_estudio.csv")
escenarios = json.load(open("datos/final/escenarios.json", encoding="utf-8"))

X = conjunto[escenarios["escenarios"]["D2"]]
y = conjunto[escenarios["objetivo"]]
```
