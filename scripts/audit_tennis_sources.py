#!/usr/bin/env python3
"""
audit_tennis_sources.py — Auditoría de las fuentes de datos para un plugin de tenis.

OBJETIVO
--------
Antes de escribir una sola línea de `sports/tennis/`, demostrar con datos
reales que la única fuente gratuita con precios históricos por casa
(tennis-data.co.uk) tiene lo que un plugin necesita:

    1. RESULTADOS — ganador, perdedor, superficie, ronda, sets, fecha.
    2. PRECIOS    — cuota de cierre por casa, Pinnacle incluido.
    3. LIQUIDACIÓN — retiros y walkovers identificables.

La lección del Bloque 11 (soccer) fue que construimos 7.594 líneas y
descubrimos en la última tarea que el modelo no batía al mercado
(ROI −5,5% en 4.582 picks). Esa medición solo necesitaba resultados, un
modelo y cuotas históricas. Este script verifica que esos insumos están
disponibles — y mide, de paso, cuán lejos está un modelo ingenuo (Elo por
superficie) del cierre de Pinnacle. Ese número decide si el plugin de
tenis se construye o no.

ARQUITECTURA
------------
Script independiente, sin dependencias del paquete `core/` ni de
`sports/`. Se ejecuta antes de que exista el plugin, así que no puede
importar nada de él. Solo stdlib + pandas + requests: los .xlsx se leen
con zipfile + xml.etree porque openpyxl NO está instalado en esta máquina.

    descubrir enlaces en alldata.php (con caché en disco)
        ↓
    descargar un .xlsx por año y circuito (con caché en disco)
        ↓
    validar resultados, cobertura de precios y margen por casa
        ↓
    contrastar Pinnacle con su relevo (Betfair Exchange) en el solape
        ↓
    retiros / walkovers — qué se puede liquidar
        ↓
    calibración del favorito de Pinnacle
        ↓
    Elo por superficie sin look-ahead vs. log-loss/Brier del cierre
        ↓
    veredicto go / no-go

FUENTE
------
tennis-data.co.uk (Joseph Buchdahl). Un libro por año y circuito: ATP en
`/AAAA/AAAA.xlsx`, WTA en `/AAAAw/AAAA.xlsx`. Hoy los enlaces de
alldata.php (que redirige a data.php) cuelgan de una ruta con token, p. ej.

    HREF="hrjk-85HytOjkhth76j_ygh4jf7/2026/2026.xlsx"

La ruta sin token devuelve 404. Por eso el script DESCUBRE los enlaces
parseando la página en vez de codificar el token: si Buchdahl lo rota,
basta con --refresh.

Según notes.txt de la propia fuente, las cuotas son "the most recent
before play starts": a efectos prácticos, cuotas de cierre. Las columnas
Max/Avg vienen de Oddsportal; PSW/PSL son Pinnacle; BFEW/BFEL, Betfair
Exchange.

Alternativa descartada: el repo JeffSackmann/tennis_atp (y tennis_wta),
la referencia habitual para resultados de tenis, devuelve 404 (comprobado
el 2026-10-15). Además nunca tuvo cuotas. El script lo comprueba en vivo
a título informativo.

RIESGOS CONOCIDOS
-----------------
  * PINNACLE DESAPARECE EN 2026. PSW/PSL cubren >99% de 2013–2024, caen
    a partir de octubre de 2025 y están vacíos desde febrero de 2026
    (probablemente por el cierre de la API pública de Pinnacle en 2025;
    la fuente no lo explica). El relevo es Betfair Exchange (BFEW/BFEL),
    presente desde agosto de 2025.
    El script mide el solape para validar que BFE sirve de referencia
    sharp; si no sirviera, el plugin no tendría contra qué medir el CLV
    en adelante.
  * 2000–2012 se publican como .xls binario (BIFF/OLE), que no es un zip.
    Leerlos exige xlrd. Si no está instalado, esos años se omiten de las
    tablas descriptivas. El Elo y la evaluación usan SIEMPRE 2013+ (el
    primer .xlsx), para que el veredicto no dependa de esa dependencia
    opcional.
  * El archivo del año en curso crece cada semana. La caché no lo
    refresca sola: usar --refresh para medir la frescura real.
  * La columna Comment trae erratas ("Rrtired", "Walkoer"). Se
    normalizan antes de contar retiros y walkovers.
  * Retiros y walkovers se liquidan distinto según la casa (unas anulan,
    otras dan ganador al que avanza si se completó un set). Se excluyen
    de la evaluación de probabilidades; el Elo sí aprende de los retiros
    porque el partido se jugó.
  * El Elo no conoce el formato al mejor de 5 ni lesiones: es ingenuo a
    propósito. Si un modelo así ya queda cerca del cierre, el margen de
    mejora está en el modelado; si queda lejos, no.

USO
---
    python scripts/audit_tennis_sources.py
    python scripts/audit_tennis_sources.py --refresh      # ignora la caché
    python scripts/audit_tennis_sources.py --cache-dir X  # otra ruta

Código de salida 0 si todas las comprobaciones pasan, 1 si alguna falla.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("ERROR: falta pandas.  pip install pandas", file=sys.stderr)
    raise SystemExit(1)

try:
    import requests
except ImportError:  # pragma: no cover
    print("ERROR: falta requests.  pip install requests", file=sys.stderr)
    raise SystemExit(1)


# ── Configuración de la fuente ────────────────────────────────────────────────

_INDEX_URL = "https://www.tennis-data.co.uk/alldata.php"

# Copia local de la página índice. Guarda también la URL final tras las
# redirecciones, que es la base contra la que se resuelven los HREF.
_INDEX_CACHE = "alldata.json"

_SACKMANN_URL = "https://api.github.com/repos/JeffSackmann/tennis_atp"

# Primera temporada a auditar. 2010–2012 solo existen como .xls y se leen
# únicamente si xlrd está instalado (ver RIESGOS CONOCIDOS).
_FIRST_SEASON = 2010

# Primera temporada que alimenta el Elo: el primer .xlsx. Fija para que el
# veredicto sea el mismo con o sin xlrd.
_MODEL_FIRST_SEASON = 2013

# Firma de archivo esperada por extensión: .xlsx es un zip ("PK"), .xls un
# documento compuesto OLE.
_SIGNATURES = {"xlsx": b"PK", "xls": b"\xd0\xcf\x11\xe0"}

# Sufijo del directorio por circuito: /2026/ es ATP, /2026w/ es WTA.
_TOURS = {"ATP": "", "WTA": "w"}

# Un HREF útil termina en  AAAA[w]/AAAA.xls[x]  — lo que haya delante es el
# token. El año se repite y debe coincidir.
_LINK_PATTERN = re.compile(r"(?:^|/)(\d{4})(w?)/\1\.(xlsx|xls)$", re.IGNORECASE)
_HREF_PATTERN = re.compile(r'href\s*=\s*"([^"]+)"', re.IGNORECASE)

# Casas a medir: prefijo de columna → nombre legible. W = cuota del
# ganador, L = cuota del perdedor.
_BOOKS = {
    "PS": "Pinnacle",
    "Avg": "Media Oddsportal",
    "B365": "Bet365",
    "BFE": "Betfair Exchange",
    "Max": "Máximo Oddsportal",
}

# Un par de cuotas cuyo margen cae fuera de este rango es un error de
# captura (cuotas cruzadas, coma decimal), no un precio real.
_SANE_OVERROUND = (-0.02, 0.15)

# Partidos mínimos por circuito para considerar completa una temporada.
# Un año normal ronda los 2.400–2.700; 2020 (pandemia) quedó cerca de 1.300
# en ATP y por debajo de 1.100 en WTA.
_MIN_MATCHES_PER_SEASON = 1_000
_MIN_FULL_SEASONS = 10

_MIN_PINNACLE_COVERAGE = 0.90
_MIN_AVG_COVERAGE = 0.95

# Para aceptar Betfair Exchange como relevo de Pinnacle: probabilidades sin
# vig casi idénticas en los partidos donde coexisten, y el mismo log-loss.
_MIN_OVERLAP_MATCHES = 500
_MAX_RELAY_PROB_GAP = 0.02
_MAX_RELAY_LOGLOSS_GAP = 0.01

# Check que documenta una limitación conocida y no bloquea por sí solo si
# el relevo queda validado (mismo criterio que "precios por casa" en NBA).
_PINNACLE_CURRENT_CHECK = "Pinnacle sigue publicándose"
_RELAY_CHECK = "Betfair Exchange releva a Pinnacle"

# Días máximos entre la descarga del archivo del año en curso y su último
# partido. tennis-data actualiza aproximadamente cada semana.
_MAX_FRESHNESS_LAG_DAYS = 14

# Tramos de probabilidad sin vig del favorito para la calibración.
_CALIBRATION_EDGES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00)
_MAX_CALIBRATION_ERROR = 0.02

# ── Elo ingenuo ──
# Parámetros fijados a priori (FiveThirtyEight), sin ajustar sobre los
# datos: un modelo "ingenuo" que se tunea deja de serlo.
_ELO_START = 1500.0
_ELO_K_NUMERATOR = 250.0
_ELO_K_OFFSET = 5.0
_ELO_K_SHAPE = 0.4
# Peso del Elo de superficie en la mezcla con el Elo global.
_ELO_SURFACE_WEIGHT = 0.5

# Ventana de evaluación fija: con o sin los .xls, el Elo tiene al menos
# dos temporadas de rodaje antes de 2015.
_EVAL_FROM = 2015
_MIN_PRIOR_MATCHES = 20
# Partidos mínimos para mostrar la brecha de una temporada o un periodo.
_MIN_PERIOD_SAMPLE = 500

# Brecha máxima de log-loss frente a Pinnacle para considerar el problema
# abordable. En NBA el umbral era 15% sobre sigma; aquí no sirve porque el
# log-loss binario vive en un rango estrecho: una moneda al aire (ln 2 ≈
# 0,693) queda a ~20% del cierre. 5% exige que el ingenuo cubra al menos
# tres cuartas partes de la distancia entre la moneda y el mercado.
_MAX_LOGLOSS_GAP = 0.05
_MIN_ELO_ACCURACY = 0.60

# Umbral de valor para la simulación informativa de apuestas.
_EDGE_THRESHOLD = 0.05

_COMPLETED = "Completed"
_RETIRED = "Retired"
_WALKOVER = "Walkover"

# Desenlaces que actualizan el Elo: el partido se jugó. Awarded,
# Disqualified, Sched o Cancelled no dicen nada del nivel de juego.
_RATED_OUTCOMES = (_COMPLETED, _RETIRED)

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_DOC_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

# Origen de las fechas seriales de Excel (sistema 1900, con el bug del
# 29-feb-1900 ya absorbido).
_EXCEL_EPOCH = "1899-12-30"


# ── Utilidades de presentación ────────────────────────────────────────────────

def _header(text: str) -> None:
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _section(text: str) -> None:
    print()
    print(f"  {text}")
    print("  " + "─" * 66)


@dataclass
class Check:
    """Una comprobación con veredicto, para el resumen final."""

    name: str
    passed: bool
    detail: str


_checks: list[Check] = []


def _check(name: str, passed: bool, detail: str) -> bool:
    _checks.append(Check(name, passed, detail))
    mark = "OK  " if passed else "FALLA"
    print(f"    [{mark}] {name}: {detail}")
    return passed


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


# ── Descubrimiento de enlaces ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Link:
    """Un libro de tennis-data: un circuito, un año."""

    tour: str
    season: int
    extension: str
    href: str
    url: str

    @property
    def cache_name(self) -> str:
        return f"{self.tour.lower()}_{self.season}.{self.extension}"


def _fetch_index(cache_dir: Path, refresh: bool) -> tuple[str, str]:
    """
    Devuelve (html, url_final) de alldata.php, con caché en disco.

    alldata.php responde 301 hacia data.php. requests sigue la redirección;
    la URL final es la base para resolver los HREF relativos.
    """
    destination = cache_dir / _INDEX_CACHE
    if destination.exists() and not refresh:
        cached = json.loads(destination.read_text(encoding="utf-8"))
        print(f"    cache  {'alldata.php':<40} (bajado {cached['fetched_at']})")
        return cached["html"], cached["final_url"]

    response = requests.get(_INDEX_URL, timeout=60)
    response.raise_for_status()
    fetched_at = datetime.now().isoformat(timespec="seconds")
    destination.write_text(
        json.dumps(
            {
                "source": _INDEX_URL,
                "final_url": response.url,
                "fetched_at": fetched_at,
                "html": response.text,
            }
        ),
        encoding="utf-8",
    )
    print(f"    baja   {'alldata.php':<40} → {response.url}")
    return response.text, response.url


def discover_links(cache_dir: Path, refresh: bool) -> list[Link]:
    """Parsea la página índice y devuelve un Link por circuito y año."""
    html, base_url = _fetch_index(cache_dir, refresh)

    links: dict[tuple[str, int], Link] = {}
    for href in _HREF_PATTERN.findall(html):
        match = _LINK_PATTERN.search(href)
        if not match:
            continue
        season = int(match.group(1))
        tour = "WTA" if match.group(2).lower() == "w" else "ATP"
        extension = match.group(3).lower()
        links[(tour, season)] = Link(
            tour=tour,
            season=season,
            extension=extension,
            href=href,
            url=urljoin(base_url, href),
        )
    return sorted(links.values(), key=lambda link: (link.tour, link.season))


def audit_discovery(links: list[Link], current_season: int) -> None:
    _section("Enlaces descubiertos en alldata.php")

    prefixes = sorted({link.href.rsplit("/", 2)[0] for link in links if link.href.count("/") >= 2})
    print(f"    ruta con token   : {', '.join(prefixes) if prefixes else '(sin token)'}")
    for tour in _TOURS:
        seasons = [link.season for link in links if link.tour == tour]
        xlsx = [link.season for link in links if link.tour == tour and link.extension == "xlsx"]
        if seasons:
            print(
                f"    {tour}              : {min(seasons)}–{max(seasons)} "
                f"({len(seasons)} libros; .xlsx desde {min(xlsx) if xlsx else '—'})"
            )

    expected = {
        (tour, season)
        for tour in _TOURS
        for season in range(_FIRST_SEASON, current_season + 1)
    }
    found = {(link.tour, link.season) for link in links}
    missing = sorted(expected - found)

    print()
    _check(
        "enlaces ATP y WTA por año",
        not missing,
        f"{_FIRST_SEASON}–{current_season} completos en ambos circuitos"
        if not missing
        else f"faltan {missing}",
    )


# ── Descarga con caché ────────────────────────────────────────────────────────

def _download(link: Link, cache_dir: Path, refresh: bool) -> Path:
    """
    Descarga un libro y lo deja en la caché local.

    Misma política que el script de NBA: si el archivo existe y no se pidió
    --refresh, se reutiliza. Los años cerrados son inmutables; el del año
    en curso se refresca con --refresh.
    """
    destination = cache_dir / link.cache_name
    if destination.exists() and not refresh:
        size_mb = destination.stat().st_size / 1e6
        print(f"    cache  {destination.name:<40} {size_mb:6.2f} MB")
        return destination

    response = requests.get(link.url, timeout=120)
    response.raise_for_status()
    # Una ruta con token caducado puede devolver una página HTML de error
    # con código 200 según el CDN. La firma del archivo lo delata.
    if not response.content.startswith(_SIGNATURES[link.extension]):
        raise RuntimeError(
            f"{link.url} no devolvió un .{link.extension} "
            f"(¿token rotado? probar --refresh)"
        )
    destination.write_bytes(response.content)
    size_mb = len(response.content) / 1e6
    print(f"    baja   {destination.name:<40} {size_mb:6.2f} MB")
    return destination


# ── Lectura de .xlsx sin openpyxl ─────────────────────────────────────────────

def _column_index(reference: str) -> int:
    """'A1' → 0, 'AB12' → 27. Solo mira las letras iniciales."""
    index = 0
    for char in reference:
        if not char.isalpha():
            break
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    """Resuelve la ruta interna de la primera hoja vía workbook.xml.rels."""
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f"{_XLSX_NS}sheets/{_XLSX_NS}sheet")
    if sheet is not None:
        relation_id = sheet.get(f"{_DOC_REL_NS}id")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        for relation in relations.iter(f"{_PKG_REL_NS}Relationship"):
            if relation.get("Id") == relation_id:
                target = relation.get("Target", "")
                return target.lstrip("/") if target.startswith("/") else f"xl/{target}"
    return "xl/worksheets/sheet1.xml"


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Tabla de cadenas compartidas. Un <si> con texto enriquecido trae varios <t>."""
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.iter(f"{_XLSX_NS}t"))
        for item in root.iter(f"{_XLSX_NS}si")
    ]


def _cell_value(cell: ET.Element, shared: list[str]) -> object:
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{_XLSX_NS}t"))
    value = cell.find(f"{_XLSX_NS}v")
    if value is None or value.text is None:
        return None
    if kind == "s":
        return shared[int(value.text)]
    if kind == "e":
        return None
    if kind == "b":
        return value.text == "1"
    if kind in ("str", "d"):
        return value.text
    try:
        return float(value.text)
    except ValueError:
        return value.text


def _read_xlsx(path: Path) -> pd.DataFrame:
    """
    Lee la primera hoja de un .xlsx con zipfile + ElementTree.

    Un .xlsx es un zip de XML. Basta con la tabla de cadenas compartidas y
    la hoja: cada <c r="AB12" t="s"><v>7</v></c> es una celda cuyo valor es
    la cadena 7 de la tabla. Las fechas llegan como número serial de Excel
    y se convierten después. iterparse + clear mantiene la memoria acotada.
    """
    with zipfile.ZipFile(path) as archive:
        sheet_path = _first_sheet_path(archive)
        shared = _shared_strings(archive)
        rows: list[list[object]] = []
        with archive.open(sheet_path) as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag != f"{_XLSX_NS}row":
                    continue
                values: dict[int, object] = {}
                position = 0
                for cell in element.iter(f"{_XLSX_NS}c"):
                    reference = cell.get("r")
                    column = _column_index(reference) if reference else position
                    values[column] = _cell_value(cell, shared)
                    position = column + 1
                if values:
                    rows.append([values.get(i) for i in range(max(values) + 1)])
                element.clear()

    if not rows:
        return pd.DataFrame()
    header = [str(name).strip() if name is not None else "" for name in rows[0]]
    width = len(header)
    body = [(row + [None] * width)[:width] for row in rows[1:]]
    return pd.DataFrame(body, columns=header)


def _read_xls(path: Path) -> pd.DataFrame:
    """Los .xls binarios solo se leen con xlrd (dependencia opcional)."""
    return pd.read_excel(path, engine="xlrd")


def _has_xlrd() -> bool:
    return importlib.util.find_spec("xlrd") is not None


# ── Carga y normalización ─────────────────────────────────────────────────────

def _to_dates(series: pd.Series) -> pd.Series:
    """Fechas seriales de Excel (números) o texto, según el año."""
    numeric = pd.to_numeric(series, errors="coerce")
    from_serial = pd.to_datetime(numeric, unit="D", origin=_EXCEL_EPOCH, errors="coerce")
    from_text = pd.to_datetime(series.where(numeric.isna()), errors="coerce")
    return from_serial.fillna(from_text).dt.normalize()


def _normalize_comment(series: pd.Series) -> pd.Series:
    """
    Unifica las erratas del archivo ("Rrtired", "Walkoer") con su forma
    canónica. Los demás desenlaces (Awarded, Disqualified…) se conservan.
    """
    text = series.astype("string").str.strip()
    lowered = text.str.lower()
    text = text.mask(lowered.str.startswith("walk", na=False), _WALKOVER)
    return text.mask(lowered.str.contains("tired", na=False), _RETIRED)


def _normalize(raw: pd.DataFrame, tour: str, season: int) -> pd.DataFrame:
    """Columnas homogéneas entre años y circuitos, tipos numéricos, nombres limpios."""
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(subset=["Winner", "Loser"], how="any")

    out = pd.DataFrame(index=frame.index)
    out["tour"] = tour
    out["season"] = season
    out["date"] = _to_dates(frame["Date"])
    for column in ("Winner", "Loser"):
        out[column.lower()] = (
            frame[column].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        )
    for column in ("Surface", "Court", "Round", "Comment"):
        values = frame[column] if column in frame.columns else pd.Series(pd.NA, index=frame.index)
        out[column.lower()] = values.astype("string").str.strip()
    out["comment"] = _normalize_comment(out["comment"])
    # ATP usa "Series"; WTA usa "Tier". Misma idea: categoría del torneo.
    category = "Series" if "Series" in frame.columns else "Tier"
    out["category"] = frame.get(category, pd.Series(pd.NA, index=frame.index)).astype("string")

    numeric_columns = ["Best of", "WRank", "LRank", "Wsets", "Lsets"]
    for book in _BOOKS:
        numeric_columns += [f"{book}W", f"{book}L"]
    for column in numeric_columns:
        values = frame[column] if column in frame.columns else pd.Series(pd.NA, index=frame.index)
        out[column.replace(" ", "_").lower()] = pd.to_numeric(values, errors="coerce")
    return out.reset_index(drop=True)


def load_matches(
    links: list[Link], cache_dir: Path, refresh: bool, current_season: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Descarga y lee todos los libros desde _FIRST_SEASON.

    Devuelve los partidos y, por circuito, la marca de tiempo de descarga del
    archivo del año en curso (para medir la frescura contra ella).
    """
    _section("Descarga")
    xlrd_available = _has_xlrd()
    frames: list[pd.DataFrame] = []
    current_downloaded: dict[str, float] = {}
    skipped_xls: list[str] = []

    for link in links:
        if link.season < _FIRST_SEASON or link.season > current_season:
            continue
        if link.extension == "xls" and not xlrd_available:
            skipped_xls.append(f"{link.tour} {link.season}")
            continue
        path = _download(link, cache_dir, refresh)
        raw = _read_xls(path) if link.extension == "xls" else _read_xlsx(path)
        frames.append(_normalize(raw, link.tour, link.season))
        if link.season == current_season:
            current_downloaded[link.tour] = path.stat().st_mtime

    if skipped_xls:
        print(
            f"    omite  {len(skipped_xls)} libros .xls ({skipped_xls[0]} … "
            f"{skipped_xls[-1]}): falta xlrd   →  pip install xlrd"
        )
    if not frames:
        raise RuntimeError("No se pudo leer ningún libro.")
    return pd.concat(frames, ignore_index=True), current_downloaded


# ── Probabilidades implícitas ─────────────────────────────────────────────────

def _valid_pair(matches: pd.DataFrame, book: str) -> pd.Series:
    """Cuotas presentes en ambos lados y con un margen físicamente posible."""
    winner = matches[f"{book.lower()}w"]
    loser = matches[f"{book.lower()}l"]
    present = (winner > 1.0) & (loser > 1.0)
    overround = 1.0 / winner + 1.0 / loser - 1.0
    low, high = _SANE_OVERROUND
    return present & overround.between(low, high)


def _devig(winner_odds: pd.Series, loser_odds: pd.Series) -> pd.Series:
    """Probabilidad sin vig del ganador, por normalización proporcional."""
    winner_implied = 1.0 / winner_odds
    loser_implied = 1.0 / loser_odds
    return winner_implied / (winner_implied + loser_implied)


def _scores(p_winner: pd.Series) -> tuple[float, float, float]:
    """
    (log-loss, Brier, acierto) desde la probabilidad asignada al ganador real.

    Con dos resultados posibles, puntuar siempre desde el ganador equivale
    a la forma simétrica habitual: el Brier de los dos lados es idéntico.
    """
    clipped = p_winner.clip(1e-9, 1 - 1e-9)
    log_loss = float(-clipped.map(math.log).mean())
    brier = float(((1 - clipped) ** 2).mean())
    accuracy = float((clipped > 0.5).mean() + 0.5 * (clipped == 0.5).mean())
    return log_loss, brier, accuracy


# ── 1. Resultados ─────────────────────────────────────────────────────────────

def audit_results(
    matches: pd.DataFrame, current_season: int, current_downloaded: dict[str, float]
) -> None:
    _header("FUENTE 1 — RESULTADOS  (tennis-data.co.uk)")

    print()
    print(f"    partidos         : {len(matches):,}")
    print(
        f"    rango de fechas  : {matches.date.min().date()} → {matches.date.max().date()}"
    )

    played = matches[matches.comment != _WALKOVER]
    print()
    print("    por temporada (cobertura de precios sobre partidos jugados;")
    print("    PS = Pinnacle, BFE = Betfair Exchange, Avg = media de mercado):")
    print("            ────────── ATP ──────────   ────────── WTA ──────────")
    print("      año   partidos  PS    BFE   Avg   partidos  PS    BFE   Avg")
    per_tour: dict[str, pd.DataFrame] = {}
    for tour in _TOURS:
        subset = played[played.tour == tour]
        per_tour[tour] = pd.DataFrame(
            {
                "partidos": matches[matches.tour == tour].groupby("season").size(),
                "ps": _valid_pair(subset, "PS").groupby(subset.season).mean(),
                "bfe": _valid_pair(subset, "BFE").groupby(subset.season).mean(),
                "avg": _valid_pair(subset, "Avg").groupby(subset.season).mean(),
            }
        )
    seasons = sorted(matches.season.unique())
    for season in seasons:
        cells = []
        for tour in _TOURS:
            table = per_tour[tour]
            if season in table.index:
                row = table.loc[season]
                cells.append(
                    f"{int(row.partidos):>7} {row.ps * 100:4.0f}% "
                    f"{row.bfe * 100:4.0f}% {row.avg * 100:4.0f}%"
                )
            else:
                cells.append(f"{'—':>7} {'':>5} {'':>5} {'':>5}")
        print(f"      {season}  {cells[0]}   {cells[1]}")

    required = ["date", "winner", "loser", "surface", "round", "best_of", "comment"]
    empty = [column for column in required if matches[column].isna().mean() > 0.01]
    print()
    _check(
        "columnas clave pobladas",
        not empty,
        "fecha, jugadores, superficie, ronda, formato y comentario ≥99%"
        if not empty
        else f"vacías >1%: {empty}",
    )

    full = [season for season in seasons if season < current_season]
    thin = [
        f"{tour} {season}"
        for tour in _TOURS
        for season in full
        if per_tour[tour].partidos.get(season, 0) < _MIN_MATCHES_PER_SEASON
    ]
    _check(
        "temporadas completas",
        len(full) >= _MIN_FULL_SEASONS and not thin,
        f"{len(full)} temporadas cerradas con ≥{_MIN_MATCHES_PER_SEASON:,} "
        f"partidos por circuito"
        if not thin
        else f"por debajo de {_MIN_MATCHES_PER_SEASON:,}: {thin}",
    )

    duplicated = matches.duplicated(["tour", "date", "winner", "loser"]).sum()
    self_match = (matches.winner == matches.loser).sum()
    _check(
        "sin duplicados ni autopartidos",
        duplicated + self_match <= len(matches) * 0.001,
        f"{duplicated} duplicados, {self_match} con ganador = perdedor",
    )

    # El orientado ganador/perdedor es la base de toda la liquidación. Si en
    # un partido completado el ganador no tiene los sets necesarios, el
    # archivo tiene las columnas cruzadas.
    completed = matches[(matches.comment == _COMPLETED) & matches.wsets.notna()]
    sets_to_win = (completed.best_of // 2 + 1)
    consistent = ((completed.wsets == sets_to_win) & (completed.lsets < sets_to_win)).mean()
    _check(
        "ganador coherente con los sets",
        consistent > 0.98,
        f"{consistent * 100:.2f}% de los completados: Wsets = sets para ganar",
    )

    _section(f"Frescura del archivo {current_season}")
    stale: list[str] = []
    for tour in _TOURS:
        subset = matches[(matches.tour == tour) & (matches.season == current_season)]
        if subset.empty or tour not in current_downloaded:
            stale.append(f"{tour}: sin datos de {current_season}")
            print(f"    {tour}: sin datos de {current_season}")
            continue
        downloaded = datetime.fromtimestamp(current_downloaded[tour])
        last_match = subset.date.max()
        lag = (pd.Timestamp(downloaded).normalize() - last_match).days
        cache_age = (datetime.now() - downloaded).days
        print(
            f"    {tour}: {len(subset):,} partidos, último {last_match.date()}; "
            f"bajado {downloaded.date()} → retraso {lag} días"
            + (f"  (caché de hace {cache_age} días: --refresh)" if cache_age > 7 else "")
        )
        if lag > _MAX_FRESHNESS_LAG_DAYS:
            stale.append(f"{tour} {lag} días")
    print()
    _check(
        "archivo del año en curso al día",
        not stale,
        f"retraso ≤ {_MAX_FRESHNESS_LAG_DAYS} días respecto a la descarga"
        if not stale
        else f"retrasado: {stale}",
    )


# ── 2. Precios ────────────────────────────────────────────────────────────────

def audit_prices(matches: pd.DataFrame, current_season: int) -> None:
    _header("FUENTE 2 — PRECIOS POR CASA  (cierre, según notes.txt)")

    played = matches[matches.comment != _WALKOVER]
    print()
    print("    margen medio (overround) y cobertura, todas las temporadas:")
    print("                             ──── ATP ────    ──── WTA ────")
    print("      casa                   cobert. margen   cobert. margen")
    for book, name in _BOOKS.items():
        cells = []
        for tour in _TOURS:
            subset = played[played.tour == tour]
            valid = _valid_pair(subset, book)
            if valid.sum() == 0:
                cells.append(f"{'—':>7} {'—':>6}")
                continue
            priced = subset[valid]
            overround = 1 / priced[f"{book.lower()}w"] + 1 / priced[f"{book.lower()}l"] - 1
            cells.append(f"{_pct(valid.mean()):>7} {overround.mean() * 100:+5.2f}%")
        print(f"      {name:<22} {cells[0]}   {cells[1]}")
    print("      (Máximo ≈ 0: el mejor precio entre decenas de casas casi anula el")
    print("       margen. Betfair Exchange solo existe desde agosto de 2025.)")

    last_full = current_season - 1
    print()
    for tour in _TOURS:
        subset = played[(played.tour == tour) & (played.season < current_season)]
        coverage = _valid_pair(subset, "PS").mean()
        recent = subset[subset.season == last_full]
        recent_coverage = _valid_pair(recent, "PS").mean() if len(recent) else 0.0
        _check(
            f"{tour}: cobertura Pinnacle histórica",
            coverage >= _MIN_PINNACLE_COVERAGE and recent_coverage >= _MIN_PINNACLE_COVERAGE,
            f"{coverage * 100:.1f}% de las temporadas cerradas, "
            f"{recent_coverage * 100:.1f}% en {last_full}",
        )
    for tour in _TOURS:
        subset = played[played.tour == tour]
        coverage = _valid_pair(subset, "Avg").mean()
        _check(
            f"{tour}: cobertura media de mercado",
            coverage >= _MIN_AVG_COVERAGE,
            f"{coverage * 100:.1f}% de los partidos jugados",
        )

    present = played[(played.psw > 1) & (played.psl > 1)]
    anomalous = (~_valid_pair(present, "PS")).mean() if len(present) else 1.0
    priced = played[_valid_pair(played, "PS")]
    margin = (1 / priced.psw + 1 / priced.psl - 1).mean()
    _check(
        "precios Pinnacle sanos",
        anomalous < 0.01 and 0.01 <= margin <= 0.05,
        f"margen medio {margin * 100:.2f}%, {anomalous * 100:.2f}% de pares "
        f"fuera de [{_SANE_OVERROUND[0]:+.0%}, {_SANE_OVERROUND[1]:+.0%}]",
    )


# ── 3. Contraste: Pinnacle y su relevo ────────────────────────────────────────

def audit_pinnacle_relay(matches: pd.DataFrame, current_season: int) -> None:
    """
    Pinnacle deja de publicarse en tennis-data a comienzos de 2026. Mismo
    espíritu que el contraste de dos fuentes de cuotas del script de NBA:
    donde Pinnacle y Betfair Exchange coexisten, ¿dicen lo mismo?
    """
    _header("CONTRASTE — Pinnacle y su relevo, Betfair Exchange")

    played = matches[matches.comment != _WALKOVER].copy()
    month = played.date.dt.to_period("M")
    monthly = pd.DataFrame(
        {
            "ps": _valid_pair(played, "PS").groupby(month).mean(),
            "bfe": _valid_pair(played, "BFE").groupby(month).mean(),
        }
    )
    pinnacle_months = monthly.index[monthly.ps >= _MIN_PINNACLE_COVERAGE]
    exchange_months = monthly.index[monthly.bfe >= _MIN_PINNACLE_COVERAGE]
    current = monthly[monthly.index.year == current_season]
    weak = [str(period) for period in current.index[current.bfe < _MIN_PINNACLE_COVERAGE]]

    print()
    print(
        f"    último mes con Pinnacle ≥{_MIN_PINNACLE_COVERAGE:.0%}  : "
        f"{pinnacle_months.max() if len(pinnacle_months) else '—'}"
    )
    print(
        f"    primer mes con Betfair ≥{_MIN_PINNACLE_COVERAGE:.0%}   : "
        f"{exchange_months.min() if len(exchange_months) else '—'}"
    )
    print(
        f"    meses de {current_season} con Betfair <{_MIN_PINNACLE_COVERAGE:.0%} : "
        f"{', '.join(weak) or 'ninguno'}"
    )

    overlap = played[
        (played.comment == _COMPLETED)
        & _valid_pair(played, "PS")
        & _valid_pair(played, "BFE")
    ]
    this_season = played[played.season == current_season]
    pinnacle_now = _valid_pair(this_season, "PS").mean() if len(this_season) else 0.0
    exchange_now = _valid_pair(this_season, "BFE").mean() if len(this_season) else 0.0

    print()
    if overlap.empty:
        print("    sin partidos con ambas cuotas: no hay solape que medir")
        _check(
            _PINNACLE_CURRENT_CHECK,
            pinnacle_now >= _MIN_PINNACLE_COVERAGE,
            f"{pinnacle_now * 100:.1f}% de los partidos de {current_season}",
        )
        _check(_RELAY_CHECK, False, "sin solape con Pinnacle")
        return

    p_pinnacle = _devig(overlap.psw, overlap.psl)
    p_exchange = _devig(overlap.bfew, overlap.bfel)
    prob_gap = float((p_pinnacle - p_exchange).abs().mean())
    pinnacle_ll, pinnacle_brier, _ = _scores(p_pinnacle)
    exchange_ll, exchange_brier, _ = _scores(p_exchange)

    print(
        f"    solape (completados con ambas): {len(overlap):,} partidos, "
        f"{overlap.date.min().date()} → {overlap.date.max().date()}"
    )
    print(f"      |p Pinnacle − p Betfair| medio : {prob_gap * 100:.2f} pp")
    print(f"      log-loss  Pinnacle {pinnacle_ll:.4f}   Betfair {exchange_ll:.4f}")
    print(f"      Brier     Pinnacle {pinnacle_brier:.4f}   Betfair {exchange_brier:.4f}")

    print()
    _check(
        _PINNACLE_CURRENT_CHECK,
        pinnacle_now >= _MIN_PINNACLE_COVERAGE,
        f"{pinnacle_now * 100:.1f}% de los partidos de {current_season}",
    )
    _check(
        _RELAY_CHECK,
        len(overlap) >= _MIN_OVERLAP_MATCHES
        and prob_gap < _MAX_RELAY_PROB_GAP
        and abs(exchange_ll - pinnacle_ll) < _MAX_RELAY_LOGLOSS_GAP
        and exchange_now >= _MIN_PINNACLE_COVERAGE,
        f"{len(overlap):,} en común, Δp {prob_gap * 100:.2f} pp, Δlog-loss "
        f"{exchange_ll - pinnacle_ll:+.4f}; cubre {exchange_now * 100:.1f}% de {current_season}",
    )


# ── 4. Liquidación: retiros y walkovers ───────────────────────────────────────

def audit_settlement(matches: pd.DataFrame) -> None:
    _header("LIQUIDACIÓN — retiros y walkovers")

    print()
    print("    comentario                 ATP              WTA")
    comments = matches.comment.fillna("(vacío)")
    for comment in comments.value_counts().index:
        cells = []
        for tour in _TOURS:
            subset = comments[matches.tour == tour]
            count = int((subset == comment).sum())
            cells.append(f"{count:>7,} {_pct(count / max(len(subset), 1))}")
        print(f"      {comment:<22} {cells[0]}   {cells[1]}")

    walkovers = matches[matches.comment == _WALKOVER]
    retired = matches[matches.comment == _RETIRED]
    print()
    print(
        f"    walkovers con cuota Pinnacle : {_valid_pair(walkovers, 'PS').mean() * 100:5.1f}%"
        "   (sin partido: la casa anula)"
    )
    print(
        f"    retiros con cuota Pinnacle   : {_valid_pair(retired, 'PS').mean() * 100:5.1f}%"
        "   (regla de liquidación depende de la casa)"
    )

    known = matches.comment.isin([_COMPLETED, _RETIRED, _WALKOVER]).mean()
    irregular = matches.comment.isin([_RETIRED, _WALKOVER]).mean()
    print()
    _check(
        "desenlace identificable",
        known > 0.99,
        f"{known * 100:.2f}% con Completed / Retired / Walkover",
    )
    _check(
        "retiros + walkovers acotados",
        irregular < 0.06,
        f"{irregular * 100:.2f}% del total — se excluyen de la evaluación",
    )


# ── 5. Calibración del favorito de Pinnacle ───────────────────────────────────

def audit_calibration(matches: pd.DataFrame) -> None:
    _header("CALIBRACIÓN — favorito de Pinnacle al cierre")

    sample = matches[(matches.comment == _COMPLETED) & _valid_pair(matches, "PS")].copy()
    sample["p_winner"] = _devig(sample.psw, sample.psl)
    # Empates exactos (cuotas iguales) no tienen favorito: se excluyen para
    # no asignar la victoria al favorito por construcción.
    sample = sample[sample.p_winner != 0.5]
    sample["favorite_won"] = sample.p_winner > 0.5
    sample["p_favorite"] = sample.p_winner.where(sample.favorite_won, 1 - sample.p_winner)

    bins = pd.cut(sample.p_favorite, list(_CALIBRATION_EDGES), right=False)
    print()
    print("    prob. sin vig          ────── ATP ──────    ────── WTA ──────")
    print("      tramo            n    previsto real       n    previsto real")
    stats: dict[str, pd.DataFrame] = {}
    for tour in _TOURS:
        mask = sample.tour == tour
        stats[tour] = sample[mask].groupby(bins[mask], observed=False).agg(
            n=("favorite_won", "size"),
            predicted=("p_favorite", "mean"),
            actual=("favorite_won", "mean"),
        )
    for interval in stats["ATP"].index:
        cells = []
        for tour in _TOURS:
            row = stats[tour].loc[interval]
            if row.n == 0:
                cells.append(f"{0:>6}  {'—':>6}  {'—':>6}")
            else:
                cells.append(f"{int(row.n):>6}  {_pct(row.predicted)} {_pct(row.actual)}")
        label = f"{interval.left:.2f}–{interval.right:.2f}"
        print(f"      {label}  {cells[0]}   {cells[1]}")

    print()
    for tour in _TOURS:
        table = stats[tour][stats[tour].n > 0]
        error = float(
            ((table.predicted - table.actual).abs() * table.n).sum() / table.n.sum()
        )
        bias = float(
            ((table.actual - table.predicted) * table.n).sum() / table.n.sum()
        )
        _check(
            f"{tour}: cierre de Pinnacle calibrado",
            error < _MAX_CALIBRATION_ERROR,
            f"error medio {error * 100:.2f} pp (sesgo del favorito {bias * 100:+.2f} pp, "
            f"n={int(table.n.sum()):,})",
        )


# ── 6. Modelo ingenuo: Elo por superficie con barrera temporal ────────────────

def _k_factor(matches_played: int) -> float:
    """K decreciente con la experiencia (FiveThirtyEight): 250 / (n + 5)^0.4."""
    return _ELO_K_NUMERATOR / (matches_played + _ELO_K_OFFSET) ** _ELO_K_SHAPE


def _expected(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def elo_predictions(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Probabilidad Elo de que gane el ganador real, sin look-ahead.

    BARRERA TEMPORAL: los partidos se procesan por fecha. Todas las
    predicciones de un día se calculan ANTES de aplicar cualquier resultado
    de ese mismo día, así que un rating nunca incluye el partido que
    predice ni otro jugado en paralelo. Misma garantía que el acumulado
    menos la fila actual del script de NBA.

    Rating usado = mezcla del Elo global y el Elo de la superficie. Carpet
    se agrupa con Hard (pista rápida cubierta; casi extinta tras 2009).
    Solo actualizan los completados y los retiros (_RATED_OUTCOMES): en un
    retiro el partido existió y el perdedor no pudo terminarlo; un
    walkover, en cambio, no se jugó.
    """
    overall: dict[tuple[str, str], float] = {}
    overall_count: dict[tuple[str, str], int] = {}
    surface_rating: dict[tuple[str, str, str], float] = {}
    surface_count: dict[tuple[str, str, str], int] = {}

    ordered = matches.sort_values(["date", "tour"], kind="stable")
    surfaces = ordered.surface.fillna("Hard").replace({"Carpet": "Hard"})

    predictions: dict[int, tuple[float, int, int]] = {}
    pending: list[tuple[int, str, str, str, str]] = []
    current_date = None

    def blended(tour: str, surface: str, player: str) -> float:
        global_rating = overall.get((tour, player), _ELO_START)
        local_rating = surface_rating.get((tour, surface, player), _ELO_START)
        return (1 - _ELO_SURFACE_WEIGHT) * global_rating + _ELO_SURFACE_WEIGHT * local_rating

    def apply_updates() -> None:
        for _, tour, surface, winner, loser in pending:
            for ratings, counts, w_key, l_key in (
                (overall, overall_count, (tour, winner), (tour, loser)),
                (surface_rating, surface_count, (tour, surface, winner), (tour, surface, loser)),
            ):
                w_rating = ratings.get(w_key, _ELO_START)
                l_rating = ratings.get(l_key, _ELO_START)
                surprise = 1.0 - _expected(w_rating, l_rating)
                ratings[w_key] = w_rating + _k_factor(counts.get(w_key, 0)) * surprise
                ratings[l_key] = l_rating - _k_factor(counts.get(l_key, 0)) * surprise
                counts[w_key] = counts.get(w_key, 0) + 1
                counts[l_key] = counts.get(l_key, 0) + 1
        pending.clear()

    for index, tour, day, surface, winner, loser, comment in zip(
        ordered.index,
        ordered.tour,
        ordered.date,
        surfaces,
        ordered.winner,
        ordered.loser,
        ordered.comment,
    ):
        if day != current_date:
            apply_updates()
            current_date = day
        probability = _expected(
            blended(tour, surface, winner), blended(tour, surface, loser)
        )
        predictions[index] = (
            probability,
            overall_count.get((tour, winner), 0),
            overall_count.get((tour, loser), 0),
        )
        if comment in _RATED_OUTCOMES:
            pending.append((index, tour, surface, winner, loser))
    apply_updates()

    return pd.DataFrame.from_dict(
        predictions, orient="index", columns=["p_elo", "prior_winner", "prior_loser"]
    )


def _flat_stake_roi(sample: pd.DataFrame) -> tuple[int, float]:
    """
    ROI informativo: apostar 1 unidad al lado con valor Elo > umbral al
    precio de cierre de Pinnacle. Como mucho un lado por partido.
    """
    edge_winner = sample.p_elo * sample.psw - 1
    edge_loser = (1 - sample.p_elo) * sample.psl - 1
    bet_winner = (edge_winner > _EDGE_THRESHOLD) & (edge_winner >= edge_loser)
    bet_loser = (edge_loser > _EDGE_THRESHOLD) & ~bet_winner
    profit = (sample.psw - 1)[bet_winner].sum() - bet_loser.sum()
    bets = int(bet_winner.sum() + bet_loser.sum())
    return bets, (float(profit) / bets if bets else 0.0)


def audit_naive_model(matches: pd.DataFrame) -> None:
    _header("MODELO INGENUO — Elo por superficie, con barrera temporal")

    history = matches[matches.season >= _MODEL_FIRST_SEASON]
    frame = history.join(elo_predictions(history))

    eligible = frame[
        (frame.season >= _EVAL_FROM)
        & (frame.comment == _COMPLETED)
        & _valid_pair(frame, "PS")
    ]
    sample = eligible[
        (eligible.prior_winner >= _MIN_PRIOR_MATCHES)
        & (eligible.prior_loser >= _MIN_PRIOR_MATCHES)
    ].copy()
    sample["p_pinnacle"] = _devig(sample.psw, sample.psl)
    avg_valid = _valid_pair(sample, "Avg")
    sample["p_avg"] = _devig(sample.avgw, sample.avgl).where(avg_valid)

    # Partidos posteriores a Pinnacle: se miden contra Betfair Exchange, el
    # relevo validado en el contraste anterior.
    exchange = frame[
        (frame.comment == _COMPLETED)
        & _valid_pair(frame, "BFE")
        & (frame.prior_winner >= _MIN_PRIOR_MATCHES)
        & (frame.prior_loser >= _MIN_PRIOR_MATCHES)
    ].copy()
    exchange["p_exchange"] = _devig(exchange.bfew, exchange.bfel)

    print()
    print(
        f"    muestra: completados {_EVAL_FROM}+ con cierre Pinnacle y ambos "
        f"jugadores con ≥{_MIN_PRIOR_MATCHES} partidos previos"
    )
    print(
        f"    historia del Elo desde {_MODEL_FIRST_SEASON}; parámetros fijos: "
        f"K = {_ELO_K_NUMERATOR:.0f}/(n+{_ELO_K_OFFSET:.0f})^{_ELO_K_SHAPE}, "
        f"mezcla {1 - _ELO_SURFACE_WEIGHT:.0%} global / "
        f"{_ELO_SURFACE_WEIGHT:.0%} superficie"
    )

    coin = math.log(2)
    for tour in _TOURS:
        subset = sample[sample.tour == tour]
        eligible_tour = eligible[eligible.tour == tour]
        _section(
            f"{tour}  (n={len(subset):,}, {len(subset) / max(len(eligible_tour), 1) * 100:.0f}% "
            f"de los elegibles)"
        )
        if subset.empty:
            _check(f"{tour}: brecha abordable con modelado", False, "muestra vacía")
            continue

        elo_ll, elo_brier, elo_acc = _scores(subset.p_elo)
        pin_ll, pin_brier, pin_acc = _scores(subset.p_pinnacle)
        priced_avg = subset[subset.p_avg.notna()]
        avg_ll, avg_brier, avg_acc = _scores(priced_avg.p_avg)

        print("                               log-loss   Brier    acierto")
        print(f"      moneda al aire            {coin:.4f}   0.2500    50.0%")
        print(f"      Elo ingenuo               {elo_ll:.4f}   {elo_brier:.4f}   {_pct(elo_acc)}")
        print(
            f"      media de mercado (cierre) {avg_ll:.4f}   {avg_brier:.4f}   {_pct(avg_acc)}"
            f"   (n={len(priced_avg):,})"
        )
        print(f"      Pinnacle (cierre)         {pin_ll:.4f}   {pin_brier:.4f}   {_pct(pin_acc)}")

        gap = elo_ll / pin_ll - 1
        brier_gap = elo_brier / pin_brier - 1
        captured = (coin - elo_ll) / (coin - pin_ll) if coin > pin_ll else 0.0
        bets, roi = _flat_stake_roi(subset)
        print()
        print(
            f"    brecha a cerrar           = {elo_ll - pin_ll:.4f} log-loss "
            f"({gap * 100:+.1f}%)   Brier {brier_gap * 100:+.1f}%"
        )
        print(
            f"    habilidad sobre la moneda = el Elo captura {captured * 100:.0f}% "
            f"de la del cierre"
        )
        print(
            f"    ROI informativo           = {roi * 100:+.1f}% en {bets:,} apuestas "
            f"(valor Elo > {_EDGE_THRESHOLD:.0%} al cierre de Pinnacle)"
        )

        # ¿La brecha se cierra con más historia? Si se estrechara año a año,
        # el rodaje del Elo explicaría parte de ella; si oscila, es estructural.
        by_season = [
            f"{season % 100:02d} "
            f"{(_scores(group.p_elo)[0] / _scores(group.p_pinnacle)[0] - 1) * 100:+.1f}%"
            for season, group in subset.groupby("season")
            if len(group) >= _MIN_PERIOD_SAMPLE
        ]
        print("    brecha por temporada      = " + " · ".join(by_season[:6]))
        if len(by_season) > 6:
            print("                                " + " · ".join(by_season[6:]))

        relay = exchange[exchange.tour == tour]
        if len(relay) >= _MIN_PERIOD_SAMPLE:
            relay_elo_ll = _scores(relay.p_elo)[0]
            relay_ll = _scores(relay.p_exchange)[0]
            print(
                f"    contra Betfair Exchange   = {(relay_elo_ll / relay_ll - 1) * 100:+.1f}% "
                f"(n={len(relay):,}, {relay.date.min().date()} → "
                f"{relay.date.max().date()}; informativo)"
            )

        print()
        _check(
            f"{tour}: el Elo lleva señal",
            elo_acc >= _MIN_ELO_ACCURACY,
            f"acierta {elo_acc * 100:.1f}% de los ganadores",
        )
        _check(
            f"{tour}: brecha abordable con modelado",
            gap < _MAX_LOGLOSS_GAP,
            f"el Elo ingenuo está {gap * 100:.1f}% por detrás del cierre "
            f"(umbral {_MAX_LOGLOSS_GAP:.0%})",
        )


# ── Fuentes alternativas ──────────────────────────────────────────────────────

def report_alternatives() -> None:
    """
    Comprobación informativa, no entra en el veredicto: la referencia
    habitual (Sackmann) ya no está publicada.
    """
    _section("Fuentes alternativas (informativo)")
    try:
        status = requests.get(_SACKMANN_URL, timeout=15).status_code
        state = "404 — repositorio retirado" if status == 404 else f"HTTP {status}"
    except requests.RequestException as error:
        state = f"no verificable ({type(error).__name__})"
    print(f"    JeffSackmann/tennis_atp : {state}")
    print("    (sin cuotas en ningún caso; tennis-data es la única fuente con precios)")


# ── Veredicto ─────────────────────────────────────────────────────────────────

def verdict() -> int:
    _header("VEREDICTO")

    failed = [check for check in _checks if not check.passed]
    print()
    print(f"    comprobaciones : {len(_checks) - len(failed)}/{len(_checks)} OK")

    if not failed:
        print()
        print("    La fuente existe, trae precios de Pinnacle, liquida y está al día.")
        print("    Un Elo ingenuo ya queda cerca del cierre: el plugin de tenis puede")
        print("    construirse sobre tennis-data.co.uk.")
        return 0

    print()
    for check in failed:
        print(f"    FALLA  {check.name}: {check.detail}")

    # Que Pinnacle haya dejado de publicarse es una limitación conocida, no
    # un fallo del entorno: si Betfair Exchange quedó validado como relevo,
    # no tumba el código de salida por sí sola.
    relay_ok = any(check.name == _RELAY_CHECK and check.passed for check in _checks)
    blocking = [
        check
        for check in failed
        if not (check.name == _PINNACLE_CURRENT_CHECK and relay_ok)
    ]
    model_failures = [
        check for check in blocking if "brecha" in check.name or "señal" in check.name
    ]
    data_failures = [check for check in blocking if check not in model_failures]

    print()
    if not blocking:
        print("    Única falla: Pinnacle ya no se publica. Betfair Exchange cubre el")
        print("    año en curso y replica a Pinnacle en el solape: sirve de cierre.")
        print("    El plugin de tenis puede construirse sobre tennis-data.co.uk.")
        return 0

    if data_failures:
        print("    Hay fallas en los datos que bloquean. Revisar antes de construir.")
        return 1

    failing_tours = sorted({check.name.split(":")[0] for check in model_failures})
    passing_tours = [tour for tour in _TOURS if tour not in failing_tours]
    print("    Los datos sirven: resultados, precios y liquidación pasan.")
    if relay_ok:
        print("    Pinnacle ya no se publica; el cierre de referencia en adelante es")
        print("    Betfair Exchange, validado en el solape.")
    print(
        f"    El Elo ingenuo queda fuera del umbral en {', '.join(failing_tours)}"
        + (f" y dentro en {', '.join(passing_tours)}." if passing_tours else ".")
    )
    print("    Mismo patrón que soccer: no construir el plugin completo para un")
    print("    circuito sin antes demostrar un modelo que cierre su brecha.")
    return 1


# ── Entrada ───────────────────────────────────────────────────────────────────

def _configure_output() -> None:
    """
    La consola de Windows acepta UTF-8, pero al redirigir a un archivo
    Python usa cp1252 y los separadores "─" rompen con UnicodeEncodeError.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _configure_output()
    parser = argparse.ArgumentParser(
        description="Audita las fuentes de datos para un plugin de tenis."
    )
    parser.add_argument(
        "--cache-dir",
        default="data/tennis/raw",
        help="Dónde cachear las descargas (por defecto data/tennis/raw).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Vuelve a descargar aunque exista caché (índice y libros).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    current_season = date.today().year

    _header("AUDITORÍA DE FUENTES — TENIS")
    print()
    print("  Fuente : tennis-data.co.uk  (Joseph Buchdahl)")
    print("  Origen : resultados ATP/WTA + cuotas de cierre por casa (Pinnacle incl.)")
    print(f"  Caché  : {cache_dir.resolve()}")
    print("  Lector : .xlsx con zipfile + xml (stdlib); .xls con xlrd "
          + ("(instalado)" if _has_xlrd() else "(NO instalado → 2010–2012 se omiten)"))

    started = time.perf_counter()
    try:
        _section("Índice")
        links = discover_links(cache_dir, args.refresh)
        audit_discovery(links, current_season)
        matches, current_downloaded = load_matches(
            links, cache_dir, args.refresh, current_season
        )
    except Exception as error:
        print()
        print(f"  ERROR descargando las fuentes: {error}")
        return 1

    audit_results(matches, current_season, current_downloaded)
    audit_prices(matches, current_season)
    audit_pinnacle_relay(matches, current_season)
    audit_settlement(matches)
    audit_calibration(matches)
    audit_naive_model(matches)
    report_alternatives()
    code = verdict()
    print()
    print(f"  ({time.perf_counter() - started:.1f} s)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
