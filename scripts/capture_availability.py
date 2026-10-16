#!/usr/bin/env python3
"""
capture_availability.py — Archiva la disponibilidad de jugadores PREVIA al partido.

OBJETIVO
--------
Capturar el informe de lesiones antes de cada jornada, para construir el
único dato que podría dar ventaja real sobre la línea de cierre.

La medición de la tarea 12.3 fue inequívoca: con el spread de cierre ya
en la ecuación, un modelo de eficiencia derivado de box scores aporta
cero información marginal (coeficiente +0,018, t = +0,31). Lo que falta
es la disponibilidad: una estrella que descansa mueve la línea 4-6
puntos.

Y ahí está el problema que este script resuelve: **esa información no se
puede reconstruir hacia atrás**. `hoopR-nba-data` no la conserva, ESPN no
publica histórico de informes de lesiones, y usar quién jugó DE VERDAD es
look-ahead — el mismo error que el clima observado durante el partido en
NFL, que infló ocho puntos el ROI de totales.

Capturada hacia adelante, en dos temporadas hay un modelo de
disponibilidad backtesteable sin look-ahead. Sin capturar, esa pregunta
no se puede ni plantear.

PRINCIPIO RECTOR: CRUDO PRIMERO, PARSEO DESPUÉS
------------------------------------------------
El artefacto primario de este script es el **JSON tal como lo devuelve
ESPN**, guardado literalmente. El CSV derivado es secundario.

La razón es de método. ESPN es una API no oficial cuya estructura puede
cambiar sin aviso, y escribir un parser contra un esquema no verificado
es cómo se cuelan los errores silenciosos. Con el crudo en disco:

  * si el parser falla hoy, el dato sigue ahí y se reprocesa mañana
  * si ESPN cambia el esquema, los archivos viejos siguen siendo válidos
  * si descubrimos que necesitábamos un campo que ignoramos, está

Por eso `capture()` guarda el crudo ANTES de intentar parsearlo, y un
fallo del parser no impide la captura. Es el mismo criterio que en
`core/odds/book_archive.py`, que archiva precios crudos en vez de
normalizados.

ARQUITECTURA
------------
    ESPN /sports/{path}/injuries
        ↓
    output/availability_archive/{sport}/raw/{YYYY-MM-DD}T{HHMM}Z.json
        │   ← artefacto primario, verbatim, nunca se sobrescribe
        ↓
    _parse_injuries()  (mejor esfuerzo, tolerante al esquema)
        ↓
    output/availability_archive/{sport}/{YYYY-MM}.csv
        ← índice derivado, reconstruible desde el crudo

COSTE
-----
Cero. ESPN no cobra, no pide clave y no consume créditos de The Odds
API. Puede correrse tantas veces al día como haga falta, lo que importa
porque el estado de un jugador cambia hasta una hora antes del tip-off.

USO
---
    # captura normal
    python scripts/capture_availability.py --sport nba

    # ver la estructura real que devuelve ESPN, sin guardar nada
    python scripts/capture_availability.py --sport nba --inspect

    # reprocesar los crudos ya guardados con el parser actual
    python scripts/capture_availability.py --sport nba --reparse

RIESGOS
-------
  * API no oficial: puede cambiar o desaparecer sin aviso. El modo
    `--inspect` existe para detectarlo pronto, y el crudo para que un
    cambio no invalide lo ya capturado.
  * El parser es de mejor esfuerzo. Si el esquema real difiere del
    esperado, el CSV saldrá pobre o vacío — y el crudo seguirá completo.
    Ejecutar `--inspect` una vez y ajustar el parser con la estructura
    verificada es el paso siguiente, no una suposición.
  * ESPN marca estado (Out, Questionable, ...) pero no minutos
    esperados. Convertir estado en impacto sobre la línea es modelado,
    no captura, y no se hace aquí.

MEJORAS FUTURAS
---------------
  * Cruzar con `player_box` histórico para ponderar cada ausencia por
    los minutos que ese jugador suele jugar. Eso convierte una lista de
    nombres en un ajuste en puntos.
  * Cuando un segundo consumidor necesite añadir a un CSV mensual, el
    apéndice de este script y el de `book_archive.py` deben promoverse a
    un `core/storage/monthly_csv.py` común. Con dos consumidores y ~25
    líneas compartidas todavía no compensa; con tres, sí.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import requests
except ImportError:  # pragma: no cover
    print("ERROR: falta requests.  pip install requests", file=sys.stderr)
    raise SystemExit(1)


# ── Rutas de ESPN por deporte ─────────────────────────────────────────────────
#
# El patrón del endpoint es idéntico para todos los deportes: solo cambia
# el segmento {liga}/{deporte}. Por eso una tabla sirve para los cuatro
# y no hace falta un módulo por deporte.

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

_SPORT_PATHS: dict[str, str] = {
    "nba": "basketball/nba",
    "nfl": "football/nfl",
    "nhl": "hockey/nhl",
    "mlb": "baseball/mlb",
    "wnba": "basketball/wnba",
}

_DEFAULT_BASE_DIR = "output/availability_archive"
_HTTP_TIMEOUT = 30

# ESPN rechaza peticiones sin User-Agent de navegador en algunos
# endpoints. No es evasión: es el encabezado que su propio sitio envía.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/json",
}

FIELDNAMES: tuple[str, ...] = (
    "captured_at",
    "sport",
    "team_id",
    "team_abbreviation",
    "team_display_name",
    "athlete_id",
    "athlete_name",
    "athlete_position",
    "status",
    "injury_type",
    "injury_detail",
    "injury_date",
)


# ── Descarga ──────────────────────────────────────────────────────────────────

def fetch_injuries(sport_path: str) -> Any:
    """
    Descarga el informe de lesiones de toda la liga en una llamada.

    Lanza si la petición falla: a diferencia del archivo de cuotas, aquí
    no hay un pipeline que proteger — este script es la operación
    completa, y un fallo debe verse en el código de salida.
    """
    url = f"{_ESPN_BASE}/{sport_path}/injuries"
    response = requests.get(url, headers=_HEADERS, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


# ── Persistencia del crudo ────────────────────────────────────────────────────

def save_raw(payload: Any, sport: str, base_dir: str, moment: datetime) -> str:
    """
    Guarda la respuesta verbatim. Nunca sobrescribe.

    El nombre lleva el instante de captura al minuto, así que dos
    capturas del mismo día conviven: la secuencia de informes a lo largo
    de la jornada ES el dato interesante — un jugador que pasa de
    Questionable a Out dos horas antes del partido mueve la línea.
    """
    stamp = moment.strftime("%Y-%m-%dT%H%MZ")
    directory = os.path.join(base_dir, sport, "raw")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{stamp}.json")

    # Si ya existe una captura de este minuto, añadir un sufijo en vez
    # de pisarla.
    if os.path.exists(path):
        suffix = 2
        while os.path.exists(path):
            path = os.path.join(directory, f"{stamp}-{suffix}.json")
            suffix += 1

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    return path


# ── Parseo de mejor esfuerzo ──────────────────────────────────────────────────

def _first(container: Any, *keys: str) -> Any:
    """Primer valor presente entre varias claves posibles."""
    if not isinstance(container, dict):
        return None
    for key in keys:
        value = container.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _as_text(value: Any) -> str:
    """
    Aplana a texto, tolerando que ESPN anide lo que en otros sitios es
    plano (un 'status' que a veces es str y a veces {'name': ...}).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("displayName", "name", "description", "abbreviation",
                    "detail", "type", "text", "shortComment"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        return ""
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value)


def _walk_teams(payload: Any) -> list[dict]:
    """
    Localiza la lista de equipos sin asumir una sola forma.

    ESPN ha servido este endpoint con la lista en la raíz, bajo
    'injuries', y bajo 'items'. En vez de fijar una, se prueban las
    conocidas y, si ninguna encaja, se recorre en busca del primer
    nivel que parezca una lista de equipos con jugadores.
    """
    for candidate in (payload,
                      _first(payload, "injuries", "items", "teams", "entries")):
        if isinstance(candidate, list) and candidate:
            if any(isinstance(item, dict) for item in candidate):
                return [item for item in candidate if isinstance(item, dict)]
    return []


def parse_injuries(payload: Any, sport: str, moment: datetime) -> list[dict]:
    """
    Convierte la respuesta en filas planas, de la forma más tolerante
    posible.

    Deliberadamente no valida ni lanza: si la estructura no es la
    esperada devuelve lo que haya podido extraer, incluso una lista
    vacía. El crudo ya está en disco, así que una extracción pobre es
    recuperable; abortar la captura por un esquema inesperado no lo
    sería.
    """
    rows: list[dict] = []
    captured = moment.isoformat(timespec="seconds").replace("+00:00", "Z")

    for team_entry in _walk_teams(payload):
        team = team_entry.get("team") if isinstance(
            team_entry.get("team"), dict
        ) else team_entry

        team_id = _as_text(_first(team, "id", "uid"))
        team_abbr = _as_text(_first(team, "abbreviation", "shortDisplayName"))
        team_name = _as_text(_first(team, "displayName", "name", "location"))

        entries = _first(team_entry, "injuries", "items", "entries", "athletes")
        if not isinstance(entries, list):
            continue

        for item in entries:
            if not isinstance(item, dict):
                continue
            athlete = item.get("athlete")
            if not isinstance(athlete, dict):
                athlete = {}

            position = athlete.get("position")
            injury_detail = _first(item, "details", "detail")

            rows.append({
                "captured_at":       captured,
                "sport":             sport,
                "team_id":           team_id,
                "team_abbreviation": team_abbr,
                "team_display_name": team_name,
                "athlete_id":        _as_text(_first(athlete, "id", "uid")),
                "athlete_name":      _as_text(
                    _first(athlete, "displayName", "fullName", "shortName")
                ),
                "athlete_position":  _as_text(position),
                "status":            _as_text(
                    _first(item, "status", "type", "injuryStatus")
                ),
                "injury_type":       _as_text(
                    _first(item, "type", "shortComment")
                ),
                "injury_detail":     _as_text(injury_detail),
                "injury_date":       _as_text(_first(item, "date")),
            })

    return rows


def append_csv(rows: list[dict], sport: str, base_dir: str,
               moment: datetime) -> str | None:
    """Añade al CSV mensual. Devuelve la ruta, o None si no había filas."""
    if not rows:
        return None
    period = moment.strftime("%Y-%m")
    directory = os.path.join(base_dir, sport)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{period}.csv")

    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return path


# ── Inspección del esquema ────────────────────────────────────────────────────

def describe(payload: Any, indent: int = 0, max_depth: int = 7) -> None:
    """
    Imprime la forma real de la respuesta.

    Existe porque escribir un parser contra un esquema no verificado es
    cómo se introducen errores silenciosos. Una ejecución de `--inspect`
    sustituye cualquier suposición sobre la estructura de ESPN.

    Los escalares se imprimen en la misma línea de su clave y solo se
    desciende a contenedores. Dos versiones anteriores cortaban justo
    en `injuries[].athlete`: la primera porque recursaba también en los
    escalares, la segunda porque sumaba 2 al indent por cada nivel de
    clave y agotaba la profundidad al doble de velocidad.
    """
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}[mas profundo, cortado]")
        return

    if isinstance(payload, dict):
        print(f"{pad}dict({len(payload)}): {', '.join(list(payload)[:14])}")
        for key, value in list(payload.items())[:14]:
            if isinstance(value, (dict, list)):
                print(f"{pad}  .{key}:")
                # Un nivel por nivel real de anidamiento. Sumar 2 aquí
                # gastaba el presupuesto de profundidad al doble de
                # velocidad y cortaba en `athlete`, que es justo el
                # nodo que hay que ver.
                describe(value, indent + 1, max_depth)
            else:
                print(f"{pad}  .{key} = {_preview(value)}")

    elif isinstance(payload, list):
        print(f"{pad}list[{len(payload)}]")
        if payload:
            # Solo el primer elemento: en estas respuestas todos
            # comparten forma, y volcar 30 equipos no añade nada.
            describe(payload[0], indent + 1, max_depth)

    else:
        print(f"{pad}{_preview(payload)}")


def _preview(value: Any) -> str:
    """Representación corta de un escalar, con su tipo."""
    text = repr(value)
    if len(text) > 64:
        text = text[:61] + "...'"
    return f"{type(value).__name__} {text}"


# ── Reproceso ─────────────────────────────────────────────────────────────────

def reparse(sport: str, base_dir: str) -> int:
    """
    Vuelve a parsear todos los crudos guardados con el parser actual.

    Es la razón de ser del principio crudo-primero: cuando el parser
    mejore, el histórico entero se reconstruye sin haber perdido nada.
    Reescribe los CSV mensuales desde cero para no duplicar filas.
    """
    raw_dir = os.path.join(base_dir, sport, "raw")
    if not os.path.isdir(raw_dir):
        print(f"  No hay crudos en {raw_dir}")
        return 2

    files = sorted(Path(raw_dir).glob("*.json"))
    if not files:
        print(f"  No hay crudos en {raw_dir}")
        return 2

    # Borrar los CSV derivados: son reconstruibles por definición.
    for stale in Path(os.path.join(base_dir, sport)).glob("*.csv"):
        stale.unlink()

    total = 0
    for path in files:
        stem = path.stem.split("-")[0:4]
        try:
            moment = datetime.strptime(
                path.stem[:16], "%Y-%m-%dT%H%MZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            moment = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"  omite {path.name}: {error}")
            continue
        rows = parse_injuries(payload, sport, moment)
        append_csv(rows, sport, base_dir, moment)
        total += len(rows)
        print(f"  {path.name}: {len(rows)} filas")

    print(f"\n  Reprocesados {len(files)} crudos, {total} filas en total.")
    return 0


# ── Entrada ───────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archiva la disponibilidad de jugadores previa al partido."
    )
    parser.add_argument("--sport", choices=sorted(_SPORT_PATHS), required=True)
    parser.add_argument("--base-dir", default=_DEFAULT_BASE_DIR)
    parser.add_argument(
        "--inspect", action="store_true",
        help="Imprime la estructura real de la respuesta y no guarda nada.",
    )
    parser.add_argument(
        "--reparse", action="store_true",
        help="Reprocesa los crudos guardados con el parser actual.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, fetcher=None) -> int:
    """
    `fetcher` permite inyectar la descarga en los tests, para ejercitar
    guardado, parseo y CSV sin tocar la red.

    Códigos de salida
    -----------------
        0  captura correcta
        1  fallo de red o de escritura
        2  la respuesta llegó pero no se extrajo ninguna fila
    """
    sport = args.sport
    sport_path = _SPORT_PATHS[sport]
    moment = datetime.now(timezone.utc)

    if args.reparse:
        print()
        print("=" * 66)
        print(f"  REPROCESO DE CRUDOS — {sport.upper()}")
        print("=" * 66)
        return reparse(sport, args.base_dir)

    print()
    print("=" * 66)
    print("  CAPTURA DE DISPONIBILIDAD")
    print("=" * 66)
    print(f"  Deporte : {sport.upper()}")
    print(f"  Fuente  : {_ESPN_BASE}/{sport_path}/injuries")
    print("  Coste   : 0 — ESPN no cobra ni pide clave")
    print()

    try:
        payload = (fetcher or fetch_injuries)(sport_path)
    except Exception as error:
        print(f"ERROR descargando de ESPN: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 1

    if args.inspect:
        print("  ESTRUCTURA DE LA RESPUESTA")
        print("  " + "-" * 62)
        describe(payload)
        print()
        print("  Nada guardado (--inspect).")
        print()
        return 0

    # El crudo primero: es el artefacto irrecuperable.
    try:
        raw_path = save_raw(payload, sport, args.base_dir, moment)
    except OSError as error:
        print(f"ERROR guardando el crudo: {error}", file=sys.stderr)
        return 1
    print(f"  Crudo guardado : {raw_path}")

    rows = parse_injuries(payload, sport, moment)

    if not rows:
        print()
        print("  El parser no extrajo ninguna fila.")
        print("  El crudo SÍ está guardado, así que no se ha perdido nada.")
        print("  Ejecutar con --inspect para ver la estructura real y")
        print("  ajustar el parser; luego --reparse reconstruye el CSV.")
        print()
        return 2

    csv_path = append_csv(rows, sport, args.base_dir, moment)

    by_status: dict[str, int] = {}
    for row in rows:
        key = row["status"] or "(sin estado)"
        by_status[key] = by_status.get(key, 0) + 1

    print(f"  Filas extraídas: {len(rows)}")
    print(f"  Equipos        : {len({r['team_abbreviation'] for r in rows})}")
    print(f"  CSV            : {csv_path}")
    print()
    print("  Por estado:")
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"    {status:<24} {count:>4}")

    if args.verbose:
        print()
        print("  Muestra:")
        for row in rows[:15]:
            print(f"    {row['team_abbreviation']:<5} "
                  f"{row['athlete_name']:<26} {row['status']:<14} "
                  f"{row['injury_type']}")

    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())