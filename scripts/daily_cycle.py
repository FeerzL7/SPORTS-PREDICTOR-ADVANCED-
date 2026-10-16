#!/usr/bin/env python3
"""
scripts/daily_cycle.py — Ciclo diario de paper trading: picks, liquidación y cierre.

OBJETIVO
--------
Que el paper trading corra solo. Sin esto, el ledger no acumula
evidencia: hubo picks de MLB pendientes desde el 27-sep que nunca se
liquidaron ni tuvieron CLV. Y el CLV es la única métrica que valida un
mercado en meses en vez de años (ver scripts/capture_closing.py).

Dos fases, pensadas para dos tareas programadas:

    picks    Una vez al día, por la mañana. Por cada deporte corre
             run_daily.py, que liquida los picks pendientes (Stage 10,
             ledger operable y de papel) y registra los nuevos.

    closing  Cada 30 minutos durante el día. Por cada deporte mira qué
             picks del día siguen sin CLV y a qué hora empiezan sus
             partidos; si alguno empieza dentro de la ventana, corre
             capture_closing.py --with-paper --max-lead-minutes.

DISCIPLINA DE CRÉDITOS
-----------------------
El plan gratuito de The Odds API da 500 créditos al mes.

    run_daily.py            3 créditos por deporte, SOLO si hay partidos
                            ese día (sin eventos no llega a pedir cuotas)
    capture_closing.py      3 créditos por llamada; con --with-paper una
                            sola llamada cubre los dos ledgers
    /events y /sports       0 créditos: se usan para decidir

La fase closing consulta /events (gratis) para conocer la hora de
inicio y solo gasta cuando un pick pendiente está por empezar. Correrla
cada 30 minutos cuesta lo mismo que correrla en los momentos justos,
sin depender de horarios fijos ni del cambio de horario.

Si quedan menos de --min-credits, el ciclo no gasta y lo deja en el log.

CAPTURA NBA (sin modelo)
-------------------------
El modelo de eficiencia de NBA dio no-go (tarea 12.3): no aporta nada
sobre el spread de cierre. Lo único con posibilidad de ventaja es la
disponibilidad de jugadores, y esa información no se puede reconstruir
hacia atrás. Por eso NBA entra como deporte SOLO DE CAPTURA:

    lesiones (ESPN, 0 créditos)   picks: si hay partidos en 24 h
                                  closing: si alguno empieza en ≤ 90 min
    precios de cierre por casa    closing: si alguno empieza en (15, 45]
    (spreads+totals, 2 créditos)  min — cada tanda una sola vez

Coste estimado: ~150-200 créditos/mes en temporada regular.

POR QUÉ SIN FÚTBOL
-------------------
Sus siete competiciones cuestan ~28 créditos por corrida (~840 al mes,
más que todo el plan) y la Fase 0 midió que su modelo no aporta
información sobre el cierre (peso óptimo 0.00 en 22.533 selecciones).
Sigue disponible con --sports soccer si se quiere.

FECHAS
-------
run_daily.py y capture_closing.py usan por defecto la fecha UTC. En
México (UTC-6) una captura a las 18:00 ya es "mañana" en UTC y no
encontraría los picks del día. Este script pasa siempre --date con la
fecha LOCAL, que en husos al oeste de la costa este coincide con la
fecha de los partidos de NFL y MLB.

Uso
----
    python scripts/daily_cycle.py picks
    python scripts/daily_cycle.py closing
    python scripts/daily_cycle.py closing --plan     # decide, no ejecuta
    python scripts/daily_cycle.py picks --sports nfl
    python scripts/daily_cycle.py closing --capture-sports     # sin capturas NBA

Registro en: logs/daily_cycle/AAAA-MM-DD.log

Códigos de salida: 0 todo bien (incluido "nada que hacer"), 1 algún paso
falló.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Clave de cada deporte en The Odds API. Es la misma que declaran los
# plugins en `odds_api_sport_id`; se repite aquí para no cargar el
# plugin entero —nflverse, pandas— solo para saber qué partidos hay.
ODDS_SPORT_KEYS: dict[str, str] = {
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
}

DEFAULT_SPORTS      = ("nfl", "mlb")
DEFAULT_WINDOW_MIN  = 45.0   # con cadencia de 30 min: captura 15-45 min antes
DEFAULT_MIN_CREDITS = 40

# Deportes SOLO DE CAPTURA: no tienen modelo ni ledger. Se archiva lo que
# no se puede reconstruir después —el informe de lesiones previo al
# partido y los precios de cierre por casa— para poder modelarlo dentro
# de una o dos temporadas sin look-ahead. Es el plan posterior al no-go
# del modelo de eficiencia de NBA (tarea 12.3).
CAPTURE_SPORTS: dict[str, dict] = {
    "nba": {"odds_key": "basketball_nba", "markets": "spreads,totals"},
}
DEFAULT_CAPTURE_SPORTS = ("nba",)

# Lesiones: foto en cada corrida con algún partido a ≤ 90 min (0
# créditos); el estado cambia hasta poco antes del inicio.
AVAILABILITY_WINDOW_MIN = 90.0
# Precios de cierre: se dispara solo si algún partido empieza en
# (15, 45] min. Con cadencia de 30 min cada tanda cae ahí exactamente
# una vez, así que no se paga dos veces la misma tanda.
CLOSING_MIN_LEAD = 15.0

# capture_availability.py: 0 = ok, 2 = crudo guardado pero sin filas.
# capture_book_odds.py: 0 = ok, 2 = sin eventos.
_CAPTURE_OK = {0, 2}

# Códigos de run_daily.py que no son error: 0 = picks, 2 = sin picks.
_RUN_DAILY_OK = {0, 2}
# capture_closing.py: 0 = capturó o nada pendiente, 2 = aún no hay ledger.
_CLOSING_OK = {0, 2}


# ── Utilidades ────────────────────────────────────────────────────────────────

def _load_capture_closing():
    """capture_closing.py como módulo, para reutilizar su emparejamiento."""
    path = _PROJECT_ROOT / "scripts" / "capture_closing.py"
    spec = importlib.util.spec_from_file_location("capture_closing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CycleLog:
    """Escribe a consola y al log del día."""

    def __init__(self, date: str, phase: str, base_dir: Path | None = None) -> None:
        base = base_dir or (_PROJECT_ROOT / "logs" / "daily_cycle")
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"{date}.log"
        self._phase = phase

    def __call__(self, message: str = "") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] [{self._phase}] {message}"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_script(args: list[str], log: CycleLog) -> int:
    """
    Ejecuta un script del proyecto y vuelca su salida al log.

    PYTHONIOENCODING=utf-8: con la salida redirigida, Python en Windows
    escribe en cp1252 y revienta con los acentos y símbolos de los
    informes (UnicodeEncodeError). CREATE_NO_WINDOW: que una tarea
    programada no abra una consola cada media hora.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # La tarea programada arranca con pythonw.exe (sin consola). Los
    # hijos van con python.exe —salida normal a la tubería— y es
    # CREATE_NO_WINDOW lo que evita que abran ventana.
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        exe = exe.with_name("python.exe")
    log(f"$ python {' '.join(args)}")
    proc = subprocess.run(
        [str(exe), *args], cwd=_PROJECT_ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=flags,
    )
    for stream in (proc.stdout, proc.stderr):
        for line in (stream or "").splitlines():
            if line.strip():
                log(f"    {line}")
    log(f"  → código de salida {proc.returncode}")
    return proc.returncode


def fetch_events(sport_key: str) -> tuple[list[dict], int | None]:
    """
    Partidos próximos y en curso de un deporte, con su hora de inicio.

    GET /sports/{sport}/events NO consume créditos (verificado: cabecera
    x-requests-last = 0), y devuelve además el saldo restante.
    """
    import requests

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("falta ODDS_API_KEY en el .env")
    response = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/events",
        params={"apiKey": api_key}, timeout=20,
    )
    response.raise_for_status()
    remaining = response.headers.get("x-requests-remaining")
    return response.json(), (int(float(remaining)) if remaining else None)


def pending_entries(sport: str, date: str, cc) -> list:
    """Picks de `date` sin CLV, en el ledger operable y en el de papel."""
    from core.bankroll.tracker import CsvLedgerStore

    args = argparse.Namespace(paper=False, with_paper=True)
    entries = []
    for _, path in cc._ledger_paths(sport, args):
        if Path(path).exists():
            entries.extend(e for e in CsvLedgerStore(path).load_all()
                           if e.date == date and e.clv is None)
    return entries


def due_for_closing(entries, events: list[dict], now: datetime,
                    window_minutes: float, cc) -> tuple[list, list]:
    """
    Separa los pendientes en (a capturar ahora, para más tarde).

    Un pick está "a capturar" si su partido empieza dentro de la
    ventana. Los partidos ya empezados o desconocidos no cuentan: el
    primer caso ya perdió su cierre y el segundo no se puede ubicar
    (capture_closing.py los reporta igual si llega a correr).
    """
    starts = {cc._event_key(e.get("away_team"), e.get("home_team")): e.get("commence_time")
              for e in events}
    due, later = [], []
    for entry in entries:
        commence = starts.get(cc._entry_event_key(entry))
        status = cc._window_status(commence, now, window_minutes) if commence else None
        if status == "capture":
            due.append(entry)
        elif status == "deferred":
            later.append(entry)
    return due, later


def minutes_until_start(events: list[dict], now: datetime) -> list[float]:
    """Minutos hasta el inicio de cada partido que aún no empezó."""
    from core.odds.book_archive import parse_iso8601

    leads = []
    for event in events:
        start = parse_iso8601(event.get("commence_time"))
        if start is not None:
            lead = (start - now).total_seconds() / 60.0
            if lead >= 0:
                leads.append(lead)
    return leads


def capture_sport(sport: str, phase: str, log: CycleLog, plan: bool,
                  window_minutes: float, min_credits: int, now: datetime,
                  runner: Callable[[list[str], CycleLog], int],
                  events_fn: Callable) -> int:
    """
    Capturas de un deporte sin modelo. Devuelve el número de fallos.

    picks    foto de lesiones si hay partidos en las próximas 24 h.
    closing  foto de lesiones si alguno empieza en ≤ 90 min, y precios de
             cierre por casa si alguno empieza en (15, ventana] min.
    """
    spec = CAPTURE_SPORTS[sport]
    try:
        events, remaining = events_fn(spec["odds_key"])
    except Exception as error:
        log(f"{sport}: no se pudo consultar /events ({error})")
        return 1

    leads = minutes_until_start(events, now)
    failures = 0

    availability_window = 24 * 60.0 if phase == "picks" else AVAILABILITY_WINDOW_MIN
    if any(lead <= availability_window for lead in leads):
        cmd = ["scripts/capture_availability.py", "--sport", sport]
        if plan:
            log(f"{sport}: [plan] correría: python {' '.join(cmd)}  (0 créditos)")
        elif runner(cmd, log) not in _CAPTURE_OK:
            failures += 1
    else:
        log(f"{sport}: ningún partido en las próximas "
            f"{availability_window / 60:.1f} h — sin foto de lesiones")

    if phase != "closing":
        return failures

    if not any(CLOSING_MIN_LEAD < lead <= window_minutes for lead in leads):
        return failures
    if remaining is not None and remaining < min_credits:
        log(f"{sport}: partidos por empezar pero quedan {remaining} créditos "
            f"(< {min_credits}); no se capturan precios")
        return failures

    cmd = ["scripts/capture_book_odds.py", "--sport", sport,
           "--markets", spec["markets"],
           "--max-lead-minutes", f"{window_minutes:.0f}",
           "--min-credits", str(min_credits)]
    cost = len(spec["markets"].split(","))
    if plan:
        log(f"{sport}: [plan] correría: python {' '.join(cmd)}  ({cost} créditos)")
    elif runner(cmd, log) not in _CAPTURE_OK:
        failures += 1
    return failures


# ── Fases ─────────────────────────────────────────────────────────────────────

def phase_picks(sports, date: str, log: CycleLog, plan: bool,
                min_credits: int,
                runner: Callable[[list[str], CycleLog], int] = run_script,
                events_fn: Callable = fetch_events,
                capture_sports=(), now: datetime | None = None) -> int:
    """
    run_daily.py por deporte, SIEMPRE, aunque no haya partidos: la misma
    corrida liquida los picks del día anterior (Stage 10). Saltarla en
    días sin partidos dejaría sin liquidar, por ejemplo, los últimos
    picks de la temporada. Sin eventos no llega a pedir cuotas, así que
    no gasta créditos.
    """
    failures = 0
    for sport in sports:
        sport_key = ODDS_SPORT_KEYS.get(sport)
        remaining = None
        if sport_key:
            try:
                _, remaining = events_fn(sport_key)
            except Exception as error:
                log(f"{sport}: no se pudo consultar el saldo ({error}); se corre igual")
        cmd = ["scripts/run_daily.py", "--sport", sport, "--date", date]
        if remaining is not None and remaining < min_credits:
            # Sin Stage 5 no se piden cuotas: no hay picks nuevos, pero
            # la liquidación de los pendientes sigue corriendo.
            log(f"{sport}: quedan {remaining} créditos (< {min_credits}); "
                f"solo se liquida, sin pedir cuotas")
            cmd += ["--skip-stages", "5"]
        if plan:
            log(f"{sport}: [plan] correría: python {' '.join(cmd)}")
            continue
        code = runner(cmd, log)
        if code not in _RUN_DAILY_OK:
            failures += 1

    now = now or datetime.now(timezone.utc)
    for sport in capture_sports:
        failures += capture_sport(sport, "picks", log, plan, DEFAULT_WINDOW_MIN,
                                  min_credits, now, runner, events_fn)
    return 1 if failures else 0


def phase_closing(sports, date: str, log: CycleLog, plan: bool,
                  window_minutes: float, min_credits: int, now: datetime | None = None,
                  runner: Callable[[list[str], CycleLog], int] = run_script,
                  events_fn: Callable = fetch_events, cc=None,
                  capture_sports=()) -> int:
    cc = cc or _load_capture_closing()
    now = now or datetime.now(timezone.utc)
    failures = 0

    for sport in sports:
        entries = pending_entries(sport, date, cc)
        if not entries:
            log(f"{sport}: sin picks pendientes de cierre (0 créditos)")
            continue

        sport_key = ODDS_SPORT_KEYS.get(sport)
        if not sport_key:
            log(f"{sport}: sin clave de The Odds API configurada en daily_cycle.py")
            continue
        try:
            events, remaining = events_fn(sport_key)
        except Exception as error:
            log(f"{sport}: no se pudo consultar /events ({error}); se reintenta en la próxima corrida")
            failures += 1
            continue

        due, later = due_for_closing(entries, events, now, window_minutes, cc)
        if not due:
            log(f"{sport}: {len(entries)} pendientes, ninguno empieza en los próximos "
                f"{window_minutes:.0f} min ({len(later)} más tarde) — 0 créditos")
            continue
        if remaining is not None and remaining < min_credits:
            log(f"{sport}: {len(due)} picks por cerrar pero quedan {remaining} créditos "
                f"(< {min_credits}); no se gastan")
            continue

        cmd = ["scripts/capture_closing.py", "--sport", sport, "--date", date,
               "--with-paper", "--max-lead-minutes", f"{window_minutes:.0f}"]
        log(f"{sport}: {len(due)} picks empiezan en ≤ {window_minutes:.0f} min")
        if plan:
            log(f"{sport}: [plan] correría: python {' '.join(cmd)}")
            continue
        if runner(cmd, log) not in _CLOSING_OK:
            failures += 1

    for sport in capture_sports:
        failures += capture_sport(sport, "closing", log, plan, window_minutes,
                                  min_credits, now, runner, events_fn)
    return 1 if failures else 0


# ── Entrada ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_dotenv()
    date = args.date or datetime.now().strftime("%Y-%m-%d")
    log = CycleLog(date, args.phase)
    log(f"inicio — deportes {', '.join(args.sports)} | captura "
        f"{', '.join(args.capture_sports) or '-'} | fecha {date}"
        f"{' | PLAN (no ejecuta)' if args.plan else ''}")

    if args.phase == "picks":
        code = phase_picks(args.sports, date, log, args.plan, args.min_credits,
                           capture_sports=args.capture_sports)
    else:
        code = phase_closing(args.sports, date, log, args.plan,
                             args.window, args.min_credits,
                             capture_sports=args.capture_sports)

    log(f"fin — {'OK' if code == 0 else 'con errores'}")
    return code


def _load_dotenv() -> None:
    """Carga el .env de la raíz sin pisar variables ya definidas."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ciclo diario de paper trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("phase", choices=["picks", "closing"])
    parser.add_argument("--sports", nargs="*", default=list(DEFAULT_SPORTS),
                        help="Deportes con modelo y ledger (default: nfl mlb)")
    parser.add_argument("--capture-sports", nargs="*", default=list(DEFAULT_CAPTURE_SPORTS),
                        choices=sorted(CAPTURE_SPORTS),
                        help="Deportes solo de captura (default: nba). "
                             "Sin valores desactiva las capturas")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD. Default: fecha LOCAL de hoy")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_MIN,
                        help="Minutos antes del inicio en que se captura el cierre")
    parser.add_argument("--min-credits", type=int, default=DEFAULT_MIN_CREDITS,
                        help="No gastar créditos por debajo de este saldo")
    parser.add_argument("--plan", action="store_true",
                        help="Decidir y registrar qué haría, sin ejecutar ni gastar")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
