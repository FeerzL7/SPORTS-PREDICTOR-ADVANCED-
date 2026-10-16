#!/usr/bin/env python3
"""
scripts/capture_closing.py

Captura las líneas de cierre para medir CLV.

Por qué el CLV es el indicador que importa
--------------------------------------------
El ROI arrastra el ruido del resultado del partido: puedes estimar
correctamente la probabilidad y perder igual. Hacen falta miles de
picks para separar la señal de ese ruido.

El CLV (Closing Line Value) elimina esa varianza. Mide la diferencia
entre el precio al que entraste y el precio al que cerró el mercado.
El cierre se observa exactamente, sin azar de por medio.

    Detectar 1% de ventaja vía ROI : ~5.000 picks   (~70 temporadas NFL)
    Detectar 1% de ventaja vía CLV : ~150-250 picks (~2-3 temporadas)

Es la diferencia entre validar el sistema en años o en meses. Por eso
el paper trading debe medir CLV, no solo resultados.

Qué es un CLV positivo
------------------------
Coger un total a 1.91 y ver cerrar a 1.83 significa que el mercado se
movió hacia tu lado: había información que tú incorporaste antes. Ese
movimiento es la evidencia de que el modelo ve algo real, aunque ese
partido concreto se pierda.

Un CLV consistentemente positivo con ROI plano indica mala suerte a
corto plazo. Un CLV negativo con ROI positivo indica lo contrario —
suerte que no se sostendrá.

Cuándo ejecutarlo
-------------------
Lo más cerca posible del inicio del partido, sin pasarse. El "cierre"
real es el último precio antes del kickoff:

    MLB  → 15-30 minutos antes del primer lanzamiento
    NFL  → 30-60 minutos antes del kickoff

Capturarlo demasiado pronto subestima el movimiento; demasiado tarde
corre el riesgo de que el mercado ya esté cerrado.

Uso
----
    # Capturar el cierre de los picks pendientes de hoy
    python scripts/capture_closing.py --sport mlb

    # Una fecha concreta
    python scripts/capture_closing.py --sport nfl --date 2026-11-08

    # Ver qué se capturaría sin escribir nada
    python scripts/capture_closing.py --sport mlb --dry-run

    # El ledger de los mercados en solo-papel ({sport}_paper_roi_tracking.csv)
    python scripts/capture_closing.py --sport nfl --paper

    # Operable + papel con una sola llamada, solo partidos que empiezan
    # en los próximos 75 minutos (los demás quedan para otra corrida)
    python scripts/capture_closing.py --sport nfl --with-paper --max-lead-minutes 75

Ventana de captura
--------------------
El CLV de un pick se registra UNA vez. Sin --max-lead-minutes, una
corrida a mediodía fija el "cierre" de un partido de la noche horas
antes de tiempo. Con la ventana, cada corrida captura solo los
partidos que están por empezar y deja los demás pendientes; los que ya
empezaron se descartan, porque su cuota actual es en vivo.

Automatización recomendada (cron)
-----------------------------------
    # MLB: 17:30 ET, antes de los partidos de tarde
    30 21 * * * cd /ruta && python scripts/capture_closing.py --sport mlb

    # NFL: domingo 12:15 ET, antes del bloque de las 13:00
    15 17 * * 0 cd /ruta && python scripts/capture_closing.py --sport nfl
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_dotenv() -> None:
    """Carga el .env de la raíz, igual que run_daily.py."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

_AVAILABLE_SPORTS: dict[str, str] = {
    "mlb": "sports.mlb.plugin.MLBPlugin",
    "nfl": "sports.nfl.plugin.NFLPlugin",
}


def main() -> int:
    args = _parse_args()
    sport = args.sport.lower()
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"\n{'=' * 62}")
    print(f"  CAPTURA DE CIERRE — {sport.upper()} | {date}")
    if args.dry_run:
        print(f"  Modo: DRY RUN (no escribe en el ledger)")
    if args.max_lead_minutes is not None:
        print(f"  Ventana: partidos que empiezan en ≤ {args.max_lead_minutes:.0f} min")
    print(f"{'=' * 62}\n")

    # ── Cargar configuración y plugin ─────────────────────────────
    from core.utils.config_loader import load_config
    config = load_config(sport=sport, base_dir="config")

    api_key = os.environ.get("ODDS_API_KEY") or config.get("ODDS_API_KEY")
    if not api_key:
        print("ERROR: ODDS_API_KEY no configurada.")
        print("  El cierre se obtiene de The Odds API, no del proveedor")
        print("  deportivo: necesita la clave aunque el resto del pipeline")
        print("  esté funcionando.")
        return 1

    import importlib
    module_path, class_name = _AVAILABLE_SPORTS[sport].rsplit(".", 1)
    plugin_cls = getattr(importlib.import_module(module_path), class_name)
    plugin = plugin_cls(config_loader=config)

    # ── Picks pendientes sin cierre registrado ────────────────────
    from core.bankroll.tracker import BankrollTracker, CsvLedgerStore

    # El ledger de papel tiene el mismo esquema: solo cambia el archivo.
    # Sin su CLV, los mercados en solo-papel no podrían salir nunca de
    # ese estado, porque el CLV es justo la evidencia que se les pide.
    # Con --with-paper se procesan los dos con UNA sola llamada a la
    # API: el precio actual de un partido es el mismo para ambos.
    work = []
    for label, path in _ledger_paths(sport, args):
        if not Path(path).exists():
            continue
        store = CsvLedgerStore(path)
        tracker = BankrollTracker(
            store=store,
            initial_bankroll=config.get("bankroll.initial_bankroll", default=1000.0),
        )
        pending = [e for e in store.load_all() if e.date == date and e.clv is None]
        work.append((label, store, tracker, pending))

    if not work:
        print(f"  No hay ledger para {sport} en output/ledger/.")
        print(f"  Ejecutar primero: python scripts/run_daily.py --sport {sport}")
        return 2

    total_pending = sum(len(pending) for *_, pending in work)
    if not total_pending:
        # Sin llamada a la API: no se gasta un crédito si no hay nada
        # que capturar.
        print(f"  Sin picks pendientes de cierre para {date}.")
        return 0

    for label, _, _, pending in work:
        print(f"  Pendientes ({label}): {len(pending)}")
    print()

    # ── Obtener las cuotas actuales ───────────────────────────────
    from core.odds.client import OddsAPIClient, OddsAPIConfig

    odds_sport = getattr(plugin, "odds_api_sport_id", plugin.sport_id)
    client = OddsAPIClient(OddsAPIConfig(
        api_key=api_key,
        regions=config.get("odds_api.regions", default="us"),
    ))
    markets = plugin.get_market_definitions().get_core_markets()

    response = client.get_events(sport=odds_sport, markets=markets)

    # OddsAPIClient modela los errores esperados (auth, rate limit,
    # deporte fuera de temporada) como resultado explícito, no como
    # excepción: el llamador decide qué hacer según error_type. Un
    # try/except aquí trataría por igual un 401 —que exige revisar la
    # clave— y un 429 —que solo pide esperar.
    if not response.success:
        print(f"  ERROR de la API ({response.error_type}): "
              f"{response.error_message}")
        if response.error_type == "rate_limit":
            print("\n  Límite de peticiones alcanzado. El cierre se puede")
            print("  capturar más tarde, pero cuanto más se tarde menos")
            print("  representativo será del precio real de cierre.")
        elif response.error_type == "auth_error":
            print("\n  Revisar ODDS_API_KEY en el .env.")
        return 1

    print(f"  Eventos con cuotas: {len(response.events)}")
    if response.requests_remaining is not None:
        print(f"  Créditos restantes: {response.requests_remaining}")
    print()

    # ── Emparejar y calcular CLV ──────────────────────────────────
    preferred = plugin.get_market_definitions().get_preferred_line("SPREAD")
    by_event = _index_prices_by_event(response.events, markets, preferred)

    if not by_event:
        print("  Ninguna cuota pudo extraerse de la respuesta.")
        print("  Verificar que los mercados solicitados coinciden con los")
        print("  que devuelve la API para este deporte.")
        return 1

    now = datetime.now(timezone.utc)
    for label, store, tracker, pending in work:
        _capture_ledger(label, store, tracker, pending, by_event, now, args)

    if args.dry_run:
        print("\n  [DRY RUN] No se escribió nada en el ledger.")

    print()
    return 0


def _ledger_paths(sport: str, args) -> list[tuple[str, str]]:
    """Ledgers a procesar: (etiqueta, ruta)."""
    live  = ("operable", f"output/ledger/{sport}_roi_tracking.csv")
    paper = ("papel",    f"output/ledger/{sport}_paper_roi_tracking.csv")
    if args.paper:
        return [paper]
    return [live, paper] if args.with_paper else [live]


def _capture_ledger(label, store, tracker, pending, by_event, now, args) -> None:
    """Calcula y guarda el CLV de los picks pendientes de un ledger."""
    from core.evaluation.clv import CLVTracker

    if not pending:
        return

    print(f"  ── Ledger {label} {'─' * (50 - len(label))}")
    captured, deferred, started, unmatched = 0, [], [], []

    for entry in pending:
        event = by_event.get(_entry_event_key(entry))
        if event is None:
            unmatched.append(entry)
            continue

        status = _window_status(event["commence_time"], now, args.max_lead_minutes)
        if status == "started":
            started.append(entry)
            continue
        if status == "deferred":
            deferred.append(entry)
            continue

        price_now = _find_price(entry, event["prices"])
        clv = (CLVTracker.calculate_clv(pick_price=entry.price, closing_price=price_now)
               if price_now is not None else None)
        if clv is None:
            unmatched.append(entry)
            continue

        arrow = "↑" if clv > 0 else ("↓" if clv < 0 else "=")
        line = f" {entry.selection}"
        print(f"    {arrow} {entry.market:7s}{line:28s} "
              f"{entry.price:5.2f} → {price_now:5.2f}   CLV {clv:+6.2f}%")

        if not args.dry_run:
            tracker.update_clv(entry_id=entry.entry_id, clv=clv)
        captured += 1

    print(f"    Capturados : {captured} de {len(pending)}")
    if deferred:
        print(f"    Aplazados  : {len(deferred)} (empiezan después de la ventana; "
              f"se capturan en una corrida posterior)")
    if started:
        print(f"    Ya empezados: {len(started)} — su cierre se perdió: la cuota "
              f"actual sería en vivo, no de cierre")
    if unmatched:
        print(f"    Sin emparejar: {len(unmatched)}")
        for e in unmatched[:5]:
            print(f"      - {e.event} | {e.market} {e.selection}")
        if len(unmatched) > 5:
            print(f"      ... y {len(unmatched) - 5} más")

    if captured and not args.dry_run:
        clvs = []
        for e in pending:
            refreshed = store.load_by_id(e.entry_id)
            if refreshed and refreshed.clv is not None:
                clvs.append(refreshed.clv)
        if clvs:
            mean = sum(clvs) / len(clvs)
            positive = sum(1 for c in clvs if c > 0)
            print(f"    CLV medio  : {mean:+.2f}%   positivos {positive}/{len(clvs)}")
    print()


# ── Emparejamiento de cuotas ─────────────────────────────────────────────────

def _norm_team(name) -> str:
    """Nombre de equipo comparable: minúsculas, sin acentos ni puntuación."""
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def _event_key(away, home) -> str:
    return f"{_norm_team(away)}@{_norm_team(home)}"


def _entry_event_key(entry) -> str | None:
    """Clave del partido de un pick, desde su campo event ("Away @ Home")."""
    away, sep, home = (entry.event or "").partition(" @ ")
    return _event_key(away, home) if sep else None


def _index_prices_by_event(events, markets: list[str],
                           preferred_line: float | None) -> dict:
    """
    Indexa las cuotas actuales POR PARTIDO.

        {clave_partido: {"commence_time": str | None,
                         "prices": {(mercado, selección, línea): precio}}}

    CORRECCIÓN (2026-10): el índice era global por (mercado, selección,
    línea). En totales la selección es solo 'over'/'under', así que dos
    partidos con la misma línea compartían clave y el pick podía quedar
    con el cierre de OTRO partido —el más alto de los dos—. Spread y
    moneyline se salvaban porque la selección lleva el nombre del equipo.

    RawOddsEvent guarda el bloque `bookmakers` crudo: extraer precios es
    trabajo de OddsNormalizer, el mismo que usa el pipeline en Stage 5.
    """
    from core.odds.normalizer import OddsNormalizer

    normalizer = OddsNormalizer()
    index: dict = {}

    for event in events:
        try:
            odds_list = normalizer.extract_best(
                raw_event=event, markets=markets,
                preferred_line=preferred_line,
            )
        except Exception:
            continue

        key = _event_key(getattr(event, "away_team", ""), getattr(event, "home_team", ""))
        slot = index.setdefault(key, {
            "commence_time": getattr(event, "commence_time", None),
            "prices": {},
        })
        for odds in odds_list:
            price_key = (odds.market.upper(), odds.selection.strip().lower(), odds.line)
            if price_key not in slot["prices"] or odds.price > slot["prices"][price_key]:
                slot["prices"][price_key] = odds.price

    return {k: v for k, v in index.items() if v["prices"]}


def _window_status(commence_time, now, max_lead_minutes) -> str:
    """
    'capture', 'deferred' (empieza más tarde que la ventana) o
    'started' (ya empezó: la cuota actual sería en vivo).

    Sin hora de inicio legible se captura: es preferible un cierre
    aproximado a ninguno, y es el comportamiento anterior.
    """
    from core.odds.book_archive import parse_iso8601

    start = parse_iso8601(commence_time)
    if start is None:
        return "capture"
    lead = (start - now).total_seconds() / 60.0
    if lead < 0:
        return "started"
    if max_lead_minutes is not None and lead > max_lead_minutes:
        return "deferred"
    return "capture"


def _find_price(entry, index: dict) -> float | None:
    """
    Busca el precio actual de un pick entre las cuotas de SU partido.

    El ledger guarda la selección con la línea embebida ('over 47.0'),
    igual que en el settlement. Hay que separarlas antes de buscar.
    """
    selection, line = _split_selection(entry.selection)
    market = entry.market.upper()

    key = (market, selection.strip().lower(), line)
    if key in index:
        return index[key]

    # Reintento sin línea, para mercados que no la usan (ML)
    key_no_line = (market, selection.strip().lower(), None)
    return index.get(key_no_line)


def _split_selection(raw: str) -> tuple[str, float | None]:
    """Separa 'over 47.0' en ('over', 47.0), preservando el signo."""
    text = (raw or "").strip()
    line: float | None = None
    words: list[str] = []
    for token in text.replace("+", " +").replace("-", " -").split():
        try:
            value = float(token)
        except ValueError:
            words.append(token)
        else:
            if line is None:
                line = value
    return " ".join(words).strip(), line


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura líneas de cierre para medir CLV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sport", required=True,
                        choices=list(_AVAILABLE_SPORTS),
                        help="Deporte cuyo cierre se captura")
    parser.add_argument("--date", default=None,
                        help="Fecha YYYY-MM-DD. Default: hoy UTC")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra el CLV sin escribirlo en el ledger")
    parser.add_argument("--paper", action="store_true",
                        help="Usa SOLO el ledger de los mercados en solo-papel")
    parser.add_argument("--with-paper", action="store_true",
                        help="Procesa también el ledger de papel con la misma "
                             "llamada a la API (sin coste extra)")
    parser.add_argument("--max-lead-minutes", type=float, default=None,
                        help="Captura solo los partidos que empiezan en esta "
                             "ventana; los demás quedan para otra corrida")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())