#!/usr/bin/env python3
"""
capture_book_odds.py — Captura los precios de TODAS las casas y los archiva.

OBJETIVO
--------
Guardar el histórico de precios por casa que hoy se descarga y se tira.
Es la Etapa 1 del plan posterior al no-go de NBA: lo único con fecha de
caducidad real, porque no hay proveedor barato de histórico de precios
por casa y lo que no se capture hoy no se compra mañana.

POR QUÉ UN SCRIPT APARTE Y NO SOLO EL PIPELINE
-----------------------------------------------
El runner ya archiva en su Stage 5 (ver `core/odds/book_archive.py` y el
parche del runner), pero eso **no basta**, por dos razones:

  1. **Cadencia.** El pipeline corre una vez al día. Una línea de
     CIERRE exige disparar cerca del tip-off, varias veces por jornada.
     Los partidos de una noche de NBA arrancan en dos a cuatro tandas.
  2. **Robustez.** El pipeline carga el plugin del deporte, que arrastra
     nflverse, soccerdata o pandas. Si cualquiera de esas falla, no se
     captura — y la captura es precisamente lo irrecuperable.

Por eso este script **no carga ningún plugin**. Solo necesita la clave
de deporte de The Odds API, que es lo único que la captura usa: el
plugin sirve para modelar, no para descargar precios.

ARQUITECTURA
------------
    OddsAPIClient.get_events(sport, markets)
        ↓  list[RawOddsEvent]  (incluye todas las casas)
    OddsNormalizer.extract_all(raw_event, markets)
        ↓  list[MarketOdds]  — uno por (selección, línea, casa)
    BookArchive.capture_many(sport, events)
        ↓
    output/book_archive/{sport}/{YYYY-MM}.csv

Se reutiliza el cliente, el normalizador y el archivo tal cual. Este
script es solo el cableado y la disciplina de créditos.

COSTE EN CRÉDITOS
-----------------
El endpoint cobra **1 crédito por mercado y por región**. Una llamada
con tres mercados en una región cuesta 3 créditos, y devuelve TODOS los
eventos del deporte con TODAS las casas. El número de partidos no
influye en el coste: lo que cuesta es el número de llamadas.

De ahí la consecuencia práctica, en el plan gratuito de 500 al mes:

    3 mercados x 3 capturas/noche x 30 noches = 270 créditos/mes
    2 mercados x 3 capturas/noche x 30 noches = 180 créditos/mes

Como el moneyline no entra en el v1 de NBA, `--markets spreads,totals`
es la opción sensata para las capturas de cierre. `--min-credits` aborta
antes de gastar si el saldo baja de un umbral, para que una captura
automática no agote la cuota del mes.

USO
---
    # captura normal
    python scripts/capture_book_odds.py --sport nba

    # captura de cierre, sin moneyline, protegiendo el saldo
    python scripts/capture_book_odds.py --sport nba \
        --markets spreads,totals --min-credits 50

    # comprobar el encadenado sin escribir ni gastar
    python scripts/capture_book_odds.py --sport nfl --dry-run

    # clave cruda de The Odds API, sin alias
    python scripts/capture_book_odds.py --sport-key icehockey_nhl

RIESGOS
-------
  * Una captura perdida no se recupera. El script devuelve código
    distinto de 0 para que un planificador pueda detectarlo y avisar.
  * El alias de deporte a clave de The Odds API se declara aquí. El
    plugin sigue siendo la fuente de verdad (`odds_api_sport_ids`); esta
    tabla es una comodidad para la línea de comandos. Mantenerla al día
    cuesta una línea por deporte, y `--sport-key` permite saltársela.
  * El archivo tolera duplicados a propósito: dos capturas de la misma
    noche son el movimiento de línea, no un error. Deduplicar es tarea
    del consolidador.

MEJORAS FUTURAS
---------------
  * Programación adaptativa: leer el calendario del día, agrupar los
    tip-offs y disparar una captura por tanda en vez de a horas fijas.
  * Registrar el `last_update` por casa, que la API entrega y
    `MarketOdds` no modela todavía: distinguiría un precio rancio de
    uno recién movido.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_dotenv() -> None:
    """
    Carga el .env de la raíz, igual que run_daily.py y capture_closing.py.

    La clave de API vive en .env y nunca en YAML: un secreto en YAML
    acaba versionado en git.
    """
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    with open(env_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


# ── Alias de conveniencia ─────────────────────────────────────────────────────
#
# La fuente de verdad es `plugin.odds_api_sport_ids`. Esta tabla existe
# solo para no tener que teclear la clave completa, y para que el script
# no tenga que importar un plugin —y con él nflverse o soccerdata— para
# hacer una llamada HTTP.

_SPORT_ALIASES: dict[str, str] = {
    "mlb":  "baseball_mlb",
    "nba":  "basketball_nba",
    "nfl":  "americanfootball_nfl",
    "nhl":  "icehockey_nhl",
    "ncaab": "basketball_ncaab",
    "ncaaf": "americanfootball_ncaaf",
    "epl":  "soccer_epl",
}

_DEFAULT_MARKETS = "h2h,spreads,totals"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura y archiva los precios de todas las casas.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sport",
        choices=sorted(_SPORT_ALIASES),
        help="Alias de deporte (nba, nfl, mlb, ...).",
    )
    group.add_argument(
        "--sport-key",
        help="Clave cruda de The Odds API, ej. 'basketball_nba'.",
    )
    parser.add_argument(
        "--markets",
        default=_DEFAULT_MARKETS,
        help=(
            "Mercados separados por coma. Default "
            f"'{_DEFAULT_MARKETS}'. Cada mercado cuesta 1 crédito por "
            "región, así que quitar los que no se usan ahorra."
        ),
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="Regiones de The Odds API. Default 'us'.",
    )
    parser.add_argument(
        "--min-credits",
        type=int,
        default=0,
        help=(
            "Aborta si los créditos restantes quedarían por debajo de "
            "este umbral. 0 desactiva la protección."
        ),
    )
    parser.add_argument(
        "--max-lead-minutes",
        type=float,
        default=None,
        help=(
            "Archiva solo eventos que empiecen dentro de estos minutos. "
            "Útil para capturas de cierre. No ahorra créditos (la "
            "respuesta es la misma), solo enfoca el archivo."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default="output/book_archive",
        help="Raíz del archivo. Default 'output/book_archive'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Llama a la API pero no escribe nada.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detalla cada evento capturado.",
    )
    return parser.parse_args(argv)


def _resolve_sport_key(args: argparse.Namespace) -> str:
    return args.sport_key or _SPORT_ALIASES[args.sport]


def run(
    args: argparse.Namespace,
    client_factory=None,
) -> int:
    """
    Ejecuta una captura.

    `client_factory` existe para los tests: permite inyectar un cliente
    falso y ejercitar toda la cadena —normalizador, filtro de ventana,
    archivo, informe— sin tocar la red ni gastar créditos. En producción
    se deja en None y se construye el cliente real.

    Códigos de salida
    -----------------
        0  captura correcta (o dry-run correcto)
        1  error: sin clave, fallo de la API, saldo insuficiente
        2  la API respondió bien pero no hay eventos
    """
    from core.odds.book_archive import BookArchive, CsvBookArchiveStore, parse_iso8601
    from core.odds.normalizer import OddsNormalizer

    sport_key = _resolve_sport_key(args)
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    if not markets:
        print("ERROR: --markets quedó vacío.", file=sys.stderr)
        return 1

    cost = len(markets) * len([r for r in args.regions.split(",") if r.strip()])

    print()
    print("=" * 66)
    print("  CAPTURA DE PRECIOS POR CASA")
    print("=" * 66)
    print(f"  Deporte  : {sport_key}")
    print(f"  Mercados : {', '.join(markets)}")
    print(f"  Regiones : {args.regions}")
    print(f"  Coste    : {cost} créditos (1 por mercado y región)")
    if args.dry_run:
        print("  Modo     : DRY RUN — llama a la API, no escribe")
    print()

    # ── Cliente ──────────────────────────────────────────────────────
    if client_factory is not None:
        client = client_factory()
    else:
        from core.odds.client import OddsAPIClient, OddsAPIConfig

        api_key = os.environ.get("ODDS_API_KEY", "").strip()
        if not api_key:
            print(
                "ERROR: falta ODDS_API_KEY. Debe estar en el .env de la "
                "raíz del proyecto, nunca en un YAML versionado.",
                file=sys.stderr,
            )
            return 1
        client = OddsAPIClient(OddsAPIConfig(
            api_key=api_key,
            regions=args.regions,
        ))

    # ── Descarga ─────────────────────────────────────────────────────
    response = client.get_events(sport=sport_key, markets=markets)

    if not response.success:
        print(
            f"ERROR de la API: {response.error_type} — "
            f"{response.error_message}",
            file=sys.stderr,
        )
        return 1

    remaining = response.requests_remaining
    if remaining is not None:
        print(f"  Créditos restantes: {remaining}")
        if args.min_credits and remaining < args.min_credits:
            print(
                f"\nABORTA: quedan {remaining} créditos, por debajo del "
                f"umbral de {args.min_credits}. No se archiva nada.\n"
                f"Subir el plan o bajar --markets.",
                file=sys.stderr,
            )
            return 1

    if not response.events:
        print()
        print("  Sin eventos próximos para este deporte.")
        print("  No es un error: es ausencia de datos. Nada que archivar.")
        print()
        return 2

    print(f"  Eventos devueltos : {len(response.events)}")

    # ── Normalización y filtro de ventana ────────────────────────────
    normalizer = OddsNormalizer()
    archive = BookArchive(
        store=CsvBookArchiveStore(args.base_dir),
        enabled=not args.dry_run,
    )

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    batch: list[tuple[str, str | None, list]] = []
    skipped_window = 0
    skipped_empty = 0

    for raw_event in response.events:
        if args.max_lead_minutes is not None:
            start = parse_iso8601(raw_event.commence_time)
            if start is not None:
                lead = (start - now).total_seconds() / 60.0
                if lead > args.max_lead_minutes:
                    skipped_window += 1
                    continue

        quotes = normalizer.extract_all(
            raw_event=raw_event,
            markets=markets,
        )
        if not quotes:
            skipped_empty += 1
            continue

        batch.append((raw_event.event_id, raw_event.commence_time, quotes))

        if args.verbose:
            books = {q.bookmaker for q in quotes}
            print(
                f"    {raw_event.home_team} vs {raw_event.away_team}"
                f"  —  {len(quotes)} precios, {len(books)} casas"
            )

    if skipped_window:
        print(
            f"  Fuera de la ventana de {args.max_lead_minutes:.0f} min: "
            f"{skipped_window}"
        )
    if skipped_empty:
        print(f"  Sin cuotas utilizables : {skipped_empty}")

    if not batch:
        print()
        print("  Ningún evento quedó para archivar.")
        print()
        return 2

    # ── Archivo ──────────────────────────────────────────────────────
    result = archive.capture_many(sport=sport_key, events=batch)

    print()
    if args.dry_run:
        total = sum(len(quotes) for _, _, quotes in batch)
        books = {q.bookmaker for _, _, quotes in batch for q in quotes}
        print(
            f"  DRY RUN: se habrían archivado {total} filas de "
            f"{len(batch)} eventos y {len(books)} casas."
        )
    else:
        print(f"  {result.summary()}")
        store = CsvBookArchiveStore(args.base_dir)
        from core.odds.book_archive import _period_of  # noqa: PLC0415
        path = store.path_for(sport_key, _period_of(now))
        print(f"  Archivo: {path}")

    for message in result.errors:
        print(f"  ERROR: {message}", file=sys.stderr)

    print()
    return 1 if result.errors else 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())