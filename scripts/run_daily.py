#!/usr/bin/env python3
"""
scripts/run_daily.py

Punto de entrada del pipeline diario de predicciones.

Uso
----
    # Ejecutar pipeline MLB para hoy
    python scripts/run_daily.py --sport mlb

    # Ejecutar para una fecha específica
    python scripts/run_daily.py --sport mlb --date 2026-07-15

    # Dry run (sin tocar el ledger)
    python scripts/run_daily.py --sport mlb --dry-run

    # Solo N eventos (debug)
    python scripts/run_daily.py --sport mlb --max-events 3

    # Omitir stages específicos
    python scripts/run_daily.py --sport mlb --skip-stages 5

    # Sin notificaciones Telegram
    python scripts/run_daily.py --sport mlb --no-notify

    # Backtesting (sin registro en ledger)
    python scripts/run_daily.py --sport mlb --date 2026-06-01 --dry-run

Flujo de ejecución
-------------------
1. Cargar configuración (base.yaml + {sport}.yaml)
2. Instanciar SportPlugin para el deporte especificado
3. Construir PipelineRunner via build_runner()
4. Ejecutar runner.run(date)
5. Mostrar resumen de picks activos en consola
6. Enviar notificaciones Telegram (si configurado y no --no-notify)
7. Exportar vistas del ledger
8. Retornar exit code: 0=éxito, 1=error, 2=sin picks

Variables de entorno requeridas
---------------------------------
    ODDS_API_KEY          -- Clave de The Odds API
    TELEGRAM_BOT_TOKEN    -- Token del bot de Telegram (opcional)
    TELEGRAM_CHAT_IDS     -- IDs de chat separados por coma (opcional)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Asegurar que el proyecto raíz está en el path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── Cargar .env desde la raíz del proyecto ────────────────────────────────────
# Sin dependencia obligatoria: usa python-dotenv si está instalado,
# o parsea el .env manualmente como fallback.
def _load_dotenv() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # Fallback manual sin dependencias externas
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv()


# ── Registro de plugins disponibles ──────────────────────────────────────────

_AVAILABLE_SPORTS: dict[str, str] = {
    "mlb": "sports.mlb.plugin.MLBPlugin",
    # Futuras extensiones:
    # "nba": "sports.nba.plugin.NBAPlugin",
    # "nfl": "sports.nfl.plugin.NFLPlugin",
}


def main() -> int:
    """
    Punto de entrada principal.

    Retorna
    -------
    0  -- Éxito con picks activos
    2  -- Éxito pero sin picks activos (no hay valor hoy)
    1  -- Error fatal
    """
    args = _parse_args()

    # ── Fecha de ejecución ────────────────────────────────────────────
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sport = args.sport.lower()

    print(f"\n{'='*60}")
    print(f"  SPORTS PREDICTOR ADVANCED")
    print(f"  Sport: {sport.upper()} | Fecha: {date}")
    print(f"  Modo: {'DRY RUN' if args.dry_run else 'PRODUCCION'}")
    print(f"{'='*60}\n")

    # ── Validar deporte ───────────────────────────────────────────────
    if sport not in _AVAILABLE_SPORTS:
        print(f"ERROR: Deporte '{sport}' no soportado.")
        print(f"Deportes disponibles: {', '.join(_AVAILABLE_SPORTS)}")
        return 1

    # ── Cargar configuración ──────────────────────────────────────────
    print("Cargando configuración...")
    try:
        config = _load_config(sport)
    except Exception as e:
        print(f"ERROR cargando configuración: {e}")
        return 1

    # ── Validar API key ───────────────────────────────────────────────
    odds_api_key = os.environ.get("ODDS_API_KEY") or config.get("ODDS_API_KEY")
    if not odds_api_key and not args.dry_run:
        print("ERROR: ODDS_API_KEY no configurada.")
        print("Definir en .env o como variable de entorno.")
        print("Tip: usar --dry-run para ejecutar sin API key.")
        return 1

    # ── Instanciar plugin ─────────────────────────────────────────────
    print(f"Cargando plugin {sport.upper()}...")
    try:
        plugin = _load_plugin(sport, config)
    except Exception as e:
        print(f"ERROR cargando plugin: {e}")
        return 1

    # ── Construir runner ──────────────────────────────────────────────
    print("Construyendo pipeline...")
    try:
        from core.pipeline.runner import build_runner, RunnerConfig

        skip = frozenset(args.skip_stages) if args.skip_stages else frozenset()

        # En dry-run sin API key, saltamos el stage de odds
        if args.dry_run and not odds_api_key:
            skip = skip | {5}

        # CORRECCIÓN (auditoría 2026-08): dos problemas en esta llamada.
        # (1) `bankroll` se pasaba directo desde `config.get(...)` sin
        #     forzar el tipo — su valor podía venir como dict/None según
        #     lo que hubiera en el YAML, violando el `bankroll: float`
        #     que espera `build_runner()`. Se envuelve en `float()`.
        # (2) `--bankroll` estaba definido como argumento CLI pero nunca
        #     se leía en ningún punto del script — quien lo pasara no
        #     tenía ningún efecto. Ahora tiene prioridad sobre el YAML
        #     si el usuario lo especifica explícitamente.
        bankroll_cfg = config.get("bankroll.initial_bankroll", default=1000.0)
        raw_bankroll = args.bankroll if args.bankroll is not None else bankroll_cfg
        # isinstance en vez de try/except float(): pyright no estrecha el
        # tipo estático de un argumento por estar dentro de un
        # try/except — necesita un chequeo explícito de tipo para saber
        # que `raw_bankroll` es realmente `int | float | str` antes de
        # pasarlo a `float()`.
        if isinstance(raw_bankroll, (int, float, str)):
            try:
                bankroll = float(raw_bankroll)
            except ValueError:
                bankroll = None
        else:
            bankroll = None

        if bankroll is None:
            print(
                f"ERROR: bankroll.initial_bankroll en el YAML no es un "
                f"número válido (valor: {bankroll_cfg!r}). Usando 1000.0 "
                f"por defecto."
            )
            bankroll = 1000.0

        runner = build_runner(
            plugin        = plugin,
            config_loader = config,
            bankroll      = bankroll,
            dry_run       = args.dry_run,
        )
        # Sobrescribir config del runner con los args CLI
        runner._config = RunnerConfig(
            dry_run       = args.dry_run,
            skip_stages   = skip,
            max_events    = args.max_events,
            log_stage_times = True,
        )
    except Exception as e:
        print(f"ERROR construyendo pipeline: {e}")
        import traceback; traceback.print_exc()
        return 1

    # ── Ejecutar pipeline ─────────────────────────────────────────────
    print(f"\nEjecutando pipeline para {date}...\n")
    try:
        result = runner.run(date=date)
    except Exception as e:
        print(f"ERROR en ejecución del pipeline: {e}")
        import traceback; traceback.print_exc()
        return 1

    # ── Mostrar resultado ─────────────────────────────────────────────
    _print_result(result)

    # ── Notificaciones Telegram ───────────────────────────────────────
    if not args.no_notify and not args.dry_run:
        _send_notifications(result, config, date)

    # ── Exportar ledger ───────────────────────────────────────────────
    if not args.dry_run:
        _export_ledger(runner, sport)

    # ── Exit code ─────────────────────────────────────────────────────
    if result.errors and not result.active_picks:
        print(f"\n{len(result.errors)} errores sin picks activos.")
        return 1

    return 0 if result.active_picks else 2


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sports Predictor Advanced — Pipeline diario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sport", required=True,
        choices=list(_AVAILABLE_SPORTS.keys()),
        help="Deporte a procesar (ej: mlb)",
    )
    parser.add_argument(
        "--date", default=None,
        help="Fecha en YYYY-MM-DD. Default: hoy UTC.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ejecutar sin modificar el ledger ni enviar notificaciones.",
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Limitar a N eventos (útil para debugging).",
    )
    parser.add_argument(
        "--skip-stages", type=int, nargs="+", default=None,
        metavar="N",
        help="Números de stages a omitir (ej: --skip-stages 5 10).",
    )
    parser.add_argument(
        "--no-notify", action="store_true",
        help="No enviar notificaciones Telegram.",
    )
    parser.add_argument(
        "--bankroll", type=float, default=None,
        help="Bankroll inicial (sobreescribe config).",
    )
    return parser.parse_args()


def _load_config(sport: str):
    """
    Carga base.yaml + {sport}.yaml con deep-merge.

    CORRECCIÓN DE CONTRATO (auditoría 2026-08): llamaba a
    `ConfigLoader.load(sport=sport)` — esa clase nunca tuvo un
    classmethod `load`; la función real es `load_config()` a nivel de
    módulo en `core/utils/config_loader.py`. El `except ImportError` de
    abajo tampoco habría salvado esto: el error real era `AttributeError`
    (atributo inexistente en la clase), no `ImportError` — así que el
    fallback a `_yaml_config_fallback` nunca se activaba y la excepción
    subía sin capturar, tumbando `run_daily.py` en el arranque para
    cualquier deporte, siempre.
    """
    try:
        from core.utils.config_loader import load_config
        return load_config(sport=sport)
    except ImportError:
        # Fallback: YAML directo si ConfigLoader no está disponible
        return _yaml_config_fallback(sport)


def _yaml_config_fallback(sport: str):
    """ConfigLoader mínimo desde YAML si el módulo no está disponible."""
    try:
        import yaml
    except ImportError:
        return _DictConfig({})

    def _deep_merge(base, override):
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    config = {}
    base_path   = _PROJECT_ROOT / "config" / "base.yaml"
    sport_path  = _PROJECT_ROOT / "config" / f"{sport}.yaml"

    if base_path.exists():
        with open(base_path) as f:
            config = yaml.safe_load(f) or {}

    if sport_path.exists():
        with open(sport_path) as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)

    # Inyectar variables de entorno
    for env_key in ("ODDS_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_IDS"):
        val = os.environ.get(env_key)
        if val:
            config[env_key] = val

    return _DictConfig(config)


class _DictConfig:
    """ConfigLoader mínimo sobre un dict."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, path: str, default=None):
        keys = path.split(".")
        cur  = self._data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur if cur is not None else default


def _load_plugin(sport: str, config):
    """Instancia el SportPlugin para el deporte dado."""
    module_path, class_name = _AVAILABLE_SPORTS[sport].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    plugin_cls = getattr(module, class_name)
    return plugin_cls(config_loader=config)


def _print_result(result) -> None:
    """Muestra el resultado del pipeline en consola."""
    print(f"\n{'-'*60}")
    print(f"  RESULTADO: {result.summary()}")
    print(f"{'-'*60}")

    if result.active_picks:
        print(f"\nPICKS ACTIVOS ({len(result.active_picks)}):\n")
        for i, pick in enumerate(result.active_picks, 1):
            line_str = f" {pick.line}" if pick.line is not None else ""
            print(
                f"  {i}. [{pick.market}] {pick.selection}{line_str} "
                f"@ {pick.price} | EV={pick.ev:+.1f}% | "
                f"Stake={pick.stake_pct}%"
            )
    else:
        print("\n  Sin picks activos para hoy.")
        print("  (Los filtros de EV y riesgo no aprobaron candidatos)")

    if result.errors:
        print(f"\nERRORES ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"  - {err}")
        if len(result.errors) > 5:
            print(f"  ... y {len(result.errors)-5} más.")

    meta = result.context.metadata
    if meta.get("risk_summary"):
        print(f"\n  {meta['risk_summary']}")

    credits = meta.get("odds_credits_remaining")
    if credits is not None:
        print(f"  Créditos API restantes: {credits}")

    print()


def _send_notifications(result, config, date: str) -> None:
    """Envía notificaciones Telegram si están configuradas."""
    try:
        from core.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier.from_config(config)
        if notifier is None:
            return  # Sin credenciales — silencioso

        # Construir roi_summary básico
        roi_summary = {
            "sport": result.sport,
            "total_apuestas": 0,
            "wins": 0, "losses": 0, "pendientes": len(result.active_picks),
            "roi": 0.0, "bankroll": 0.0,
        }

        notif_result = notifier.send_picks(
            picks       = result.active_picks,
            roi_summary = roi_summary,
            date        = date,
        )
        print(f"Notificaciones: {notif_result.summary()}")
    except Exception as e:
        print(f"Notificaciones fallaron (no crítico): {e}")


def _export_ledger(runner, sport: str) -> None:
    """Exporta vistas del ledger por deporte."""
    try:
        roi_tracker = runner._roi_tracker
        paths = roi_tracker.export_sport_views(
            base_dir=f"output/ledger"
        )
        if paths:
            print(f"Ledger exportado: {', '.join(paths[:2])}")
    except Exception as e:
        print(f"Export ledger fallido (no crítico): {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())