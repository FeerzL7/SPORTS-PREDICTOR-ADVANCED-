#!/usr/bin/env python3
"""
scripts/backtest_nfl.py

Backtest walk-forward del plugin NFL sobre temporadas históricas.

Qué hace posible este backtest
--------------------------------
Un backtest de apuestas necesita las líneas de mercado históricas, no
solo los resultados. The Odds API no las ofrece de forma asequible para
temporadas pasadas.

nflverse sí las incluye en el schedule: `spread_line`, `total_line`,
`home_moneyline` y `away_moneyline`. Eso convierte un ejercicio
imposible en uno directo.

Advertencia importante sobre esas líneas
------------------------------------------
Son líneas de CIERRE. Apostar contra el cierre es considerablemente más
difícil que apostar en vivo durante la semana:

    El cierre incorpora toda la información pública y todo el dinero
    profesional. Es el punto de máxima eficiencia del mercado.

    En producción el pipeline opera con líneas de miércoles a domingo
    por la mañana, que son sensiblemente más blandas.

Por tanto: el ROI de este backtest es un SUELO, no una estimación
central. Un resultado plano contra el cierre es compatible con un ROI
positivo en producción; uno negativo contra el cierre es una señal
inequívoca de que el modelo no tiene edge.

Metodología walk-forward
--------------------------
Para cada partido de la semana N, el modelo solo ve datos hasta la
semana N-1. Esa barrera la impone NFLDataProvider._cutoff_week(), no
este script — así el backtest ejercita exactamente el mismo código que
producción, en vez de una reimplementación que podría divergir.

La semana 1 se excluye: no hay datos previos de la temporada y el
modelo proyectaría con la media de liga para todos los equipos.

Uso
----
    # Temporada completa
    python scripts/backtest_nfl.py --seasons 2023 2024

    # Una temporada, solo hasta la semana 10
    python scripts/backtest_nfl.py --seasons 2024 --max-week 10

    # Guardar el detalle de cada pick
    python scripts/backtest_nfl.py --seasons 2023 2024 --output backtest.csv

Requisitos
-----------
    pip install nfl_data_py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.contracts.event import Event, EventStatus
from core.contracts.market_odds import MarketOdds


# Cuota decimal por defecto para spread y total cuando nflverse no
# expone el juice. -110 es el estándar del mercado NFL.
_DEFAULT_SPREAD_PRICE: float = 1.9091
_DEFAULT_TOTAL_PRICE:  float = 1.9091

# La semana 1 no se apuesta: sin datos previos de la temporada, el
# modelo proyecta con la media de liga y no aporta información.
_FIRST_BETTABLE_WEEK: int = 2


# ── Registro de resultados ───────────────────────────────────────────────────

@dataclass
class BetRecord:
    """Un pick simulado y su resolución."""
    season:     int
    week:       int
    game_id:    str
    matchup:    str
    market:     str
    selection:  str
    line:       float | None
    price:      float
    model_prob: float
    ev:         float
    stake_pct:  int
    result:     str
    profit:     float
    confidence: float


@dataclass
class BacktestMetrics:
    """Métricas agregadas de un conjunto de picks."""
    n_bets:     int = 0
    wins:       int = 0
    losses:     int = 0
    pushes:     int = 0
    voids:      int = 0
    staked:     float = 0.0
    profit:     float = 0.0
    records:    list[BetRecord] = field(default_factory=list)

    def add(self, record: BetRecord) -> None:
        self.records.append(record)
        self.n_bets += 1
        self.staked += record.stake_pct
        self.profit += record.profit
        if record.result == "win":
            self.wins += 1
        elif record.result == "lose":
            self.losses += 1
        elif record.result == "null":
            self.pushes += 1
        else:
            self.voids += 1

    @property
    def resolved(self) -> int:
        """Apuestas con ganador: los push no cuentan para el hit rate."""
        return self.wins + self.losses

    @property
    def hit_rate(self) -> float:
        return 100.0 * self.wins / self.resolved if self.resolved else 0.0

    @property
    def roi(self) -> float:
        """ROI sobre el total apostado, en unidades de stake."""
        return 100.0 * self.profit / self.staked if self.staked else 0.0

    @property
    def breakeven_rate(self) -> float:
        """
        Hit rate necesario para no perder a la cuota media.

        A -110 (1.9091) el break-even es 52.38%. Comparar el hit rate
        contra este número dice más que el ROI aislado: un 51% a -110
        pierde dinero por mucho que parezca "casi la mitad".
        """
        if not self.records:
            return 0.0
        avg_price = sum(r.price for r in self.records) / len(self.records)
        return 100.0 / avg_price


# ── Construcción de cuotas desde nflverse ────────────────────────────────────

def _american_to_decimal(american) -> float | None:
    """
    Convierte cuota americana a decimal.

    nflverse expone los moneylines en formato americano (-150, +130),
    mientras que el resto del sistema trabaja en decimal.
    """
    value = _safe_float(american)
    if value is None or value == 0:
        return None
    if value > 0:
        return round(value / 100.0 + 1.0, 4)
    return round(100.0 / abs(value) + 1.0, 4)


def _odds_from_game(game, event: Event) -> list[MarketOdds]:
    """
    Construye MarketOdds desde las líneas de cierre de nflverse.

    Convención de `spread_line` en nflverse: es el handicap desde la
    perspectiva del LOCAL, positivo cuando el local es favorito. Un
    spread_line de 3.5 significa que el local estaba favorito por 3.5,
    lo que en notación de mercado es local -3.5 y visitante +3.5.

    El resto del sistema usa la línea PROPIA de cada selección (misma
    convención que MarketOdds.line y que NormalModel tras la corrección
    de la tarea 10.11), así que hay que invertir el signo.
    """
    odds: list[MarketOdds] = []
    ts = f"{event.date}T12:00:00Z"
    eid = event.event_id
    home, away = event.home_team_id, event.away_team_id

    # ── Moneyline ──
    home_ml = _american_to_decimal(game.home_moneyline)
    away_ml = _american_to_decimal(game.away_moneyline)
    if home_ml and away_ml:
        odds.append(MarketOdds(event_id=eid, market="ML", selection=home,
                               line=None, price=home_ml,
                               bookmaker="nflverse_close", timestamp=ts))
        odds.append(MarketOdds(event_id=eid, market="ML", selection=away,
                               line=None, price=away_ml,
                               bookmaker="nflverse_close", timestamp=ts))

    # ── Spread ──
    spread = _safe_float(game.spread_line)
    if spread is not None:
        home_price = (_american_to_decimal(game.home_spread_odds)
                      or _DEFAULT_SPREAD_PRICE)
        away_price = (_american_to_decimal(game.away_spread_odds)
                      or _DEFAULT_SPREAD_PRICE)
        odds.append(MarketOdds(event_id=eid, market="SPREAD", selection=home,
                               line=-spread, price=home_price,
                               bookmaker="nflverse_close", timestamp=ts))
        odds.append(MarketOdds(event_id=eid, market="SPREAD", selection=away,
                               line=spread, price=away_price,
                               bookmaker="nflverse_close", timestamp=ts))

    # ── Total ──
    total = _safe_float(game.total_line)
    if total is not None:
        over_price  = (_american_to_decimal(game.over_odds)
                       or _DEFAULT_TOTAL_PRICE)
        under_price = (_american_to_decimal(game.under_odds)
                       or _DEFAULT_TOTAL_PRICE)
        odds.append(MarketOdds(event_id=eid, market="TOTAL", selection="over",
                               line=total, price=over_price,
                               bookmaker="nflverse_close", timestamp=ts))
        odds.append(MarketOdds(event_id=eid, market="TOTAL", selection="under",
                               line=total, price=under_price,
                               bookmaker="nflverse_close", timestamp=ts))

    return odds


def _model_probs(projection, event: Event, prob_model, odds) -> dict[str, float]:
    """
    Mapea la proyección a probabilidades por selección.

    El ValueEngine exige que las claves coincidan EXACTAMENTE con
    MarketOdds.selection, así que se construyen desde las cuotas en vez
    de asumir nombres.
    """
    home, away = event.home_team_id, event.away_team_id
    probs: dict[str, float] = {}

    for o in odds:
        if o.market == "ML":
            probs[o.selection] = (
                projection.home_win_prob if o.selection == home
                else projection.away_win_prob
            )
        elif o.market == "SPREAD" and o.line is not None:
            side = "home" if o.selection == home else "away"
            probs[o.selection] = prob_model.spread_probability(
                projection, o.line, side
            )
        elif o.market == "TOTAL" and o.line is not None:
            probs[o.selection] = prob_model.total_probability(
                projection, o.line, o.selection
            )

    # El engine rechaza probabilidades fuera de (0,1)
    return {k: min(max(v, 0.001), 0.999) for k, v in probs.items() if v is not None}


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_season(season: int, max_week: int, config, verbose: bool,
               neutralize_weather: bool = False) -> BacktestMetrics:
    """Ejecuta el backtest de una temporada completa."""
    from core.value.blending import BlendingEngine
    from core.value.engine import EvaluationRequest, ValueEngine
    from core.value.filters import MarketFilters
    from core.value.kelly import KellyCriterion
    from sports.nfl.plugin import NFLPlugin

    plugin     = NFLPlugin(config_loader=config, season=season)
    provider   = plugin.get_data_provider()
    model      = plugin.get_projection_model()
    prob_model = plugin.get_probability_model()
    settlement = plugin.get_settlement_provider()
    schedule   = plugin._get_schedule()

    engine = ValueEngine(
        blending = BlendingEngine(config=config),
        kelly    = KellyCriterion(config=config),
        filters  = MarketFilters(config=config),
    )

    metrics = BacktestMetrics()
    games = schedule._games()

    # Contadores de descarte.
    #
    # Sin esto, un backtest que produce cero picks es indistinguible de
    # uno donde el modelo simplemente no encontró valor — y la primera
    # ejecución real falló exactamente así: NFLGameInfo no cargaba las
    # líneas de mercado, `_odds_from_game` devolvía lista vacía y el
    # informe decía "muestra insuficiente" como si fuera un resultado
    # estadístico.
    skipped = {"sin_lineas": 0, "proyeccion": 0, "evaluacion": 0,
               "sin_resultado": 0, "filtrados": 0, "evaluados": 0}

    for week in range(_FIRST_BETTABLE_WEEK, max_week + 1):
        week_games = [g for g in games
                      if g.week == week and g.game_type == "REG" and g.is_final]
        if not week_games:
            continue

        for game in week_games:
            event = _game_to_event(game, season)
            odds = _odds_from_game(game, event)
            if not odds:
                skipped["sin_lineas"] += 1
                continue

            try:
                home_f, away_f = provider.enrich_event(event)
                context = provider.get_context(event)

                # El clima observado de nflverse es información que NO
                # estaba disponible al apostar: se mide durante el
                # partido. Usarlo en el backtest es look-ahead bias, y
                # afecta desproporcionadamente al mercado de totales
                # porque el viento suprime la anotación de ambos
                # equipos sin mover el margen.
                #
                # --neutralize-weather permite falsar la hipótesis de
                # que el edge observado en TOTAL viene de ahí: si el
                # rendimiento se desploma al desactivarlo, el edge no
                # era del modelo.
                if neutralize_weather:
                    context["weather_factor"] = 1.0

                projection = model.project(home_f, away_f, context)
            except Exception as e:
                skipped["proyeccion"] += 1
                if verbose:
                    print(f"    ⚠️  {game.game_id}: proyección falló ({e})")
                continue

            probs = _model_probs(projection, event, prob_model, odds)
            if not probs:
                continue

            try:
                result = engine.evaluate(EvaluationRequest(
                    event=event, model_probs=probs, market_odds=odds,
                    projected_value=projection.expected_home - projection.expected_away,
                ))
            except Exception as e:
                skipped["evaluacion"] += 1
                if verbose:
                    print(f"    ⚠️  {game.game_id}: evaluación falló ({e})")
                continue

            event_result = settlement.get_event_result(event)
            if event_result is None:
                skipped["sin_resultado"] += 1
                continue

            skipped["evaluados"] += 1
            skipped["filtrados"] += len(result.picks_rejected)

            for candidate in result.picks_passed:
                stake = candidate.stake_pct or 1
                outcome = settlement.settle_pick(candidate, event_result)
                profit = _profit(outcome, stake, candidate.price)

                metrics.add(BetRecord(
                    season=season, week=week, game_id=game.game_id,
                    matchup=f"{game.away_team} @ {game.home_team}",
                    market=candidate.market, selection=candidate.selection,
                    line=candidate.line, price=candidate.price,
                    model_prob=candidate.blended_prob, ev=candidate.ev,
                    stake_pct=stake, result=outcome, profit=profit,
                    confidence=projection.confidence,
                ))

        if verbose and metrics.n_bets:
            print(f"    semana {week:2d}: {metrics.n_bets:3d} picks acumulados, "
                  f"ROI {metrics.roi:+6.2f}%")

    if metrics.n_bets == 0:
        _explain_empty(skipped, season)

    return metrics


def _explain_empty(skipped: dict, season: int) -> None:
    """
    Explica por qué una temporada no produjo ningún pick.

    Cero picks puede significar dos cosas muy distintas: que el modelo
    no encontró valor (un resultado legítimo) o que algo se rompió
    antes de llegar a evaluar (un fallo). Confundirlas lleva a
    interpretar un error de integración como una conclusión sobre el
    edge del modelo.
    """
    print(f"    Ningún pick en {season}. Desglose:")
    print(f"      partidos evaluados      : {skipped['evaluados']}")
    print(f"      sin líneas de mercado   : {skipped['sin_lineas']}")
    print(f"      proyección falló        : {skipped['proyeccion']}")
    print(f"      evaluación falló        : {skipped['evaluacion']}")
    print(f"      sin resultado final     : {skipped['sin_resultado']}")
    print(f"      candidatos filtrados    : {skipped['filtrados']}")

    if skipped["evaluados"] == 0 and skipped["sin_lineas"] > 0:
        print()
        print("      → Ningún partido tenía líneas de mercado. nflverse las")
        print("        expone en el schedule (spread_line, total_line,")
        print("        home_moneyline). Verificar que NFLGameInfo las carga.")
    elif skipped["evaluados"] > 0 and skipped["filtrados"] > 0:
        print()
        print("      → Se evaluaron candidatos pero los filtros los")
        print("        descartaron todos. Eso SÍ es un resultado del modelo:")
        print("        no encontró valor suficiente contra el cierre.")


def _profit(outcome: str, stake: float, price: float) -> float:
    """Beneficio en unidades de stake."""
    if outcome == "win":
        return stake * (price - 1.0)
    if outcome == "lose":
        return -stake
    return 0.0   # push y void devuelven el stake


def _game_to_event(game, season: int) -> Event:
    """NFLGameInfo → Event, con el formato que espera el pipeline."""
    return Event(
        event_id=game.game_id, sport="nfl", league="NFL",
        season_start=season, season_end=season + 1,
        date=game.gameday, start_time=f"{game.gameday}T18:00:00Z",
        home_team_id=game.home_team, away_team_id=game.away_team,
        home_team=game.home_team, away_team=game.away_team,
        venue_id=game.home_team, venue_name=game.stadium or "",
        status=EventStatus.FINAL,
        provider_ids={"nfl_game_id": game.game_id},
    )


# ── Informe ──────────────────────────────────────────────────────────────────

def _report(metrics: BacktestMetrics, label: str) -> None:
    print(f"\n{'─' * 66}")
    print(f"  {label}")
    print(f"{'─' * 66}")
    if not metrics.n_bets:
        print("  Sin picks generados.")
        return

    print(f"    Picks              : {metrics.n_bets}")
    print(f"    Ganados / Perdidos : {metrics.wins} / {metrics.losses}")
    print(f"    Push / Anulados    : {metrics.pushes} / {metrics.voids}")
    print(f"    Hit rate           : {metrics.hit_rate:.2f}%")
    print(f"    Break-even         : {metrics.breakeven_rate:.2f}%  "
          f"(a la cuota media)")
    print(f"    Margen sobre BE    : {metrics.hit_rate - metrics.breakeven_rate:+.2f} pp")
    print(f"    Stake total        : {metrics.staked:.0f} unidades")
    print(f"    Beneficio          : {metrics.profit:+.2f} unidades")
    print(f"    ROI                : {metrics.roi:+.2f}%")

    # Desglose por mercado
    by_market: dict[str, BacktestMetrics] = {}
    for r in metrics.records:
        by_market.setdefault(r.market, BacktestMetrics()).add(r)
    if len(by_market) > 1:
        print(f"\n    Por mercado:")
        for market, m in sorted(by_market.items()):
            print(f"      {market:8s} n={m.n_bets:3d}  "
                  f"hit={m.hit_rate:5.1f}%  ROI={m.roi:+6.2f}%")


def _interpret(metrics: BacktestMetrics) -> None:
    """Lectura honesta del resultado."""
    print(f"\n{'=' * 66}")
    print("  LECTURA DEL RESULTADO")
    print(f"{'=' * 66}")

    if metrics.n_bets < 50:
        print(f"  Muestra insuficiente ({metrics.n_bets} picks).")
        print("  Con menos de ~150 picks el ROI observado es indistinguible")
        print("  del ruido. Ampliar el rango de temporadas antes de sacar")
        print("  cualquier conclusión.")
        return

    margin = metrics.hit_rate - metrics.breakeven_rate

    print(f"  Las líneas usadas son de CIERRE, el punto de máxima")
    print(f"  eficiencia del mercado. En producción el pipeline opera con")
    print(f"  líneas más blandas, así que este resultado es un SUELO.")
    print()

    if margin > 2.0:
        print(f"  Hit rate {margin:+.2f} pp sobre break-even contra el cierre.")
        print("  Es un resultado fuerte. Conviene verificar que no proviene")
        print("  de un mercado o temporada concretos antes de confiar en él.")
    elif margin > 0:
        print(f"  Hit rate {margin:+.2f} pp sobre break-even. Positivo pero")
        print("  dentro del rango donde el ruido domina con esta muestra.")
    else:
        print(f"  Hit rate {margin:+.2f} pp por DEBAJO del break-even.")
        print("  Contra líneas de cierre esto no descarta edge en producción,")
        print("  pero sí obliga a revisar la calibración antes de arriesgar")
        print("  capital real.")

    if metrics.n_bets < 150:
        print()
        print(f"  Nota: {metrics.n_bets} picks siguen por debajo del umbral de")
        print("  150 que config/nfl.yaml fija para considerar recalibrada la")
        print("  configuración.")


def _export(metrics: BacktestMetrics, path: str) -> None:
    """Exporta el detalle de cada pick a CSV."""
    if not metrics.records:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(vars(metrics.records[0])))
        writer.writeheader()
        for r in metrics.records:
            writer.writerow(vars(r))
    print(f"\n  Detalle exportado: {path} ({len(metrics.records)} filas)")


# ── Utilidades ───────────────────────────────────────────────────────────────

def _is_nan(value) -> bool:
    try:
        return value != value
    except Exception:
        return False


def _safe_float(value) -> float | None:
    if value is None or _is_nan(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ── Entrada ──────────────────────────────────────────────────────────────────

def _verify_imports() -> str | None:
    """
    Comprueba que los módulos del plugin exponen lo que el backtest usa.

    Retorna un diagnóstico accionable si algo falta, o None si todo
    está en su sitio.

    Existe porque un ImportError capturado dentro del bucle de
    temporadas produce un informe engañoso: el script termina
    reportando "0 picks — muestra insuficiente", que parece un
    resultado del modelo cuando en realidad el modelo nunca se
    ejecutó. Distinguir un fallo de instalación de un resultado
    estadístico importa más que la elegancia del manejo de errores.
    """
    required = [
        ("sports.nfl.provider",    "NFLDataProvider"),
        ("sports.nfl.plugin",      "NFLPlugin"),
        ("sports.nfl.projections", "NFLProjectionModel"),
        ("sports.nfl.settlement",  "NFLSettlementProvider"),
        ("sports.nfl.schedule",    "NFLScheduleFetcher"),
        ("core.value.engine",      "ValueEngine"),
    ]

    import importlib

    for module_name, symbol in required:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            return (
                f"  No se pudo importar {module_name}\n"
                f"    {type(e).__name__}: {e}\n\n"
                f"  Revisar que el archivo exista y que sus propias\n"
                f"  dependencias estén instaladas."
            )

        if not hasattr(module, symbol):
            exported = sorted(
                n for n in dir(module) if n[0].isupper() and not n.startswith("_")
            )
            path = getattr(module, "__file__", "(desconocido)")
            return (
                f"  El módulo {module_name} no expone '{symbol}'.\n\n"
                f"    Archivo cargado : {path}\n"
                f"    Sí exporta      : {', '.join(exported) or '(nada)'}\n\n"
                f"  Causa habitual: el archivo en disco es una versión\n"
                f"  anterior a la que define esa clase. Verificar que se\n"
                f"  copió la última versión de {module_name.replace('.', '/')}.py"
            )

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest walk-forward del plugin NFL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                        help="Temporadas a backtestear (ej: 2023 2024)")
    parser.add_argument("--max-week", type=int, default=18,
                        help="Última semana a incluir. Default 18.")
    parser.add_argument("--output", default=None,
                        help="CSV donde exportar el detalle de cada pick")
    parser.add_argument("--neutralize-weather", action="store_true",
                        help="Ignora el clima (factor 1.0). Sirve para "
                             "comprobar si el edge depende de conocer las "
                             "condiciones reales del partido, que al "
                             "apostar no se conocen.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from sports.nfl.plugin import NFLPlugin
    if not NFLPlugin.is_available():
        print("ERROR: el backtest requiere nfl_data_py.")
        print("  Instalar con: pip install nfl_data_py")
        return 1

    # Verificar que los componentes del plugin se importan antes de
    # empezar. Un ImportError es un problema de instalación, no de
    # datos: repetirlo por cada temporada oculta la causa y produce un
    # informe de "0 picks" que parece un resultado del modelo cuando
    # en realidad el modelo nunca llegó a ejecutarse.
    problem = _verify_imports()
    if problem:
        print("ERROR: el plugin NFL no se puede cargar.\n")
        print(problem)
        return 1

    from core.utils.config_loader import load_config
    config = load_config(sport="nfl", base_dir="config")

    print("=" * 66)
    print("  BACKTEST NFL — walk-forward contra líneas de cierre")
    print("=" * 66)
    print(f"  Temporadas : {', '.join(str(s) for s in args.seasons)}")
    print(f"  Semanas    : {_FIRST_BETTABLE_WEEK} a {args.max_week}")
    print()
    if args.neutralize_weather:
        print("  ⚠️  CLIMA NEUTRALIZADO (factor 1.0)")
        print("      El clima de nflverse se mide DURANTE el partido, así")
        print("      que usarlo es información del futuro. Comparar este")
        print("      resultado con el normal revela cuánto del edge venía")
        print("      de ahí.")
        print()

    print("  Metodología: para cada partido de la semana N el modelo solo")
    print("  ve datos hasta N-1. Esa barrera la impone el propio provider")
    print("  (NFLDataProvider._cutoff_week), así que el backtest ejercita")
    print("  el mismo código que producción.")
    print()

    overall = BacktestMetrics()

    for season in args.seasons:
        print(f"  Temporada {season}...")
        try:
            metrics = run_season(season, args.max_week, config, args.verbose,
                                 neutralize_weather=args.neutralize_weather)
        except (ImportError, AttributeError) as e:
            # Fallo estructural: afectaría igual a todas las temporadas.
            # Abortar en vez de repetir el mismo error N veces y cerrar
            # con un informe de cero picks que parece un resultado.
            print(f"    ERROR ESTRUCTURAL: {type(e).__name__}: {e}")
            print("\n  El backtest se detiene: este fallo no depende de los")
            print("  datos de la temporada y se repetiría en todas.")
            return 1
        except Exception as e:
            # Fallo de datos: la temporada se omite, las demás siguen.
            print(f"    ERROR en datos de {season}: {type(e).__name__}: {e}")
            continue
        _report(metrics, f"TEMPORADA {season}")
        for r in metrics.records:
            overall.add(r)

    if len(args.seasons) > 1:
        _report(overall, "AGREGADO")

    _interpret(overall)

    if args.output:
        _export(overall, args.output)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())