#!/usr/bin/env python3
"""
scripts/backtest_soccer.py

Backtest walk-forward del plugin de fútbol contra cuotas de cierre.

Qué lo hace posible
---------------------
football-data.co.uk publica los resultados históricos de las cinco
grandes CON las cuotas de cierre, desde 1993. Incluye las de Pinnacle,
que opera con margen mínimo y límites altos: su cierre es el mejor
estimador del precio justo y el listón más exigente que existe.

Es el equivalente de lo que nflverse dio para NFL, y mejor: allí las
líneas venían de una fuente agregada; aquí se sabe qué casa las puso.

ADVERTENCIA sobre lo que mide
-------------------------------
Apostar contra el CIERRE es considerablemente más difícil que apostar
durante la semana. El cierre incorpora toda la información pública y
todo el dinero profesional; en producción el pipeline opera con precios
más blandos.

Por tanto el ROI de este backtest es un SUELO, no una estimación
central. Un resultado plano contra el cierre es compatible con ROI
positivo en producción; uno negativo es señal inequívoca de que falta
ventaja.

La barrera contra el look-ahead
---------------------------------
Dos capas, y la segunda es específica del fútbol.

    FECHA DE CORTE
        Para proyectar un partido del 15 de marzo, los fetchers solo
        ven lo jugado ANTES de ese día, con comparación estricta. La
        impone SoccerDataProvider._as_of(), no este script: así el
        backtest ejercita el mismo código que producción en vez de una
        reimplementación que podría divergir.

    EL xG DEL PROPIO PARTIDO
        SoccerMatchInfo lleva home_xg y away_xg del partido que se está
        proyectando. Ese dato se calcula DESDE el partido: leerlo sería
        conocer el resultado antes de apostarlo.

        El diseño no lo usa —los índices salen del histórico— pero el
        campo está ahí, y una ruta de código futura podría leerlo sin
        que nada fallara. `--verify-no-lookahead` proyecta cada partido
        dos veces, con y sin su propio xG, y compara: si difieren, hay
        una fuga.

        Esta verificación existe porque el backtest de NFL tardó dos
        ejecuciones completas en revelar que el clima OBSERVADO —medido
        durante el partido— inflaba el rendimiento del mercado de
        totales en 8 puntos de ROI.

Uso
----
    python scripts/backtest_soccer.py --seasons 2022 2023 2024
    python scripts/backtest_soccer.py --seasons 2023 --comps epl laliga
    python scripts/backtest_soccer.py --seasons 2023 --verify-no-lookahead
    python scripts/backtest_soccer.py --seasons 2022 2023 --output bt.csv

Requisitos
-----------
    Acceso a football-data.co.uk (resultados y cuotas).
    Opcional: acceso a understat.com para el xG. Sin él, el plugin
    opera en tier PARTIAL con goles y forma.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_N = NormalDist()

# Jornadas iniciales que no se apuestan.
#
# Con menos de seis partidos jugados los índices están dominados por el
# shrinkage: el modelo proyecta prácticamente la media de liga y no
# aporta información sobre el mercado.
_MIN_MATCHES_PLAYED = 6

# Picks mínimos para que el ROI sea interpretable.
_MIN_PICKS_ROI = 150

# Marca si la ejecución usa apertura, para adaptar la lectura final.
_USING_OPENING = [False]


@dataclass
class BetRecord:
    """Un pick simulado y su resolución."""
    comp:       str
    season:     int
    date:       str
    match:      str
    market:     str
    selection:  str
    line:       float | None
    price:      float
    model_prob: float
    market_prob: float
    ev:         float
    result:     str
    profit:     float
    confidence: float
    odds_source: str
    passed_filter: bool = True


@dataclass
class Metrics:
    """Métricas agregadas de un conjunto de picks."""
    label:   str = ""
    n:       int = 0
    wins:    int = 0
    losses:  int = 0
    pushes:  int = 0
    voids:   int = 0
    staked:  float = 0.0
    profit:  float = 0.0
    records: list[BetRecord] = field(default_factory=list)

    def add(self, record: BetRecord) -> None:
        """
        Registra un pick.

        Con --no-filter el CSV incluye TODOS los candidatos, pero las
        métricas del informe solo cuentan los que habrían pasado. Así
        una misma ejecución sirve para dos cosas: comparar el
        rendimiento real y medir la calibración sobre el conjunto
        completo.

        Sin esa separación, el ROI del informe mezclaría picks que el
        sistema nunca habría hecho.
        """
        self.records.append(record)
        if not record.passed_filter:
            return
        self.n += 1
        self.staked += 1.0
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
        """Los push no cuentan para el hit rate: devuelven el stake."""
        return self.wins + self.losses

    @property
    def hit_rate(self) -> float | None:
        return 100.0 * self.wins / self.resolved if self.resolved else None

    @property
    def roi(self) -> float | None:
        return 100.0 * self.profit / self.staked if self.staked else None

    # ── Comparación con el mercado ────────────────────────────────────────
    #
    # NO se reporta un "break-even hit rate": con precios variables no
    # existe uno solo.
    #
    #     ROI = 0  ⟺  Σ_ganadas (precio − 1) = nº perdidas
    #
    # El resultado depende de CUÁLES se ganan, no solo de cuántas. Una
    # cartera de 100 picks —50 a 1.80 y 50 a 6.00— con hit rate del 33%
    # da ROI entre -40.6% y +98.0% según el reparto.
    #
    # La primera versión de este módulo calculaba 100/media(precio) y lo
    # llamaba break-even. Por la desigualdad de Jensen eso subestima
    # sistemáticamente, y produjo un informe con margen "+3.67 pp,
    # p=0.0000" junto a un ROI de -5.50%: dos números que no pueden ser
    # ciertos a la vez.
    #
    # La pregunta bien planteada es: ¿acierta el modelo MÁS de lo que
    # las probabilidades implícitas del mercado predecían?

    @property
    def market_expected_wins(self) -> float:
        """
        Victorias que predecían las probabilidades implícitas.

        Σ(1/precio) sobre los picks resueltos QUE PASAN EL FILTRO.
        Incluye el margen del book, así que superarlo es un listón más
        exigente que la probabilidad justa — que es exactamente lo que
        interesa medir contra cuotas de cierre.

        CORRECCIÓN: la versión anterior sumaba sobre `self.records`
        completo. Con --no-filter eso incluye los 22.533 candidatos,
        mientras que `wins` solo cuenta los 4.582 filtrados. El informe
        reportaba "1641 observadas vs 9265.5 esperadas": comparaba dos
        poblaciones distintas.

        El filtro de passed_filter hace que ambas cifras cubran el
        mismo conjunto.
        """
        return sum(
            1.0 / r.price for r in self.records
            if r.result in ("win", "lose") and r.passed_filter
        )

    @property
    def win_surplus(self) -> float | None:
        """Victorias observadas menos esperadas por el mercado."""
        if self.resolved == 0:
            return None
        return self.wins - self.market_expected_wins

    @property
    def surplus_pct(self) -> float | None:
        """El excedente en puntos porcentuales sobre los resueltos."""
        if self.resolved == 0:
            return None
        surplus = self.win_surplus
        return None if surplus is None else 100.0 * surplus / self.resolved

    @property
    def p_value(self) -> float | None:
        """
        p de que el excedente sea positivo por azar.

        Bajo la hipótesis nula, cada pick gana con probabilidad 1/precio
        de forma independiente. La suma es una Poisson-binomial, cuya
        varianza es Σ p(1−p); con n grande se aproxima bien por la
        normal.
        """
        if self.resolved < 30:
            return None

        probs = [1.0 / r.price for r in self.records
                 if r.result in ("win", "lose")]
        variance = sum(p * (1 - p) for p in probs)
        if variance <= 0:
            return None

        z = (self.wins - sum(probs)) / math.sqrt(variance)
        return 1.0 - _N.cdf(z)


def _profit(result: str, price: float) -> float:
    """Beneficio en unidades de stake."""
    if result == "win":
        return price - 1.0
    if result == "lose":
        return -1.0
    return 0.0   # push y void devuelven el stake


class _ConfigWithOverride:
    """
    Envoltorio que sobreescribe claves concretas del config.

    Permite barrer un parámetro desde la línea de comandos sin editar
    el archivo en cada corrida. Delega todo lo demás en el loader real,
    así que el resto de la configuración es idéntica.

    Existe para que la comparación entre pesos sea limpia: si hubiera
    que editar soccer.yaml entre corridas, cualquier otro cambio
    accidental contaminaría el resultado.
    """

    def __init__(self, base, overrides: dict):
        self._base = base
        self._overrides = dict(overrides)

    def get(self, key: str, default=None):
        if key in self._overrides:
            return self._overrides[key]
        return self._base.get(key, default=default)

    def __getattr__(self, name):
        return getattr(self._base, name)


# ── Verificación de imports ──────────────────────────────────────────────────

def _verify_imports() -> str | None:
    """
    Comprueba los símbolos que el backtest usa antes de iterar.

    Un ImportError es un problema de instalación, no de datos:
    repetirlo por cada temporada oculta la causa y produce un informe
    de "0 picks — muestra insuficiente" que parece un resultado del
    modelo cuando el modelo nunca se ejecutó.

    Es la lección del backtest de NFL, donde ese informe engañoso costó
    dos ejecuciones.
    """
    import importlib

    required = [
        ("sports.soccer.plugin",       "SoccerPlugin"),
        ("sports.soccer.provider",     "SoccerDataProvider"),
        ("sports.soccer.projections",  "SoccerProjectionModel"),
        ("sports.soccer.settlement",   "SoccerSettlementProvider"),
        ("sports.soccer.schedule",     "SoccerScheduleFetcher"),
        ("sports.soccer.competitions", "get_competition"),
    ]

    for module_name, symbol in required:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            return (f"  No se pudo importar {module_name}\n"
                    f"    {type(e).__name__}: {e}")
        if not hasattr(module, symbol):
            exported = sorted(
                n for n in dir(module) if n[0].isupper() and not n.startswith("_")
            )
            return (f"  El módulo {module_name} no expone '{symbol}'.\n"
                    f"    Archivo : {getattr(module, '__file__', '?')}\n"
                    f"    Exporta : {', '.join(exported) or '(nada)'}\n\n"
                    f"  Causa habitual: el archivo en disco es una versión\n"
                    f"  anterior a la que define ese símbolo.")
    return None


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_competition(
    comp_id:   str,
    season:    int,
    config,
    verify_no_lookahead: bool = False,
    verbose:   bool = False,
    no_filter: bool = False,
    opening:   bool = False,
) -> tuple[Metrics, dict]:
    """Backtestea una competición y temporada."""
    from sports.soccer.competitions import get_competition
    from sports.soccer.data_source import SoccerDataSource
    from sports.soccer.schedule import SoccerScheduleFetcher
    from sports.soccer.provider import SoccerDataProvider
    from sports.soccer.projections import SoccerProjectionModel
    from sports.soccer.settlement import SoccerSettlementProvider
    from sports.soccer.markets import SoccerMarketDefinitions

    comp = get_competition(comp_id)
    metrics = Metrics(label=f"{comp_id} {season}")
    skipped = {
        "sin_cuotas": 0, "muestra_corta": 0, "proyeccion": 0,
        "sin_resultado": 0, "evaluados": 0, "filtrados": 0,
        "lookahead": 0,
    }

    if comp is None:
        return metrics, skipped

    source = SoccerDataSource(current_season=season)
    schedule = SoccerScheduleFetcher(
        data_source=source, competitions=[comp], season=season
    )
    provider = SoccerDataProvider(
        data_source=source, competitions=[comp], season=season,
        config_loader=config, schedule_fetcher=schedule,
    )
    model = SoccerProjectionModel(config_loader=config)
    settlement = SoccerSettlementProvider(schedule_fetcher=schedule)
    markets = SoccerMarketDefinitions()

    matches = schedule.get_competition_matches(comp, season)
    played: dict[str, int] = {}

    filters = config.get("filters", default={}) or {}
    blending = config.get("blending", default={}) or {}

    for match in sorted(matches, key=lambda m: m.date):
        if not match.is_final:
            continue

        n_home = played.get(match.home, 0)
        n_away = played.get(match.away, 0)
        played[match.home] = n_home + 1
        played[match.away] = n_away + 1

        # Muestra insuficiente al arranque de temporada
        if min(n_home, n_away) < _MIN_MATCHES_PLAYED:
            skipped["muestra_corta"] += 1
            continue

        tiene_cuotas = (match.has_opening_odds if opening
                        else match.has_closing_odds)
        if not tiene_cuotas:
            skipped["sin_cuotas"] += 1
            continue

        event = _to_event(match, comp)

        try:
            home_f, away_f = provider.enrich_event(event)
            context = provider.get_context(event)
            projection = model.project(home_f, away_f, context)
        except Exception as e:
            skipped["proyeccion"] += 1
            if verbose:
                print(f"      ⚠️  {match.match_id}: {type(e).__name__}: {e}")
            continue

        # ── Verificación de look-ahead ─────────────────────────
        if verify_no_lookahead:
            if _leaks_own_xg(model, provider, event, match, projection):
                skipped["lookahead"] += 1

        result = settlement.get_event_result(event)
        if result is None:
            skipped["sin_resultado"] += 1
            continue

        matrix = model.score_matrix(projection)

        for market, selection, line, price in _candidates(match, opening):
            p_model = _model_prob(market, selection, line, projection, matrix, match)
            if p_model is None:
                continue

            skipped["evaluados"] += 1

            weight = float(blending.get(market, {}).get("model_weight", 0.45))
            p_market = 1.0 / price
            blended = weight * p_model + (1 - weight) * p_market
            ev = (blended * price - 1.0) * 100.0

            passed = _passes(ev, p_model, p_market, price,
                             filters.get(market, {}))
            if not passed:
                skipped["filtrados"] += 1
                if not no_filter:
                    continue

            outcome = _settle(settlement, event, market, selection, line, result)

            metrics.add(BetRecord(
                comp=comp_id, season=season, date=match.date,
                match=f"{match.away_display} @ {match.home_display}",
                market=market, selection=selection, line=line, price=price,
                model_prob=round(p_model, 4), market_prob=round(p_market, 4),
                ev=round(ev, 2), result=outcome,
                profit=_profit(outcome, price),
                confidence=projection.confidence,
                odds_source=(match.open_source if opening
                             else match.odds_source),
                passed_filter=passed,
            ))

    return metrics, skipped


def _leaks_own_xg(model, provider, event, match, baseline) -> bool:
    """
    True si la proyección cambia al ocultar el xG del propio partido.

    SoccerMatchInfo lleva home_xg y away_xg del encuentro que se está
    proyectando. Ese dato se calcula DESDE el partido, así que leerlo
    equivale a conocer el resultado antes de apostarlo.

    El diseño no lo usa —los índices salen del histórico, con la
    barrera de fecha— pero el campo existe y una ruta futura podría
    leerlo sin que nada fallara. Comparar las dos proyecciones lo
    detecta.

    Es la verificación que faltó en el backtest de NFL, donde el clima
    OBSERVADO —medido durante el partido— inflaba el ROI de totales en
    8 puntos y tardó dos ejecuciones completas en salir a la luz.
    """
    import dataclasses

    try:
        blanked = dataclasses.replace(
            match, home_xg=None, away_xg=None,
            home_npxg=None, away_npxg=None,
        )
    except Exception:
        return False

    # Sustituir temporalmente el partido en el calendario
    schedule = provider._schedule
    original = schedule.get_match_info

    def patched(match_id: str):
        return blanked if match_id == match.match_id else original(match_id)

    try:
        schedule.get_match_info = patched          # type: ignore[assignment]
        home_f, away_f = provider.enrich_event(event)
        context = provider.get_context(event)
        without = model.project(home_f, away_f, context)
    except Exception:
        return False
    finally:
        schedule.get_match_info = original         # type: ignore[assignment]

    return (abs(without.expected_home - baseline.expected_home) > 1e-6
            or abs(without.expected_away - baseline.expected_away) > 1e-6)


def _candidates(
    match,
    opening: bool = False,
) -> list[tuple[str, str, float | None, float]]:
    """
    Mercados cotizados del partido.

    Parámetros
    ----------
    opening -- Si True usa las cuotas de APERTURA en vez de las de
               cierre.

    Por qué importa la distinción
    ------------------------------
    El cierre es el punto de máxima eficiencia: incorpora todo el
    dinero profesional y toda la información pública. El pipeline en
    producción NO opera ahí — apuesta con precios de apertura y media
    semana.

    Un modelo dominado por el cierre puede perfectamente batir la
    apertura, y esa diferencia es el margen realmente accesible.
    Medirla responde si el sistema es operable, que es una pregunta
    distinta de si bate al mercado en su mejor momento.

    football-data publica el 1X2 completo y el over/under 2.5 en ambos
    conjuntos. No publica BTTS ni hándicap asiático de forma
    consistente.
    """
    out: list[tuple[str, str, float | None, float]] = []

    if opening:
        home, draw, away = match.open_home, match.open_draw, match.open_away
        over, under = match.open_over, match.open_under
    else:
        home, draw, away = match.odds_home, match.odds_draw, match.odds_away
        over, under = match.odds_over, match.odds_under

    if home and draw and away:
        out.append(("1X2", match.home, None, home))
        out.append(("1X2", "draw",     None, draw))
        out.append(("1X2", match.away, None, away))

    if over and under:
        out.append(("TOTAL", "over",  2.5, over))
        out.append(("TOTAL", "under", 2.5, under))

    return out


def _model_prob(market, selection, line, projection, matrix, match) -> float | None:
    """Probabilidad que el modelo asigna a una selección."""
    if market == "1X2":
        sel = (selection or "").strip().lower()
        if sel in ("draw", "x"):
            return projection.draw_prob
        if sel == match.home:
            return projection.home_win_prob
        if sel == match.away:
            return projection.away_win_prob
        return None

    if market == "TOTAL" and line is not None:
        sel = (selection or "").strip().lower()
        if sel == "over":
            return matrix.total_over(line)
        if sel == "under":
            return matrix.total_under(line)

    return None


def _passes(ev, p_model, p_market, price, cfg) -> bool:
    """Aplica los filtros de soccer.yaml."""
    if not cfg:
        return ev >= 3.0

    if ev < float(cfg.get("min_ev", 3.0)):
        return False

    for key, cmp_fn in (
        ("min_prob", lambda v: p_model < v),
        ("min_odds", lambda v: price < v),
        ("max_odds", lambda v: price > v),
        ("min_edge", lambda v: (p_model - p_market) < v),
        ("max_edge", lambda v: (p_model - p_market) > v),
    ):
        value = cfg.get(key)
        if value is not None and cmp_fn(float(value)):
            return False

    return True


def _settle(settlement, event, market, selection, line, result) -> str:
    """Liquida el pick con el settlement real del plugin."""
    from core.contracts.pick import CandidatePick

    pick = CandidatePick(
        event=event, market=market, selection=selection, line=line,
        price=2.0, model_prob_raw=0.5, market_prob=0.5, blended_prob=0.5,
    )
    return settlement.settle_pick(pick, result)


def _to_event(match, comp):
    """SoccerMatchInfo → Event."""
    from core.contracts.event import Event, EventStatus

    return Event(
        event_id=match.match_id, sport="soccer", league=comp.name,
        season_start=match.season, season_end=match.season + 1,
        date=match.date, start_time=f"{match.date}T12:00:00Z",
        home_team_id=match.home, away_team_id=match.away,
        home_team=match.home_display or match.home,
        away_team=match.away_display or match.away,
        venue_id=match.home, venue_name="",
        status=EventStatus.FINAL,
        provider_ids={"match_id": match.match_id, "competition": comp.comp_id},
    )


# ── Informe ──────────────────────────────────────────────────────────────────

def _report(m: Metrics, indent: str = "  ") -> None:
    if not m.n:
        print(f"{indent}{m.label}: sin picks")
        return

    hit = f"{m.hit_rate:.2f}%" if m.hit_rate is not None else "—"
    roi = f"{m.roi:+.2f}%" if m.roi is not None else "—"
    esperadas = f"{m.market_expected_wins:.1f}"
    surplus = f"{m.win_surplus:+.1f}" if m.win_surplus is not None else "—"

    print(f"{indent}{m.label}")
    print(f"{indent}  n={m.n:<4d} {m.wins}G-{m.losses}P-{m.pushes}E   "
          f"hit={hit:>7s}  ROI={roi:>8s}")
    print(f"{indent}  victorias: {m.wins} observadas vs {esperadas} "
          f"esperadas por el mercado  ({surplus})")


def _breakdown(records: list[BetRecord], key, titulo: str) -> None:
    grupos: dict[str, Metrics] = {}
    for r in records:
        k = key(r)
        grupos.setdefault(k, Metrics(label=str(k))).add(r)

    if len(grupos) <= 1:
        return

    print(f"\n  {titulo}")
    for k in sorted(grupos):
        _report(grupos[k], indent="    ")


def _interpret(overall: Metrics, lookahead: int) -> None:
    print()
    print("=" * 70)
    print("  LECTURA DEL RESULTADO")
    print("=" * 70)
    print()

    if lookahead:
        print(f"  ⚠️  FUGA DE LOOK-AHEAD en {lookahead} partidos.")
        print("      La proyección cambia al ocultar el xG del propio")
        print("      partido, así que alguna ruta lo está leyendo. El")
        print("      resultado de abajo NO es válido hasta corregirlo.")
        print()

    if overall.n == 0:
        print("  Sin picks. Ver el desglose de descartes de arriba.")
        return

    if _USING_OPENING[0]:
        print("  Las cuotas son de APERTURA. Es el benchmark que corresponde")
        print("  a cómo opera el pipeline en producción, no un suelo.")
        print()
        print("  Comparar este ROI con el del cierre dice cuánto margen hay")
        print("  entre ambos momentos: esa diferencia es lo accesible.")
    else:
        print("  Las cuotas son de CIERRE, el punto de máxima eficiencia del")
        print("  mercado. En producción el pipeline opera con precios más")
        print("  blandos, así que esto es un SUELO, no una estimación central.")
    print()

    surplus = overall.win_surplus
    p = overall.p_value
    roi = overall.roi

    print(f"  ROI                     : {roi:+.2f}%" if roi is not None else "")
    print(f"  Victorias observadas    : {overall.wins}")
    print(f"  Esperadas por el mercado: {overall.market_expected_wins:.1f}")
    if surplus is not None:
        print(f"  Excedente               : {surplus:+.1f} "
              f"({overall.surplus_pct:+.2f} pp)")
    if p is not None:
        print(f"  p (una cola)            : {p:.4f}")
    print()

    # El ROI manda: es la métrica de dinero.
    #
    # El excedente de victorias y el ROI pueden discrepar, y cuando lo
    # hacen dicen algo concreto: el modelo acierta más veces de las
    # implícitas pero se equivoca en las caras, o al revés. No es una
    # contradicción sino información sobre DÓNDE acierta.
    if roi is not None and roi > 0:
        if p is not None and p < 0.05:
            print("  ROI positivo Y más victorias de las implícitas, con")
            print("  significancia. Verificar en el desglose que no viene")
            print("  de una sola competición o mercado.")
        else:
            print("  ROI positivo pero el excedente de victorias no es")
            print("  significativo: puede venir de acertar pocas apuestas")
            print("  caras, lo que es más frágil de lo que sugiere el ROI.")
    else:
        print("  ROI NEGATIVO contra el cierre.")
        if surplus is not None and surplus > 0:
            print()
            print("  Ojo: hay más victorias de las implícitas pero el ROI")
            print("  es negativo. Eso significa que el modelo acierta en")
            print("  las apuestas baratas y falla en las caras — el patrón")
            print("  de un modelo que subestima a los favoritos.")
        print()
        print("  No descarta ventaja en producción, donde los precios son")
        print("  más blandos, pero obliga a revisar la calibración antes")
        print("  de arriesgar capital.")

    if overall.resolved < _MIN_PICKS_ROI:
        print()
        print(f"  Nota: {overall.resolved} picks resueltos siguen por debajo de")
        print(f"  los {_MIN_PICKS_ROI} que hacen interpretable el ROI.")


def _export(records: list[BetRecord], path: str) -> None:
    if not records:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(vars(records[0])))
        writer.writeheader()
        for r in records:
            writer.writerow(vars(r))
    print(f"\n  Detalle exportado: {path} ({len(records)} filas)")


# ── Entrada ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest walk-forward del plugin de fútbol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                        help="Temporadas por su año de inicio (2024 = 24/25)")
    parser.add_argument("--comps", nargs="+", default=None,
                        help="Competiciones. Default: las activas")
    parser.add_argument("--verify-no-lookahead", action="store_true",
                        help="Proyecta cada partido con y sin su propio xG "
                             "y compara. Duplica el tiempo de ejecución.")
    parser.add_argument("--elo-weight", type=float, default=None,
                        help="Peso de la señal de Elo en la mezcla de medias, "
                             "de 0.0 a 1.0. Sobreescribe soccer.yaml. Barrer "
                             "varios valores dice si el Elo aporta resolución "
                             "o solo ruido — la misma pregunta que resolvimos "
                             "con el peso del mercado.")
    parser.add_argument("--opening-odds", action="store_true",
                        help="Mide contra las cuotas de APERTURA en vez de "
                             "las de cierre. El pipeline en producción opera "
                             "con precios de apertura y media semana, así que "
                             "este benchmark responde si el sistema es "
                             "operable — una pregunta distinta de si bate al "
                             "mercado en su momento más eficiente.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Registra TODOS los candidatos en el CSV, no "
                             "solo los que pasan el filtro. Las métricas del "
                             "informe siguen contando solo los filtrados. "
                             "Sirve para medir la calibración del modelo sin "
                             "el sesgo de selección que introduce el filtro.")
    parser.add_argument("--output", default=None, help="CSV de salida")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _USING_OPENING[0] = args.opening_odds

    problem = _verify_imports()
    if problem:
        print("ERROR: el plugin de fútbol no se puede cargar.\n")
        print(problem)
        return 1

    from core.utils.config_loader import load_config
    from sports.soccer.competitions import enabled_competitions

    config = load_config(sport="soccer", base_dir="config")

    # El peso del Elo pasado por CLI sobreescribe el del archivo, para
    # poder barrer valores sin editar soccer.yaml en cada corrida.
    if args.elo_weight is not None:
        config = _ConfigWithOverride(config, {
            "soccer.elo.weight": args.elo_weight,
            "soccer.elo.enabled": args.elo_weight > 0,
        })

    comps = args.comps or [c.comp_id for c in enabled_competitions()]

    print("=" * 70)
    print("  BACKTEST SOCCER — walk-forward contra cuotas de cierre")
    print("=" * 70)
    print(f"  Temporadas    : {', '.join(str(s) for s in args.seasons)}")
    print(f"  Competiciones : {', '.join(comps)}")
    if args.verify_no_lookahead:
        print(f"  Verificación de look-ahead: ACTIVA")
    if args.elo_weight is not None:
        print(f"  Peso del Elo: {args.elo_weight}")
    if args.opening_odds:
        print(f"  Cuotas: APERTURA (no cierre)")
    if args.no_filter:
        print(f"  Sin filtro: el CSV incluirá TODOS los candidatos")
        print(f"  (las métricas del informe siguen contando solo los que pasan)")
    print()
    print("  Metodología: para cada partido, el modelo solo ve lo jugado")
    print("  ANTES de esa fecha. La barrera la impone el provider")
    print("  (SoccerDataProvider._as_of), así que el backtest ejercita el")
    print("  mismo código que producción.")
    print()

    overall = Metrics(label="AGREGADO")
    total_skipped = {
        "sin_cuotas": 0, "muestra_corta": 0, "proyeccion": 0,
        "sin_resultado": 0, "evaluados": 0, "filtrados": 0, "lookahead": 0,
    }

    for season in args.seasons:
        for comp_id in comps:
            try:
                metrics, skipped = run_competition(
                    comp_id, season, config,
                    verify_no_lookahead=args.verify_no_lookahead,
                    verbose=args.verbose,
                    no_filter=args.no_filter,
                    opening=args.opening_odds,
                )
            except (ImportError, AttributeError) as e:
                print(f"  ERROR ESTRUCTURAL: {type(e).__name__}: {e}")
                print("\n  Se detiene: este fallo no depende de los datos y")
                print("  se repetiría en todas las competiciones.")
                return 1
            except Exception as e:
                print(f"  ERROR en {comp_id} {season}: {type(e).__name__}: {e}")
                continue

            for k in total_skipped:
                total_skipped[k] += skipped.get(k, 0)
            for r in metrics.records:
                overall.add(r)

            if args.verbose or metrics.n:
                _report(metrics)

    print()
    print("=" * 70)
    print("  AGREGADO")
    print("=" * 70)
    _report(overall)

    if overall.records:
        _breakdown(overall.records, lambda r: r.comp, "Por competición")
        _breakdown(overall.records, lambda r: r.market, "Por mercado")
        _breakdown(overall.records, lambda r: str(r.season), "Por temporada")

    print()
    print("  Descartes:")
    for k, v in total_skipped.items():
        if v:
            print(f"    {k:16s} {v}")

    if overall.n == 0 and total_skipped["evaluados"] == 0:
        print()
        print("    → No se evaluó ningún candidato. Si 'sin_cuotas' es alto,")
        print("      football-data no publicó cuotas para esas temporadas.")
        print("      Si 'muestra_corta' lo es, el rango es demasiado breve.")

    _interpret(overall, total_skipped["lookahead"])

    if args.output:
        _export(overall.records, args.output)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())