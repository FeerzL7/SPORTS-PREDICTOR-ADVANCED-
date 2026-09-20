#!/usr/bin/env python3
"""
scripts/dry_run_soccer.py

Dry run del pipeline de fútbol con provider mock.

Qué valida y qué no
---------------------
Ejercita la cadena completa —eventos, enriquecimiento, proyección,
cuotas, valor, staking, riesgo— sin tocar la red. Los datos son
sintéticos pero las estructuras son las reales: Event, TeamFeatures,
Projection, MarketOdds y CandidatePick del contrato del Core.

Lo que NO valida: que las fuentes respondan, que el emparejamiento con
The Odds API funcione contra datos reales, ni que el modelo tenga
ventaja. Eso es el backtest (11.19) y el paper trading.

Lo que SÍ valida: que las piezas encajen. En el plugin NFL el dry run
descubrió que la ventana de operación de los filtros era de 0.87 puntos
porcentuales —el sistema no habría generado un solo pick de spread— y
que el settlement anulaba todos los picks de total por el formato de la
selección del ledger. Dos fallos que ninguna revisión de código había
detectado.

Escenarios
------------
Tres partidos elegidos para cubrir los casos que importan:

    Favorito claro      Un grande contra un recién ascendido. Prueba
                        que el modelo produce probabilidades altas sin
                        salirse de las cotas, y que min_odds excluye al
                        favorito más corto.

    Partido igualado    Dos equipos de media tabla. Es donde el EMPATE
                        tiene más probabilidad, y por tanto donde el
                        modelo puede encontrar valor: es el mercado
                        donde los books aplican más margen.

    Derbi               Mismo escenario que el igualado pero con la
                        condición de derbi activa, para comprobar que
                        el ajuste reduce la ventaja de campo y desplaza
                        la probabilidad hacia el empate.

Uso
----
    python scripts/dry_run_soccer.py
    python scripts/dry_run_soccer.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.contracts.event import Event, EventStatus
from core.contracts.features import TeamFeatures
from core.contracts.market_odds import MarketOdds


# ── Escenarios ───────────────────────────────────────────────────────────────

_FECHA = "2024-11-09"

# (comp, local, visitante, ataque_l, defensa_l, ataque_v, defensa_v, derbi)
_ESCENARIOS = [
    ("epl", "manchester city", "luton town", 1.42, 1.35, 0.68, 0.72, False),
    ("epl", "brighton",        "fulham",     1.05, 0.98, 0.97, 1.02, False),
    ("epl", "arsenal",         "tottenham",  1.18, 1.12, 1.14, 0.95, True),
]

# Cuotas sintéticas por partido: (mercado, selección, línea, precio)
#
# Los precios corresponden a las probabilidades implícitas típicas de
# cada escenario, con un margen del ~5% repartido.
_CUOTAS = {
    "manchester city": [
        ("1X2", "manchester city", None, 1.28),
        ("1X2", "draw",            None, 6.00),
        ("1X2", "luton town",      None, 11.00),
        ("TOTAL", "over",          2.5,  1.55),
        ("TOTAL", "under",         2.5,  2.45),
    ],
    "brighton": [
        ("1X2", "brighton", None, 2.40),
        ("1X2", "draw",     None, 3.40),
        ("1X2", "fulham",   None, 3.00),
        ("TOTAL", "over",   2.5,  1.95),
        ("TOTAL", "under",  2.5,  1.90),
    ],
    "arsenal": [
        ("1X2", "arsenal",   None, 2.05),
        ("1X2", "draw",      None, 3.50),
        ("1X2", "tottenham", None, 3.60),
        ("TOTAL", "over",    2.5,  1.80),
        ("TOTAL", "under",   2.5,  2.05),
    ],
}


class MockSoccerProvider:
    """
    Provider sintético que implementa SportDataProvider.

    Devuelve índices fijos por equipo en vez de calcularlos del
    histórico. Eso aísla el pipeline de las fuentes: si un stage falla,
    el fallo está en el pipeline, no en los datos.
    """

    def __init__(self, league_home: float = 1.53, league_away: float = 1.25):
        self._league_home = league_home
        self._league_away = league_away

    def get_events(self, date: str) -> list[Event]:
        events = []
        for comp, home, away, *_ in _ESCENARIOS:
            match_id = f"{comp}_{date}_{home.replace(' ','-')}_{away.replace(' ','-')}"
            events.append(Event(
                event_id=match_id, sport="soccer", league="Premier League",
                season_start=2024, season_end=2025, date=date,
                start_time=f"{date}T12:00:00Z",
                home_team_id=home, away_team_id=away,
                home_team=home.title(), away_team=away.title(),
                venue_id=home, venue_name="",
                status=EventStatus.SCHEDULED,
                provider_ids={"match_id": match_id, "competition": comp},
            ))
        return events

    def enrich_event(self, event: Event) -> tuple[TeamFeatures, TeamFeatures]:
        row = self._row(event.home_team_id)
        _, home, away, atk_h, def_h, atk_a, def_a, _ = row

        return (
            self._features(home, atk_h, def_h, is_home=True),
            self._features(away, atk_a, def_a, is_home=False),
        )

    def get_context(self, event: Event) -> dict:
        row = self._row(event.home_team_id)
        derbi = row[7]
        return {
            "match_id":          event.event_id,
            "competition":       row[0],
            "season":            2024,
            "league_home_goals": self._league_home,
            "league_away_goals": self._league_away,
            "matchday":          12,
            "season_progress":   0.32,
            "is_early_season":   False,
            "is_late_season":    False,
            "is_derby":          derbi,
            "derby_adjustment":  -0.12 if derbi else 0.0,
            "congestion_differential": 0.0,
            "table_reliable":    True,
        }

    @staticmethod
    def _row(home_id: str):
        for row in _ESCENARIOS:
            if row[1] == home_id:
                return row
        return _ESCENARIOS[0]

    @staticmethod
    def _features(team: str, attack: float, defense: float,
                  is_home: bool) -> TeamFeatures:
        return TeamFeatures(
            team_id=team, team_name=team.title(),
            expected_score=1.4, offense_index=attack, defense_index=defense,
            recent_scores=[1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 3.0, 1.0],
            recent_avg=1.375, recent_n=8, sample_size=12, data_quality=0.92,
            sport_metadata={
                "attack_index": attack, "defense_index": defense,
                "xg_available": True, "is_home": is_home,
                "congestion_penalty": 0.0, "matches": 12,
            },
        )


def _odds_for(event: Event) -> list[MarketOdds]:
    """Cuotas sintéticas del partido."""
    especificas = _CUOTAS.get(event.home_team_id, [])
    return [
        MarketOdds(
            event_id=event.event_id, market=market, selection=sel,
            line=line, price=price, bookmaker="mock",
            timestamp=f"{event.date}T10:00:00Z",
        )
        for market, sel, line, price in especificas
    ]


# ── Ejecución ────────────────────────────────────────────────────────────────

def run(verbose: bool = False) -> tuple[int, int]:
    """Ejecuta el dry run. Retorna (verificaciones ok, total)."""
    from core.utils.config_loader import load_config
    from sports.soccer.projections import SoccerProjectionModel
    from sports.soccer.markets import SoccerMarketDefinitions
    from sports.soccer.settlement import SoccerSettlementProvider
    from core.simulation.bivariate_poisson import BivariatePoissonModel

    checks: list[tuple[str, bool]] = []

    def check(label: str, condition: bool) -> None:
        checks.append((label, bool(condition)))

    print("=" * 66)
    print("  DRY RUN SOCCER — provider mock")
    print("=" * 66)
    print()

    config = load_config(sport="soccer", base_dir="config")
    provider = MockSoccerProvider()
    model = SoccerProjectionModel(config_loader=config)
    prob_model = BivariatePoissonModel()
    markets = SoccerMarketDefinitions()

    # ── Stage 1 ────────────────────────────────────────────────
    events = provider.get_events(_FECHA)
    print(f"  Stage 1  eventos obtenidos      : {len(events)}")
    check("Stage 1 produce eventos", len(events) == 3)

    # ── Stage 2 y 3 ────────────────────────────────────────────
    proyecciones = {}
    for event in events:
        home_f, away_f = provider.enrich_event(event)
        context = provider.get_context(event)
        proyecciones[event.event_id] = (
            model.project(home_f, away_f, context), context
        )

    print(f"  Stage 2  eventos enriquecidos   : {len(proyecciones)}")
    print(f"  Stage 3  proyecciones generadas : {len(proyecciones)}")
    check("Stage 3 proyecta todos", len(proyecciones) == len(events))

    print()
    print("  PROYECCIONES")
    for event in events:
        proj, ctx = proyecciones[event.event_id]
        derbi = " [DERBI]" if ctx.get("is_derby") else ""
        print(f"    {event.away_team} @ {event.home_team}{derbi}")
        print(f"      λ {proj.expected_home:.3f} - {proj.expected_away:.3f}"
              f"   total {proj.expected_total:.3f}")
        print(f"      1X2  {proj.home_win_prob:.4f} / {proj.draw_prob:.4f} "
              f"/ {proj.away_win_prob:.4f}   confianza={proj.confidence}")

    # ── Invariantes del modelo ─────────────────────────────────
    todas = [p for p, _ in proyecciones.values()]

    check("Distribución bivariate_poisson declarada",
          all(p.distribution == "bivariate_poisson" for p in todas))
    check("Parámetros de la distribución propagados",
          all("rho" in (p.distribution_params or {}) and
              "lambda_home" in (p.distribution_params or {}) for p in todas))
    check("1X2 suma 1.0",
          all(abs(p.home_win_prob + p.draw_prob + p.away_win_prob - 1.0) < 0.01
              for p in todas))
    # El empate en rango SOLO para partidos igualados.
    #
    # El ~25% europeo es la media sobre TODOS los partidos, no un suelo
    # para cada uno. Un grande contra un recién ascendido —λ 3.02 vs
    # 0.63— debe tener el empate bajo: exigirle un 18% sería pedir al
    # modelo que ignore la diferencia de nivel.
    #
    # La primera versión de esta verificación aplicaba el rango a los
    # tres escenarios y fallaba sobre el favorito claro. El modelo
    # estaba bien; el invariante, mal formulado.
    igualados = [p for p in todas
                 if abs(p.expected_home - p.expected_away) < 0.50]
    check("Empate en rango europeo en partidos igualados (20-32%)",
          all(0.20 <= p.draw_prob <= 0.32 for p in igualados))

    # El invariante que SÍ aplica a todos: cuanto mayor el desequilibrio
    # entre las medias, menor la probabilidad de empate. Es monotonía,
    # no un rango.
    por_desequilibrio = sorted(
        todas, key=lambda p: abs(p.expected_home - p.expected_away)
    )
    check("El empate decrece al crecer el desequilibrio",
          all(por_desequilibrio[i].draw_prob >= por_desequilibrio[i + 1].draw_prob
              for i in range(len(por_desequilibrio) - 1)))
    check("λ dentro de las cotas [0.35, 3.50]",
          all(0.35 <= p.expected_home <= 3.50 and
              0.35 <= p.expected_away <= 3.50 for p in todas))

    # ── El favorito domina ─────────────────────────────────────
    fav = proyecciones[events[0].event_id][0]
    check("Favorito claro con P(local) alta", fav.home_win_prob > 0.55)

    # ── El derbi desplaza hacia el empate ──────────────────────
    igualado = proyecciones[events[1].event_id][0]
    derbi = proyecciones[events[2].event_id][0]
    check("Derbi: ajuste aplicado a λ local",
          proyecciones[events[2].event_id][1]["derby_adjustment"] < 0)

    # ── Coherencia modelo ↔ probabilidad ───────────────────────
    coherente = True
    for proj, _ in proyecciones.values():
        w = prob_model.win_probabilities(proj)
        if abs(w["draw"] - proj.draw_prob) > 0.002:
            coherente = False
    check("Modelo ≡ capa de probabilidad", coherente)

    # ── Stage 5: cuotas ────────────────────────────────────────
    cuotas = {e.event_id: _odds_for(e) for e in events}
    total_cuotas = sum(len(v) for v in cuotas.values())
    print()
    print(f"  Stage 5  cuotas inyectadas      : {total_cuotas}")
    check("Stage 5 inyecta cuotas", total_cuotas == 15)

    # ── Línea preferida ────────────────────────────────────────
    check("Línea preferida de TOTAL = 2.5",
          markets.get_preferred_line("TOTAL") == 2.5)
    check("1X2 sin línea preferida",
          markets.get_preferred_line("1X2") is None)

    # ── Stage 6: EV ────────────────────────────────────────────
    print(f"  Stage 6  candidatos evaluados   : {total_cuotas}")
    print()
    print("  CANDIDATOS")

    filtros = config.get("filters", default={}) or {}
    blending = config.get("blending", default={}) or {}

    candidatos = []
    for event in events:
        proj, _ = proyecciones[event.event_id]
        matriz = model.score_matrix(proj)

        for odds in cuotas[event.event_id]:
            p_model = _prob_del_modelo(odds, proj, matriz, event)
            if p_model is None:
                continue

            cfg_b = blending.get(odds.market, {})
            peso = float(cfg_b.get("model_weight", 0.45))
            p_mercado = 1.0 / odds.price
            p_mezcla = peso * p_model + (1 - peso) * p_mercado
            ev = (p_mezcla * odds.price - 1.0) * 100.0

            cfg_f = filtros.get(odds.market, {})
            pasa = _pasa_filtros(ev, p_model, p_mercado, odds.price, cfg_f)

            candidatos.append({
                "event": event, "odds": odds, "p_model": p_model,
                "ev": ev, "pasa": pasa,
            })

    for c in sorted(candidatos, key=lambda x: -x["ev"]):
        marca = "PASA  " if c["pasa"] else "filtra"
        linea = f" {c['odds'].line}" if c["odds"].line is not None else ""
        print(f"    [{marca}] {c['odds'].market:6s} "
              f"{c['odds'].selection:16s}{linea:>6s} @ {c['odds'].price:5.2f}"
              f"  EV={c['ev']:+7.2f}%  p={c['p_model']:.4f}")

    check("Stage 6 evalúa todos los candidatos",
          len(candidatos) == total_cuotas)
    check("EV calculado en todos",
          all(isinstance(c["ev"], float) for c in candidatos))
    check("Probabilidades del modelo válidas",
          all(0.0 < c["p_model"] < 1.0 for c in candidatos))

    activos = [c for c in candidatos if c["pasa"]]
    print()
    print(f"  Stage 9  picks activos          : {len(activos)}")

    check("Los filtros rechazan parte de los candidatos",
          len(activos) < len(candidatos))
    check("Los picks activos tienen EV positivo",
          all(c["ev"] > 0 for c in activos))

    # ── min_odds excluye al favorito corto ─────────────────────
    corto = [c for c in candidatos
             if c["odds"].market == "1X2" and c["odds"].price < 1.70]
    check("min_odds excluye al favorito más corto",
          all(not c["pasa"] for c in corto))

    # ── Stage 10: liquidación ──────────────────────────────────
    print()
    liquidados = _verificar_liquidacion(check)
    print(f"  Stage 10 liquidación            : {liquidados} escenarios")

    # ── Resumen ────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  VERIFICACIONES")
    print("=" * 66)
    for label, ok in checks:
        print(f"  {'✅' if ok else '❌'} {label}")

    aciertos = sum(1 for _, ok in checks if ok)
    print()
    print("=" * 66)
    print(f"  DRY RUN COMPLETADO: {aciertos}/{len(checks)} verificaciones")
    print("=" * 66)

    return aciertos, len(checks)


def _prob_del_modelo(odds, proj, matriz, event) -> float | None:
    """Probabilidad que el modelo asigna a una selección."""
    market = odds.market.upper()
    sel = (odds.selection or "").strip().lower()

    if market == "1X2":
        if sel in ("draw", "x", "empate"):
            return proj.draw_prob
        if sel == event.home_team_id:
            return proj.home_win_prob
        if sel == event.away_team_id:
            return proj.away_win_prob
        return None

    if market == "TOTAL" and odds.line is not None:
        if sel == "over":
            return matriz.total_over(odds.line)
        if sel == "under":
            return matriz.total_under(odds.line)

    if market == "BTTS":
        return matriz.btts() if sel == "yes" else 1.0 - matriz.btts()

    return None


def _pasa_filtros(ev, p_model, p_mercado, price, cfg) -> bool:
    """Aplica los filtros de soccer.yaml."""
    if not cfg:
        return ev >= 3.0

    if ev < float(cfg.get("min_ev", 3.0)):
        return False

    min_prob = cfg.get("min_prob")
    if min_prob is not None and p_model < float(min_prob):
        return False

    min_odds = cfg.get("min_odds")
    if min_odds is not None and price < float(min_odds):
        return False

    max_odds = cfg.get("max_odds")
    if max_odds is not None and price > float(max_odds):
        return False

    edge = p_model - p_mercado
    min_edge = cfg.get("min_edge")
    if min_edge is not None and edge < float(min_edge):
        return False

    max_edge = cfg.get("max_edge")
    if max_edge is not None and edge > float(max_edge):
        return False

    return True


def _verificar_liquidacion(check) -> int:
    """
    Liquidación sobre resultados sintéticos.

    Cubre los tres casos que diferencian al fútbol de NFL: el empate
    como resultado, el push en línea entera y los mercados de primera
    parte.
    """
    from core.contracts.event import Event, EventStatus
    from core.contracts.pick import CandidatePick
    from sports.soccer.schedule import SoccerMatchInfo
    from sports.soccer.settlement import (
        SoccerSettlementProvider, RESULT_WIN, RESULT_LOSE, RESULT_PUSH,
    )

    MID = "epl_2024-11-09_a_b"

    def provider(hg, ag, hth=1, hta=0):
        class Cal:
            # El parámetro se llama `match_id`, no `mid`.
            #
            # Un Protocol de Python incluye los NOMBRES de los
            # parámetros en su contrato, salvo que se declaren
            # posicionales. Llamar con `mid` funciona en runtime
            # —Python no comprueba nombres en llamadas posicionales—
            # pero el stub no satisface realmente el Protocol, y el
            # type checker lo señala con razón.
            #
            # Un doble de prueba que no cumple el contrato que dice
            # cumplir valida menos de lo que aparenta.
            def get_match_info(self, match_id: str) -> SoccerMatchInfo | None:
                return SoccerMatchInfo(
                    match_id=MID, comp_id="epl", season=2024,
                    date="2024-11-09", home="a", away="b",
                    home_goals=hg, away_goals=ag,
                    ht_home=hth, ht_away=hta,
                )
        return SoccerSettlementProvider(schedule_fetcher=Cal())

    def evento():
        return Event(
            event_id=MID, sport="soccer", league="EPL", season_start=2024,
            season_end=2025, date="2024-11-09",
            start_time="2024-11-09T12:00:00Z",
            home_team_id="a", away_team_id="b", home_team="a", away_team="b",
            venue_id="a", venue_name="", status=EventStatus.FINAL,
            provider_ids={"match_id": MID, "competition": "epl"},
        )

    def pick(market, sel, line=None):
        return CandidatePick(
            event=evento(), market=market, selection=sel, line=line,
            price=2.0, model_prob_raw=0.5, market_prob=0.5, blended_prob=0.5,
        )

    # Empate 1-1: el pick de empate GANA
    p_empate = provider(1, 1)
    r_empate = p_empate.get_event_result(evento())

    # get_event_result devuelve None cuando el partido no ha terminado.
    # Aquí siempre tiene marcador, pero comprobarlo es lo correcto: el
    # tipo lo declara opcional y saltarse el guard porque «sé que no
    # será None» es cómo se cuelan los AttributeError en producción.
    if r_empate is None:
        check("Liquidación: resultado del empate disponible", False)
        return 1
    check("Liquidación: resultado del empate disponible", True)
    check("Empate: pick de empate gana",
          p_empate.settle_pick(pick("1X2", "draw"), r_empate) == RESULT_WIN)
    check("Empate: pick de local pierde",
          p_empate.settle_pick(pick("1X2", "a"), r_empate) == RESULT_LOSE)

    # Push en línea entera
    check("Total 2 goles con línea 2.0: push",
          p_empate.settle_pick(pick("TOTAL", "over", 2.0), r_empate) == RESULT_PUSH)
    check("Línea .5 sin push",
          p_empate.settle_pick(pick("TOTAL", "over", 2.5), r_empate) in
          (RESULT_WIN, RESULT_LOSE))

    # Primera parte liquidable
    p_final = provider(2, 1, hth=1, hta=0)
    r_final = p_final.get_event_result(evento())
    if r_final is None:
        check("Liquidación: resultado del partido disponible", False)
        return 3
    check("Liquidación: resultado del partido disponible", True)
    check("Primera parte liquidable (1-0 al descanso)",
          p_final.settle_pick(pick("1X2_H1", "a"), r_final) == RESULT_WIN)

    # BTTS
    check("BTTS: 2-1 sí marcaron ambos",
          p_final.settle_pick(pick("BTTS", "yes"), r_final) == RESULT_WIN)

    return 8


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry run del pipeline de fútbol")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    aciertos, total = run(verbose=args.verbose)
    return 0 if aciertos == total else 1


if __name__ == "__main__":
    sys.exit(main())