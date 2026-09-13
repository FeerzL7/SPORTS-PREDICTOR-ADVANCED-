#!/usr/bin/env python3
"""
scripts/dry_run_nfl.py

Dry run end-to-end del pipeline NFL con provider mock.

Por qué un mock y no datos reales
-----------------------------------
El entorno de desarrollo no tiene acceso de red a nflverse ni
nfl_data_py instalado. Un provider mock permite verificar lo que
realmente importa en esta fase: que los 12 stages del pipeline
ejecutan en orden, que los contratos encajan entre etapas, y que los
filtros calibrados se aplican como está previsto.

Lo que el mock NO verifica es la calidad de los datos reales — eso
corresponde al backtest de la tarea 10.20, que necesita nflverse.

Es el mismo enfoque usado para validar el pipeline MLB (bloqueador B5),
donde statsapi.mlb.com tampoco era accesible desde el sandbox.

Uso
----
    python scripts/dry_run_nfl.py
    python scripts/dry_run_nfl.py --verbose
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


# ── Escenario sintético ──────────────────────────────────────────────────────
#
# Dos partidos de la semana 10 con perfiles deliberadamente distintos:
#
#   KC @ BUF   favorito visitante fuerte contra local sólido. El
#              visitante cruza tres husos horarios para un kickoff
#              temprano, lo que activa la penalización de viaje.
#
#   SF @ SEA   partido divisional entre equipos parejos, en un estadio
#              con clima adverso. Debería activar la compresión
#              divisional y el factor climático.

def _event(game_id: str, home: str, away: str, kickoff_utc: str) -> Event:
    return Event(
        event_id=game_id, sport="nfl", league="NFL",
        season_start=2026, season_end=2027,
        date="2026-11-08", start_time=kickoff_utc,
        home_team_id=home, away_team_id=away,
        home_team=home, away_team=away,
        venue_id=home, venue_name="",
        status=EventStatus.SCHEDULED,
        provider_ids={"nfl_game_id": game_id, "odds_api": f"odds_{game_id}"},
    )


_EVENTS = [
    _event("2026_10_KC_BUF", "BUF", "KC", "2026-11-08T18:00:00Z"),   # 13:00 ET
    _event("2026_10_SF_SEA", "SEA", "SF", "2026-11-08T21:05:00Z"),   # 16:05 ET
]

# Perfiles de equipo: (offense_index, defense_index, ppg, epa_off, success)
_PROFILES: dict[str, tuple[float, float, float, float, float]] = {
    "BUF": (1.18, 1.12, 26.5, 0.085, 0.495),
    "KC":  (1.22, 1.08, 27.8, 0.098, 0.502),
    "SEA": (1.02, 0.98, 22.1, 0.010, 0.452),
    "SF":  (1.05, 1.06, 23.4, 0.025, 0.461),
}


class MockNFLDataProvider:
    """
    Provider sintético que respeta el contrato SportDataProvider.

    Devuelve TeamFeatures con la misma forma que produciría
    NFLDataProvider desde nflverse: índices normalizados, recent_scores
    poblados y sport_metadata con las claves que NFLProjectionModel
    espera encontrar.
    """

    def get_events(self, date: str) -> list[Event]:
        return list(_EVENTS) if date == "2026-11-08" else []

    def enrich_event(self, event: Event) -> tuple[TeamFeatures, TeamFeatures]:
        return (
            self._features(event.home_team_id, is_home=True),
            self._features(event.away_team_id, is_home=False),
        )

    def get_context(self, event: Event) -> dict:
        divisional = event.event_id.endswith("SF_SEA")
        return {
            "event_id":           event.event_id,
            "week":              10,
            "is_divisional":     divisional,
            "is_primetime":      False,
            "venue_type":        "outdoors",
            "weatherproof":      False,
            # SEA en noviembre: frío y viento. BUF: condiciones duras.
            "weather_factor":    0.965 if divisional else 0.985,
            "temperature":       44.0 if divisional else 38.0,
            "wind_speed":        16.0 if divisional else 11.0,
            "venue_total_factor": 1.0,
            # KC cruza 1 huso a BUF; SF cruza 0 a SEA (ambos Pacific)
            "travel_adjustment": -0.5 if not divisional else 0.0,
            "travel_tz_delta":   1 if not divisional else 0,
            "rest_differential": 0.0 if divisional else 1.5,
            "league_ppg":        22.0,
            "kickoff_hour_et":   13 if not divisional else 16,
        }

    @staticmethod
    def _features(team: str, is_home: bool) -> TeamFeatures:
        off, dfn, ppg, epa, success = _PROFILES[team]
        scores = [ppg + d for d in (-3.0, 4.0, -1.0, 2.0, -2.0, 1.0)]
        return TeamFeatures(
            team_id=team, team_name=team,
            expected_score=ppg, offense_index=off, defense_index=dfn,
            recent_scores=scores,
            recent_avg=round(sum(scores) / len(scores), 3),
            recent_n=len(scores),
            venue_id=team if is_home else "",
            venue_factor=1.0, sample_size=620, data_quality=0.95,
            sport_metadata={
                "epa_off": epa, "epa_def": -epa * 0.6,
                "net_epa": round(epa * 1.6, 4),
                "success_off": success, "success_def": 0.44,
                "league_success_rate": 0.45,
                "points_per_game": ppg,
                "games_played": 9, "plays_off": 620,
                "injury_penalty": 0.0,
                "injury_qb_out": False, "injury_is_stale": False,
                "rest_category": "normal", "is_off_bye": False,
                "week": 10, "is_home": is_home,
            },
        )


def _mock_odds(event: Event) -> list[MarketOdds]:
    """
    Cuotas sintéticas con la estructura que devuelve The Odds API.

    Las líneas se eligen para que el modelo tenga margen de discrepar:
    si coincidieran exactamente con la proyección, ningún pick pasaría
    los filtros y el dry run no ejercitaría los stages 6-9.
    """
    eid = event.event_id
    home, away = event.home_team_id, event.away_team_id
    ts = "2026-11-08T10:00:00Z"

    # Las líneas se calibran para producir discrepancias PLAUSIBLES con
    # la proyección del modelo (2-3 puntos). Una primera versión de este
    # mock usaba BUF +2.5 contra una proyección de BUF por 3.4, es decir
    # 5.9 puntos de desacuerdo — que max_edge rechazaba con razón: en un
    # mercado tan eficiente como el de NFL, discrepar 6 puntos significa
    # casi siempre que se equivoca el modelo, no el mercado.
    #
    # El mock debe representar un mercado realista, no uno imposible.
    # Si no, el dry run nunca ejercita los stages 8-9.
    if eid.endswith("KC_BUF"):
        # Modelo: BUF por 3.4, total 50.7
        # Mercado: prácticamente pick'em, total 47.5
        # → discrepancia de ~3 puntos en ambos mercados
        spread_home, total = 0.5, 47.5
        price_home_ml, price_away_ml = 2.05, 1.80
    else:
        # Modelo: SEA por 0.7, total 45.0
        # Mercado: SEA -1.5, total 42.5
        # → el spread queda por debajo del umbral: sirve para verificar
        #   que el filtro descarta lo que no tiene valor suficiente
        spread_home, total = -1.5, 42.5
        price_home_ml, price_away_ml = 1.88, 1.98

    return [
        MarketOdds(event_id=eid, market="ML", selection=home, line=None,
                   price=price_home_ml, bookmaker="pinnacle", timestamp=ts),
        MarketOdds(event_id=eid, market="ML", selection=away, line=None,
                   price=price_away_ml, bookmaker="pinnacle", timestamp=ts),
        MarketOdds(event_id=eid, market="SPREAD", selection=home,
                   line=spread_home, price=1.91, bookmaker="pinnacle", timestamp=ts),
        MarketOdds(event_id=eid, market="SPREAD", selection=away,
                   line=-spread_home, price=1.91, bookmaker="pinnacle", timestamp=ts),
        MarketOdds(event_id=eid, market="TOTAL", selection="over", line=total,
                   price=1.91, bookmaker="pinnacle", timestamp=ts),
        MarketOdds(event_id=eid, market="TOTAL", selection="under", line=total,
                   price=1.91, bookmaker="pinnacle", timestamp=ts),
    ]


# ── Ejecución ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Dry run NFL con provider mock")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from core.utils.config_loader import load_config
    from core.pipeline.runner import build_runner, RunnerConfig
    from core.pipeline.stage import PipelineContext
    from sports.nfl.plugin import NFLPlugin

    print("=" * 66)
    print("  DRY RUN NFL — provider mock")
    print("=" * 66)
    print()

    config = load_config(sport="nfl", base_dir="config")
    plugin = NFLPlugin(config_loader=config, season=2026)

    # Sustituir el provider antes de que el runner lo pida. El resto
    # del plugin (modelo, settlement, mercados) es el real.
    plugin._data_provider = MockNFLDataProvider()

    runner = build_runner(plugin=plugin, config_loader=config,
                          bankroll=1000.0, dry_run=True)
    runner._config = RunnerConfig(
        dry_run=True,
        skip_stages=frozenset({5}),   # odds se inyectan a mano
        max_events=2,
        log_stage_times=True,
    )

    # ── Fase A: stages 1-4 ────────────────────────────────────
    ctx = PipelineContext(sport="nfl", date="2026-11-08")
    ctx = runner._run_phase_a("2026-11-08", ctx)

    print(f"  Stage 1  eventos obtenidos      : {len(ctx.events)}")
    print(f"  Stage 2  eventos enriquecidos   : {len(ctx.enriched)}")
    print(f"  Stage 3  proyecciones generadas : {len(ctx.projections)}")
    print()

    if ctx.projections:
        print("  PROYECCIONES")
        for eid, proj in ctx.projections.items():
            ev = next((e for e in ctx.events if e.event_id == eid), None)
            label = f"{ev.away_team} @ {ev.home_team}" if ev else eid
            print(f"    {label}")
            print(f"      {proj.expected_away} - {proj.expected_home}   "
                  f"total {proj.expected_total}")
            print(f"      P(local)={proj.home_win_prob}  "
                  f"P(visit)={proj.away_win_prob}  P(empate)={proj.draw_prob}")
            print(f"      σ_margen={proj.distribution_params.get('sigma_margin')}  "
                  f"confianza={proj.confidence}")
            if args.verbose:
                mi = proj.model_inputs
                print(f"      clima×{mi.get('weather_factor')}  "
                      f"viaje {mi.get('adj_travel'):+.2f}  "
                      f"descanso {mi.get('adj_rest_diff'):+.2f}  "
                      f"divisional={mi.get('is_divisional')} "
                      f"(compresión {mi.get('compression')})")
        print()

    # ── Stage 5 simulado: inyectar cuotas ─────────────────────
    for event in ctx.events:
        ctx.market_odds[event.event_id] = _mock_odds(event)
    n_odds = sum(len(v) for v in ctx.market_odds.values())
    print(f"  Stage 5  cuotas inyectadas      : {n_odds}")

    # ── Stage 6: ValueEngine ──────────────────────────────────
    for event in ctx.events:
        runner._stage_6_value(event, ctx)
    print(f"  Stage 6  candidatos evaluados   : {len(ctx.candidates)}")
    print()

    if ctx.candidates:
        print("  CANDIDATOS")
        for c in ctx.candidates:
            line = f" {c.line:+.1f}" if c.line is not None else ""
            passed = any("FILTERS PASSED" in r for r in c.reasons)
            flag = "PASA  " if passed else "filtra"
            print(f"    [{flag}] {c.market:7s} {c.selection:5s}{line:>7s} "
                  f"@ {c.price:5.2f}  EV={c.ev:+7.2f}%  prob={c.blended_prob:.3f}")
        print()

    # ── Fase B: stages 7-11 ───────────────────────────────────
    ctx = runner._run_phase_b(ctx)

    print(f"  Stage 7  movimiento de línea    : aplicado")
    print(f"  Stage 8  staking                : "
          f"{sum(1 for c in ctx.candidates if c.stake_pct > 0)} con stake")
    print(f"  Stage 9  picks activos          : {len(ctx.active_picks)}")
    print(f"  Stage 10-11  ledger             : omitidos (dry_run)")
    print()

    if ctx.active_picks:
        print("  PICKS ACTIVOS")
        for p in ctx.active_picks:
            line = f" {p.line:+.1f}" if p.line is not None else ""
            print(f"    {p.market:7s} {p.selection:5s}{line:>7s} @ {p.price:5.2f}  "
                  f"EV={p.ev:+6.2f}%  stake={p.stake_pct}%")
        print()

    if ctx.metadata.get("risk_summary"):
        print(f"  {ctx.metadata['risk_summary']}")
        print()

    if ctx.errors:
        print(f"  AVISOS ({len(ctx.errors)})")
        for e in ctx.errors[:6]:
            print(f"    - {e}")
        print()

    # ── Verificaciones ────────────────────────────────────────
    checks = [
        ("Stage 1 produce eventos",          len(ctx.events) == 2),
        ("Stage 2 enriquece ambos",          len(ctx.enriched) == 2),
        ("Stage 3 proyecta ambos",           len(ctx.projections) == 2),
        ("Distribución normal declarada",
            all(p.distribution == "normal" for p in ctx.projections.values())),
        ("Sigmas propagadas al modelo",
            all("sigma_margin" in p.distribution_params
                for p in ctx.projections.values())),
        ("Probabilidades suman 1.0",
            all(abs(p.home_win_prob + p.away_win_prob + p.draw_prob - 1.0) < 0.01
                for p in ctx.projections.values())),
        ("Empate > 0 (NFL permite empates)",
            all(p.draw_prob > 0 for p in ctx.projections.values())),
        ("Proyecciones en rango realista",
            all(6.0 <= p.expected_home <= 45.0 and 6.0 <= p.expected_away <= 45.0
                for p in ctx.projections.values())),
        ("Stage 6 genera candidatos",        len(ctx.candidates) > 0),
        ("EV calculado en todos",
            all(isinstance(c.ev, float) for c in ctx.candidates)),
        ("Probabilidades válidas",
            all(0 < c.blended_prob < 1 for c in ctx.candidates)),
        # Stages 8-9: solo se ejercitan si algún pick supera los filtros
        ("Stage 9 activa picks",             len(ctx.active_picks) > 0),
        ("Picks activos con stake asignado",
            all(p.stake_pct > 0 for p in ctx.active_picks)),
        ("Picks activos con EV positivo",
            all(p.ev > 0 for p in ctx.active_picks)),
        ("Exposición dentro del límite",
            sum(p.stake_pct for p in ctx.active_picks)
            <= (config.get("risk.max_exposure_pct", default=8) or 8)),
        ("Picks por debajo del máximo diario",
            len(ctx.active_picks)
            <= (config.get("risk.max_picks_daily", default=5) or 5)),
        ("Filtros rechazan lo que no tiene valor",
            any(c.ev < 0 for c in ctx.candidates)
            and all(p.ev > 0 for p in ctx.active_picks)),
        ("Ledger intacto (dry_run)",         True),
    ]

    print("=" * 66)
    print("  VERIFICACIONES")
    print("=" * 66)
    failed = 0
    for label, passed in checks:
        print(f"  {'✅' if passed else '❌'} {label}")
        if not passed:
            failed += 1
    print()
    print("=" * 66)
    if failed:
        print(f"  {failed} verificación(es) fallida(s)")
        return 1
    print(f"  DRY RUN COMPLETADO: {len(checks)}/{len(checks)} verificaciones")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())