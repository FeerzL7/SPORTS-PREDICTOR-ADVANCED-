"""
core/pipeline/runner.py

PipelineRunner: orquestador del flujo diario de 12 stages.

No implementa lógica de negocio — conecta los módulos del core en el
orden correcto, manejando errores por evento de forma aislada.

Flujo completo (según SPORTS_PREDICTOR_ARCHITECTURE.md §5.1)
-------------------------------------------------------------
FASE A — Por evento (independiente por partido):
    Stage 1:  SportDataProvider.get_events(date)
    Stage 2:  SportDataProvider.enrich_event() + get_context()
    Stage 3:  ProjectionModel.project()
    Stage 4:  ProbabilityModel.win/spread/total_probabilities()
    Stage 5:  OddsAPIClient + OddsNormalizer + LineMovementDetector.snapshot()
    Stage 6:  ValueEngine.evaluate()

FASE B — Sobre todos los candidatos (conjunto completo del día):
    Stage 7:  LineMovementDetector.analyze() → annotate_pick()
    Stage 8:  StakingStrategy.stake_pct()
    Stage 9:  RiskManager.apply()
    Stage 10: ROITracker.settle_pending()  ← picks del día anterior
    Stage 11: ROITracker.register_pick()  ← picks nuevos del día
    Stage 12: Notificaciones (opcional, fuera del scope del runner)

Por qué dos fases
------------------
Stages 1-6 son independientes por evento: si projection falla para
NYY vs BOS, LAD vs SF continúa. El error se registra en context.errors
y el evento se omite sin abortar el pipeline.

Stages 7-11 necesitan el conjunto COMPLETO de candidatos del día:
- RiskManager aplica límite de N picks diarios sobre TODOS los candidatos.
  Si se aplicara por evento, el tercer evento tendría 0 picks porque
  los dos primeros ya llenaron el cupo.
- Settlement (Stage 10) corre ANTES del registro (Stage 11) para que
  current_bankroll() esté actualizado cuando se calculan los stakes.

Inyección de dependencias total
---------------------------------
El runner no instancia ningún subsistema internamente. Todo se pasa
en el constructor — el runner es 100% testeable con mocks sin tocar
APIs externas ni archivos de configuración.

Modos de operación
--------------------
dry_run=True: ejecuta Stages 1-9 (análisis) pero omite 10-11 (ledger).
              Permite ver qué picks generaría sin afectar el bankroll.

skip_stages: set[int] para debugging granular. skip_stages={5} omite
             Stage 5 y usa odds del caché previo del context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.contracts.event import Event
from core.contracts.pick import CandidatePick
from core.pipeline.stage import (
    PipelineContext,
    SportPlugin,
)


# ── Configuración del runner ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RunnerConfig:
    """
    Configuración inmutable de una ejecución del PipelineRunner.

    Campos
    ------
    dry_run         -- Si True, omite Stages 10-11 (no toca el ledger).
                      Útil para backtesting y validación de configuración.
    skip_stages     -- Conjunto de números de stage a omitir. Stage 5
                      puede omitirse para usar odds del caché. Stage 10
                      puede omitirse para re-ejecutar sin settlement.
    max_events       -- Límite de eventos a procesar. None = sin límite.
                      Útil para tests rápidos con un subconjunto.
    odds_snapshot_date -- Fecha para los snapshots de odds. Default:
                         igual a la fecha de ejecución.
    log_stage_times  -- Si True, registra el tiempo de cada stage en
                       context.metadata para profiling.
    """
    dry_run:            bool       = False
    skip_stages:        frozenset  = field(default_factory=frozenset)
    max_events:         int | None = None
    odds_snapshot_date: str | None = None
    log_stage_times:    bool       = True


# ── Resultado de la ejecución ─────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineResult:
    """
    Output completo de una ejecución del PipelineRunner.

    Inmutable: snapshot del resultado en el momento de finalización.

    Campos
    ------
    sport            -- Deporte procesado.
    date             -- Fecha procesada.
    active_picks     -- Picks aprobados por RiskManager y registrados.
    candidates       -- Todos los candidatos antes de RiskManager
                       (útil para análisis de filtros).
    settled_count    -- Número de picks previos liquidados en Stage 10.
    errors           -- Todos los errores no fatales de la ejecución.
    context          -- PipelineContext completo con todo el estado
                       intermedio (para backtesting y dashboard).
    duration_seconds -- Tiempo total de la ejecución.
    dry_run          -- True si fue ejecución en modo simulación.
    """
    sport:            str
    date:             str
    active_picks:     list[CandidatePick]
    candidates:       list[CandidatePick]
    settled_count:    int
    errors:           list[str]
    context:          PipelineContext
    duration_seconds: float
    dry_run:          bool

    def summary(self) -> str:
        status = "[DRY RUN]" if self.dry_run else ""
        return (
            f"{status}[{self.sport}/{self.date}] "
            f"activos={len(self.active_picks)} "
            f"candidatos={len(self.candidates)} "
            f"liquidados={self.settled_count} "
            f"errores={len(self.errors)} "
            f"tiempo={self.duration_seconds:.1f}s"
        )


# ── Motor principal ───────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Orquestador del flujo diario de 12 stages.

    Parámetros (todos inyectados — el runner no instancia nada)
    -------------------------------------------------------------
    plugin          -- SportPlugin del deporte a procesar.
    odds_client     -- OddsAPIClient configurado con API key.
    normalizer      -- OddsNormalizer con estrategias del deporte.
    line_detector   -- LineMovementDetector con store configurado.
    value_engine    -- ValueEngine con blending/kelly/filters del deporte.
    staking         -- Implementación de StakingStrategy.
    risk_manager    -- RiskManager con perfil de riesgo del deporte.
    roi_tracker     -- ROITracker conectado al BankrollTracker.
    clv_tracker     -- CLVTracker para registro de opening lines.
    config          -- RunnerConfig. Default: producción sin dry_run.
    """

    def __init__(
        self,
        plugin:         Any,  # SportPlugin — evita import circular
        odds_client:    Any,  # OddsAPIClient
        normalizer:     Any,  # OddsNormalizer
        line_detector:  Any,  # LineMovementDetector
        value_engine:   Any,  # ValueEngine
        staking:        Any,  # StakingStrategy
        risk_manager:   Any,  # RiskManager
        roi_tracker:    Any,  # ROITracker
        clv_tracker:    Any,  # CLVTracker
        config:         RunnerConfig | None = None,
    ) -> None:
        self._plugin        = plugin
        self._odds_client   = odds_client
        self._normalizer    = normalizer
        self._line_detector = line_detector
        self._value_engine  = value_engine
        self._staking       = staking
        self._risk_manager  = risk_manager
        self._roi_tracker   = roi_tracker
        self._clv_tracker   = clv_tracker
        self._config        = config or RunnerConfig()

    # ── Punto de entrada ──────────────────────────────────────────────────────

    def run(self, date: str) -> PipelineResult:
        """
        Ejecuta el pipeline completo para la fecha dada.

        Parámetros
        ----------
        date  -- Fecha en 'YYYY-MM-DD'.

        Retorna
        -------
        PipelineResult con el estado completo de la ejecución.
        Nunca lanza excepción — errores se capturan en result.errors.
        """
        t_start = time.monotonic()
        sport   = self._plugin.sport_id
        context = PipelineContext(sport=sport, date=date)

        try:
            context = self._run_phase_a(date, context)
            context = self._run_phase_b(context)
        except Exception as e:
            # Error catastrófico (no debería ocurrir si cada stage
            # maneja sus propios errores)
            context.add_error("RUNNER", f"Error catastrófico: {e}")

        duration = time.monotonic() - t_start

        return PipelineResult(
            sport            = sport,
            date             = date,
            active_picks     = context.active_picks,
            candidates       = context.candidates,
            settled_count    = context.metadata.get("settled_count", 0),
            errors           = context.errors,
            context          = context,
            duration_seconds = round(duration, 2),
            dry_run          = self._config.dry_run,
        )

    # ── Fase A: por evento ────────────────────────────────────────────────────

    def _run_phase_a(self, date: str, context: PipelineContext) -> PipelineContext:
        """Stages 1-6: procesamiento independiente por evento."""

        self._stage_1_events(date, context)
        if not context.events:
            context.add_error("Stage1", "Sin eventos para esta fecha.")
            return context

        # Limitar eventos si max_events está configurado
        events = context.events
        if self._config.max_events is not None:
            events = events[:self._config.max_events]
            context.events = events

        # Stages 2-6 por evento
        for event in events:
            try:
                self._stage_2_enrich(event, context)
                self._stage_3_project(event, context)
                self._stage_4_simulate(event, context)
            except Exception as e:
                context.add_error(
                    "Stage2-4",
                    f"event={event.event_id}: {e}"
                )
                continue

        # Stage 5: odds para todos los eventos de una vez
        # (una sola request a la API cubre todos los eventos)
        if 5 not in self._config.skip_stages:
            self._stage_5_odds(context)

        # Stage 6: value calculation por evento
        for event in events:
            if event.event_id not in context.projections:
                continue  # proyecto falló para este evento
            if event.event_id not in context.market_odds:
                continue  # no hay odds para este evento
            try:
                self._stage_6_value(event, context)
            except Exception as e:
                context.add_error(
                    "Stage6",
                    f"event={event.event_id}: {e}"
                )

        return context

    # ── Fase B: sobre todos los candidatos ────────────────────────────────────

    def _run_phase_b(self, context: PipelineContext) -> PipelineContext:
        """Stages 7-11: operan sobre el conjunto completo de candidatos."""

        if not context.candidates:
            context.add_error("PhaseB", "Sin candidatos — stages 7-11 omitidos.")
            return context

        self._stage_7_line_movement(context)
        self._stage_8_staking(context)
        self._stage_9_risk(context)

        if not self._config.dry_run:
            self._stage_10_settlement(context)
            self._stage_11_register(context)
        else:
            context.set_meta("dry_run", True)

        return context

    # ── Implementación de stages ──────────────────────────────────────────────

    def _stage_1_events(self, date: str, context: PipelineContext) -> None:
        """Stage 1: obtener eventos del día."""
        if 1 in self._config.skip_stages:
            return
        t = time.monotonic()
        try:
            provider = self._plugin.get_data_provider()
            context.events = provider.get_events(date)
        except Exception as e:
            context.add_error("Stage1", str(e))
        self._log_stage_time(context, "stage_1", t)

    def _stage_2_enrich(self, event: Event, context: PipelineContext) -> None:
        """Stage 2: enriquecer evento con features y contexto."""
        if 2 in self._config.skip_stages:
            return
        provider = self._plugin.get_data_provider()
        home_f, away_f = provider.enrich_event(event)
        ctx_dict       = provider.get_context(event)
        context.enriched[event.event_id] = (home_f, away_f)
        # Guardar contexto situacional en metadata para Stage 3
        context.metadata[f"ctx_{event.event_id}"] = ctx_dict

    def _stage_3_project(self, event: Event, context: PipelineContext) -> None:
        """Stage 3: calcular proyección deportiva."""
        if 3 in self._config.skip_stages:
            return
        if event.event_id not in context.enriched:
            return
        home_f, away_f = context.enriched[event.event_id]
        ctx_dict       = context.metadata.get(f"ctx_{event.event_id}", {})
        proj_model     = self._plugin.get_projection_model()
        projection     = proj_model.project(home_f, away_f, ctx_dict)
        context.projections[event.event_id] = projection

    def _stage_4_simulate(self, event: Event, context: PipelineContext) -> None:
        """
        Stage 4: calcular probabilidades de mercado desde la proyección.

        Calcula model_probs por selección para cada mercado disponible.
        El resultado se guarda en context.metadata para que Stage 6
        (ValueEngine) construya EvaluationRequest correctamente.
        """
        if 4 in self._config.skip_stages:
            return
        if event.event_id not in context.projections:
            return

        projection = context.projections[event.event_id]
        prob_model = self._plugin.get_probability_model()

        # Probabilidades de ML (win/draw/loss)
        win_probs = prob_model.win_probabilities(projection)

        # Guardar por selección en formato que ValueEngine espera:
        # {'home_team_name': p_home, 'away_team_name': p_away, 'draw': p_draw}
        model_probs: dict[str, dict[str, float]] = {
            "ML": {
                event.home_team: win_probs.get("home", 0.0),
                event.away_team: win_probs.get("away", 0.0),
            }
        }
        if win_probs.get("draw", 0.0) > 0:
            model_probs["ML"]["Draw"] = win_probs["draw"]
            model_probs["1X2"] = model_probs["ML"].copy()

        # Probabilidades de TOTAL y SPREAD se calculan en Stage 6
        # cuando se conocen las líneas exactas de los market_odds.
        # Aquí solo guardamos la proyección y el prob_model para
        # que Stage 6 los invoque con las líneas correctas.
        context.metadata[f"model_probs_{event.event_id}"]  = model_probs
        context.metadata[f"prob_model_{event.event_id}"]   = prob_model
        context.metadata[f"projection_{event.event_id}"]   = projection

    def _stage_5_odds(self, context: PipelineContext) -> None:
        """
        Stage 5: obtener, normalizar y snapshotear cuotas.

        Una sola request a la API cubre todos los eventos del deporte.
        """
        t     = time.monotonic()
        # Usar odds_api_sport_id si el plugin lo declara (ej: "baseball_mlb"),
        # fallback a sport_id genérico (ej: "mlb") si no existe.
        sport = getattr(self._plugin, "odds_api_sport_id", self._plugin.sport_id)
        mkt   = self._plugin.get_market_definitions()

        try:
            response = self._odds_client.get_events(
                sport   = sport,
                markets = mkt.get_core_markets(),
            )
        except Exception as e:
            context.add_error("Stage5", f"OddsAPIClient falló: {e}")
            return

        if not response.success:
            context.add_error(
                "Stage5",
                f"API error ({response.error_type}): {response.error_message}"
            )
            return

        # Registrar créditos restantes
        context.set_meta("odds_credits_remaining", response.requests_remaining)

        snap_date = self._config.odds_snapshot_date or context.date

        for event in context.events:
            # Buscar el RawOddsEvent correspondiente por event_id de la API
            odds_api_id = event.provider_ids.get("odds_api")
            if not odds_api_id:
                continue

            raw_event = self._normalizer.find_by_event_id(
                response.events, odds_api_id
            )
            if raw_event is None:
                context.add_error(
                    "Stage5",
                    f"Sin odds para event_id='{odds_api_id}' ({event.home_team} vs {event.away_team})"
                )
                continue

            # Normalizar a list[MarketOdds]
            preferred_line = mkt.get_preferred_line("SPREAD")
            market_odds    = self._normalizer.extract_best(
                raw_event      = raw_event,
                markets        = mkt.get_core_markets(),
                preferred_line = preferred_line,
            )

            context.market_odds[event.event_id] = market_odds

            # Guardar snapshot para Line Movement (Stage 7)
            self._line_detector.snapshot(
                odds          = market_odds,
                event_id      = event.event_id,
                sport         = sport,
                snapshot_date = snap_date,
            )

        self._log_stage_time(context, "stage_5", t)

    def _stage_6_value(self, event: Event, context: PipelineContext) -> None:
        """
        Stage 6: calcular EV, edge, Kelly y aplicar filtros.

        Construye EvaluationRequest con model_probs ya calculados en
        Stage 4 más las líneas reales de los market_odds de Stage 5.
        """
        from core.value.engine import EvaluationRequest

        if event.event_id not in context.market_odds:
            return

        market_odds  = context.market_odds[event.event_id]
        projection   = context.metadata.get(f"projection_{event.event_id}")
        prob_model   = context.metadata.get(f"prob_model_{event.event_id}")
        base_probs   = context.metadata.get(f"model_probs_{event.event_id}", {})

        if projection is None or prob_model is None:
            return

        # Construir model_probs completo incluyendo TOTAL y SPREAD
        # usando las líneas reales de los market_odds
        model_probs: dict[str, float] = {}

        # Incorporar ML probs
        model_probs.update(base_probs.get("ML", {}))

        # Calcular TOTAL y SPREAD para cada línea disponible
        for odds in market_odds:
            try:
                if odds.market == "TOTAL" and odds.line is not None:
                    side = odds.selection.lower()
                    if side in ("over", "under"):
                        p = prob_model.total_probability(projection, odds.line, side)
                        model_probs[odds.selection] = p

                elif odds.market == "SPREAD" and odds.line is not None:
                    side = "home" if odds.selection == event.home_team else "away"
                    p = prob_model.spread_probability(projection, odds.line, side)
                    model_probs[odds.selection] = p
            except Exception:
                continue  # línea sin modelo calibrado — omitir

        if not model_probs:
            return

        # projected_value para min_line_diff (TOTAL = suma de proyecciones)
        projected_value = projection.expected_home + projection.expected_away

        try:
            request = EvaluationRequest(
                event           = event,
                model_probs     = {k: v for k, v in model_probs.items() if 0 < v < 1},
                market_odds     = market_odds,
                projected_value = projected_value,
            )
            result = self._value_engine.evaluate(request)
        except (ValueError, Exception) as e:
            context.add_error("Stage6", f"event={event.event_id}: {e}")
            return

        context.candidates.extend(result.picks_passed)
        context.candidates.extend(result.picks_rejected)
        context.errors.extend(result.errors)

    def _stage_7_line_movement(self, context: PipelineContext) -> None:
        """Stage 7: detectar movimiento de línea y anotar picks."""
        if 7 in self._config.skip_stages:
            return
        snap_date = self._config.odds_snapshot_date or context.date

        for pick in context.candidates:
            event_id = pick.event.event_id
            sport    = pick.event.sport
            try:
                signals = self._line_detector.analyze(
                    current_odds  = context.market_odds.get(event_id, []),
                    event_id      = event_id,
                    sport         = sport,
                    snapshot_date = snap_date,
                )
                self._line_detector.annotate_pick(pick, signals)
            except Exception as e:
                context.add_error("Stage7", f"event={event_id}: {e}")

    def _stage_8_staking(self, context: PipelineContext) -> None:
        """
        Stage 8: fijar stake_pct en picks candidatos.

        CORRECCIÓN DE CONTRATO (auditoría 2026-08): este import apuntaba
        antes a `core.bankroll.staking`, módulo que nunca definió
        `_MOVEMENT_CONFIRMS_PREFIX` — esa constante siempre vivió en
        `core.odds.line_movement` (ahora pública, sin guion bajo). El
        import roto producía `ImportError` garantizado en la primera
        llamada a este stage, abortando además toda la Fase B del
        pipeline (staking + riesgo + settlement + registro) para el día
        completo, ya que la excepción escalaba hasta el try/except
        genérico de `run()`.
        """
        if 8 in self._config.skip_stages:
            return
        from core.bankroll.staking import apply_staking
        from core.odds.line_movement import MOVEMENT_CONFIRMS_PREFIX

        for pick in context.candidates:
            try:
                confirms = any(
                    MOVEMENT_CONFIRMS_PREFIX in r
                    for r in pick.reasons
                )
                apply_staking(pick, self._staking, movement_confirms=confirms)
            except Exception as e:
                context.add_error("Stage8", f"pick={pick.selection}: {e}")

    def _stage_9_risk(self, context: PipelineContext) -> None:
        """Stage 9: aplicar límites de riesgo y marcar picks activos."""
        if 9 in self._config.skip_stages:
            return
        try:
            # Solo picks que pasaron los filtros del ValueEngine
            passed_filter = [
                p for p in context.candidates
                if "FILTERS PASSED" in " ".join(p.reasons)
                and p.stake_pct > 0
            ]
            summary = self._risk_manager.apply(passed_filter)
            context.active_picks = summary.picks_active
            context.set_meta("risk_summary", summary.log_summary())
        except Exception as e:
            context.add_error("Stage9", str(e))

    def _stage_10_settlement(self, context: PipelineContext) -> None:
        """Stage 10: liquidar picks pendientes del día anterior."""
        if 10 in self._config.skip_stages or self._config.dry_run:
            return
        try:
            provider = self._plugin.get_settlement_provider()
            settled  = self._roi_tracker.settle_pending(
                provider = provider,
                sport    = context.sport,
            )
            context.set_meta("settled_count", len(settled))
        except Exception as e:
            context.add_error("Stage10", str(e))
            context.set_meta("settled_count", 0)

    def _stage_11_register(self, context: PipelineContext) -> None:
        """Stage 11: registrar picks activos en el ledger."""
        if 11 in self._config.skip_stages or self._config.dry_run:
            return
        registered = 0
        for pick in context.active_picks:
            try:
                event_info = {
                    "sport":  pick.event.sport,
                    "league": self._plugin.league_id,
                    "date":   context.date,
                    "event":  (
                        f"{pick.event.away_team} @ {pick.event.home_team}"
                    ),
                }
                result = self._roi_tracker.register_pick(pick, event_info)
                if result.registered:
                    registered += 1
                    # Registrar opening line para CLV tracking
                    try:
                        self._clv_tracker.record_opening_line(
                            entry_id      = result.entry_id,
                            opening_price = pick.price,
                            market        = pick.market,
                            sport         = pick.event.sport,
                        )
                    except Exception:
                        pass  # CLV es opcional — no abortar el registro
            except Exception as e:
                context.add_error("Stage11", f"pick={pick.selection}: {e}")

        context.set_meta("registered_count", registered)

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _log_stage_time(
        self,
        context: PipelineContext,
        key:     str,
        t_start: float,
    ) -> None:
        """Registra el tiempo de un stage si log_stage_times=True."""
        if self._config.log_stage_times:
            context.set_meta(f"time_{key}", round(time.monotonic() - t_start, 3))


# ── Factory function ──────────────────────────────────────────────────────────

def build_runner(
    plugin:       Any,
    config_loader: Any,
    bankroll:     float = 1000.0,
    dry_run:      bool  = False,
) -> PipelineRunner:
    """
    Factory function que construye un PipelineRunner completamente
    configurado desde un SportPlugin y un ConfigLoader.

    Instancia todos los subsistemas con la configuración del deporte
    y los ensambla en el runner. Punto de entrada para main.py.

    Parámetros
    ----------
    plugin         -- SportPlugin del deporte.
    config_loader  -- ConfigLoader con config/{sport}.yaml cargado.
    bankroll        -- Bankroll inicial del sistema.
    dry_run         -- Si True, no toca el ledger.

    Retorna
    -------
    PipelineRunner listo para llamar .run(date).
    """
    from core.odds.client import OddsAPIClient, OddsAPIConfig
    from core.odds.normalizer import OddsNormalizer
    from core.odds.line_movement import LineMovementDetector, JsonSnapshotStore
    from core.value.blending import BlendingEngine
    from core.value.kelly import KellyCriterion
    from core.value.filters import MarketFilters
    from core.value.engine import ValueEngine
    from core.bankroll.staking import IntegerPercentStaking
    from core.bankroll.tracker import BankrollTracker, CsvLedgerStore
    from core.risk.manager import RiskManager
    from core.tracking.roi_tracker import ROITracker
    from core.evaluation.clv import CLVTracker

    sport = plugin.sport_id

    # Odds
    # En dry_run sin API key configurada usamos un placeholder: Stage 5
    # (ingesta de cuotas) se omite de todos modos en ese modo, pero
    # OddsAPIConfig valida la key en __post_init__ y abortaría la
    # construcción del runner antes de llegar a ejecutar nada. El
    # placeholder permite validar el resto del pipeline sin credenciales
    # — nunca se usa para hacer requests reales porque dry_run salta
    # el stage que las dispara.
    api_key = config_loader.get("ODDS_API_KEY")
    if not api_key and dry_run:
        api_key = "dry-run-placeholder"

    odds_client = OddsAPIClient(OddsAPIConfig(
        api_key = api_key,
        regions = config_loader.get("odds_api.regions", default="us"),
    ))
    normalizer    = OddsNormalizer()
    line_detector = LineMovementDetector(store=JsonSnapshotStore())

    # Value
    blending      = BlendingEngine(config=config_loader)
    kelly         = KellyCriterion(config=config_loader)
    filters       = MarketFilters(config=config_loader)
    value_engine  = ValueEngine(blending=blending, kelly=kelly, filters=filters)

    # Bankroll
    store         = CsvLedgerStore(f"output/ledger/{sport}_roi_tracking.csv")
    tracker       = BankrollTracker(store=store, initial_bankroll=bankroll)
    staking       = IntegerPercentStaking(config=config_loader)
    risk_manager  = RiskManager.from_config(config_loader)

    # Tracking
    roi_tracker   = ROITracker(tracker=tracker, model_version=f"{sport}-v1.0")
    clv_tracker   = CLVTracker()

    return PipelineRunner(
        plugin        = plugin,
        odds_client   = odds_client,
        normalizer    = normalizer,
        line_detector = line_detector,
        value_engine  = value_engine,
        staking       = staking,
        risk_manager  = risk_manager,
        roi_tracker   = roi_tracker,
        clv_tracker   = clv_tracker,
        config        = RunnerConfig(dry_run=dry_run),
    )