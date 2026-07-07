"""
core/value/engine.py

ValueEngine: orquestador del Bloque 3 — conecta no_vig → blending →
kelly → filters en un único punto de entrada para el pipeline.

Posición en el pipeline (Stage 6)
-----------------------------------
    Stage 5: list[MarketOdds]  (cuotas normalizadas)
        ↓
    ValueEngine.evaluate(EvaluationRequest)
        ├── no_vig_probabilities()      → market_prob por selección
        ├── BlendingEngine.blend()      → blended_prob
        ├── KellyCriterion.calculate()  → kelly_fraction
        ├── CandidatePick(...)          → pick con ev/edge calculados
        └── MarketFilters.evaluate()    → passed/rejected + reasons
        ↓
    EvaluationResult
        ├── picks_passed:   list[CandidatePick]  → Stage 7, 8, 9
        └── picks_rejected: list[CandidatePick]  → backtester/calibración

Principio rector
-----------------
El ValueEngine no sabe de qué deporte provienen los datos. No conoce
qué es ERA, OPS, park factor, proyección de carreras, ni ningún concepto
deportivo. Recibe únicamente:
    - model_probs: dict[str, float]  — probabilidades ya calculadas
    - market_odds: list[MarketOdds]  — cuotas ya normalizadas
    - projected_value: float | None  — para min_line_diff, ya extraído

La responsabilidad de mapear "proyección del modelo" → model_probs y
Projection.expected_home + expected_away → projected_value pertenece
al sport plugin. El engine no contiene lógica deportiva.

Inyección de dependencias
--------------------------
BlendingEngine, KellyCriterion y MarketFilters se pasan en el
constructor, no se instancian internamente. Esto permite:
    1. Tests determinísticos con mocks sin YAML real.
    2. Sustituir cualquier subcomponente sin modificar el engine.
    3. Compartir instancias cacheadas entre invocaciones del pipeline.

Input/Output tipados
---------------------
EvaluationRequest — frozen dataclass: valida inputs en construcción,
    documenta el contrato de entrada explícitamente.
EvaluationResult  — frozen dataclass: retorna picks_passed Y
    picks_rejected para que el backtester pueda analizar rechazos
    sin re-ejecutar el pipeline.

Robustez operacional
---------------------
- Selecciones sin model_prob → error registrado, evento continúa.
- market_odds mezclados (ML + TOTAL juntos) → mercado omitido con
  error, otros mercados continúan.
- BlendingEngine o KellyCriterion lanzan ValueError → selección
  omitida con error, evento continúa.
- Ningún error parcial aborta el evento completo.

Uso típico
-----------
    from core.value.engine import ValueEngine, EvaluationRequest
    from core.value.blending import BlendingEngine
    from core.value.kelly import KellyCriterion
    from core.value.filters import MarketFilters
    from core.utils.config_loader import load_config

    config = load_config(sport='mlb')
    engine = ValueEngine(
        blending=BlendingEngine(config=config),
        kelly=KellyCriterion(config=config),
        filters=MarketFilters(config=config),
    )

    request = EvaluationRequest(
        event=event,
        model_probs={'over': 0.58, 'under': 0.42},
        market_odds=[over_odds, under_odds],
        projected_value=10.2,   # proj_home + proj_away para TOTAL
    )

    result = engine.evaluate(request)
    # result.picks_passed  → Stage 7 (line movement)
    # result.picks_rejected → backtester para calibración
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from core.contracts.event import Event
from core.contracts.market_odds import MarketOdds
from core.contracts.pick import CandidatePick
from core.odds.no_vig import no_vig_probabilities
from core.value.blending import BlendingEngine
from core.value.filters import MarketFilters
from core.value.kelly import KellyCriterion


# ── Tipos de entrada y salida ─────────────────────────────────────────────────

@dataclass(frozen=True)
class EvaluationRequest:
    """
    Input tipado al ValueEngine.

    Frozen: el engine nunca modifica la solicitud de entrada.
    La validación ocurre en __post_init__ antes de que el engine
    procese cualquier dato.

    Campos
    ------
    event           -- Evento deportivo de origen. Provee sport y
                      league para que BlendingEngine, KellyCriterion
                      y MarketFilters resuelvan su configuración YAML.
    model_probs      -- Probabilidades del modelo por selección.
                      Las claves deben coincidir EXACTAMENTE con
                      MarketOdds.selection de los odds pasados.
                      El sport plugin es responsable del mapeo:
                          Prediction.home_win_prob → 'New York Yankees'
                          Prediction.away_win_prob → 'Boston Red Sox'
                          Prediction.over_prob     → 'over'
                      Selecciones sin model_prob se omiten con error
                      registrado en EvaluationResult.errors.
    market_odds      -- Lista de MarketOdds del evento. Puede mezclar
                      mercados distintos (ML + TOTAL + SPREAD) — el
                      engine agrupa por market antes de calcular no-vig.
                      Las cuotas deben ser las mejores disponibles
                      (ya normalizadas por OddsNormalizer en Stage 5).
    projected_value  -- Proyección numérica del modelo para este evento,
                      ya extraída y adaptada al mercado consultado por
                      el sport plugin:
                        TOTAL MLB:  proj_home + proj_away (carreras)
                        TOTAL Soccer: proj_home + proj_away (goles)
                        SPREAD NBA: proj_home - proj_away (puntos)
                        ML:         None (no aplica min_line_diff)
                      Si None, el filtro min_line_diff se omite aunque
                      esté configurado para este (sport, market).
    """
    event:           Event
    model_probs:     dict[str, float]
    market_odds:     list[MarketOdds]
    projected_value: float | None = None

    def __post_init__(self) -> None:
        if not self.model_probs:
            raise ValueError(
                "model_probs no puede estar vacío. El sport plugin debe "
                "proveer al menos una probabilidad de modelo antes de "
                "llamar a ValueEngine.evaluate()."
            )
        if not self.market_odds:
            raise ValueError(
                "market_odds no puede estar vacío. Verificar que Stage 5 "
                "(OddsIngestion) produjo cuotas antes de llamar al engine."
            )
        for sel, prob in self.model_probs.items():
            if not 0.0 < prob < 1.0:
                raise ValueError(
                    f"model_probs['{sel}']={prob} fuera de (0, 1). "
                    f"Las probabilidades del modelo deben ser estrictamente "
                    f"positivas y menores que 1."
                )


@dataclass(frozen=True)
class EvaluationResult:
    """
    Output tipado del ValueEngine.

    Retorna picks_passed Y picks_rejected — no solo los aprobados.
    El backtester necesita los rechazados para calibrar umbrales:
    distinguir "rechazado por EV bajo" de "rechazado por cuota fuera
    de rango" es información crítica que se perdería si solo se
    retornaran los picks activos.

    Campos
    ------
    picks_passed    -- Picks que pasaron todos los filtros activos.
                      Se pasan a Stage 7 (LineMovement), Stage 8
                      (Staking) y Stage 9 (RiskManager).
                      active=False por defecto — RiskManager en Stage
                      9 es el único que puede marcar active=True.
    picks_rejected  -- Picks que fallaron al menos un filtro.
                      Contienen el trail completo de reasons incluyendo
                      qué criterio específico los rechazó.
                      Útiles para calibración de umbrales con backtesting.
    errors          -- Selecciones o mercados omitidos por errores de
                      datos (sin model_prob, odds mezclados, ValueError
                      en BlendingEngine). No son picks rechazados por
                      filtros — son datos que no llegaron a generar un
                      pick. Registrados como strings descriptivos.
    event_id        -- ID del evento evaluado. Para correlacionar el
                      resultado con el evento en el pipeline sin
                      necesidad de acceder al Event completo.
    """
    picks_passed:   list[CandidatePick]
    picks_rejected: list[CandidatePick]
    errors:         list[str]
    event_id:       str

    @property
    def total_picks(self) -> int:
        """Total de picks generados (pasados + rechazados)."""
        return len(self.picks_passed) + len(self.picks_rejected)

    @property
    def has_value(self) -> bool:
        """True si hay al menos un pick que pasó los filtros."""
        return len(self.picks_passed) > 0


# ── Motor principal ───────────────────────────────────────────────────────────

class ValueEngine:
    """
    Orquestador del Bloque 3: no_vig → blending → kelly → filters.

    No contiene lógica deportiva. No sabe qué es ERA, OPS, xG, ni
    ningún concepto específico de deporte. Recibe probabilidades y
    cuotas ya procesadas y produce CandidatePick con trazabilidad
    completa.

    Parámetros
    ----------
    blending  -- BlendingEngine configurado para el deporte activo.
                Inyectado en el constructor para permitir mocks en
                tests y compartir la caché de configuración YAML
                entre llamadas del pipeline.
    kelly      -- KellyCriterion configurado para el deporte activo.
    filters    -- MarketFilters configurados para el deporte activo.
    """

    def __init__(
        self,
        blending: BlendingEngine,
        kelly: KellyCriterion,
        filters: MarketFilters,
    ) -> None:
        self._blending = blending
        self._kelly    = kelly
        self._filters  = filters

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evalúa todas las selecciones del evento y produce CandidatePick.

        Flujo por selección (dentro de un grupo de mercado):
            1. no_vig_probabilities(grupo_mercado) → market_prob
            2. blend(model_prob, market_prob, market, sport)
            3. kelly.calculate_for_market(blended_prob, price, ...)
            4. CandidatePick(...) con ev/edge calculados automáticamente
            5. add_reason(blend.to_reason()), add_reason(kelly.to_reason())
            6. filters.evaluate(pick, projected_value)
            7. add_reason(filter.to_reason())
            8. → picks_passed si passed, → picks_rejected si no

        Parámetros
        ----------
        request  -- EvaluationRequest validado (ver EvaluationRequest).

        Retorna
        -------
        EvaluationResult con picks_passed, picks_rejected y errors.
        Nunca lanza excepción por datos parciales — errores individuales
        se registran en EvaluationResult.errors.
        """
        picks_passed:   list[CandidatePick] = []
        picks_rejected: list[CandidatePick] = []
        errors:         list[str]           = []

        sport    = request.event.sport
        event_id = request.event.event_id

        # Agrupar market_odds por market — defensa contra inputs mezclados
        groups: dict[str, list[MarketOdds]] = defaultdict(list)
        for odds in request.market_odds:
            groups[odds.market].append(odds)

        for market, group_odds in groups.items():

            # ── 1. Normalización no-vig ────────────────────────────────
            try:
                normalized = no_vig_probabilities(group_odds)
            except ValueError as e:
                errors.append(
                    f"market={market}: no_vig falló ({e}). "
                    f"Mercado omitido."
                )
                continue

            # Índice rápido selection → MarketOdds normalizado
            no_vig_by_selection: dict[str, MarketOdds] = {
                o.selection: o for o in normalized
            }

            # ── 2. Procesar cada selección del mercado ─────────────────
            for odds_normalized in normalized:
                selection = odds_normalized.selection
                market_prob = odds_normalized.no_vig_prob

                # Sin market_prob no se puede construir nada
                if market_prob is None:
                    errors.append(
                        f"market={market}, selection='{selection}': "
                        f"no_vig_prob es None tras normalización. "
                        f"Selección omitida."
                    )
                    continue

                # Sin model_prob el engine no puede blend ni evaluar
                model_prob = request.model_probs.get(selection)
                if model_prob is None:
                    errors.append(
                        f"market={market}, selection='{selection}': "
                        f"sin model_prob en request.model_probs. "
                        f"Claves disponibles: "
                        f"{sorted(request.model_probs.keys())}. "
                        f"Selección omitida."
                    )
                    continue

                pick = self._build_pick(
                    event=request.event,
                    market=market,
                    selection=selection,
                    odds=odds_normalized,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    projected_value=request.projected_value,
                    picks_passed=picks_passed,
                    picks_rejected=picks_rejected,
                    errors=errors,
                )

        return EvaluationResult(
            picks_passed=picks_passed,
            picks_rejected=picks_rejected,
            errors=errors,
            event_id=event_id,
        )

    # ── Pipeline interno por selección ────────────────────────────────────────

    def _build_pick(
        self,
        event:           Event,
        market:          str,
        selection:       str,
        odds:            MarketOdds,
        model_prob:      float,
        market_prob:     float,
        projected_value: float | None,
        picks_passed:    list[CandidatePick],
        picks_rejected:  list[CandidatePick],
        errors:          list[str],
    ) -> None:
        """
        Ejecuta los stages 2-7 para una única selección y deposita el
        resultado en picks_passed o picks_rejected.

        Separado de evaluate() para mantener el loop principal legible.
        Los errores de cálculo se registran en errors y retornan sin
        producir pick — el evento continúa con las selecciones restantes.
        """
        sport = event.sport

        # ── Stage 2: Blending ──────────────────────────────────────────
        try:
            blend_result = self._blending.blend(
                model_prob=model_prob,
                market_prob=market_prob,
                market=market,
                sport=sport,
            )
        except ValueError as e:
            errors.append(
                f"market={market}, selection='{selection}': "
                f"BlendingEngine.blend() falló ({e}). "
                f"Selección omitida."
            )
            return

        # ── Stage 3: Kelly ─────────────────────────────────────────────
        try:
            kelly_result = self._kelly.calculate_for_market(
                blended_prob=blend_result.blended_prob,
                price=odds.price,
                market=market,
                sport=sport,
            )
        except ValueError as e:
            errors.append(
                f"market={market}, selection='{selection}': "
                f"KellyCriterion.calculate() falló ({e}). "
                f"Selección omitida."
            )
            return

        # ── Stage 4: CandidatePick ─────────────────────────────────────
        # ev y edge son propiedades calculadas — imposible desincronizarlas
        try:
            pick = CandidatePick(
                event=event,
                market=market,
                selection=selection,
                line=odds.line,
                price=odds.price,
                model_prob_raw=model_prob,
                market_prob=market_prob,
                blended_prob=blend_result.blended_prob,
                kelly_fraction=kelly_result.kelly_final,
                price_at_pick=odds.price,
            )
        except (ValueError, TypeError) as e:
            errors.append(
                f"market={market}, selection='{selection}': "
                f"CandidatePick() rechazó los valores ({e}). "
                f"blend={blend_result.blended_prob:.4f}, "
                f"model={model_prob:.4f}, market={market_prob:.4f}. "
                f"Selección omitida."
            )
            return

        # ── Stage 5: Trazabilidad ──────────────────────────────────────
        pick.add_reason(blend_result.to_reason())
        pick.add_reason(kelly_result.to_reason())

        # ── Stage 6: Filtros ───────────────────────────────────────────
        filter_result = self._filters.evaluate(
            pick=pick,
            projected_value=projected_value,
        )
        pick.add_reason(filter_result.to_reason())

        # ── Stage 7: Clasificar ────────────────────────────────────────
        if filter_result.passed:
            picks_passed.append(pick)
        else:
            picks_rejected.append(pick)