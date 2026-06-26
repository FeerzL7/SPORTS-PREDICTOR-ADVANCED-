"""
core.contracts
==============
Tipos de datos compartidos entre el Core y los sport plugins.

Todos los contratos son dataclasses con tipado completo. Reemplazan
los diccionarios mutables del sistema MLB-PREDICTOR-ADVANCED,
eliminando la categoría de bugs silenciosos causada por acceso a
claves inexistentes, tipos incorrectos o campos que podían divergir
entre sí sin que nada lo detectara (ver CRITICAL_FINDINGS_VALIDATION.md,
hallazgos F1/F2 sobre runs_last_5/runs_recientes_lista).

Regla de oro: ningún contrato importa nada de ``sports/``. La
dependencia va en una sola dirección: plugins → contratos, nunca al
revés.

Inmutabilidad por contrato
---------------------------
Event y MarketOdds son ``frozen=True``: representan hechos fijos
(un partido programado, una cuota observada en un instante) que
viajan sin modificación a través del pipeline.

TeamFeatures, Projection y CandidatePick son mutables: se construyen
o enriquecen progresivamente en múltiples stages del pipeline.

BetLedgerEntry es mutable únicamente a través de su método settle(),
que es la única vía legítima para liquidar un pick y que rechaza
explícitamente cualquier intento de re-liquidación.

Exports públicos
-----------------
Dataclasses:
    Event             Partido o evento deportivo (inmutable).
    TeamFeatures      Resumen estadístico normalizado de un equipo.
    Projection        Salida del ProjectionModel de un sport plugin.
    MarketOdds        Cuota de mercado normalizada para una selección
                      (inmutable).
    CandidatePick     Decisión de apuesta con trazabilidad completa;
                      ev y edge son propiedades calculadas, nunca
                      campos asignables directamente.
    BetLedgerEntry    Registro financiero permanente; profit_amount,
                      bankroll_after y yield_pct solo se pueblan vía
                      settle().

Constantes de validación (API pública reutilizable por otros módulos
del Core y por sport plugins):
    EventStatus        Constantes de estado de Event
                      (SCHEDULED/LIVE/FINAL/POSTPONED/CANCELLED) y su
                      conjunto EventStatus.ALL.
    VALID_DISTRIBUTIONS  Set MUTABLE de distribuciones de probabilidad
                      reconocidas por Projection/DistributionFactory.
                      Registrar una distribución nueva es
                      VALID_DISTRIBUTIONS.add("nombre") antes de
                      construir una Projection que la use.
    VALID_RESULTS       Frozenset de resultados válidos para
                      BetLedgerEntry.result
                      (pending/win/lose/null/void).
    TERMINAL_RESULTS    Frozenset de resultados terminales
                      (win/lose/null/void) — usado por ROITracker
                      para identificar picks liquidables.
    MIN_RECENT_SAMPLE   Umbral mínimo de partidos recientes para que
                      TeamFeatures.has_sufficient_sample sea True.
                      Referenciado por EnsembleModel (Bloque 2 del
                      roadmap) para decidir si aplica regresión
                      lineal sobre la forma reciente de un equipo.

Constantes deliberadamente NO exportadas (detalles internos de
implementación de cada contrato, sin uso documentado fuera de su
propio módulo en ningún sprint del roadmap):
    PROBABILITY_SUM_TOLERANCE, BLENDED_PROB_RANGE_TOLERANCE,
    MARKETS_WITHOUT_LINE, MIN_DATA_QUALITY,
    NO_RECENT_DATA_QUALITY_PENALTY. Importar directamente desde el
    submódulo si un caso de uso futuro genuinamente las requiere
    (ej. ``from core.contracts.projection import
    PROBABILITY_SUM_TOLERANCE``) — ampliar este __init__.py en ese
    momento es un cambio de una línea.
"""

from core.contracts.event import Event, EventStatus
from core.contracts.features import TeamFeatures, MIN_RECENT_SAMPLE
from core.contracts.projection import Projection, VALID_DISTRIBUTIONS
from core.contracts.market_odds import MarketOdds
from core.contracts.pick import CandidatePick
from core.contracts.ledger import (
    BetLedgerEntry,
    VALID_RESULTS,
    TERMINAL_RESULTS,
)

__all__ = [
    # Dataclasses
    "Event",
    "TeamFeatures",
    "Projection",
    "MarketOdds",
    "CandidatePick",
    "BetLedgerEntry",
    # Constantes de validación — API pública
    "EventStatus",
    "VALID_DISTRIBUTIONS",
    "VALID_RESULTS",
    "TERMINAL_RESULTS",
    "MIN_RECENT_SAMPLE",
]