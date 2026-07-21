"""
core/pipeline/stage.py

Protocolos del pipeline multi-deporte y contexto de ejecución.

Este módulo define los contratos que:
    1. Cada sport plugin DEBE implementar para integrarse al pipeline.
    2. El PipelineRunner usa para orquestar el flujo diario.
    3. El Core usa para invocar modelos deportivos sin conocer su implementación.

Principio rector de todo el proyecto
---------------------------------------
El Core no debe saber que existe un pitcher, un xG, o un park factor.
Estos protocolos son la frontera exacta entre el Core (agnóstico al
deporte) y los sport plugins (específicos). Cualquier concepto
deportivo que cruce esta frontera en el código del Core es una violación
arquitectónica.

Dependencia unidireccional estricta:
    plugins → stage.py → contracts → core

Los sport plugins importan desde contracts y stage.
stage.py importa desde contracts.
El core nunca importa desde plugins.

Protocolos del sport plugin (§4.1 - §4.5 de la arquitectura)
--------------------------------------------------------------
SportPlugin          — punto de entrada de cada deporte. El runner
                       solo interactúa con esta interfaz.
SportDataProvider    — obtiene eventos del día y los enriquece con
                       features. Implementación específica de cada
                       deporte (MLB Stats API, NBA API, etc.).
ProjectionModel      — convierte TeamFeatures en Projection. Contiene
                       toda la lógica predictiva deportiva.
ProbabilityModel     — convierte Projection en probabilidades de mercado.
                       El Core resuelve la implementación via
                       DistributionFactory.
SettlementProvider   — determina el resultado deportivo de un evento
                       y resuelve picks. Versión "deportiva" —
                       distinta de tracking/protocols.py que liquida
                       el ledger financiero.
MarketDefinitions    — catálogo de mercados disponibles para el deporte.
                       Wrappea MarketRegistry con contexto del deporte.

Protocolo del pipeline
-----------------------
PipelineStage        — cualquier stage del pipeline diario.
                       run(context) → PipelineContext.

Contexto compartido
--------------------
PipelineContext      — estado mutable que fluye a través de todos los
                       stages de una ejecución. Acumula el resultado de
                       cada stage para el siguiente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from core.contracts.event import Event
from core.contracts.features import TeamFeatures
from core.contracts.pick import CandidatePick
from core.contracts.projection import Projection

# Import circular evitado con TYPE_CHECKING:
# MarketRegistry está en core/odds/ que puede importar desde core/contracts/
# Si importáramos directamente, habría riesgo de ciclo en runtime.
if TYPE_CHECKING:
    from core.odds.market_registry import MarketRegistry


# ── MarketDefinitions ─────────────────────────────────────────────────────────

@runtime_checkable
class MarketDefinitions(Protocol):
    """
    Catálogo de mercados disponibles para un deporte específico.

    Implementado por cada sport plugin — generalmente como un wrapper
    de MarketRegistry con la configuración específica del deporte
    (preferred_line, mercados activos, thresholds de línea, etc.).

    El runner usa esta interfaz para saber qué mercados pedir al
    OddsAPIClient sin conocer los detalles de cada deporte.
    """

    def get_core_markets(self) -> list[str]:
        """
        Retorna los API keys de mercados CORE para este deporte.

        Son los mercados que se piden en el endpoint principal de la
        API (bajo costo, todos los eventos en una request).
        Ej: ['h2h', 'totals', 'spreads'] para MLB.
        """
        ...

    def get_extended_markets(self) -> list[str]:
        """
        Retorna los API keys de mercados EXTENDED para este deporte.

        Requieren endpoint por evento o mayor costo en créditos.
        El runner decide si solicitarlos según créditos disponibles.
        Ej: ['h2h_1st_5_innings', 'pitcher_strikeouts'] para MLB.
        """
        ...

    def get_preferred_line(self, market: str) -> float | None:
        """
        Retorna la línea preferida para un mercado de spread/total.

        Usada por OddsNormalizer.extract_best() para seleccionar
        la línea más representativa cuando hay múltiples disponibles.

        Ejemplos:
            MLB SPREAD: -1.5 (runline estándar)
            NBA SPREAD: None (usar línea de mayor consenso)
            TOTAL:      None (usar línea de mayor consenso)

        Retorna None si no hay preferencia para este mercado.
        """
        ...


# ── SportDataProvider ─────────────────────────────────────────────────────────

@runtime_checkable
class SportDataProvider(Protocol):
    """
    Responsable de obtener eventos del día y enriquecerlos con features.

    El Core llama get_events() y enrich_event() sin saber cómo se
    obtienen los datos (MLB Stats API, NBA API, web scraping, etc.).

    Garantías del protocolo
    ------------------------
    - get_events() nunca lanza excepción por eventos sin datos —
      retorna lista vacía si no hay partidos.
    - enrich_event() nunca retorna TeamFeatures con campos None que
      sean críticos para ProjectionModel — usa fallbacks documentados.
    - get_context() siempre retorna un dict (vacío si no aplica).
    """

    def get_events(self, date: str) -> list[Event]:
        """
        Retorna los partidos del día en formato Event.

        Parámetros
        ----------
        date  -- Fecha en 'YYYY-MM-DD'. El provider obtiene eventos
                de esta fecha en el timezone del deporte (ET para MLB
                y NBA, CET para ligas europeas).

        Retorna
        -------
        list[Event] — vacía si no hay partidos o el provider falla.
        No propaga excepciones de APIs externas.
        """
        ...

    def enrich_event(
        self,
        event: Event,
    ) -> tuple[TeamFeatures, TeamFeatures]:
        """
        Enriquece un evento con features estadísticas de ambos equipos.

        Retorna (home_features, away_features) con todos los datos
        necesarios para ProjectionModel. El provider maneja su propio
        caché y fallbacks para datos faltantes.

        Parámetros
        ----------
        event  -- Event a enriquecer. event.home_team_id y
                 event.away_team_id identifican los equipos en la
                 API deportiva del provider.

        Retorna
        -------
        (TeamFeatures_home, TeamFeatures_away). Los campos de
        TeamFeatures con datos insuficientes usan values conservadores
        documentados en el plugin, nunca None para campos críticos.
        """
        ...

    def get_context(self, event: Event) -> dict:
        """
        Contexto situacional del partido para el ProjectionModel.

        Información que afecta la proyección pero no está en
        TeamFeatures: clima, tipo de estadio, hora local, travel.

        Retorna {} si no aplica (estadio indoor, deporte indoor,
        contexto no disponible).

        Claves comunes (no todas aplican a todos los deportes):
            'temperature': float   — temperatura en °F
            'wind_speed':  float   — velocidad del viento en mph
            'venue_type':  str     — 'outdoor', 'retractable', 'indoor'
            'day_night':   str     — 'day', 'night'
            'travel_days_home': int — días de descanso del equipo local
            'travel_days_away': int — días de descanso del visitante
        """
        ...


# ── ProjectionModel ───────────────────────────────────────────────────────────

@runtime_checkable
class ProjectionModel(Protocol):
    """
    Convierte TeamFeatures en una Projection de puntuación esperada.

    Contiene TODA la lógica predictiva específica del deporte.
    El Core nunca implementa ProjectionModel — es responsabilidad
    exclusiva de cada sport plugin.

    Ejemplos de implementación:
        MLBProjectionModel: usa ERA, FIP, xFIP, park factors, bullpen
        NBAProjectionModel: usa OffRtg, DefRtg, pace, rest, travel
        SoccerProjectionModel: usa xG, xGA, forma reciente, H2H

    El Core solo recibe la Projection — no sabe qué variables se usaron.
    """

    def project(
        self,
        home_features: TeamFeatures,
        away_features: TeamFeatures,
        context:       dict,
    ) -> Projection:
        """
        Calcula la proyección de puntuación para ambos equipos.

        Parámetros
        ----------
        home_features  -- Features del equipo local.
        away_features  -- Features del equipo visitante.
        context        -- Contexto situacional de SportDataProvider.

        Retorna
        -------
        Projection con expected_home, expected_away, home_win_prob,
        away_win_prob, draw_prob y opcionalmente distribution y
        distribution_params para que DistributionFactory use el
        modelo más apropiado para este partido específico.
        """
        ...

    def model_version(self) -> str:
        """
        Identificador de la versión del modelo.

        Se registra en cada BetLedgerEntry.model_version para
        permitir comparación de versiones en backtesting.

        Formato sugerido: '{sport}-v{major}.{minor}.{patch}'
        Ejemplo: 'mlb-v2.1.0'
        """
        ...


# ── ProbabilityModel ──────────────────────────────────────────────────────────

@runtime_checkable
class ProbabilityModel(Protocol):
    """
    Convierte una Projection en probabilidades de mercado.

    El Core resuelve qué implementación usar via DistributionFactory
    según (sport, market) y Projection.distribution.

    Implementaciones disponibles (core/simulation/):
        PoissonModel         — MLB, NHL
        BivariatePoissonModel — Soccer (con corrección Dixon-Coles)
        NormalModel          — NFL, NBA totales
        SkellamModel         — NBA spread
        BradleyTerryModel    — Tennis
    """

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        Calcula P(home wins), P(away wins), P(draw).

        Retorna {'home': p, 'away': q, 'draw': r} donde p+q+r ≈ 1.0.
        draw=0.0 para deportes sin empate posible (MLB, NBA, NFL, Tennis).
        """
        ...

    def spread_probability(
        self,
        projection: Projection,
        line:       float,
        side:       str,
    ) -> float:
        """
        P(selección cubre el spread).

        Parámetros
        ----------
        line  -- Handicap de la selección en el mercado. Negativo para
                favoritos (ej. -4.5), positivo para dogs (+4.5).
        side  -- 'home' o 'away'.
        """
        ...

    def total_probability(
        self,
        projection: Projection,
        line:       float,
        side:       str,
    ) -> float:
        """
        P(total de puntos/carreras/goles cruza la línea).

        Parámetros
        ----------
        line  -- Línea del total (ej. 8.5 carreras, 221.5 puntos).
        side  -- 'over' o 'under'.
        """
        ...


# ── SettlementProvider (versión deportiva) ────────────────────────────────────

@runtime_checkable
class SettlementProvider(Protocol):
    """
    Obtiene el resultado deportivo de un evento y resuelve picks.

    NOTA — Dos SettlementProvider en el sistema:
    Esta interfaz (core/pipeline/stage.py) es la "deportiva" — el
    sport plugin la implementa y el runner la llama en Stage 10 para
    obtener el score real y determinar el resultado de cada pick.

    La interfaz en core/tracking/protocols.py es la "financiera" —
    recibe un BetLedgerEntry y retorna un SettlementResult con el
    CLV incluido. El sport plugin MLB implementará ambas interfaces.

    Separación de responsabilidades:
        get_event_result()  → score crudo del partido (datos deportivos)
        settle_pick()       → win/lose/null/void (interpretación del pick)
    """

    def get_event_result(
        self,
        event: Event,
    ) -> dict | None:
        """
        Obtiene el resultado final del partido.

        Retorna dict con el score y status del partido, o None si
        el partido no ha terminado o el resultado no está disponible.

        El formato exacto lo define cada sport plugin — el Core no
        interpreta el dict directamente. settle_pick() lo consume.

        Ejemplos por deporte:
            MLB:    {'home_score': 5, 'away_score': 3, 'status': 'final',
                     'innings': 9}
            NBA:    {'home_score': 112, 'away_score': 108,
                     'status': 'final', 'overtime': False}
            Soccer: {'home_score': 2, 'away_score': 1, 'status': 'FT',
                     'home_ht': 1, 'away_ht': 0}

        Retorna None si:
            - El partido no ha terminado
            - La API no está disponible
            - El evento fue postponed/cancelled (settle_pick retornará
              'void' cuando lo reciba posteriormente)
        """
        ...

    def settle_pick(
        self,
        pick:         CandidatePick,
        event_result: dict,
    ) -> str:
        """
        Determina el resultado del pick dado el resultado del partido.

        Contiene TODA la lógica de interpretación del mercado para
        el deporte: runline de MLB, overtime de NBA, tiempo reglamentario
        de soccer, reglas de void por cancelación, etc.

        Parámetros
        ----------
        pick          -- CandidatePick con market, selection y line.
        event_result  -- Dict retornado por get_event_result().

        Retorna
        -------
        'win'     — el pick ganó
        'lose'    — el pick perdió
        'null'    — push/empate (stake devuelto)
        'void'    — anulado (evento cancelado, reglas de book)
        'pending' — el partido terminó pero la resolución requiere
                   información adicional (ej. confirmación de stats
                   oficiales para props de jugadores)
        """
        ...


# ── SportPlugin ───────────────────────────────────────────────────────────────

@runtime_checkable
class SportPlugin(Protocol):
    """
    Punto de entrada de cada deporte al pipeline.

    El PipelineRunner solo interactúa con esta interfaz — nunca
    importa desde sports/mlb/, sports/nba/, etc. directamente.

    Cada sport plugin provee sus componentes via factory methods.
    El runner obtiene las instancias una vez por ejecución y las
    reutiliza para todos los eventos del día.

    Atributos obligatorios
    -----------------------
    sport_id   -- Identificador del deporte en minúsculas.
                 Debe coincidir con Event.sport de los eventos que
                 retorna get_data_provider().get_events().
                 Ejemplos: 'mlb', 'nba', 'nfl', 'soccer', 'tennis'

    league_id  -- Identificador de la liga en mayúsculas.
                 Usado en BetLedgerEntry.league.
                 Ejemplos: 'MLB', 'NBA', 'EPL', 'NBA', 'ATP'
    """
    sport_id:  str
    league_id: str

    def get_data_provider(self) -> SportDataProvider:
        """Retorna el provider de datos para este deporte."""
        ...

    def get_projection_model(self) -> ProjectionModel:
        """
        Retorna el modelo de proyección deportivo.

        El runner llama project() por cada evento del día.
        El plugin es responsable de inicializar el modelo con
        sus propios datos (ERA histórico, ratings Elo, etc.)
        antes de retornarlo.
        """
        ...

    def get_probability_model(self) -> ProbabilityModel:
        """
        Retorna el modelo de probabilidad estadístico.

        El plugin puede retornar un modelo específico o delegar
        en DistributionFactory para que el core resuelva según
        el sport_id y el mercado.
        """
        ...

    def get_settlement_provider(self) -> SettlementProvider:
        """Retorna el provider de liquidación para este deporte."""
        ...

    def get_market_definitions(self) -> MarketDefinitions:
        """
        Retorna el catálogo de mercados disponibles para este deporte.

        El runner usa esto para saber qué pedir al OddsAPIClient.
        """
        ...

    def get_config(self) -> dict:
        """
        Retorna la configuración del deporte como dict.

        El runner puede pasar esta config a los subsistemas del
        Core (BlendingEngine, MarketFilters, etc.) si no tienen
        acceso directo al ConfigLoader del deporte.

        Retorna {} si el plugin no tiene configuración adicional.
        """
        ...


# ── PipelineStage ─────────────────────────────────────────────────────────────

@dataclass
class PipelineContext:
    """
    Estado mutable que fluye a través de los stages del pipeline diario.

    Un PipelineContext es creado al inicio de cada ejecución del
    PipelineRunner y va siendo enriquecido por cada stage hasta
    producir el resultado final (picks activos y ledger actualizado).

    Mutable por diseño: cada stage añade su output sin necesidad de
    crear nuevos objetos. El runner es responsable de no reutilizar
    un context entre ejecuciones distintas.

    Campos
    ------
    sport           -- Identificador del deporte de esta ejecución.
    date            -- Fecha de la ejecución en 'YYYY-MM-DD'.
    events          -- Eventos del día (Stage 1: SportDataProvider).
    enriched        -- {event_id: (home_features, away_features)}
                      (Stage 2: enrich_event).
    projections     -- {event_id: Projection} (Stage 3: project).
    raw_odds        -- {event_id: list[RawOddsEvent]} (Stage 4: OddsAPIClient).
    market_odds     -- {event_id: list[MarketOdds]} (Stage 5: OddsNormalizer).
    candidates      -- Picks candidatos antes de filtros (Stage 6: ValueEngine).
    active_picks    -- Picks aprobados por RiskManager (Stage 9).
    errors          -- Errores no fatales acumulados por todos los stages.
    metadata        -- Metadatos de la ejecución (tiempos, créditos API, etc.).
    """
    sport:        str
    date:         str
    events:       list[Event]                              = field(default_factory=list)
    enriched:     dict[str, tuple[TeamFeatures, TeamFeatures]] = field(default_factory=dict)
    projections:  dict[str, Projection]                    = field(default_factory=dict)
    raw_odds:     dict[str, list]                          = field(default_factory=dict)
    market_odds:  dict[str, list]                          = field(default_factory=dict)
    candidates:   list[CandidatePick]                      = field(default_factory=list)
    active_picks: list[CandidatePick]                      = field(default_factory=list)
    errors:       list[str]                                = field(default_factory=list)
    metadata:     dict                                     = field(default_factory=dict)

    def add_error(self, stage: str, message: str) -> None:
        """Añade un error no fatal al log del contexto."""
        self.errors.append(f"[{stage}] {message}")

    def set_meta(self, key: str, value) -> None:
        """Registra metadato de la ejecución (tiempo, créditos, etc.)."""
        self.metadata[key] = value

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_active(self) -> int:
        return len(self.active_picks)

    def summary(self) -> str:
        return (
            f"[{self.sport}/{self.date}] "
            f"eventos={self.n_events} "
            f"candidatos={len(self.candidates)} "
            f"activos={self.n_active} "
            f"errores={len(self.errors)}"
        )


@runtime_checkable
class PipelineStage(Protocol):
    """
    Protocolo para cualquier stage del pipeline diario.

    Cada stage recibe el PipelineContext del stage anterior,
    realiza su trabajo y retorna el context enriquecido.

    La interfaz es intencionalmente simple: un único método run().
    Los stages del pipeline (ValueEngine, RiskManager, etc.) no
    necesitan implementar este Protocol explícitamente — son clases
    autónomas que el runner invoca. Este Protocol existe para
    permitir que el runner trate los stages uniformemente cuando
    se necesita composición genérica.
    """

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Ejecuta el stage y retorna el context actualizado.

        El stage debe:
        - Leer los datos que necesita desde context
        - Añadir sus resultados al context
        - Registrar errores no fatales con context.add_error()
        - Nunca lanzar excepción por datos faltantes en el context
          (registrar el error y retornar el context sin modificar)

        Retorna el mismo context (mutado), no una copia.
        """
        ...