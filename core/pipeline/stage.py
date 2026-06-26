"""
core/pipeline/stage.py

Protocols (interfaces) entre el Core y los sport plugins.

Define los 5 contratos de comportamiento que todo sport plugin debe
satisfacer para integrarse con el PipelineRunner. Ningún módulo del
Core implementa estos Protocols — solo los consume. Los sport plugins
(sports/mlb/, sports/soccer/, etc.) los implementan mediante duck
typing estructural, sin herencia explícita.

Por qué Protocol y no ABC (ver SPORTS_PREDICTOR_ARCHITECTURE.md §13.3):
    Python Protocol permite duck typing estructural: un sport plugin que
    implementa los métodos correctos pasa la verificación de tipo de
    mypy/pyright sin necesidad de importar ni heredar ninguna clase base
    del Core. Esto reduce acoplamiento y facilita añadir deportes sin
    modificar ningún archivo del Core.

    ABC requeriría que cada plugin herede explícitamente de la clase base
    del Core, creando un acoplamiento de importación que viola el
    principio de dependencia unidireccional (plugins → core, nunca
    core → sports/).

Dependencias hacia contratos del Core:
    Estos Protocols referencian Event, TeamFeatures, Projection y
    CandidatePick del Bloque 0. MarketDefinitions (core/odds/
    market_registry.py, tarea C1.3.8) se referencia como string
    forward reference para evitar dependencia circular con core/odds/,
    que aún no existe en este punto del roadmap.

Uso típico (en un sport plugin):
    from core.pipeline.stage import SportPlugin  # solo para type hints

    class MLBPlugin:
        sport_id  = "mlb"
        league_id = "MLB"

        def get_data_provider(self)    -> SportDataProvider:    ...
        def get_projection_model(self) -> ProjectionModel:      ...
        def get_probability_model(self)-> ProbabilityModel:     ...
        def get_settlement_provider(self) -> SettlementProvider: ...
        def get_market_definitions(self) -> MarketDefinitions:  ...
        def get_config(self)           -> dict:                 ...

    # mypy/pyright verificará que MLBPlugin satisface SportPlugin
    # sin que MLBPlugin importe ni herede nada del Core.

Uso en el PipelineRunner:
    def run(plugin: SportPlugin, date: str) -> None:
        provider = plugin.get_data_provider()
        events   = provider.get_events(date)
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from core.contracts import (
    CandidatePick,
    Event,
    Projection,
    TeamFeatures,
)

# MarketDefinitions vive en core/odds/market_registry.py (tarea C1.3.8).
# Se importa solo para type checking para evitar dependencia circular
# en tiempo de ejecución — core/pipeline/ no debe importar core/odds/
# hasta que ambos módulos existan y estén estabilizados.
if TYPE_CHECKING:
    from core.odds.market_registry import MarketDefinitions # type: ignore


# ── 4.1 SportPlugin ───────────────────────────────────────────────────────────

@runtime_checkable
class SportPlugin(Protocol):
    """
    Punto de entrada de cada deporte hacia el Core.

    El PipelineRunner (core/pipeline/runner.py, tarea C1.4.1) solo
    interactúa con esta interfaz — nunca con las implementaciones
    concretas de sports/mlb/, sports/soccer/, etc. Esto garantiza
    que añadir un nuevo deporte en cualquier Fase futura no requiere
    modificar ningún módulo del Core.

    Atributos
    ---------
    sport_id   -- Identificador canónico del deporte en minúsculas.
                  Coincide con el nombre del directorio en sports/
                  y con el parámetro que load_config() espera.
                  Ejemplos: 'mlb', 'soccer', 'nba', 'nfl', 'nhl'.
    league_id   -- Identificador de la liga en mayúsculas.
                  Coincide con BetLedgerEntry.league.
                  Ejemplos: 'MLB', 'EPL', 'NBA', 'NFL', 'NHL'.
    """

    sport_id:  str
    league_id: str

    def get_data_provider(self) -> SportDataProvider:
        """Proveedor de datos para este deporte."""
        ...

    def get_projection_model(self) -> ProjectionModel:
        """Modelo de proyección de puntuación para este deporte."""
        ...

    def get_probability_model(self) -> ProbabilityModel:
        """Modelo de simulación estadística para este deporte."""
        ...

    def get_settlement_provider(self) -> SettlementProvider:
        """Proveedor de resolución de picks para este deporte."""
        ...

    def get_market_definitions(self) -> MarketDefinitions:
        """
        Catálogo de mercados soportados por este deporte.
        Define qué mercados pueden apostarse y cómo se resuelven.
        """
        ...

    def get_config(self) -> dict:
        """
        Configuración completa del plugin como dict plano.
        Normalmente retorna el resultado de
        load_config(sport=self.sport_id)._data para que el
        PipelineRunner pueda loggearlo/auditarlo sin depender
        del ConfigLoader directamente.
        """
        ...


# ── 4.2 SportDataProvider ─────────────────────────────────────────────────────

@runtime_checkable
class SportDataProvider(Protocol):
    """
    Obtención y enriquecimiento de datos deportivos.

    El Core llama get_events() y enrich_event() sin saber cómo
    se implementan — MLB usará statsapi, Soccer usará football-data.org,
    NBA usará stats.nba.com. El CacheManager del Bloque 1 es
    responsabilidad de cada implementación concreta, no del Core.
    """

    def get_events(self, date: str) -> list[Event]:
        """
        Retorna los partidos del día para date (formato 'YYYY-MM-DD').

        Responsabilidades del implementador:
        - Paginación y retry ante fallos de API.
        - Filtrar eventos sin datos suficientes (ej. pitchers TBD en MLB).
        - Retornar lista vacía (no lanzar excepción) si no hay partidos.
        """
        ...

    def enrich_event(
        self,
        event: Event,
    ) -> tuple[TeamFeatures, TeamFeatures]:
        """
        Retorna (home_features, away_features) con todos los datos
        necesarios para la proyección.

        Responsabilidades del implementador:
        - Caché propio (via CacheManager) para evitar llamadas API
          repetidas dentro del mismo pipeline diario.
        - Fallbacks documentados cuando los datos no están disponibles
          (retornar TeamFeatures con data_quality reducido, no fallar).
        - Separación entre datos de entrada (OPS, ERA, etc.) y la
          proyección resultante — enrich_event() NO proyecta, solo
          recolecta features.
        """
        ...

    def get_context(self, event: Event) -> dict:
        """
        Contexto ambiental del partido: clima, tipo de venue, hora local.

        Retorna {} si no aplica (estadio indoor, deporte indoor como NBA).
        El Core pasa este dict a ProjectionModel.project() — la
        interpretación de su contenido es responsabilidad del plugin.
        """
        ...


# ── 4.3 ProjectionModel ───────────────────────────────────────────────────────

@runtime_checkable
class ProjectionModel(Protocol):
    """
    Modelo de proyección de puntuación esperada.

    Convierte TeamFeatures (datos crudos del deporte) en una Projection
    (medias esperadas de anotación). Cada deporte implementa su propia
    fórmula:
        MLB:    ERA/FIP × OPS/wRC+ × park_factor
        Soccer: xG ofensivo vs xGA defensivo
        NBA:    ORtg × pace × DRtg_rival
        NFL:    EPA/play × snaps × situaciones

    El Core recibe solo la Projection resultante — nunca accede a ERA,
    OPS, xG ni ningún concepto deportivo directamente.
    """

    def project(
        self,
        home_features: TeamFeatures,
        away_features: TeamFeatures,
        context: dict,
    ) -> Projection:
        """
        Retorna proyección de puntuación esperada para home y away.

        context es el dict retornado por SportDataProvider.get_context()
        — la implementación decide si lo usa (ej. MLB ajusta por
        temperatura y tipo de estadio; NBA lo ignora porque los
        estadios son indoor y sin variación climática).
        """
        ...

    def model_version(self) -> str:
        """
        Identificador de la versión del modelo, para trazabilidad en
        BetLedgerEntry.model_version.

        Ejemplos: 'mlb-v1.0', 'soccer-xg-v2.1', 'nba-pace-v1.0'.
        Permite comparar rendimiento entre versiones del modelo en
        el backtesting (BacktestEngine filtra por model_version).
        """
        ...


# ── 4.4 ProbabilityModel ─────────────────────────────────────────────────────

@runtime_checkable
class ProbabilityModel(Protocol):
    """
    Simulación estadística: convierte una Projection en probabilidades
    de mercado.

    El Core resuelve qué implementación usar via DistributionFactory
    (core/simulation/factory.py, tarea C1.2.5), basándose en
    Projection.distribution. El plugin sugiere la distribución en la
    Projection; el Core la honra si existe en el Factory, o usa un
    fallback documentado.

    Distribuciones por deporte (Fases del roadmap):
        MLB, NHL, Soccer: PoissonModel / BivariatePoissonModel
        NBA, NFL:         NormalModel / SkellamModel
        Golf, Tennis:     BradleyTerryModel / NormalModel
    """

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        Retorna {'home': p, 'away': q, 'draw': r}.

        draw=0.0 para deportes sin empate (MLB, NBA, NFL).
        home + away + draw debe sumar 1.0 (dentro de tolerancia).
        """
        ...

    def spread_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        Probabilidad de cubrir el spread/handicap.

        side: 'home' o 'away'.
        line: handicap publicado (ej. -1.5 para RL en MLB,
              -4.5 para spread en NBA).
        Retorna P(team_score + line > opponent_score).
        """
        ...

    def total_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        Probabilidad de Over o Under en el total de anotaciones.

        side: 'over' o 'under'.
        line: línea de totales publicada (ej. 8.5, 215.5).
        Excluye push cuando la línea es entera (ej. P(X<9) para
        Under 9, no P(X<=9) — igual que cdf(linea-1, mu) en
        poisson_math.py).
        """
        ...


# ── 4.5 SettlementProvider ───────────────────────────────────────────────────

@runtime_checkable
class SettlementProvider(Protocol):
    """
    Resolución de resultados reales para liquidar picks.

    Cada deporte implementa su propia lógica de settlement porque
    las reglas son completamente distintas:
        MLB:    ML (winner), RL (spread entero), TOTAL (suma carreras)
        Soccer: 1X2, AH (Asian Handicap), BTTS, TOTAL
        NBA:    ML, Spread (-4.5), TOTAL
        NFL:    ML, Spread, TOTAL, Half/Quarter lines
        Tennis: ML sets, Spread sets/games, TOTAL games
    """

    def get_event_result(self, event: Event) -> dict | None:
        """
        Obtiene el resultado real del evento desde la fuente de datos.

        Retorna None si el evento aún no terminó o los datos no están
        disponibles — el ROITracker reintenta en la siguiente ejecución.

        El formato del dict lo define cada plugin. Ejemplo MLB:
            {'home_score': 5, 'away_score': 3, 'status': 'final',
             'innings': 9}
        Ejemplo Soccer:
            {'home_score': 2, 'away_score': 1, 'status': 'final',
             'home_score_ht': 1, 'away_score_ht': 0}
        """
        ...

    def settle_pick(
        self,
        pick: CandidatePick,
        event_result: dict,
    ) -> str:
        """
        Aplica las reglas del deporte para resolver un pick.

        Retorna uno de: 'win', 'lose', 'null', 'void', 'pending'.
        'void' = evento cancelado/pospuesto; stake se devuelve.
        'null' = push (empate en handicap); stake se devuelve.
        'pending' = datos insuficientes; el ROITracker reintenta.

        Maneja toda la lógica de ML/Spread/Total para su deporte —
        el Core recibe solo el string de resultado y lo pasa a
        BetLedgerEntry.settle().
        """
        ...