"""
sports/nfl/plugin.py

NFLPlugin: punto de entrada del plugin NFL al pipeline.
Implementa core/pipeline/stage.py:SportPlugin.

El PipelineRunner solo interactúa con esta clase. Nunca importa desde
sports/nfl/provider.py, projections.py ni ningún otro módulo del
plugin — toda la composición ocurre aquí.

Instancias compartidas
------------------------
Los componentes NFL tienen dependencias cruzadas que importa resolver
una sola vez:

    NFLDataSource      lo usan schedule, team_stats e injuries. Sin
                       compartirlo, cada uno descargaría su propia copia
                       del play-by-play (~50 MB por temporada).

    NFLScheduleFetcher lo usan rest, context, h2h y settlement. Todos
                       consultan el mismo calendario.

    NFLInjuryFetcher   lo usan el provider (para las features) y el
    NFLRestFetcher     modelo de proyección (para los ajustes en
    NFLH2HFetcher      puntos). Si el plugin creara instancias
                       separadas, el modelo recalcularía penalizaciones
                       que el provider ya resolvió, y ambos podrían
                       divergir si sus cachés se desincronizaran.

Por eso los factory methods construyen perezosamente pero memorizan:
la primera llamada crea, las siguientes devuelven la misma instancia.

El identificador de The Odds API
----------------------------------
`odds_api_sport_id = "americanfootball_nfl"` existe por el mismo motivo
que su equivalente en MLB: el sport_id interno ('nfl') no es el
identificador que usa The Odds API.

Sin ese atributo, el runner pediría cuotas a
/v4/sports/nfl/odds y recibiría un 404 — exactamente el fallo que
apareció en MLB con 'mlb' vs 'baseball_mlb'. El runner lee
`odds_api_sport_id` con `getattr` y cae al `sport_id` si no existe,
así que declararlo es lo único necesario.

Dependencia opcional
----------------------
El plugin NFL requiere nfl_data_py, que arrastra pandas y pyarrow
(~150 MB). Esa dependencia es exclusiva de NFL: el Core y el plugin MLB
no la tocan.

`is_available()` permite a run_daily.py comprobar si el plugin puede
funcionar antes de intentar construirlo, y dar un mensaje útil en vez
de una traza de ImportError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Importes solo para anotaciones. Van bajo TYPE_CHECKING porque el
    # plugin usa imports diferidos dentro de cada factory method — así
    # el Core no carga módulos del plugin hasta que se piden, y añadir
    # estas anotaciones no revierte esa decisión.
    #
    # Con `from __future__ import annotations` las anotaciones no se
    # evalúan en tiempo de ejecución, así que el bloque nunca se
    # ejecuta fuera del type checker.
    from core.pipeline.stage import (
        MarketDefinitions,
        ProbabilityModel,
        ProjectionModel,
        SettlementProvider,
        SportDataProvider,
    )


class NFLPlugin:
    """
    Punto de entrada del plugin NFL.

    Parámetros
    ----------
    config_loader    -- ConfigLoader con base.yaml + nfl.yaml.
                        Se propaga a todos los componentes que leen
                        parámetros calibrados.
    season           -- Temporada NFL. None = actual. La temporada NFL
                        se identifica por su año de inicio: la 2026 va
                        de septiembre 2026 a febrero 2027.
    include_props    -- Si False, excluye las props de jugador del
                        catálogo de mercados. Ahorra créditos de API.
    include_periods  -- Si False (por defecto), excluye los mercados de
                        primera mitad, que NFLSettlementProvider no
                        puede liquidar con el marcador final.
    """

    sport_id:          str = "nfl"
    league_id:         str = "NFL"
    odds_api_sport_id: str = "americanfootball_nfl"

    def __init__(
        self,
        config_loader   = None,
        season:         int | None = None,
        include_props:  bool = True,
        include_periods: bool = False,
    ) -> None:
        self._config          = config_loader
        self._season          = season
        self._include_props   = include_props
        self._include_periods = include_periods

        # Instancias memorizadas. None = aún no construida.
        #
        # Los cinco componentes que el PipelineRunner consume se anotan
        # con los PROTOCOLS del Core, no con las clases concretas del
        # plugin. La diferencia importa:
        #
        #   _data_provider: NFLDataProvider | None
        #       obliga a que sea exactamente esa clase. Rechaza
        #       cualquier otra implementación aunque cumpla el contrato
        #       — incluidos dobles de prueba, un provider con caché
        #       offline o un backend alternativo.
        #
        #   _data_provider: SportDataProvider | None
        #       acepta cualquier cosa que cumpla el Protocol, que es
        #       exactamente lo que el runner necesita y lo único que
        #       este plugin garantiza.
        #
        # Es el mismo criterio que ScheduleSource en rest.py y
        # context.py: depender del contrato, no de la implementación.
        self._data_source       = None
        self._schedule          = None
        self._venues            = None
        self._injuries          = None
        self._rest              = None
        self._h2h               = None
        self._data_provider:     SportDataProvider | None  = None
        self._projection_model:  ProjectionModel | None    = None
        self._probability_model: ProbabilityModel | None   = None
        self._settlement:        SettlementProvider | None = None
        self._market_defs:       MarketDefinitions | None  = None

    # ── SportPlugin Protocol ──────────────────────────────────────────────────

    def get_data_provider(self):
        """
        NFLDataProvider con todos los fetchers compartidos.

        Se le inyectan las instancias ya construidas en vez de dejar
        que las cree: así el modelo de proyección y el provider usan
        exactamente los mismos fetchers de lesiones, descanso y H2H,
        con la misma caché.
        """
        if self._data_provider is None:
            from sports.nfl.provider import NFLDataProvider
            self._data_provider = NFLDataProvider(
                data_source        = self._get_data_source(),
                season             = self._season,
                config_loader      = self._config,
                schedule_fetcher   = self._get_schedule(),
                injury_fetcher     = self._get_injuries(),
                rest_fetcher       = self._get_rest(),
                venue_factors      = self._get_venues(),
                h2h_fetcher        = self._get_h2h(),
            )
        return self._data_provider

    def get_projection_model(self):
        """
        NFLProjectionModel con los fetchers de ajuste compartidos.

        El modelo recibe las mismas instancias de lesiones, descanso y
        H2H que el provider. Compartirlas evita que recalcule
        penalizaciones que el provider ya dejó en sport_metadata, y
        garantiza que ambos vean el mismo estado si algún fetcher
        cachea resultados.
        """
        if self._projection_model is None:
            from sports.nfl.projections import NFLProjectionModel

            # Los tres fetchers son OPCIONALES para el modelo: su firma
            # los declara así y cae a sport_metadata cuando faltan, que
            # es justo lo que el provider ya deja poblado.
            #
            # Construirlos con _try() en vez de directamente respeta esa
            # opcionalidad. Sin ello, el modelo sería inconstruible en
            # cualquier entorno sin nfl_data_py — incluido el caso
            # legítimo de proyectar desde features precalculadas, donde
            # no hace falta descargar nada.
            #
            # Es el mismo criterio aplicado en NFLDataProvider: no
            # acoplar un componente a una dependencia que su propio
            # contrato declara prescindible.
            self._projection_model = NFLProjectionModel(
                config_loader  = self._config,
                injury_fetcher = self._try(self._get_injuries),
                rest_fetcher   = self._try(self._get_rest),
                h2h_fetcher    = self._try(self._get_h2h),
            )
        return self._projection_model

    def get_probability_model(self):
        """
        NormalModel resuelto por DistributionFactory.

        La factory mapea ('nfl', cualquier mercado) a NormalModel. Se
        usa `get_model`, que es el método real de la factory — llamar a
        un inexistente `build` fue el bloqueador B2 en el plugin MLB.

        Tras la corrección de la tarea 10.11, NormalModel consume
        `sigma_margin` y `sigma_total` directamente desde
        distribution_params, así que la sigma adaptativa que calcula
        NFLProjectionModel (13.5 base, 15.12 con el QB fuera) llega
        intacta al cálculo de probabilidades.
        """
        if self._probability_model is None:
            from core.simulation.factory import DistributionFactory
            factory = DistributionFactory(config=self._config)
            self._probability_model = factory.get_model("nfl", "SPREAD")
        return self._probability_model

    def get_settlement_provider(self):
        """NFLSettlementProvider sobre el calendario compartido."""
        if self._settlement is None:
            from sports.nfl.settlement import NFLSettlementProvider
            self._settlement = NFLSettlementProvider(
                schedule_fetcher=self._get_schedule()
            )
        return self._settlement

    def get_market_definitions(self):
        """NFLMarketDefinitions con los flags del constructor."""
        if self._market_defs is None:
            from sports.nfl.markets import NFLMarketDefinitions
            self._market_defs = NFLMarketDefinitions(
                include_props   = self._include_props,
                include_periods = self._include_periods,
            )
        return self._market_defs

    def get_config(self) -> dict:
        """
        Configuración NFL como dict, para subsistemas del Core que no
        reciben el ConfigLoader directamente.
        """
        if self._config is None:
            return {}
        result: dict = {}
        for section in ("blending", "kelly", "filters", "staking",
                        "risk", "line_movement", "ensemble", "nfl"):
            try:
                value = self._config.get(section, default=None)
            except Exception:
                continue
            if value is not None:
                result[section] = value
        return result

    # ── Componentes compartidos ───────────────────────────────────────────────

    def _get_data_source(self):
        """
        NFLDataSource único para schedule, team_stats e injuries.

        Es la instancia que hace posible el caché en dos niveles
        descrito en data_source.py: memoria durante la ejecución,
        parquet entre ejecuciones.
        """
        if self._data_source is None:
            from sports.nfl.data_source import NFLDataSource, _current_nfl_season
            self._data_source = NFLDataSource(
                current_season=self._season or _current_nfl_season()
            )
        return self._data_source

    def _get_schedule(self):
        """NFLScheduleFetcher, consumido por rest, context, h2h y settlement."""
        if self._schedule is None:
            from sports.nfl.schedule import NFLScheduleFetcher
            self._schedule = NFLScheduleFetcher(
                data_source=self._get_data_source(),
                season=self._season,
            )
        return self._schedule

    def _get_venues(self):
        """NFLVenueFactors: catálogo estático de los 32 estadios."""
        if self._venues is None:
            from sports.nfl.venue_factors import NFLVenueFactors
            self._venues = NFLVenueFactors(config_loader=self._config)
        return self._venues

    def _get_injuries(self):
        """NFLInjuryFetcher, compartido entre provider y modelo."""
        if self._injuries is None:
            from sports.nfl.injuries import NFLInjuryFetcher
            self._injuries = NFLInjuryFetcher(
                data_source   = self._get_data_source(),
                season        = self._season,
                config_loader = self._config,
            )
        return self._injuries

    def _get_rest(self):
        """NFLRestFetcher, compartido entre provider y modelo."""
        if self._rest is None:
            from sports.nfl.rest import NFLRestFetcher
            self._rest = NFLRestFetcher(
                schedule_fetcher=self._get_schedule(),
                config_loader=self._config,
            )
        return self._rest

    def _get_h2h(self):
        """NFLH2HFetcher, compartido entre provider y modelo."""
        if self._h2h is None:
            from sports.nfl.h2h import NFLH2HFetcher
            self._h2h = NFLH2HFetcher(
                schedule_fetcher=self._get_schedule(),
                config_loader=self._config,
            )
        return self._h2h

    @staticmethod
    def _try(factory):
        """
        Construye un componente devolviendo None si no es posible.

        Se reserva a los componentes que el consumidor declara
        opcionales. No se usa con el data provider ni con el
        settlement: esos SÍ necesitan acceso a datos, y enmascarar su
        fallo produciría un pipeline que se ejecuta sin obtener nada y
        reporta cero picks como si no hubiera valor, en vez de señalar
        que falta una dependencia.
        """
        try:
            return factory()
        except Exception:
            return None

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """
        True si las dependencias del plugin están instaladas.

        Permite a run_daily.py comprobar la disponibilidad antes de
        construir el plugin y emitir un mensaje con la instrucción de
        instalación, en vez de dejar escapar una traza de ImportError
        que no dice qué hacer.
        """
        try:
            from sports.nfl.data_source import is_available
            return is_available()
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"NFLPlugin(season={self._season}, "
            f"props={self._include_props}, periods={self._include_periods})"
        )