"""
sports/mlb/plugin.py

MLBPlugin: punto de entrada del plugin MLB para el PipelineRunner.

Implementa core/pipeline/stage.py:SportPlugin.

El PipelineRunner solo interactúa con esta clase — nunca importa
directamente desde sports/mlb/provider.py, projections.py, etc.
MLBPlugin provee todos los componentes via factory methods.

Ensamblaje
-----------
MLBPlugin
    ├── get_data_provider()      → MLBDataProvider (8.13)
    ├── get_projection_model()   → MLBProjectionModel (8.10)
    ├── get_probability_model()  → DistributionFactory → PoissonModel (2.2)
    ├── get_settlement_provider()→ MLBSettlementProvider (8.11)
    ├── get_market_definitions() → MLBMarketDefinitions (8.12)
    └── get_config()             → dict desde mlb.yaml

Uso típico
-----------
    from sports.mlb.plugin import MLBPlugin
    from core.pipeline.runner import build_runner
    from core.utils.config_loader import load_config

    config = load_config(sport='mlb')
    plugin = MLBPlugin(config_loader=config)
    runner = build_runner(plugin=plugin, config_loader=config)
    result = runner.run(date='2026-07-15')
"""

from __future__ import annotations


class MLBPlugin:
    """
    Punto de entrada del plugin MLB.

    Implementa SportPlugin Protocol. El PipelineRunner obtiene todos
    los componentes de este objeto — nunca instancia módulos MLB
    directamente.

    Parámetros
    ----------
    config_loader  -- ConfigLoader con base.yaml + mlb.yaml merged.
                     Si None, usa defaults hardcodeados de cada módulo.
    season         -- Temporada MLB. Default: año actual.
    include_props  -- Si True, incluye prop markets en el catálogo.
                     Default True.
    """

    sport_id:  str = "mlb"
    league_id: str = "MLB"

    def __init__(
        self,
        config_loader = None,
        season:        int | None = None,
        include_props: bool = True,
    ) -> None:
        self._config        = config_loader
        self._season        = season
        self._include_props = include_props

        # Componentes compartidos entre módulos (instancia única)
        # StatcastFetcher y VenueFactorProvider son costosos de crear
        # y usados por múltiples submódulos — se crean una sola vez.
        self._statcast_fetcher  = None
        self._venue_provider    = None
        self._data_provider     = None
        self._projection_model  = None
        self._probability_model = None
        self._settlement        = None
        self._market_defs       = None

    # ── SportPlugin Protocol ──────────────────────────────────────────────────

    def get_data_provider(self):
        """
        Retorna MLBDataProvider.

        Singleton por instancia — el mismo provider se reutiliza
        para todos los eventos del día (comparte caché de statcast).
        """
        if self._data_provider is None:
            from sports.mlb.provider import MLBDataProvider
            self._data_provider = MLBDataProvider(
                statcast_fetcher = self._get_statcast(),
                venue_provider   = self._get_venue(),
                season           = self._season,
                config_loader    = self._config,
            )
        return self._data_provider

    def get_projection_model(self):
        """
        Retorna MLBProjectionModel.

        Comparte StatcastFetcher y VenueFactorProvider con el DataProvider.
        """
        if self._projection_model is None:
            from sports.mlb.projections import MLBProjectionModel
            from sports.mlb.bullpen import BullpenFetcher
            from core.simulation.ensemble import EnsembleModel

            bullpen = BullpenFetcher(
                statcast_fetcher = self._get_statcast(),
                config_loader    = self._config,
            )
            ensemble = EnsembleModel(
                proj_min = self._cfg("ensemble.proj_min", 1.5),
                proj_max = self._cfg("ensemble.proj_max", 12.0),
            )
            self._projection_model = MLBProjectionModel(
                venue_provider  = self._get_venue(),
                bullpen_fetcher = bullpen,
                ensemble        = ensemble,
                config_loader   = self._config,
            )
        return self._projection_model

    def get_probability_model(self):
        """
        Retorna PoissonModel para MLB via DistributionFactory.

        El core resuelve PoissonModel para (sport='mlb', market='*').
        La factory lee simulation.mlb.poisson.max_score del config.
        """
        if self._probability_model is None:
            from core.simulation.factory import DistributionFactory
            factory = DistributionFactory(config=self._config)
            # Retornar el modelo para MLB (Poisson con max_score configurado)
            self._probability_model = factory.build("mlb", "ML")
        return self._probability_model

    def get_settlement_provider(self):
        """Retorna MLBSettlementProvider."""
        if self._settlement is None:
            from sports.mlb.settlement import MLBSettlementProvider
            self._settlement = MLBSettlementProvider()
        return self._settlement

    def get_market_definitions(self):
        """Retorna MLBMarketDefinitions."""
        if self._market_defs is None:
            from sports.mlb.markets import MLBMarketDefinitions
            self._market_defs = MLBMarketDefinitions(
                include_props=self._include_props,
            )
        return self._market_defs

    def get_config(self) -> dict:
        """
        Retorna la configuración MLB como dict.

        El runner puede pasarla a subsistemas del Core que no tienen
        acceso directo al ConfigLoader del deporte.
        """
        if self._config is None:
            return {}
        try:
            # Retornar las secciones relevantes del YAML como dict plano
            sections = [
                "blending", "kelly", "filters", "staking",
                "risk", "line_movement", "ensemble", "mlb",
            ]
            result = {}
            for section in sections:
                val = self._config.get(section, default=None)
                if val is not None:
                    result[section] = val
            return result
        except Exception:
            return {}

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _get_statcast(self):
        """StatcastFetcher singleton compartido entre módulos."""
        if self._statcast_fetcher is None:
            from sports.mlb.statcast import StatcastFetcher
            self._statcast_fetcher = StatcastFetcher(
                config_loader = self._config,
            )
        return self._statcast_fetcher

    def _get_venue(self):
        """VenueFactorProvider singleton compartido entre módulos."""
        if self._venue_provider is None:
            from sports.mlb.venue_factors import VenueFactorProvider
            self._venue_provider = VenueFactorProvider(
                config_loader=self._config
            )
        return self._venue_provider

    def _cfg(self, key: str, default):
        """Lee un valor del ConfigLoader con fallback a default."""
        if self._config is None:
            return default
        try:
            val = self._config.get(key, default=default)
            return val if val is not None else default
        except Exception:
            return default

    def __repr__(self) -> str:
        return (
            f"MLBPlugin(season={self._season}, "
            f"include_props={self._include_props})"
        )