"""
sports/mlb/plugin.py

MLBPlugin: punto de entrada del plugin MLB para el PipelineRunner.

Implementa core/pipeline/stage.py:SportPlugin.

FIX B2: get_probability_model() usaba factory.build() que no existe.
Corregido a factory.get_model(sport, market) — método real de
DistributionFactory (core/simulation/factory.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Solo para anotaciones: el plugin usa imports diferidos en cada
    # factory method y este bloque no se ejecuta en runtime.
    from core.pipeline.stage import (
        MarketDefinitions,
        ProbabilityModel,
        ProjectionModel,
        SettlementProvider,
        SportDataProvider,
    )


class MLBPlugin:
    """
    Punto de entrada del plugin MLB.

    Implementa SportPlugin Protocol. El PipelineRunner obtiene todos
    los componentes de este objeto — nunca importa desde sports/mlb/*
    directamente.

    Parámetros
    ----------
    config_loader  -- ConfigLoader con base.yaml + mlb.yaml merged.
    season         -- Temporada MLB. Default: año actual.
    include_props  -- Si True, incluye prop markets. Default True.
    """

    sport_id:          str = "mlb"
    league_id:         str = "MLB"
    odds_api_sport_id: str = "baseball_mlb"  # identificador en The Odds API

    def __init__(
        self,
        config_loader = None,
        season:        int | None = None,
        include_props: bool = True,
    ) -> None:
        self._config        = config_loader
        self._season        = season
        self._include_props = include_props

        # Singletons — instanciados lazy en los factory methods
        self._statcast_fetcher  = None
        self._venue_provider    = None
        # Anotados con los PROTOCOLS del Core, no con las clases
        # concretas del plugin. Sin la anotación, Pyright infiere el
        # tipo de las asignaciones —MLBDataProvider | None— y rechaza
        # cualquier otra implementación aunque cumpla el contrato:
        # dobles de prueba, un provider con caché offline o un backend
        # alternativo.
        #
        # Lo que el PipelineRunner necesita es el Protocol; es también
        # lo único que este plugin garantiza.
        self._data_provider:     SportDataProvider | None  = None
        self._projection_model:  ProjectionModel | None    = None
        self._probability_model: ProbabilityModel | None   = None
        self._settlement:        SettlementProvider | None = None
        self._market_defs:       MarketDefinitions | None  = None

    # ── SportPlugin Protocol ──────────────────────────────────────────────────

    def get_data_provider(self):
        """Retorna MLBDataProvider (singleton por instancia)."""
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
        """Retorna MLBProjectionModel (singleton por instancia)."""
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

        FIX B2: usa factory.get_model() — no factory.build() que no existe.
        """
        if self._probability_model is None:
            from core.simulation.factory import DistributionFactory
            factory = DistributionFactory(config=self._config)
            # get_model(sport, market) — firma correcta del factory
            self._probability_model = factory.get_model("mlb", "ML")
        return self._probability_model

    def get_settlement_provider(self):
        """Retorna MLBSettlementProvider (singleton)."""
        if self._settlement is None:
            from sports.mlb.settlement import MLBSettlementProvider
            self._settlement = MLBSettlementProvider()
        return self._settlement

    def get_market_definitions(self):
        """Retorna MLBMarketDefinitions (singleton)."""
        if self._market_defs is None:
            from sports.mlb.markets import MLBMarketDefinitions
            self._market_defs = MLBMarketDefinitions(
                include_props=self._include_props,
            )
        return self._market_defs

    def get_odds_matcher(self):
        """
        Empareja un Event con su RawOddsEvent por fecha y equipos.

        Por qué hace falta
        -------------------
        El pipeline empareja cada evento con sus cuotas usando
        `provider_ids['odds_api']`, que debe contener el id del evento
        EN THE ODDS API.

        MLBDataProvider lo deja vacío con el comentario "se mapea
        externamente", pero ese paso nunca llegó a existir. El runner
        hace `if not odds_api_id: continue` y salta TODOS los eventos
        sin registrar un solo error — el pipeline llega a Stage 6 con
        cero candidatos y el informe solo dice "sin picks".

        Es el mismo caso que en fútbol, donde football-data y The Odds
        API son proveedores sin relación: la MLB Stats API tampoco
        conoce los identificadores de la casa de apuestas.

        Por qué basta con los nombres
        ------------------------------
        Ambas fuentes usan el nombre oficial completo —"Washington
        Nationals", no "Nationals" ni "WSH"— así que la normalización
        resuelve la mayoría de casos sin tabla de alias.

        No se usa coincidencia difusa por el mismo motivo que en
        fútbol: emparejar un partido con las cuotas de otro daría
        precios plausibles del rival equivocado, y nada en el sistema
        lo detectaría. Ante la duda, None.
        """

        def matcher(event, raw_events):
            if not raw_events:
                return None

            home = _normalize_mlb_team(event.home_team)
            away = _normalize_mlb_team(event.away_team)
            if not home or not away:
                return None

            date = (event.date or "")[:10]

            for raw in raw_events:
                raw_home = _normalize_mlb_team(getattr(raw, "home_team", ""))
                raw_away = _normalize_mlb_team(getattr(raw, "away_team", ""))
                if raw_home != home or raw_away != away:
                    continue

                # La fecha de The Odds API viene en UTC. Un partido
                # nocturno en la costa oeste —19:00 en Los Ángeles— es
                # las 02:00 UTC del día siguiente, así que se acepta
                # ±1 día.
                #
                # Ampliarlo más arriesgaría cruzar partidos de una
                # serie: dos equipos juegan tres o cuatro días
                # seguidos, y a dos días de distancia ya sería otro
                # encuentro.
                raw_date = str(getattr(raw, "commence_time", ""))[:10]
                if not raw_date or _within_a_day(date, raw_date):
                    return raw

            return None

        return matcher

    @staticmethod
    def is_available() -> bool:
        """
        True si el plugin puede operar.

        MLB consulta la MLB Stats API por HTTP directo, así que solo
        necesita `requests` —que ya es requisito del Core para The Odds
        API. No hay dependencias adicionales que instalar, al contrario
        que NFL, que exige nfl_data_py.

        Por qué existe este método
        ---------------------------
        Lo declaran SoccerPlugin y NFLPlugin, y el CLI lo consulta
        antes de construir el pipeline para dar un mensaje útil en vez
        de fallar al descargar datos.

        MLB no lo tenía, así que cualquier consumidor que lo llamara de
        forma uniforme sobre los tres plugins obtenía un AttributeError
        en vez de una respuesta. La verificación de entorno lo destapó
        al recorrerlos en bucle.

        Que un plugin no necesite dependencias externas no es razón
        para omitir el método: la interfaz debe ser la misma aunque la
        respuesta sea siempre True.
        """
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def get_config(self) -> dict:
        """Retorna configuración MLB como dict para subsistemas del Core."""
        if self._config is None:
            return {}
        try:
            result = {}
            for section in ("blending", "kelly", "filters", "staking",
                            "risk", "line_movement", "ensemble", "mlb"):
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
                config_loader=self._config,
            )
        return self._statcast_fetcher

    def _get_venue(self):
        """VenueFactorProvider singleton compartido entre módulos."""
        if self._venue_provider is None:
            from sports.mlb.venue_factors import VenueFactorProvider
            self._venue_provider = VenueFactorProvider(
                config_loader=self._config,
            )
        return self._venue_provider

    def _cfg(self, key: str, default):
        """Lee valor del ConfigLoader con fallback a default."""
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


# ── Utilidades ───────────────────────────────────────────────────────────────

def _normalize_mlb_team(name: str) -> str:
    """
    Normaliza un nombre de equipo de MLB para comparar entre fuentes.

    La MLB Stats API y The Odds API usan ambas el nombre oficial
    completo, así que basta con minúsculas, sin acentos y espacios
    colapsados. Los alias que necesita el fútbol —'Man City' frente a
    'Manchester City'— no tienen equivalente aquí.

    Se mantiene la ciudad en el nombre a propósito: hay dos equipos en
    Nueva York, dos en Chicago y dos en Los Ángeles. Recortar a
    'Yankees' o 'Mets' funcionaría, pero 'Sox' no distinguiría Boston
    de Chicago.
    """
    import unicodedata

    if not name:
        return ""

    decomposed = unicodedata.normalize("NFD", str(name))
    stripped = "".join(ch for ch in decomposed
                       if unicodedata.category(ch) != "Mn")
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " "
                      for ch in stripped)
    return " ".join(cleaned.split())


def _within_a_day(date_a: str, date_b: str) -> bool:
    """
    True si dos fechas ISO distan como mucho un día.

    Las cuotas de The Odds API llevan `commence_time` en UTC, y un
    partido nocturno en la costa oeste cae en el día siguiente.
    """
    from datetime import datetime

    try:
        a = datetime.strptime(date_a[:10], "%Y-%m-%d")
        b = datetime.strptime(date_b[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    return abs((a - b).days) <= 1