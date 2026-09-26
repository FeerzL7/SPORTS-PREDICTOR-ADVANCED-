"""
sports/soccer/plugin.py

SoccerPlugin: punto de entrada del deporte.
Implementa core/pipeline/stage.py:SportPlugin.

Dos problemas que MLB y NFL no tenían
---------------------------------------

1. CINCO CLAVES DE THE ODDS API, NO UNA
   MLB declara `odds_api_sport_id = "baseball_mlb"` y NFL
   `"americanfootball_nfl"`. Una clave, una request, todos los partidos
   del día.

   Fútbol tiene una clave POR COMPETICIÓN: soccer_epl,
   soccer_spain_la_liga, soccer_italy_serie_a... Una sola request no
   cubriría más que una liga.

   `odds_api_sport_ids` (plural) expone la lista. `odds_api_sport_id`
   (singular) se conserva devolviendo la primera, para que un
   consumidor que solo conozca el atributo antiguo obtenga algo
   utilizable en vez de fallar — pero cubriendo solo una liga, así que
   el runner debe usar el plural.

2. THE ODDS API NO DA SUS IDS EN NUESTRA FUENTE
   El pipeline empareja cada Event con su RawOddsEvent por
   `provider_ids['odds_api']`, que debe contener el id del evento EN
   THE ODDS API.

   football-data.co.uk no publica ese id —ni tiene por qué: son
   proveedores sin relación— así que el plugin no puede rellenarlo al
   construir el Event en Stage 1, antes de que las cuotas existan.

   `get_odds_matcher()` devuelve una función que empareja por FECHA y
   NOMBRES CANÓNICOS. La reconciliación de teams.py hace el trabajo:
   'Man City' de football-data y 'Manchester City' de The Odds API
   convergen al mismo nombre canónico.

   Sin ese emparejamiento, ningún partido de fútbol recibiría cuotas y
   el pipeline produciría cero picks — el mismo modo de fallo
   silencioso que dio cero picks en el primer backtest de NFL.

Instancias compartidas
------------------------
Un SoccerDataSource, un SoccerScheduleFetcher, un juego de fetchers.
Sin esto, cada componente descargaría su propia copia de los partidos
de las cinco ligas: cinco CSV por temporada multiplicados por cada
consumidor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Solo para anotaciones. Los imports reales son diferidos dentro de
    # cada factory method, de forma que cargar el plugin no arrastre la
    # cadena completa de fetchers.
    from core.pipeline.stage import (
        MarketDefinitions, ProbabilityModel, ProjectionModel,
        SettlementProvider, SportDataProvider,
    )
    from sports.soccer.competitions import Competition


__all__ = ["SoccerPlugin"]


class SoccerPlugin:
    """
    Plugin de fútbol para las cinco grandes ligas europeas.

    Parámetros
    ----------
    config_loader -- ConfigLoader con soccer.yaml.
    season        -- Temporada por su año de inicio (2024 = 2024/25).
                     None la deduce de la fecha.
    competitions  -- Competiciones a cubrir. None usa las activas del
                     registro, que es lo que hace el pipeline.
    """

    sport_id:  str = "soccer"
    league_id: str = "SOCCER"

    def __init__(
        self,
        config_loader = None,
        season: int | None = None,
        competitions: list["Competition"] | None = None,
    ) -> None:
        self._config = config_loader
        self._season = season

        from sports.soccer.competitions import enabled_competitions
        self._competitions = (competitions if competitions is not None
                              else enabled_competitions())

        # Instancias memorizadas. Anotadas con los PROTOCOLS del Core,
        # no con las clases concretas: lo que el runner necesita es el
        # contrato, y es lo único que este plugin garantiza.
        self._data_source = None
        self._schedule = None
        self._data_provider:     "SportDataProvider | None"  = None
        self._projection_model:  "ProjectionModel | None"    = None
        self._probability_model: "ProbabilityModel | None"   = None
        self._settlement:        "SettlementProvider | None" = None
        self._market_defs:       "MarketDefinitions | None"  = None

    # ── Identificadores de The Odds API ───────────────────────────────────────

    @property
    def odds_api_sport_ids(self) -> list[str]:
        """
        Claves de The Odds API de las competiciones activas.

        Cinco en la configuración por defecto. El runner debe iterar
        sobre ellas: una sola request no cubriría más que una liga.
        """
        return [c.odds_api_key for c in self._competitions if c.odds_api_key]

    @property
    def odds_api_sport_id(self) -> str:
        """
        Primera clave, para compatibilidad con consumidores antiguos.

        El runner lee este atributo con getattr y cae al sport_id si no
        existe. Devolver la primera clave hace que un consumidor que no
        conozca el plural obtenga algo utilizable —cuotas de una liga—
        en vez de pedir 'soccer', que no es una clave válida de la API.

        Es una degradación parcial declarada, no una solución: quien
        use el singular cubre el 20% de los partidos.
        """
        keys = self.odds_api_sport_ids
        return keys[0] if keys else self.sport_id

    def competition_for_odds_key(self, key: str):
        """Competición de una clave de The Odds API."""
        from sports.soccer.competitions import by_odds_api_key
        return by_odds_api_key(key)

    # ── Emparejamiento de eventos con cuotas ──────────────────────────────────

    def get_odds_matcher(self):
        """
        Función que empareja un Event con su RawOddsEvent.

        Firma: matcher(event, raw_events) -> RawOddsEvent | None

        Por qué el plugin la provee
        ----------------------------
        El normalizador del Core empareja por `provider_ids['odds_api']`,
        que debe traer el id del evento en The Odds API. MLB y NFL lo
        obtienen porque sus fuentes de calendario lo publican o porque
        un paso previo lo resuelve.

        football-data.co.uk no lo da. El plugin no puede rellenarlo al
        construir el Event en Stage 1, porque las cuotas todavía no se
        han pedido.

        Lo que sí tenemos es fecha y equipos, y la reconciliación de
        teams.py hace que 'Man City' y 'Manchester City' converjan. Eso
        basta para emparejar de forma inequívoca: dos equipos no juegan
        dos veces el mismo día.

        La alternativa —coincidencia difusa por similitud de cadenas—
        se descartó por lo mismo que en teams.py: produce falsos
        positivos silenciosos, y emparejar un partido con las cuotas de
        otro sería peor que no emparejarlo.
        """
        from sports.soccer.teams import canonical_team

        def matcher(event, raw_events):
            if not raw_events:
                return None

            comp_id = (event.provider_ids or {}).get("competition", "")
            home = canonical_team(event.home_team_id, comp_id)
            away = canonical_team(event.away_team_id, comp_id)
            if not home or not away:
                return None

            date = (event.date or "")[:10]

            for raw in raw_events:
                raw_home = canonical_team(getattr(raw, "home_team", ""), comp_id)
                raw_away = canonical_team(getattr(raw, "away_team", ""), comp_id)
                if raw_home != home or raw_away != away:
                    continue

                # La fecha de The Odds API viene en UTC y puede caer un
                # día antes o después de la fecha local del partido —un
                # encuentro de las 21:00 en España es las 19:00 UTC del
                # mismo día, pero uno de las 00:30 ya es del siguiente.
                #
                # Se acepta ±1 día. Ampliarlo más arriesgaría cruzar dos
                # jornadas: en fútbol un mismo emparejamiento se repite
                # dos veces por temporada, pero nunca con dos días de
                # diferencia.
                raw_date = str(getattr(raw, "commence_time", ""))[:10]
                if not raw_date or _within_a_day(date, raw_date):
                    return raw

            return None

        return matcher

    # ── SportPlugin Protocol ──────────────────────────────────────────────────

    def get_data_provider(self) -> "SportDataProvider":
        if self._data_provider is None:
            from sports.soccer.provider import SoccerDataProvider
            self._data_provider = SoccerDataProvider(
                data_source=self._get_data_source(),
                competitions=self._competitions,
                season=self._season,
                config_loader=self._config,
                schedule_fetcher=self._get_schedule(),
            )
        return self._data_provider

    def get_projection_model(self) -> "ProjectionModel":
        if self._projection_model is None:
            from sports.soccer.projections import SoccerProjectionModel
            self._projection_model = SoccerProjectionModel(
                config_loader=self._config
            )
        return self._projection_model

    def get_probability_model(self) -> "ProbabilityModel":
        """
        Modelo de probabilidad: Poisson bivariado con Dixon-Coles.

        Se obtiene de la factory del Core en vez de instanciarlo
        directamente, para que un cambio de modelo por deporte sea
        configuración y no código.

        Los parámetros calibrados —rho, lambda_3— viajan en la
        proyección, no en el constructor: BivariatePoissonModel los lee
        de distribution_params desde la tarea 11.10. Antes de esa
        corrección usaba los suyos y las probabilidades del pick
        diferían de las que produjeron la proyección.
        """
        if self._probability_model is None:
            from core.simulation.factory import DistributionFactory
            self._probability_model = DistributionFactory().get_model(
                self.sport_id, "1X2"
            )
        return self._probability_model

    def get_settlement_provider(self) -> "SettlementProvider":
        if self._settlement is None:
            from sports.soccer.settlement import SoccerSettlementProvider
            self._settlement = SoccerSettlementProvider(
                schedule_fetcher=self._get_schedule()
            )
        return self._settlement

    def get_market_definitions(self) -> "MarketDefinitions":
        if self._market_defs is None:
            from sports.soccer.markets import SoccerMarketDefinitions
            self._market_defs = SoccerMarketDefinitions()
        return self._market_defs

    def get_config(self) -> dict:
        """
        Configuración como dict, para subsistemas del Core que no
        reciben el ConfigLoader directamente.

        La sección `simulation` se incluye además de las que usa NFL:
        ahí viven los parámetros del Poisson bivariado —rho, lambda_3,
        max_goals— que el modelo de proyección lee y deja en
        distribution_params para que la capa de probabilidad
        reconstruya exactamente la misma matriz.
        """
        if self._config is None:
            return {}

        result: dict = {}
        for section in ("blending", "kelly", "filters", "staking",
                        "risk", "line_movement", "ensemble",
                        "simulation", "soccer"):
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
        SoccerDataSource compartido, creado de forma diferida.

        Una sola instancia para todo el plugin: su caché de dos niveles
        pierde sentido si cada consumidor tiene la suya, y un backtest
        de diez temporadas descargaría cincuenta CSV por cada copia.
        """
        if self._data_source is None:
            from sports.soccer.data_source import SoccerDataSource
            self._data_source = SoccerDataSource(
                current_season=self._season
            )
        return self._data_source

    def _get_schedule(self):
        """
        SoccerScheduleFetcher compartido.

        Lo consumen team_stats, congestion, h2h, context y settlement.
        Compartirlo hace que el cruce de xG con los resultados —que
        ocurre una vez por competición y temporada— se aproveche en
        todos ellos.
        """
        if self._schedule is None:
            from sports.soccer.schedule import SoccerScheduleFetcher
            self._schedule = SoccerScheduleFetcher(
                data_source=self._get_data_source(),
                competitions=self._competitions,
                season=self._season,
            )
        return self._schedule

    # ── Disponibilidad ────────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """
        True si el plugin puede operar.

        A diferencia de NFL —que exige nfl_data_py— aquí basta
        `requests`, que el Core ya necesita para The Odds API. Los
        resultados y las cuotas históricas vienen de CSV planos y el xG
        del cliente propio de Understat.

        Esa decisión se tomó tras comprobar que `soccerdata` arrastraba
        pandas>=2.0 y rompía el plugin NFL, que exige pandas<2.0.
        """
        from sports.soccer.data_source import SoccerDataSource
        return SoccerDataSource.is_available()

    def describe(self) -> dict:
        """Resumen del plugin, para logs y diagnóstico."""
        from sports.soccer.data_source import current_soccer_season

        source = self._get_data_source()
        return {
            "sport":        self.sport_id,
            "league":       self.league_id,
            "season":       self._season or current_soccer_season(),
            "competitions": [c.comp_id for c in self._competitions],
            "odds_api_keys": self.odds_api_sport_ids,
            "n_competitions": len(self._competitions),
            "effective_tiers": {
                c.comp_id: source.effective_tier(c) for c in self._competitions
            },
            "has_xg_backend": source.has_xg_backend(),
        }


# ── Utilidades ───────────────────────────────────────────────────────────────

def _within_a_day(date_a: str, date_b: str) -> bool:
    """
    True si dos fechas ISO distan como mucho un día.

    Las cuotas de The Odds API llevan `commence_time` en UTC, que puede
    caer un día antes o después de la fecha local del partido. Un
    encuentro de las 21:00 en España es las 19:00 UTC del mismo día;
    uno de las 00:30, del siguiente.
    """
    from datetime import datetime

    try:
        a = datetime.strptime(date_a[:10], "%Y-%m-%d")
        b = datetime.strptime(date_b[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    return abs((a - b).days) <= 1