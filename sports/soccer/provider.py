"""
sports/soccer/provider.py

SoccerDataProvider: orquesta los fetchers del plugin de fútbol.
Implementa core/pipeline/stage.py:SportDataProvider.

Composición
------------
    SoccerScheduleFetcher    (11.4)  calendario multicompetición
    SoccerTeamStatsFetcher   (11.5)  índices de ataque y defensa
    SoccerCongestionFetcher  (11.6)  carga de calendario
    SoccerH2HFetcher         (11.7)  historial y derbis
    SoccerContextFetcher     (11.8)  clasificación y contexto

La barrera temporal
---------------------
En NFL la barrera era la SEMANA de corte: para proyectar la jornada 10
el modelo solo veía hasta la 9.

Aquí es la FECHA del partido, y con comparación estricta: para proyectar
un encuentro del 15 de marzo, el modelo solo ve lo jugado ANTES de ese
día. La diferencia con `<=` importa más de lo que parece — las cinco
ligas juegan sábado y domingo, así que un partido del mismo día es o
bien el que se está proyectando, o bien uno simultáneo cuyo resultado
tampoco se conoce al apostar.

`_as_of()` resuelve esa fecha en un único punto. El modo de fallo no
lanza excepción ni deja campos vacíos: produce un backtest inflado que
no se reproduce en producción, y el diagnóstico llega meses después
cuando el ledger real diverge del histórico simulado.

La competición viaja en el evento
-----------------------------------
A diferencia de MLB y NFL, aquí cada evento pertenece a una de cinco
competiciones, con sus propias medias de goles y su propio catálogo de
equipos. El provider la extrae de `provider_ids['competition']`, que
SoccerScheduleFetcher rellena al construir el Event.

Sin esa resolución, los índices se calcularían contra la media de liga
equivocada: la Bundesliga promedia 3.10 goles por partido y La Liga
2.53, así que un equipo medio de una parecería flojo en la otra.

Degradación por capas
-----------------------
Cada fetcher es independiente. Si Understat cae, los partidos y las
cuotas siguen llegando y el plugin opera en tier PARTIAL con goles y
forma. `data_quality` lo refleja, el modelo baja su confianza y los
filtros de EV hacen el resto.

Ese encadenamiento es deliberado: la respuesta correcta a "no tengo el
xG de este equipo" es apostar menos, no apostar a ciegas.
"""

from __future__ import annotations

from core.contracts.event import Event
from core.contracts.features import TeamFeatures

from sports.soccer.competitions import (
    Competition, TIER_FULL, TIER_MINIMAL, TIER_PARTIAL,
    enabled_competitions, get_competition,
)
from sports.soccer.data_source import current_soccer_season
from sports.soccer.h2h import h2h_metadata
from sports.soccer.schedule import SoccerMatchInfo
from sports.soccer.team_stats import SoccerTeamStats


__all__ = ["SoccerDataProvider"]


# Media de goles por equipo cuando no hay datos de la competición.
_DEFAULT_GOALS = 1.40

# ── Pesos de calidad de datos ────────────────────────────────────────────────
#
# Reflejan cuánto aporta cada fuente a la calidad de la proyección, no
# cuántos campos rellena. Las estadísticas de equipo pesan más que todo
# lo demás junto porque el xG es la señal principal: su correlación con
# resultados futuros ronda 0.60, frente al ~0.40 de los goles.
_DQ_TEAM_STATS = 0.50
_DQ_SCHEDULE   = 0.15
_DQ_CONGESTION = 0.15
_DQ_H2H        = 0.05
_DQ_CONTEXT    = 0.15


class SoccerDataProvider:
    """
    Proveedor de datos de fútbol. Implementa SportDataProvider.

    Parámetros
    ----------
    data_source   -- SoccerDataSource compartido. Si None, crea uno.
    competitions  -- Competiciones a cubrir. None usa las activas del
                     registro.
    season        -- Temporada. None la deduce de cada fecha.
    config_loader -- ConfigLoader con soccer.yaml, propagado a los
                     fetchers que leen parámetros calibrados.

    Los fetchers se pueden inyectar ya construidos. Si se omiten, el
    provider los crea compartiendo la misma fuente y el mismo
    calendario — sin eso, cada uno descargaría su propia copia de los
    partidos de las cinco ligas.
    """

    def __init__(
        self,
        data_source          = None,
        competitions:  list[Competition] | None = None,
        season:        int | None = None,
        config_loader        = None,
        schedule_fetcher     = None,
        team_stats_fetcher   = None,
        congestion_fetcher   = None,
        h2h_fetcher          = None,
        context_fetcher      = None,
    ) -> None:
        self._competitions = (competitions if competitions is not None
                              else enabled_competitions())
        self._season = season
        self._config = config_loader

        # SoccerDataSource solo se crea si hace falta: si el calendario
        # viene inyectado, nadie más lo necesita. Eso permite usar el
        # provider con un backend alternativo o con dobles de prueba
        # sin tocar la red — el mismo criterio que en NFLDataProvider.
        self._source = data_source
        if self._source is None and schedule_fetcher is None:
            from sports.soccer.data_source import SoccerDataSource
            self._source = SoccerDataSource()

        if schedule_fetcher is None:
            from sports.soccer.schedule import SoccerScheduleFetcher
            schedule_fetcher = SoccerScheduleFetcher(
                data_source=self._source,
                competitions=self._competitions,
                season=season,
            )
        self._schedule = schedule_fetcher

        if team_stats_fetcher is None:
            from sports.soccer.team_stats import SoccerTeamStatsFetcher
            team_stats_fetcher = SoccerTeamStatsFetcher(
                schedule_fetcher=self._schedule, config_loader=config_loader,
            )
        self._team_stats = team_stats_fetcher

        if congestion_fetcher is None:
            from sports.soccer.congestion import SoccerCongestionFetcher
            congestion_fetcher = SoccerCongestionFetcher(
                schedule_fetcher=self._schedule,
                config_loader=config_loader,
                tracked_competitions=[c.comp_id for c in self._competitions],
            )
        self._congestion = congestion_fetcher

        if h2h_fetcher is None:
            from sports.soccer.h2h import SoccerH2HFetcher
            h2h_fetcher = SoccerH2HFetcher(
                schedule_fetcher=self._schedule, config_loader=config_loader,
            )
        self._h2h = h2h_fetcher

        if context_fetcher is None:
            from sports.soccer.context import SoccerContextFetcher
            context_fetcher = SoccerContextFetcher(
                schedule_fetcher=self._schedule,
                congestion_fetcher=self._congestion,
                h2h_fetcher=self._h2h,
            )
        self._context = context_fetcher

    # ── SportDataProvider Protocol ────────────────────────────────────────────

    def get_events(self, date: str) -> list[Event]:
        """
        Partidos de la fecha en todas las competiciones activas.

        Un sábado de temporada puede devolver veinte partidos
        repartidos entre las cinco ligas.
        """
        return self._safe(lambda: self._schedule.get_events(date), default=[]) or []

    def enrich_event(self, event: Event) -> tuple[TeamFeatures, TeamFeatures]:
        """
        Construye las TeamFeatures de ambos equipos.

        Nunca lanza. Ante el fallo de cualquier fuente, ese bloque de
        datos queda ausente, `data_quality` baja y el resto del
        pipeline sigue. Un partido con datos parciales produce una
        proyección de baja confianza que los filtros probablemente
        rechacen — que es la respuesta correcta.
        """
        comp = self._competition_of(event)
        match = self._match_of(event)
        season = self._season_of(event, match)
        as_of = self._as_of(event, match)

        home_id = event.home_team_id
        away_id = event.away_team_id

        # ── Estadísticas, con la barrera temporal ──────────────────
        home_stats = self._safe(
            lambda: self._team_stats.fetch(home_id, comp, season, as_of)
        ) if comp else None
        away_stats = self._safe(
            lambda: self._team_stats.fetch(away_id, comp, season, as_of)
        ) if comp else None

        # ── Congestión ─────────────────────────────────────────────
        home_cong = self._safe(
            lambda: self._congestion.fetch(home_id, comp, season, as_of)
        ) if comp else None
        away_cong = self._safe(
            lambda: self._congestion.fetch(away_id, comp, season, as_of)
        ) if comp else None

        home_penalty = self._penalty(home_id, comp, season, as_of)
        away_penalty = self._penalty(away_id, comp, season, as_of)

        # ── H2H ────────────────────────────────────────────────────
        h2h = self._safe(
            lambda: self._h2h.get_h2h(home_id, away_id, comp, season, as_of)
        ) if comp else None

        # ── Medias de liga, compartidas por ambos equipos ──────────
        league = (home_stats.league if home_stats and home_stats.league
                  else (away_stats.league if away_stats else None))

        return (
            self._build(home_id, event.home_team, home_stats, home_cong,
                        home_penalty, h2h, match, comp, league, is_home=True),
            self._build(away_id, event.away_team, away_stats, away_cong,
                        away_penalty, h2h, match, comp, league, is_home=False),
        )

    def get_context(self, event: Event) -> dict:
        """
        Contexto situacional, enriquecido con lo que el modelo necesita.

        SoccerContextFetcher aporta clasificación, jornada, derbi y
        congestión. El provider añade las medias de liga del corte
        temporal, que el modelo de proyección usa como base de la
        fórmula de Maher.

        Calcularlas aquí y no en el modelo evita que este necesite
        conocer la competición y la temporada, que no recibe.
        """
        context = self._safe(lambda: dict(self._context.get_context(event)),
                             default={}) or {}

        comp = self._competition_of(event)
        match = self._match_of(event)
        if comp is None:
            return context

        season = self._season_of(event, match)
        as_of = self._as_of(event, match)

        league = self._safe(
            lambda: self._team_stats.league_averages(comp, season, as_of)
        )

        context.setdefault("match_id", event.event_id)
        context["competition"] = comp.comp_id
        context["season"] = season

        if league is not None:
            # Una media no positiva no describe ninguna liga; en ese
            # caso se usan las históricas del catálogo, que es la
            # referencia correcta al arranque de temporada.
            context["league_home_goals"] = (
                league.home_goals if league.home_goals > 0 else comp.avg_home_goals
            )
            context["league_away_goals"] = (
                league.away_goals if league.away_goals > 0 else comp.avg_away_goals
            )
            context["league_matches"] = league.matches
        else:
            context["league_home_goals"] = comp.avg_home_goals
            context["league_away_goals"] = comp.avg_away_goals

        return context

    # ── La barrera temporal ───────────────────────────────────────────────────

    @staticmethod
    def _as_of(event: Event, match: SoccerMatchInfo | None) -> str:
        """
        Fecha de corte: el modelo solo ve lo jugado ANTES.

        Se devuelve la fecha del propio partido, y los fetchers aplican
        comparación ESTRICTA contra ella. Un encuentro del mismo día es
        o bien el que se está proyectando, o bien uno simultáneo cuyo
        resultado tampoco se conoce al apostar — las cinco ligas juegan
        sábado y domingo, así que el caso es habitual, no excepcional.

        Por qué esto tiene su propio método
        ------------------------------------
        Es la única defensa contra el look-ahead bias, y su modo de
        fallo es silencioso: no lanza excepción, no deja campos vacíos,
        no aparece en ningún log. Produce un backtest con rendimiento
        inflado que no se reproduce en producción.

        Un `event.date` suelto en medio del ensamblaje sería
        indistinguible de un acceso cualquiera. Aquí queda explícito
        qué es y por qué importa.
        """
        if match is not None and match.date:
            return match.date
        return event.date or ""

    # ── Resolución de competición y temporada ─────────────────────────────────

    def _competition_of(self, event: Event) -> Competition | None:
        """
        Competición del evento, desde provider_ids.

        SoccerScheduleFetcher la rellena al construir el Event. Sin
        ella los índices se calcularían contra la media de liga
        equivocada: la Bundesliga promedia 3.10 goles por partido y La
        Liga 2.53, así que un equipo medio de una parecería flojo en la
        otra.
        """
        provider_ids = event.provider_ids or {}
        comp_id = provider_ids.get("competition", "")

        if not comp_id and event.event_id:
            # Respaldo: el comp_id es el primer segmento del match_id
            comp_id = event.event_id.split("_", 1)[0]

        comp = get_competition(comp_id)
        if comp is not None:
            return comp

        # Último respaldo: si solo hay una competición activa, es esa
        return self._competitions[0] if len(self._competitions) == 1 else None

    def _season_of(self, event: Event, match: SoccerMatchInfo | None) -> int:
        if self._season is not None:
            return self._season
        if match is not None and match.season:
            return match.season
        if event.season_start:
            return event.season_start
        return current_soccer_season()

    def _match_of(self, event: Event) -> SoccerMatchInfo | None:
        result = self._safe(lambda: self._schedule.get_match_info(event.event_id))
        return result if isinstance(result, SoccerMatchInfo) else None

    def _penalty(self, team: str, comp, season: int, as_of: str) -> float:
        """Penalización de congestión en goles, con los valores del config."""
        if comp is None:
            return 0.0
        value = self._safe(
            lambda: self._congestion.penalty_for(team, comp, season, as_of),
            default=0.0,
        )
        return float(value) if value is not None else 0.0

    # ── Ensamblaje de TeamFeatures ────────────────────────────────────────────

    def _build(
        self,
        team_id:   str,
        team_name: str,
        stats:     SoccerTeamStats | None,
        congestion,
        penalty:   float,
        h2h,
        match:     SoccerMatchInfo | None,
        comp:      Competition | None,
        league,
        is_home:   bool,
    ) -> TeamFeatures:
        """Construye las TeamFeatures de un equipo desde las fuentes."""

        # Índices normalizados. Sin estadísticas, 1.0 = media de liga:
        # el supuesto neutro correcto, no una degradación silenciosa.
        attack = stats.attack_index() if stats else 1.0
        defense = stats.defense_index() if stats else 1.0

        expected = self._expected_goals(stats, comp, league, is_home)
        recent = list(stats.recent_goals) if stats else []

        # ── sport_metadata ─────────────────────────────────────────
        metadata: dict = {}
        if stats is not None:
            metadata.update(stats.to_metadata())
        if congestion is not None:
            metadata.update(congestion.to_metadata())
        metadata["congestion_penalty"] = round(float(penalty), 4)
        if h2h is not None:
            metadata.update(h2h_metadata(h2h))
        if match is not None:
            metadata.update(match.to_metadata())

        metadata["is_home"] = is_home
        metadata["venue_team"] = match.home if match else ""
        metadata["competition"] = comp.comp_id if comp else ""
        metadata["xg_available"] = bool(stats.xg_available) if stats else False

        # Índices por localía, que el modelo puede preferir al general
        if stats is not None:
            metadata["attack_index_venue"] = stats.attack_index_at(is_home)
            metadata["defense_index_venue"] = stats.defense_index_at(is_home)

        # Medias de liga separadas por localía, base de la fórmula
        if league is not None:
            metadata["league_home_goals"] = league.home_goals
            metadata["league_away_goals"] = league.away_goals

        quality, missing = self._assess(stats, congestion, h2h, match, comp)

        return TeamFeatures(
            team_id=team_id,
            team_name=team_name or team_id,
            expected_score=round(float(expected), 3),
            offense_index=attack,
            defense_index=defense,
            recent_scores=[float(g) for g in recent],
            recent_avg=(round(sum(recent) / len(recent), 3) if recent else 0.0),
            recent_n=len(recent),
            # TeamFeatures no tiene campo de sede: en MLB y NFL el
            # estadio importa por sus dimensiones y su clima, y allí lo
            # captura venue_factor. En fútbol el terreno es
            # reglamentariamente uniforme y el efecto de la localía ya
            # está en las medias de liga separadas por casa y fuera.
            #
            # Se deja en 1.0 neutro y la sede viaja en sport_metadata,
            # donde el modelo la consulta si la necesita.
            venue_factor=1.0,
            sample_size=stats.matches if stats else 0,
            data_quality=quality,
            missing_fields=missing,
            sport_metadata=metadata,
        )

    @staticmethod
    def _expected_goals(
        stats:   SoccerTeamStats | None,
        comp:    Competition | None,
        league,
        is_home: bool,
    ) -> float:
        """
        Goles esperados de referencia para el contrato del Core.

        No es la proyección del partido —esa la calcula el modelo con
        la fórmula de Maher— sino una estimación de contexto para los
        stages que la consultan antes de proyectar.

        Se prefiere la media del propio equipo; en su defecto, la de la
        competición separada por localía, que ya incluye la ventaja de
        campo.
        """
        if stats is not None and stats.goals_for > 0:
            return stats.goals_for

        if league is not None:
            value = league.home_goals if is_home else league.away_goals
            if value > 0:
                return value

        if comp is not None:
            return comp.avg_home_goals if is_home else comp.avg_away_goals

        return _DEFAULT_GOALS

    @staticmethod
    def _assess(stats, congestion, h2h, match, comp) -> tuple[float, list[str]]:
        """
        Calidad de datos como suma ponderada de las fuentes presentes.

        El tier de la competición acota el máximo alcanzable: una
        competición sin xG no puede llegar a 1.0 por mucho que todas
        sus fuentes respondan, porque le falta la señal principal.

        Sobre el xG y la congestión
        ----------------------------
        Se comprueba `stats is not None`, no que tenga xG: un equipo
        con estadísticas pero sin xG SÍ tiene datos —goles, forma,
        resultados— y penalizarlo como si no tuviera nada sería
        confundir "señal más débil" con "sin señal". La ausencia de xG
        se refleja por separado, en `xg_available`, que el modelo usa
        para bajar su confianza.

        La congestión incompleta penaliza a medias: ver solo los
        partidos de liga es peor que verlo todo, pero mejor que no ver
        nada.
        """
        quality = 0.0
        missing: list[str] = []

        if stats is not None and stats.matches > 0:
            weight = _DQ_TEAM_STATS
            if not stats.xg_available:
                # Sin xG la fuente está pero informa menos: la
                # correlación con resultados futuros baja de ~0.60 a
                # ~0.40.
                weight *= 0.60
            if not stats.has_sufficient_sample:
                weight *= 0.65
            quality += weight
        else:
            missing.append("team_stats")

        if match is not None:
            quality += _DQ_SCHEDULE
        else:
            missing.append("schedule")

        if congestion is not None:
            weight = _DQ_CONGESTION
            if congestion.coverage_warning:
                weight *= 0.60
            quality += weight
        else:
            missing.append("congestion")

        if h2h is not None and h2h.is_reliable:
            quality += _DQ_H2H
        else:
            missing.append("h2h")

        quality += _DQ_CONTEXT   # la clasificación se deriva del calendario

        # El tier de la competición acota el máximo
        if comp is not None:
            quality = min(quality, comp.data_quality_base)

        return round(max(0.0, min(1.0, quality)), 3), missing

    # ── Utilidad ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(fn, default=None):
        """
        Ejecuta fn() devolviendo `default` si lanza.

        Cada fetcher es independiente: el fallo de uno no debe impedir
        que el resto aporte sus datos. Un partido con Understat caído
        pero cuotas y resultados intactos sigue siendo proyectable, con
        la incertidumbre declarada en data_quality.
        """
        try:
            return fn()
        except Exception:
            return default