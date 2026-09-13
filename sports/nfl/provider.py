"""
sports/nfl/provider.py

NFLDataProvider: orquesta los fetchers del plugin NFL.
Implementa core/pipeline/stage.py:SportDataProvider.

Composición
------------
    NFLScheduleFetcher   (10.3)  → get_events, metadatos del partido
    NFLTeamStatsFetcher  (10.4)  → EPA, success rate, puntos
    NFLInjuryFetcher     (10.5)  → penalización por lesiones
    NFLRestFetcher       (10.6)  → diferencial de descanso
    NFLVenueFactors      (10.7)  → techo, altitud, viaje
    NFLContextFetcher    (10.8)  → clima y contexto situacional
    NFLH2HFetcher        (10.9)  → historial de enfrentamientos

Responsabilidad crítica: la semana de corte
---------------------------------------------
El provider de MLB solo ensambla datos. Este además decide QUÉ DATOS
puede ver el modelo, y ahí está el punto más delicado del plugin.

`NFLTeamStatsFetcher.fetch(team, up_to_week)` exige la semana de corte
sin valor por defecto. Para proyectar un partido de la semana 10 hay
que pasar 9: incluir la 10 significaría que el modelo ve el resultado
del propio partido que está proyectando.

Un error aquí no produce ninguna excepción ni dato faltante. Produce un
backtest con rendimiento inflado que no se reproduce en producción, y
el diagnóstico llega meses después, cuando el ledger real diverge del
histórico simulado.

Por eso `_cutoff_week()` es un método propio con su propia
justificación documentada, en vez de un `week - 1` incrustado en medio
del ensamblaje.

Degradación ante fallos
-------------------------
Cada fetcher es independiente. Si el injury report no carga, el resto
del pipeline continúa con `injury_penalty = 0` y `data_quality`
reducido. El modelo refleja esa incertidumbre bajando `confidence` y
ensanchando sigma, lo que a su vez hace que los filtros de EV rechacen
el pick.

Ese encadenamiento es deliberado: la respuesta correcta a "no sé el
estado del quarterback" es no apostar, no apostar a ciegas.
"""

from __future__ import annotations

from core.contracts.event import Event
from core.contracts.features import TeamFeatures

from sports.nfl.context import NFLContextFetcher
from sports.nfl.data_source import NFLDataSource, _current_nfl_season
from sports.nfl.h2h import NFLH2HFetcher, h2h_metadata
from sports.nfl.injuries import NFLInjuryFetcher
from sports.nfl.rest import NFLRestFetcher
from sports.nfl.schedule import NFLGameInfo, NFLScheduleFetcher
from sports.nfl.team_stats import NFLTeamStats, NFLTeamStatsFetcher
from sports.nfl.venue_factors import NFLVenueFactors


# Media de liga para los fallbacks cuando no hay datos del equipo.
_LEAGUE_PPG: float = 22.0

# data_quality se construye sumando la contribución de cada fuente.
# Los pesos reflejan cuánto aporta cada una a la calidad de la
# proyección, no cuántos campos rellena.
_DQ_WEIGHTS: dict[str, float] = {
    "team_stats": 0.45,   # EPA es el 70% del modelo
    "injuries":   0.25,   # un QB fuera mueve el spread 7 puntos
    "schedule":   0.15,   # descanso, divisional, metadatos
    "h2h":        0.05,   # señal débil en NFL por rotación de plantilla
    "venue":      0.10,   # techo y altitud
}


class NFLDataProvider:
    """
    Proveedor de datos NFL. Implementa SportDataProvider.

    Parámetros
    ----------
    data_source      -- NFLDataSource compartido. Si None, crea uno.
    season           -- Temporada. None = actual.
    config_loader    -- ConfigLoader con nfl.yaml, propagado a los
                        fetchers que leen parámetros calibrados.
    schedule_fetcher -- Permite inyectar fetchers ya construidos. Si
    team_stats_fetcher  se omiten, el provider los crea compartiendo
    injury_fetcher      el mismo NFLDataSource, que es lo que evita
    rest_fetcher        descargas redundantes.
    venue_factors
    context_fetcher
    h2h_fetcher
    """

    def __init__(
        self,
        data_source:        NFLDataSource | None = None,
        season:             int | None = None,
        config_loader       = None,
        schedule_fetcher:   NFLScheduleFetcher | None = None,
        team_stats_fetcher: NFLTeamStatsFetcher | None = None,
        injury_fetcher:     NFLInjuryFetcher | None = None,
        rest_fetcher:       NFLRestFetcher | None = None,
        venue_factors:      NFLVenueFactors | None = None,
        context_fetcher:    NFLContextFetcher | None = None,
        h2h_fetcher:        NFLH2HFetcher | None = None,
    ) -> None:
        self._season = season or _current_nfl_season()
        self._config = config_loader

        # NFLDataSource se crea SOLO si hace falta.
        #
        # Los tres fetchers que descargan de nflverse (schedule,
        # team_stats, injuries) lo necesitan; el resto deriva de ellos.
        # Si los tres vienen inyectados, no se instancia — y eso
        # importa más allá de los tests: permite usar el provider con
        # un backend alternativo, una caché offline o dobles de prueba
        # sin tener nfl_data_py instalado, que es una dependencia de
        # ~150 MB entre pandas y pyarrow.
        #
        # Cuando sí se crea, es UNO SOLO compartido por los tres. Sin
        # esa compartición cada fetcher descargaría su propia copia del
        # play-by-play (~50 MB por temporada) y un domingo de 16
        # partidos generaría cientos de megas de tráfico redundante.
        needs_source = (
            schedule_fetcher is None
            or team_stats_fetcher is None
            or injury_fetcher is None
        )
        source = data_source
        if source is None and needs_source:
            source = NFLDataSource(current_season=self._season)
        self._source = source

        self._schedule = schedule_fetcher or NFLScheduleFetcher(
            data_source=source, season=self._season
        )
        self._team_stats = team_stats_fetcher or NFLTeamStatsFetcher(
            data_source=source, season=self._season,
            schedule_fetcher=self._schedule,
        )
        self._injuries = injury_fetcher or NFLInjuryFetcher(
            data_source=source, season=self._season,
            config_loader=config_loader,
        )
        self._rest = rest_fetcher or NFLRestFetcher(
            schedule_fetcher=self._schedule, config_loader=config_loader
        )
        self._venues = venue_factors or NFLVenueFactors(
            config_loader=config_loader
        )
        self._context = context_fetcher or NFLContextFetcher(
            schedule_fetcher=self._schedule, venue_factors=self._venues
        )
        self._h2h = h2h_fetcher or NFLH2HFetcher(
            schedule_fetcher=self._schedule, config_loader=config_loader
        )

    # ── SportDataProvider Protocol ────────────────────────────────────────────

    def get_events(self, date: str) -> list[Event]:
        """
        Partidos NFL de la fecha dada.

        Delega en NFLScheduleFetcher, que devuelve solo los partidos de
        ESE día — no de la semana completa. La justificación está en
        schedule.py: el control de riesgo del pipeline opera sobre
        exposición diaria.
        """
        try:
            return self._schedule.get_events(date)
        except Exception:
            return []

    def enrich_event(self, event: Event) -> tuple[TeamFeatures, TeamFeatures]:
        """
        Construye las TeamFeatures de ambos equipos.

        Nunca lanza. Ante el fallo de cualquier fuente, ese bloque de
        datos queda ausente, `data_quality` baja y el resto del
        pipeline sigue. Un partido con datos parciales produce una
        proyección de baja confianza que los filtros probablemente
        rechacen — que es la respuesta correcta.
        """
        home_id = event.home_team_id
        away_id = event.away_team_id

        game = self._safe(lambda: self._schedule.get_game_info(event.event_id))
        week = game.week if isinstance(game, NFLGameInfo) else None

        # ── La barrera contra el look-ahead bias ───────────────────
        cutoff = self._cutoff_week(week)

        # ── Estadísticas de equipo ─────────────────────────────────
        home_stats = self._safe(lambda: self._team_stats.fetch(home_id, cutoff))
        away_stats = self._safe(lambda: self._team_stats.fetch(away_id, cutoff))

        # ── Lesiones ───────────────────────────────────────────────
        report_week = week if week is not None else cutoff
        home_injury = self._safe(lambda: self._injuries.fetch(home_id, report_week))
        away_injury = self._safe(lambda: self._injuries.fetch(away_id, report_week))
        home_penalty = self._safe(
            lambda: self._injuries.penalty_for(home_id, report_week), default=0.0
        ) or 0.0
        away_penalty = self._safe(
            lambda: self._injuries.penalty_for(away_id, report_week), default=0.0
        ) or 0.0

        # ── Descanso ───────────────────────────────────────────────
        home_rest, away_rest = self._safe(
            lambda: self._rest.fetch_for_game(event.event_id),
            default=(None, None),
        ) or (None, None)

        # ── Estadio ────────────────────────────────────────────────
        venue_factor = self._safe(
            lambda: self._venues.total_factor(home_id), default=1.0
        ) or 1.0

        # ── H2H ────────────────────────────────────────────────────
        is_divisional = bool(game.div_game) if isinstance(game, NFLGameInfo) else None
        h2h = self._safe(
            lambda: self._h2h.get_h2h(home_id, away_id, self._season, is_divisional)
        )

        # ── Ensamblaje ─────────────────────────────────────────────
        home_features = self._build_features(
            team_id=home_id, team_name=event.home_team,
            stats=home_stats, injury=home_injury, injury_penalty=home_penalty,
            rest=home_rest, venue_id=event.venue_id, venue_factor=venue_factor,
            game=game, h2h=h2h, is_home=True,
        )
        away_features = self._build_features(
            team_id=away_id, team_name=event.away_team,
            stats=away_stats, injury=away_injury, injury_penalty=away_penalty,
            rest=away_rest, venue_id=event.venue_id, venue_factor=venue_factor,
            game=game, h2h=h2h, is_home=False,
        )

        return home_features, away_features

    def get_context(self, event: Event) -> dict:
        """
        Contexto situacional, enriquecido con lo que el modelo necesita.

        NFLContextFetcher aporta clima, techo, viaje y metadatos del
        calendario. El provider añade tres cosas que el modelo consume
        pero que el fetcher de contexto no conoce:

            event_id            para que el modelo pueda resolver el
                                descanso por su cuenta si hiciera falta
            rest_differential   ya calculado, con el game_id correcto
            league_ppg          media de liga del corte temporal

        Calcular aquí el diferencial de descanso —en vez de dejar que
        lo haga el modelo— evita que este necesite conocer el game_id,
        que no siempre está en el contexto.
        """
        try:
            context = dict(self._context.get_context(event))
        except Exception:
            context = {}

        context["event_id"] = event.event_id

        rest_diff = self._safe(
            lambda: self._rest.differential(event.event_id), default=0.0
        )
        context["rest_differential"] = rest_diff if rest_diff is not None else 0.0

        week = context.get("week")
        cutoff = self._cutoff_week(week if isinstance(week, int) else None)
        league = self._safe(lambda: self._team_stats.league_averages(cutoff))
        if league is not None:
            context["league_ppg"] = getattr(league, "points_per_game", _LEAGUE_PPG)

        return context

    # ── La barrera temporal ───────────────────────────────────────────────────

    def _cutoff_week(self, game_week: int | None) -> int:
        """
        Última semana cuyos datos puede ver el modelo.

        Para un partido de la semana N, el corte es N-1: incluir la
        semana N significaría que el modelo observa el resultado del
        propio partido que está proyectando.

        Por qué esto tiene su propio método
        ------------------------------------
        Un `week - 1` incrustado en medio del ensamblaje sería
        indistinguible de un off-by-one accidental para quien lea el
        código después. Y el modo de fallo es particularmente
        traicionero: no lanza excepción, no deja campos vacíos, no
        aparece en ningún log. Produce un backtest con rendimiento
        inflado que no se reproduce en producción, y el diagnóstico
        llega meses más tarde cuando el ledger real diverge del
        histórico simulado.

        Casos límite
        ------------
        Semana 1 → corte 1. No hay temporada previa que consultar y
        NFLTeamStatsFetcher devolverá estadísticas vacías con
        shrinkage total hacia la media de liga. Correcto: en la semana
        1 el modelo no sabe nada que el mercado no sepa.

        Semana desconocida → se usa la semana actual del calendario
        menos uno. Si tampoco se puede determinar, 1 — el valor más
        conservador, que no expone ningún dato futuro.
        """
        if game_week is not None and game_week > 0:
            return max(1, game_week - 1)

        current = self._safe(lambda: self._schedule.current_week())
        if isinstance(current, int) and current > 0:
            return max(1, current - 1)

        return 1

    # ── Ensamblaje de TeamFeatures ────────────────────────────────────────────

    def _build_features(
        self,
        team_id:        str,
        team_name:      str,
        stats:          NFLTeamStats | None,
        injury,
        injury_penalty: float,
        rest,
        venue_id:       str,
        venue_factor:   float,
        game:           NFLGameInfo | None,
        h2h,
        is_home:        bool,
    ) -> TeamFeatures:
        """Construye las TeamFeatures de un equipo desde las fuentes."""

        # Índices normalizados. Sin estadísticas, 1.0 = media de liga:
        # es el supuesto neutro correcto, no una degradación silenciosa.
        offense_index = stats.offense_index() if stats else 1.0
        defense_index = stats.defense_index() if stats else 1.0

        recent_scores = list(stats.recent_scores) if stats else []
        recent_avg = (
            round(sum(recent_scores) / len(recent_scores), 3)
            if recent_scores else 0.0
        )
        expected_score = (
            recent_avg or (stats.points_per_game if stats else None) or _LEAGUE_PPG
        )

        # ── sport_metadata: todo lo específico de NFL ──────────────
        metadata: dict = {}

        if stats is not None:
            metadata.update(stats.to_metadata())
        if injury is not None:
            metadata.update(injury.to_metadata())
        metadata["injury_penalty"] = round(float(injury_penalty), 3)
        if rest is not None:
            metadata.update(rest.to_metadata())
        if game is not None:
            metadata.update(game.to_metadata())
        if h2h is not None:
            metadata.update(h2h_metadata(h2h))
        metadata["is_home"] = is_home

        # Media de liga del success rate, que NFLProjectionModel usa
        # para normalizar esa señal.
        if stats is not None and stats.league is not None:
            metadata["league_success_rate"] = stats.league.success_rate

        data_quality, missing = self._assess_quality(stats, injury, rest, game, h2h)

        return TeamFeatures(
            team_id        = team_id,
            team_name      = team_name,
            expected_score = round(float(expected_score), 3),
            offense_index  = offense_index,
            defense_index  = defense_index,
            recent_scores  = recent_scores,
            recent_avg     = recent_avg,
            recent_n       = len(recent_scores),
            venue_id       = venue_id or "",
            venue_factor   = round(float(venue_factor), 4),
            sample_size    = stats.plays_off if stats else 0,
            data_quality   = data_quality,
            missing_fields = missing,
            sport_metadata = metadata,
        )

    @staticmethod
    def _assess_quality(
        stats, injury, rest, game, h2h,
    ) -> tuple[float, list[str]]:
        """
        Calidad de datos como suma ponderada de las fuentes disponibles.

        Los pesos reflejan el aporte de cada fuente a la calidad de la
        proyección, no cuántos campos rellena. Las estadísticas de
        equipo pesan 0.45 porque EPA es el 70% del modelo; el H2H pesa
        0.05 porque en NFL la rotación de plantilla lo convierte en
        señal marginal.

        Se penaliza aparte el injury report desactualizado: tener el
        parte de la semana anterior es mejor que no tenerlo, pero peor
        que tener el actual.
        """
        quality = 0.0
        missing: list[str] = []

        if stats is not None and stats.games_played > 0:
            quality += _DQ_WEIGHTS["team_stats"]
            if not stats.has_sufficient_sample:
                # Muestra corta: la fuente está, pero informa poco
                quality -= _DQ_WEIGHTS["team_stats"] * 0.35
        else:
            missing.append("team_stats")

        # Se comprueba `is not None`, NO `has_data`.
        #
        # InjuryReport.has_data devuelve len(entries) > 0, así que un
        # equipo sin lesionados daría False — y contarlo como dato
        # faltante sería un error semántico: un parte vacío obtenido
        # con éxito significa "plantilla sana", que es información
        # valiosa, no su ausencia.
        #
        # La distinción real es entre "el fetch falló" (el provider
        # recibe None desde _safe) y "el fetch funcionó y no hay
        # lesionados" (un InjuryReport con entries vacío). Penalizar el
        # segundo caso castigaría precisamente a los equipos en mejor
        # estado.
        if injury is not None:
            weight = _DQ_WEIGHTS["injuries"]
            if injury.is_stale:
                # Parte de la semana anterior: informativo pero no fresco
                weight *= 0.6
            quality += weight
        else:
            missing.append("injuries")

        if game is not None:
            quality += _DQ_WEIGHTS["schedule"]
        else:
            missing.append("schedule")

        if rest is not None:
            quality += _DQ_WEIGHTS["venue"] * 0.5
        else:
            missing.append("rest")

        quality += _DQ_WEIGHTS["venue"] * 0.5   # el catálogo nunca falla

        if h2h is not None and h2h.is_reliable:
            quality += _DQ_WEIGHTS["h2h"]
        else:
            missing.append("h2h")

        return round(max(0.0, min(1.0, quality)), 3), missing

    # ── Utilidad ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(fn, default=None):
        """
        Ejecuta fn() devolviendo `default` si lanza.

        Cada fetcher es independiente: el fallo de uno no debe impedir
        que el resto aporte sus datos. Un partido con injury report
        caído pero EPA completo sigue siendo proyectable, con la
        incertidumbre declarada en data_quality.
        """
        try:
            return fn()
        except Exception:
            return default