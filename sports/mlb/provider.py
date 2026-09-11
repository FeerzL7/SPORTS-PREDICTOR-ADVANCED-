"""
sports/mlb/provider.py

MLBDataProvider: proveedor de datos MLB para el pipeline.

Implementa core/pipeline/stage.py:SportDataProvider.

Orquesta todos los módulos de datos del plugin MLB:
    StatcastFetcher    → ERA, FIP del abridor (8.2)
    VenueFactorProvider → park factor (8.3)
    PitchingFetcher    → abridor probable (8.4)
    BullpenFetcher     → ERA del bullpen, defense_index (8.5)
    OffenseFetcher     → OPS, recent_scores (8.6) [bugs F1/F2 corregidos]
    DefenseFetcher     → fielding stats (8.7)
    MLBContextFetcher  → clima y venue_type (8.8)
    MLBH2HFetcher      → historial H2H (8.9)
    MLBMarketDefinitions → mercados (8.12)

Flujo de enrich_event(event)
-------------------------------
    1. Fetch del schedule del día para obtener game_pk
    2. PitchingFetcher.get_starters(game_pk)
       → home_pitcher, away_pitcher (ProbablePitcher)
    3. BullpenFetcher.fetch(team_id) → BullpenStats (home y away)
    4. OffenseFetcher.fetch(team_id) → OffenseStats (home y away)
    5. OffenseFetcher.fetch_recent_scores(team_id) → list[float]
    6. DefenseFetcher.fetch(team_id) → FieldingStats (home y away)
    7. VenueFactorProvider.get(venue_id) → float
    8. MLBH2HFetcher.get_stats(home_id, away_id) → H2HMetrics
    9. BullpenFetcher.defense_index(pitcher_era, bullpen) → float
    10. Ensamblar TeamFeatures para home y away

Manejo de fallos
-----------------
Cada fetch es independiente. Si el fetch de estatcast falla para
un pitcher, el provider usa ERA de liga como fallback documentado
en TeamFeatures.data_quality. El pipeline nunca aborta por datos
faltantes de un equipo — degrada graciosamente con fallbacks.

data_quality refleja qué porcentaje de campos están poblados:
    1.0 = todos los datos disponibles
    0.5 = mitad de los datos disponibles (pitcher sin confirmar, etc.)
    0.0 = sin datos (API completamente caída) — el evento se omite
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

T = TypeVar("T")

from core.contracts.event import Event, EventStatus
from core.contracts.features import TeamFeatures

from sports.mlb.bullpen import BullpenFetcher, BullpenStats
from sports.mlb.context import MLBContextFetcher
from sports.mlb.defense import DefenseFetcher, FieldingStats
from sports.mlb.h2h import MLBH2HFetcher, h2h_metadata
from sports.mlb.offense import OffenseFetcher, OffenseStats
from sports.mlb.pitching import PitchingFetcher, ProbablePitcher, _bullpen_day
from sports.mlb.statcast import StatcastFetcher, _LEAGUE_ERA, _current_season
from sports.mlb.venue_factors import VenueFactorProvider

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

# CORRECCIÓN (auditoría 2026-08): antes _requests solo se asignaba en la
# rama try, dejando la variable "possibly unbound" para el type checker
# en cualquier punto donde se usara tras el try/except (Pylance/pyright
# marcaba esto en cada uno de los ~10 archivos que repiten este patrón
# de dependencia opcional). Ahora _requests siempre está definida (como
# None si el import falla), y _REQUESTS_AVAILABLE se deriva de eso en
# vez de ser una bandera independiente que podía desincronizarse.
_REQUESTS_AVAILABLE = _requests is not None

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Umbral de data_quality mínimo para incluir el evento en el pipeline
_MIN_DATA_QUALITY: float = 0.30

# MLB team IDs — para validación de IDs
_VALID_TEAM_ID_RANGE = range(100, 200)  # IDs MLB están en este rango aprox.


class MLBDataProvider:
    """
    Proveedor de datos MLB. Implementa SportDataProvider Protocol.

    Parámetros
    ----------
    statcast_fetcher   -- StatcastFetcher configurado. None = default.
    venue_provider     -- VenueFactorProvider. None = default.
    pitching_fetcher   -- PitchingFetcher. None = default.
    bullpen_fetcher    -- BullpenFetcher. None = default.
    offense_fetcher    -- OffenseFetcher. None = default.
    defense_fetcher    -- DefenseFetcher. None = default.
    context_fetcher    -- MLBContextFetcher. None = default.
    h2h_fetcher        -- MLBH2HFetcher. None = default.
    season             -- Temporada a consultar. None = actual.
    config_loader      -- ConfigLoader con mlb.yaml.
    """

    def __init__(
        self,
        statcast_fetcher:  StatcastFetcher | None    = None,
        venue_provider:    VenueFactorProvider | None = None,
        pitching_fetcher:  PitchingFetcher | None     = None,
        bullpen_fetcher:   BullpenFetcher | None      = None,
        offense_fetcher:   OffenseFetcher | None      = None,
        defense_fetcher:   DefenseFetcher | None      = None,
        context_fetcher:   MLBContextFetcher | None   = None,
        h2h_fetcher:       MLBH2HFetcher | None       = None,
        season:            int | None                 = None,
        config_loader                                 = None,
    ) -> None:
        cfg = config_loader
        self._statcast  = statcast_fetcher or StatcastFetcher(config_loader=cfg)
        self._venue     = venue_provider   or VenueFactorProvider(config_loader=cfg)
        self._pitching  = pitching_fetcher or PitchingFetcher(
            statcast_fetcher=self._statcast
        )
        self._bullpen   = bullpen_fetcher  or BullpenFetcher()
        self._offense   = offense_fetcher  or OffenseFetcher()
        self._defense   = defense_fetcher  or DefenseFetcher()
        self._context   = context_fetcher  or MLBContextFetcher()
        self._h2h       = h2h_fetcher      or MLBH2HFetcher()
        self._season    = season or _current_season()
        self._config    = cfg

    # ── SportDataProvider Protocol ────────────────────────────────────────────

    def get_events(self, date: str) -> list[Event]:
        """
        Retorna los partidos MLB del día en formato Event.

        Consulta MLB Stats API schedule para obtener todos los partidos
        de temporada regular del día especificado.

        Parámetros
        ----------
        date  -- Fecha en 'YYYY-MM-DD'.

        Retorna
        -------
        list[Event] — vacía si no hay partidos o la API falla.
        """
        raw_games = self._fetch_schedule(date)
        events = []
        for game in raw_games:
            event = self._parse_game_to_event(game, date)
            if event is not None:
                events.append(event)
        return events

    def enrich_event(
        self,
        event: Event,
    ) -> tuple[TeamFeatures, TeamFeatures]:
        """
        Enriquece el evento con features estadísticas de ambos equipos.

        Retorna (home_features, away_features).

        Nunca lanza excepción — usa fallbacks documentados cuando
        algún fetch falla. data_quality refleja la calidad de los datos.
        """
        home_id = self._parse_team_id(event.home_team_id)
        away_id = self._parse_team_id(event.away_team_id)

        # game_pk para pitching
        game_pk = self._extract_game_pk(event)

        # ── 1. Abridores probables ─────────────────────────────────
        if game_pk is not None:
            home_pitcher, away_pitcher = self._pitching.get_starters(game_pk)
        else:
            home_pitcher = _bullpen_day("Unknown")
            away_pitcher = _bullpen_day("Unknown")

        # ── 2. Bullpen stats ───────────────────────────────────────
        home_bullpen = self._safe_fetch(
            self._bullpen.fetch, home_id, self._season,
            fallback=BullpenStats(team_id=home_id, season=self._season)
        )
        away_bullpen = self._safe_fetch(
            self._bullpen.fetch, away_id, self._season,
            fallback=BullpenStats(team_id=away_id, season=self._season)
        )

        # ── 3. Ofensiva ────────────────────────────────────────────
        home_offense = self._safe_fetch(
            self._offense.fetch, home_id, self._season,
            fallback=OffenseStats(team_id=home_id, season=self._season)
        )
        away_offense = self._safe_fetch(
            self._offense.fetch, away_id, self._season,
            fallback=OffenseStats(team_id=away_id, season=self._season)
        )

        # ── 4. Carreras recientes (bug F1 corregido) ───────────────
        # event.date ya es str 'YYYY-MM-DD' por contrato (Event.date) —
        # el parche defensivo hasattr(event.date, 'isoformat') que había
        # aquí antes era el síntoma de que _parse_game_to_event violaba
        # ese contrato en otro punto; corregido en origen, ya no hace
        # falta adivinar el tipo en runtime.
        date_str = event.date
        home_recent = self._safe_fetch(
            self._offense.fetch_recent_scores, home_id, date_str,
            fallback=[]
        )
        away_recent = self._safe_fetch(
            self._offense.fetch_recent_scores, away_id, date_str,
            fallback=[]
        )

        # ── 5. Fielding ────────────────────────────────────────────
        home_fielding = self._safe_fetch(
            self._defense.fetch, home_id, self._season,
            fallback=FieldingStats(team_id=home_id, season=self._season)
        )
        away_fielding = self._safe_fetch(
            self._defense.fetch, away_id, self._season,
            fallback=FieldingStats(team_id=away_id, season=self._season)
        )

        # ── 6. Park factor ─────────────────────────────────────────
        venue_factor = self._venue.get(event.venue_id or "")

        # ── 7. H2H ────────────────────────────────────────────────
        h2h_stats = self._safe_fetch(
            self._h2h.get_stats, home_id, away_id,
            fallback=None
        )

        # ── 8. defense_index para cada equipo ─────────────────────
        # defense_index = LEAGUE_ERA / ERA_combinada(starter + bullpen)
        home_def_idx = self._bullpen.defense_index(
            starter_era    = home_pitcher.effective_era,
            bullpen        = home_bullpen,
            is_bullpen_day = home_pitcher.is_bullpen_day,
        )
        away_def_idx = self._bullpen.defense_index(
            starter_era    = away_pitcher.effective_era,
            bullpen        = away_bullpen,
            is_bullpen_day = away_pitcher.is_bullpen_day,
        )

        # Ajuste fielding sobre defense_index
        if home_fielding.fielding_pct is not None:
            home_def_idx = self._defense.combined_defense_index(
                home_def_idx, home_fielding
            )
        if away_fielding.fielding_pct is not None:
            away_def_idx = self._defense.combined_defense_index(
                away_def_idx, away_fielding
            )

        # ── 9. Ensamblaje de TeamFeatures ─────────────────────────
        home_features = self._build_features(
            team_id       = home_id,
            team_name     = event.home_team,
            pitcher       = home_pitcher,
            bullpen       = home_bullpen,
            offense       = home_offense,
            fielding      = home_fielding,
            recent_scores = home_recent,
            defense_index = home_def_idx,
            venue_factor  = venue_factor,
            venue_id      = event.venue_id or "",
            h2h_stats     = h2h_stats,
            rival_pitcher_hand = away_pitcher.hand,
        )
        away_features = self._build_features(
            team_id       = away_id,
            team_name     = event.away_team,
            pitcher       = away_pitcher,
            bullpen       = away_bullpen,
            offense       = away_offense,
            fielding      = away_fielding,
            recent_scores = away_recent,
            defense_index = away_def_idx,
            venue_factor  = venue_factor,
            venue_id      = event.venue_id or "",
            h2h_stats     = h2h_stats,
            rival_pitcher_hand = home_pitcher.hand,
            is_away       = True,
        )

        return home_features, away_features

    def get_context(self, event: Event) -> dict:
        """
        Retorna el contexto situacional del partido (clima, venue).

        Delega en MLBContextFetcher. Retorna {} si la API falla.
        """
        try:
            return self._context.get_context(event)
        except Exception:
            return {}

    # ── Construcción de TeamFeatures ──────────────────────────────────────────

    def _build_features(
        self,
        team_id:            int,
        team_name:          str,
        pitcher:            ProbablePitcher,
        bullpen:            BullpenStats,
        offense:            OffenseStats,
        fielding:           FieldingStats,
        recent_scores:      list[float],
        defense_index:      float,
        venue_factor:       float,
        venue_id:           str,
        h2h_stats,
        rival_pitcher_hand: str | None,
        is_away:            bool = False,
    ) -> TeamFeatures:
        """Ensambla un TeamFeatures completo desde los datos fetched."""

        # Offense index con split por mano del rival
        offense_idx = offense.offense_index_vs_hand(rival_pitcher_hand)

        # Recent avg desde recent_scores.
        # 0.0 en vez de None cuando no hay datos: TeamFeatures.recent_avg
        # está tipado como float (no Optional) porque __post_init__ lo
        # recalcula siempre desde recent_scores de todos modos — pasar
        # None aquí violaba ese contrato sin aportar nada, ya que el
        # valor se descarta y se recalcula igual dentro de TeamFeatures.
        # `expected_score` de abajo sigue funcionando igual: 0.0 es
        # falsy en Python, el fallback a runs_per_game se activa igual
        # que antes con None.
        recent_avg = (
            round(sum(recent_scores) / len(recent_scores), 3)
            if recent_scores else 0.0
        )

        # expected_score: recent_avg o runs_per_game como fallback
        expected_score = recent_avg or offense.runs_per_game or 4.5

        # sport_metadata: todo lo específico de MLB
        metadata: dict = {}
        metadata.update(pitcher.to_metadata())
        metadata.update(bullpen.to_metadata())
        metadata.update(offense.to_metadata())
        metadata.update(fielding.to_metadata())
        if h2h_stats is not None:
            metadata.update(h2h_metadata(h2h_stats))

        # data_quality: proporción de campos críticos no-None
        critical_fields = [
            pitcher.era, offense.ops, bullpen.era,
            offense.obp, offense.slg,
        ]
        populated = sum(1 for f in critical_fields if f is not None)
        dq = round(populated / len(critical_fields), 2)

        return TeamFeatures(
            team_id        = str(team_id),
            team_name      = team_name,
            expected_score = round(float(expected_score), 3),
            offense_index  = round(offense_idx, 4),
            defense_index  = round(defense_index, 4),
            recent_scores  = recent_scores,
            recent_avg     = recent_avg,
            venue_id       = venue_id,
            venue_factor   = round(venue_factor, 4),
            data_quality   = dq,
            sport_metadata = metadata,
        )

    # ── Fetch del schedule ────────────────────────────────────────────────────

    def _fetch_schedule(self, date: str) -> list[dict]:
        """Fetch del schedule MLB para una fecha."""
        if not _REQUESTS_AVAILABLE or _requests is None:
            return []
        url    = f"{_MLB_API_BASE}/schedule"
        params = {
            "sportId":   1,
            "date":      date,
            "gameType":  "R",
            "hydrate":   "team,venue,probablePitcher",
        }
        try:
            resp = _requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
            games = []
            for date_entry in data.get("dates", []):
                games.extend(date_entry.get("games", []))
            return games
        except Exception:
            return []

    def _parse_game_to_event(self, game: dict, date: str) -> Event | None:
        """
        Convierte un game del schedule MLB a un Event del contrato.

        CORRECCIÓN DE CONTRATO (auditoría 2026-08): esta función
        construía `season_start`/`season_end` como objetos `datetime.date`
        y `date` como objeto `date` también, pero `Event` los tipa como
        `int` (season_start/season_end) y `str` 'YYYY-MM-DD' (date)
        respectivamente. El síntoma downstream era un parche defensivo
        en `enrich_event()` (`hasattr(event.date, 'isoformat')`) que
        adivinaba en runtime qué tipo tenía `event.date` en vez de
        confiar en el contrato — y dos propiedades de `Event`
        (`is_single_year_season`, `season_label`) quedaban rotas de forma
        silenciosa porque comparar dos `date` distintos nunca da True
        aunque el año coincida. Se corrige aquí, en el origen, para que
        el contrato se respete desde el primer punto de construcción.
        """
        try:
            game_pk    = game.get("gamePk")
            teams      = game.get("teams", {})
            home       = teams.get("home", {}).get("team", {})
            away       = teams.get("away", {}).get("team", {})
            venue      = game.get("venue", {})
            start_time = game.get("gameDate", "")

            # Status
            abstract_state = game.get("status", {}).get("abstractGameState", "")
            status_map = {
                "Preview": EventStatus.SCHEDULED,
                "Live":    EventStatus.LIVE,
                "Final":   EventStatus.FINAL,
            }
            status = status_map.get(abstract_state, EventStatus.SCHEDULED)

            # Validar que `date` tiene el formato esperado — si no,
            # queremos que la excepción caiga al except Exception de
            # abajo y el evento se descarte, no que se cuele un string
            # malformado dentro del Event.
            datetime.strptime(date, "%Y-%m-%d")

            return Event(
                event_id      = str(game_pk),
                sport         = "mlb",
                league        = "MLB",
                season_start  = self._season,
                season_end    = self._season,
                date          = date,
                start_time    = start_time,
                home_team_id  = str(home.get("id", "")),
                away_team_id  = str(away.get("id", "")),
                home_team     = home.get("name", ""),
                away_team     = away.get("name", ""),
                venue_id      = str(venue.get("id", "")),
                venue_name    = venue.get("name", ""),
                status        = status,
                provider_ids  = {
                    "mlb_game_pk": str(game_pk),
                    "odds_api":    "",  # se mapea externamente
                },
            )
        except Exception:
            return None

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_team_id(team_id_str: Any) -> int:
        """Convierte team_id (str o int) a int."""
        try:
            return int(team_id_str)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _extract_game_pk(event: Event) -> int | None:
        """Extrae game_pk de Event.provider_ids."""
        pk = event.provider_ids.get("mlb_game_pk") or event.event_id
        try:
            return int(pk)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_fetch(
        fn: Callable[..., T],
        *args,
        fallback: T,
    ) -> T:
        """
        Ejecuta fn(*args) y retorna fallback si lanza excepción.

        Garantiza que un fallo de API no propague al pipeline.

        CORRECCIÓN DE TIPADO (auditoría 2026-08): antes `fn` y `fallback`
        no tenían anotación de tipo, así que Pylance/pyright inferían el
        retorno de cada llamada como `Unknown | None` — eso se propagaba
        a cada sitio de uso (`home_bullpen`, `home_offense`,
        `home_fielding`, etc.) y de ahí a `_build_features()`, generando
        ~14 errores de tipo en cascada, todos con la misma causa raíz.
        Con `Callable[..., T]` + `fallback: T` sin default, el tipo de
        retorno queda ligado exactamente al tipo del `fallback` que cada
        llamada ya provee (ej. `fallback=BullpenStats(...)` → retorno
        `BullpenStats`, nunca `None`) — que es además la garantía real en
        tiempo de ejecución: esta función NUNCA retorna `None` a menos
        que el propio `fallback` pasado sea `None` (como en el caso de
        `h2h_stats`, donde sí se pasa `fallback=None` explícitamente).
        """
        try:
            return fn(*args)
        except Exception:
            return fallback