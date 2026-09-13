"""
sports/nfl/schedule.py

NFLScheduleFetcher: calendario de partidos NFL desde nflverse.

Responsabilidad
----------------
Convertir el DataFrame de `nfl_data_py.import_schedules()` en objetos
`Event` del contrato del Core. Es el Stage 1 del pipeline NFL.

Además expone metadatos de calendario que otros fetchers necesitan:
    - Semana actual de la temporada (rest.py, injuries.py)
    - Días de descanso desde el último partido (rest.py)
    - Bye weeks por equipo (rest.py)
    - Historial de enfrentamientos (h2h.py)

Diferencia estructural con MLB
--------------------------------
El calendario MLB se consulta por fecha: pides '2026-07-15' y obtienes
los 15 partidos de ese día. El calendario NFL se organiza por SEMANA:
la semana 3 abarca de jueves a lunes, con partidos repartidos en 5 días.

Decisión de diseño: `get_events(date)` retorna solo los partidos de ESE
día exacto, no de toda la semana. Razón: el pipeline registra picks con
exposición diaria y RiskManager limita `max_picks_daily`. Si get_events
devolviera los 16 partidos de la semana completa, el control de riesgo
diario perdería sentido — el operador ejecutaría el pipeline el domingo
y recibiría también los partidos del jueves siguiente.

El operador ejecuta el pipeline el día de los partidos que quiere
evaluar. Para el Thursday Night Football, ejecuta el jueves.

Temporada cruzando el año calendario
--------------------------------------
La temporada NFL 2026 va de septiembre 2026 a febrero 2027. Por eso:
    season_start = 2026
    season_end   = 2027

nflverse identifica la temporada por su año de inicio (campo `season`),
así que los partidos de enero-febrero 2027 tienen season=2026.

Identificadores de equipo
---------------------------
nflverse usa abreviaciones de 2-3 letras: 'KC', 'SF', 'NE', 'LAR'.
Son estables entre temporadas y sirven como `team_id` canónico.

Para `home_team` / `away_team` (display) se resuelve el nombre completo
desde `load_team_descriptions()`, necesario para el matching con The
Odds API que usa nombres completos ('Kansas City Chiefs').
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from core.contracts.event import Event, EventStatus

from sports.nfl.data_source import NFLDataSource, _current_nfl_season


# Tipos de partido en nflverse (`game_type`)
GAME_TYPE_REGULAR = "REG"
GAME_TYPE_WILDCARD = "WC"
GAME_TYPE_DIVISIONAL = "DIV"
GAME_TYPE_CONFERENCE = "CON"
GAME_TYPE_SUPERBOWL = "SB"

_POSTSEASON_TYPES = frozenset({
    GAME_TYPE_WILDCARD,
    GAME_TYPE_DIVISIONAL,
    GAME_TYPE_CONFERENCE,
    GAME_TYPE_SUPERBOWL,
})

# Semanas de temporada regular NFL (2021+: 18 semanas, 17 partidos por equipo)
_REGULAR_SEASON_WEEKS = 18


@dataclass(frozen=True)
class NFLGameInfo:
    """
    Metadatos de un partido NFL que el Event genérico no captura.

    El contrato Event es sport-agnóstico: no tiene campos para 'week',
    'div_game' ni 'roof'. Esa información vive aquí y se inyecta en
    TeamFeatures.sport_metadata por el NFLDataProvider.

    Campos
    ------
    game_id     -- ID de nflverse: '2026_03_KC_LAC'.
    season      -- Año de inicio de temporada.
    week        -- Semana (1-18 regular, 19+ playoffs).
    game_type   -- 'REG', 'WC', 'DIV', 'CON', 'SB'.
    gameday     -- Fecha 'YYYY-MM-DD'.
    gametime    -- Hora local del venue 'HH:MM'.
    weekday     -- Día de la semana ('Sunday', 'Thursday', 'Monday').
    div_game    -- True si es partido divisional.
    roof        -- 'outdoors', 'dome', 'closed', 'open'.
    surface     -- 'grass', 'fieldturf', 'astroturf', etc.
    home_rest   -- Días de descanso del local desde su último partido.
    away_rest   -- Días de descanso del visitante.
    overtime    -- True si el partido fue a tiempo extra (solo finalizados).
    """
    game_id:    str
    season:     int
    week:       int
    game_type:  str
    gameday:    str
    home_team:  str = ""
    away_team:  str = ""
    gametime:   str | None = None
    weekday:    str | None = None
    div_game:   bool = False
    roof:       str | None = None
    surface:    str | None = None
    stadium:    str | None = None
    home_rest:  int | None = None
    away_rest:  int | None = None
    home_score: float | None = None
    away_score: float | None = None
    overtime:   bool = False

    # ── Clima observado ──────────────────────────────────────────
    # nflverse registra la temperatura y el viento medidos en el
    # estadio para partidos ya jugados. Es el dato REAL, no un
    # pronóstico, así que NFLContextFetcher lo prefiere sobre
    # Open-Meteo: usar un forecast donde existe la medición metería
    # en el backtest un error de predicción que no ocurrió.
    temp: float | None = None
    wind: float | None = None

    # ── Líneas de cierre del mercado ─────────────────────────────
    # nflverse incluye las líneas históricas en el schedule. Son de
    # CIERRE — el punto de máxima eficiencia del mercado — y son lo
    # que hace posible un backtest de apuestas sin pagar por datos
    # históricos de odds.
    #
    # Convención de spread_line: handicap desde la perspectiva del
    # LOCAL, positivo cuando el local es favorito. Un valor de 3.5
    # significa local favorito por 3.5, que en notación de mercado
    # es local -3.5 y visitante +3.5.
    #
    # Los moneylines vienen en formato americano (-150, +130).
    spread_line:      float | None = None
    total_line:       float | None = None
    home_moneyline:   float | None = None
    away_moneyline:   float | None = None
    home_spread_odds: float | None = None
    away_spread_odds: float | None = None
    over_odds:        float | None = None
    under_odds:       float | None = None

    @property
    def is_postseason(self) -> bool:
        """True si es partido de playoffs."""
        return self.game_type in _POSTSEASON_TYPES

    @property
    def is_primetime(self) -> bool:
        """
        True si es partido de horario estelar.

        Thursday Night, Sunday Night y Monday Night tienen dinámicas
        distintas: audiencia nacional, más descanso o menos según el
        caso, y equipos que ajustan su preparación.
        """
        if self.weekday in ("Thursday", "Monday"):
            return True
        # Sunday Night: kickoff a las 20:20 ET
        if self.weekday == "Sunday" and self.gametime:
            try:
                hour = int(self.gametime.split(":")[0])
                return hour >= 19
            except (ValueError, IndexError):
                return False
        return False

    @property
    def has_market_lines(self) -> bool:
        """
        True si nflverse trae líneas de cierre para este partido.

        El backtest omite los partidos sin líneas: sin precio de
        mercado no hay EV que calcular ni apuesta que simular.
        """
        return self.spread_line is not None or self.total_line is not None

    @property
    def has_observed_weather(self) -> bool:
        """True si hay mediciones reales de clima (partido ya jugado)."""
        return self.temp is not None or self.wind is not None

    @property
    def is_final(self) -> bool:
        """True si el partido ya terminó (tiene marcador registrado)."""
        return self.home_score is not None and self.away_score is not None

    @property
    def is_indoor(self) -> bool:
        """True si el partido se juega bajo techo (clima irrelevante)."""
        return self.roof in ("dome", "closed")

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "nfl_game_id":  self.game_id,
            "week":         self.week,
            "game_type":    self.game_type,
            "div_game":     self.div_game,
            "weekday":      self.weekday,
            "is_primetime": self.is_primetime,
            "is_indoor":    self.is_indoor,
            "roof":         self.roof,
            "surface":      self.surface,
            "home_rest":    self.home_rest,
            "away_rest":    self.away_rest,
            "spread_line":  self.spread_line,
            "total_line":   self.total_line,
        }


class NFLScheduleFetcher:
    """
    Obtiene y convierte el calendario NFL desde nflverse.

    Parámetros
    ----------
    data_source      -- NFLDataSource compartido. Si None, crea uno.
    season           -- Temporada a consultar. None = actual.
    include_playoffs -- Si True, incluye partidos de postemporada.
                       Default True: a diferencia de MLB, los playoffs
                       NFL son solo 13 partidos y tienen mercados muy
                       líquidos, así que vale la pena cubrirlos.
    """

    def __init__(
        self,
        data_source:      NFLDataSource | None = None,
        season:           int | None = None,
        include_playoffs: bool = True,
    ) -> None:
        self._source           = data_source or NFLDataSource()
        self._season           = season or _current_nfl_season()
        self._include_playoffs = include_playoffs
        self._team_names:  dict[str, str] | None = None
        # Caché de la temporada completa como objetos de dominio.
        # None = aún no cargada; [] = cargada pero vacía o falló.
        self._games_cache: list[NFLGameInfo] | None = None

    # ── Carga y caché de la temporada ─────────────────────────────────────────

    def _games(self) -> list[NFLGameInfo]:
        """
        Todos los partidos de la temporada como objetos de dominio.

        Este es el ÚNICO punto del módulo que toca un DataFrame. El
        resto de la clase opera sobre list[NFLGameInfo] — objetos
        tipados que el type checker entiende completamente.

        Motivo de este diseño (auditoría de tipado):
            La versión anterior hacía boolean masking de pandas
            directamente en cada método:

                past = df[df["gameday"].astype(str) <= ref]
                reg[(reg["home_team"] == team) | (reg["away_team"] == team)]

            Pyright no modela la semántica de pandas: ve que
            `df["col"] == valor` retorna `bool` (no `Series[bool]`) y
            concluye que `df[mask]` es un acceso con clave booleana,
            lo que viola la firma `__getitem__(key: str)`. Eran 8
            ocurrencias del mismo patrón en el módulo.

            Silenciar los errores con `# type: ignore` habría escondido
            el problema real: la lógica de dominio estaba acoplada a la
            representación en DataFrame. Con esta capa, pandas queda
            confinado a data_source.py (la frontera con la librería
            externa) y schedule.py trabaja con el modelo de dominio.

        El resultado se cachea en memoria: una carga por instancia,
        no por llamada.
        """
        if self._games_cache is not None:
            return self._games_cache

        try:
            df = self._source.load_schedules([self._season])
        except Exception:
            self._games_cache = []
            return self._games_cache

        if df is None or len(df) == 0:
            self._games_cache = []
            return self._games_cache

        games: list[NFLGameInfo] = []
        for row in df.itertuples(index=False):
            try:
                games.append(self._row_to_game_info(row))
            except Exception:
                continue  # fila malformada: se descarta, no aborta la carga

        self._games_cache = games
        return games

    # ── Stage 1: get_events ───────────────────────────────────────────────────

    def get_events(self, date: str) -> list[Event]:
        """
        Retorna los partidos NFL de la fecha dada como objetos Event.

        Parámetros
        ----------
        date  -- Fecha en 'YYYY-MM-DD'.

        Retorna
        -------
        list[Event] — vacía si no hay partidos ese día o falla la carga.
        Un martes o miércoles de temporada devolverá [] correctamente:
        no hay partidos NFL esos días.
        """
        games = [g for g in self._games() if g.gameday == date]

        if not self._include_playoffs:
            games = [g for g in games if g.game_type == GAME_TYPE_REGULAR]

        events: list[Event] = []
        for game in games:
            event = self._game_to_event(game)
            if event is not None:
                events.append(event)
        return events

    def get_week_events(self, week: int) -> list[Event]:
        """
        Retorna todos los partidos de una semana de la temporada.

        Útil para backtesting y para pre-cargar el contexto de la
        semana completa, pero NO se usa en el pipeline diario —
        el control de riesgo opera sobre exposición diaria.
        """
        events: list[Event] = []
        for game in self._games():
            if game.week != week:
                continue
            event = self._game_to_event(game)
            if event is not None:
                events.append(event)
        return events

    # ── Metadatos de calendario ───────────────────────────────────────────────

    def get_game_info(self, game_id: str) -> NFLGameInfo | None:
        """
        Metadatos NFL-específicos de un partido.

        Consumido por NFLDataProvider para poblar sport_metadata con
        week, div_game, roof, rest days, etc.
        """
        for game in self._games():
            if game.game_id == game_id:
                return game
        return None

    def current_week(self, date: str | None = None) -> int | None:
        """
        Semana de temporada que contiene la fecha dada.

        Necesaria para NFLInjuryFetcher (el injury report es semanal)
        y NFLTeamStatsFetcher (acumular EPA solo hasta la semana previa,
        evitando look-ahead bias en backtests).

        Retorna None si no hay calendario disponible.
        """
        ref   = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        games = self._games()
        if not games:
            return None

        # Semanas de partidos ya jugados o en curso a la fecha dada
        weeks = [g.week for g in games if g.gameday and g.gameday <= ref]
        if not weeks:
            return 1  # antes del primer partido de la temporada
        return max(weeks)

    def get_bye_weeks(self) -> dict[str, int]:
        """
        Semana de descanso (bye) de cada equipo en la temporada.

        Un equipo saliendo de bye tiene una semana extra de preparación
        y recuperación física — efecto bien documentado y estable
        históricamente (config: nfl.rest.bye_week_bonus).

        Retorna
        -------
        dict {team_abbr: bye_week}. Vacío si no se puede determinar.
        """
        regular = [g for g in self._games() if g.game_type == GAME_TYPE_REGULAR]
        if not regular:
            return {}

        # Semanas en las que juega cada equipo
        weeks_played: dict[str, set[int]] = {}
        for g in regular:
            for team in (g.home_team, g.away_team):
                if team:
                    weeks_played.setdefault(team, set()).add(g.week)

        all_weeks = set(range(1, _REGULAR_SEASON_WEEKS + 1))
        byes: dict[str, int] = {}
        for team, played in weeks_played.items():
            missing = all_weeks - played
            # Exactamente una semana sin partido = el bye.
            # Si hay más de una, el calendario está incompleto (temporada
            # en curso) y no podemos afirmar cuál es el bye.
            if len(missing) == 1:
                byes[team] = missing.pop()
        return byes

    def get_team_schedule(self, team: str) -> list[NFLGameInfo]:
        """
        Todos los partidos de un equipo en la temporada, ordenados por fecha.

        Usado por NFLRestFetcher para calcular días desde el último
        partido, y por NFLTeamStatsFetcher para la ventana reciente.
        """
        return sorted(
            (
                g for g in self._games()
                if g.home_team == team or g.away_team == team
            ),
            key=lambda g: g.gameday,
        )

    # ── Conversión de filas ───────────────────────────────────────────────────

    def _game_to_event(self, game: NFLGameInfo) -> Event | None:
        """
        Convierte un NFLGameInfo al contrato Event del Core.

        Retorna None si faltan los campos mínimos de identidad.
        """
        if not game.game_id or not game.home_team or not game.away_team:
            return None

        status = EventStatus.FINAL if game.is_final else EventStatus.SCHEDULED

        return Event(
            event_id     = game.game_id,
            sport        = "nfl",
            league       = "NFL",
            # La temporada NFL cruza el año calendario:
            # 2026 va de septiembre 2026 a febrero 2027.
            season_start = self._season,
            season_end   = self._season + 1,
            date         = game.gameday,
            start_time   = _build_start_time(game.gameday, game.gametime),
            home_team_id = game.home_team,
            away_team_id = game.away_team,
            home_team    = self._team_name(game.home_team),
            away_team    = self._team_name(game.away_team),
            # nflverse no expone un venue_id estable, pero cada franquicia
            # juega en un estadio fijo por temporada — la abreviación del
            # equipo local es un proxy válido. NFLVenueFactors lo mapea a
            # coordenadas y tipo de techo.
            venue_id     = game.home_team,
            venue_name   = game.stadium or "",
            status       = status,
            provider_ids = {
                "nfl_game_id": game.game_id,
                "odds_api":    "",  # se mapea por nombres en Stage 5
            },
        )

    @staticmethod
    def _row_to_game_info(row) -> NFLGameInfo:
        """Convierte una fila del schedule a NFLGameInfo."""
        return NFLGameInfo(
            game_id    = str(getattr(row, "game_id", "")),
            season     = _safe_int(getattr(row, "season", 0)) or 0,
            week       = _safe_int(getattr(row, "week", 0)) or 0,
            game_type  = str(getattr(row, "game_type", GAME_TYPE_REGULAR)),
            gameday    = str(getattr(row, "gameday", "")),
            home_team  = _safe_str(getattr(row, "home_team", None)) or "",
            away_team  = _safe_str(getattr(row, "away_team", None)) or "",
            gametime   = _safe_str(getattr(row, "gametime", None)),
            weekday    = _safe_str(getattr(row, "weekday", None)),
            div_game   = bool(getattr(row, "div_game", 0) or 0),
            roof       = _safe_str(getattr(row, "roof", None)),
            surface    = _safe_str(getattr(row, "surface", None)),
            stadium    = _safe_str(getattr(row, "stadium", None)),
            home_rest  = _safe_int(getattr(row, "home_rest", None)),
            away_rest  = _safe_int(getattr(row, "away_rest", None)),
            home_score = _safe_float(getattr(row, "home_score", None)),
            away_score = _safe_float(getattr(row, "away_score", None)),
            overtime   = bool(getattr(row, "overtime", 0) or 0),
            # Clima observado
            temp = _safe_float(getattr(row, "temp", None)),
            wind = _safe_float(getattr(row, "wind", None)),
            # Líneas de cierre
            spread_line      = _safe_float(getattr(row, "spread_line", None)),
            total_line       = _safe_float(getattr(row, "total_line", None)),
            home_moneyline   = _safe_float(getattr(row, "home_moneyline", None)),
            away_moneyline   = _safe_float(getattr(row, "away_moneyline", None)),
            home_spread_odds = _safe_float(getattr(row, "home_spread_odds", None)),
            away_spread_odds = _safe_float(getattr(row, "away_spread_odds", None)),
            over_odds        = _safe_float(getattr(row, "over_odds", None)),
            under_odds       = _safe_float(getattr(row, "under_odds", None)),
        )

    # ── Nombres de equipo ─────────────────────────────────────────────────────

    def _team_name(self, abbr: str) -> str:
        """
        Nombre completo del equipo desde su abreviación.

        Necesario para el matching con The Odds API, que usa nombres
        completos ('Kansas City Chiefs') mientras nflverse usa
        abreviaciones ('KC').

        Si la tabla de equipos no carga, retorna la abreviación —
        degradación aceptable: el pipeline sigue funcionando, solo
        el display es menos legible.
        """
        if self._team_names is None:
            self._team_names = self._load_team_names()
        return self._team_names.get(abbr, abbr)

    def _load_team_names(self) -> dict[str, str]:
        """Carga el mapa abreviación → nombre completo."""
        try:
            df = self._source.load_team_descriptions()
        except Exception:
            return {}

        names: dict[str, str] = {}
        for row in df.itertuples(index=False):
            abbr = _safe_str(getattr(row, "team_abbr", None))
            name = _safe_str(getattr(row, "team_name", None))
            if abbr and name:
                names[abbr] = name
        return names


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _is_nan(value) -> bool:
    """True si el valor es NaN (pandas usa NaN para celdas vacías)."""
    try:
        return value != value  # NaN es el único valor != a sí mismo
    except Exception:
        return False


def _safe_str(value) -> str | None:
    """Convierte a str, retornando None para NaN o vacíos."""
    if value is None or _is_nan(value):
        return None
    s = str(value).strip()
    return s if s and s.lower() != "nan" else None


def _safe_float(value) -> float | None:
    """Convierte a float de forma segura (NaN → None)."""
    if value is None or _is_nan(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value) -> int | None:
    """Convierte a int de forma segura."""
    if value is None or _is_nan(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _build_start_time(date: str, gametime) -> str:
    """
    Construye start_time ISO-8601 UTC desde fecha y hora local.

    nflverse da `gametime` en hora del ESTE (ET), el estándar de la
    liga para horarios de kickoff. La conversión a UTC usa el offset
    de ET: -4 en horario de verano (septiembre-octubre), -5 en
    horario estándar (noviembre-febrero).

    Si gametime no está disponible, usa 17:00 UTC (13:00 ET) como
    aproximación del horario más común de kickoff dominical.
    """
    time_str = _safe_str(gametime)
    if not time_str:
        return f"{date}T17:00:00Z"

    try:
        hh, mm = time_str.split(":")[:2]
        hour   = int(hh)
        minute = int(mm)
    except (ValueError, IndexError):
        return f"{date}T17:00:00Z"

    # Offset de ET según el mes: DST termina el primer domingo de
    # noviembre, así que septiembre-octubre son EDT (-4) y
    # noviembre-febrero son EST (-5).
    try:
        month = int(date.split("-")[1])
    except (ValueError, IndexError):
        month = 10
    offset = 4 if month in (9, 10) else 5

    utc_hour = hour + offset
    day_shift = 0
    if utc_hour >= 24:
        utc_hour -= 24
        day_shift = 1

    if day_shift:
        try:
            d = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
            date = d.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return f"{date}T{utc_hour:02d}:{minute:02d}:00Z"