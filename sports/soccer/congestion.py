"""
sports/soccer/congestion.py

SoccerCongestionFetcher: congestión de calendario.

El factor que no existe en MLB ni NFL
--------------------------------------
En aquellos deportes un equipo juega UNA competición. El descanso entre
partidos es regular —diario en MLB, semanal en NFL— y lo único que
varía son las excepciones, que NFLRestFetcher ya cubría.

En fútbol un equipo puede jugar liga, copa nacional y competición
europea en la misma semana. Y el efecto no es uno sino dos, con
mecanismos distintos:

    FATIGA      Rendimiento físico reducido con menos de tres días de
                recuperación. Afecta sobre todo a la segunda parte, y
                se manifiesta más en la presión que en la posesión.

    ROTACIÓN    El cuerpo técnico reserva titulares para la
                competición que prioriza. Baja el nivel del once más
                de lo que lo baja la fatiga, y es menos predecible
                porque depende de una decisión, no de la fisiología.

Ambos reducen la media esperada de goles del equipo afectado, y por eso
las penalizaciones se expresan en GOLES sobre λ, no como multiplicador.

LIMITACIÓN DE ALCANCE — declarada, no disimulada
--------------------------------------------------
Con solo las cinco grandes ligas activas, este módulo SOLO VE LOS
PARTIDOS DE LIGA.

Un equipo que juega Champions el martes y liga el sábado aparece aquí
con siete días de descanso cuando realmente tuvo tres. La congestión
queda SUBESTIMADA, y de forma sesgada: afecta justo a los equipos que
compiten en Europa, que son los mejores de cada liga.

Consecuencia práctica: el modelo sobreestimará ligeramente a los
grandes en jornadas de competición europea. `coverage_warning` marca
los perfiles donde eso puede estar ocurriendo, para que el provider
pueda bajar `data_quality` y el modelo su confianza.

Activar la Champions en competitions.py cerraría este hueco sin tocar
este módulo: `get_team_matches` ya acepta varias competiciones.

Lo que importa es el DIFERENCIAL
----------------------------------
Igual que con el descanso en NFL: lo que mueve la línea no es la
congestión de un equipo sino la DIFERENCIA entre ambos. Dos equipos que
vienen de jugar entre semana están igualados; uno fresco contra uno
cargado, no.

`differential()` es el método que consume el modelo de proyección.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from sports.soccer.competitions import Competition, get_competition
from sports.soccer.schedule import SoccerMatchInfo
from sports.soccer.teams import canonical_team


# ── Dependencia inyectada ────────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """Interfaz mínima del calendario que este módulo consume."""

    def get_team_matches(
        self, team: str, comp, season: int,
        before: str | None = None, only_final: bool = True,
    ) -> list[SoccerMatchInfo]:
        ...


# ── Umbrales ─────────────────────────────────────────────────────────────────

# Días de descanso por debajo de los cuales hay fatiga medible.
#
# Tres es el umbral habitual en la literatura de rendimiento: por
# debajo, los marcadores de recuperación muscular no vuelven a la
# línea base. Un partido el sábado y otro el martes son tres días.
_SHORT_REST_DAYS = 3

# Descanso a partir del cual no hay penalización de ningún tipo.
_NORMAL_REST_DAYS = 6

# Carga alta: tres partidos en ocho días.
_HEAVY_LOAD_MATCHES = 3
_HEAVY_LOAD_WINDOW  = 8

# Penalizaciones por defecto, en GOLES. Coinciden con soccer.yaml.
_DEFAULT_SHORT_REST    = -0.12
_DEFAULT_MIDWEEK_EURO  = -0.08
_DEFAULT_HEAVY_LOAD    = -0.15
_DEFAULT_MAX_PENALTY   = -0.30

# Descanso por defecto cuando no hay partido previo (arranque de
# temporada). Se asume pretemporada completa, sin penalización.
_OPENER_REST_DAYS = 14


@dataclass(frozen=True)
class CongestionProfile:
    """
    Carga de calendario de un equipo antes de un partido.

    Campos
    ------
    team           -- Equipo en forma canónica.
    match_date     -- Fecha del partido a proyectar.
    days_rest      -- Días desde el partido anterior. None si no hay
                      partido previo conocido.
    previous_date  -- Fecha del último partido jugado.
    matches_7      -- Partidos en los 7 días previos.
    matches_14     -- Partidos en los 14 días previos.
    is_opener      -- True si es el primer partido registrado, donde el
                      descanso no discrimina entre equipos.
    coverage_warning
                   -- True si el perfil puede estar subestimando la
                      congestión real, porque el equipo juega
                      competiciones que el sistema no observa.
    """
    team:          str
    comp_id:       str
    match_date:    str
    days_rest:     int | None = None
    previous_date: str | None = None
    matches_7:     int = 0
    matches_14:    int = 0
    is_opener:     bool = False
    coverage_warning: bool = False

    # ── Clasificación ─────────────────────────────────────────────────────────

    @property
    def is_short_rest(self) -> bool:
        """True si viene de jugar con menos de tres días."""
        return (self.days_rest is not None
                and self.days_rest <= _SHORT_REST_DAYS
                and not self.is_opener)

    @property
    def is_heavy_load(self) -> bool:
        """True si acumula tres partidos en ocho días."""
        return self.matches_7 >= _HEAVY_LOAD_MATCHES - 1

    @property
    def is_fresh(self) -> bool:
        """True si llega con descanso pleno y sin acumulación."""
        return (self.days_rest is None
                or self.days_rest >= _NORMAL_REST_DAYS) and self.matches_7 <= 1

    @property
    def category(self) -> str:
        """Etiqueta legible de la carga."""
        if self.is_opener:
            return "opener"
        if self.is_short_rest and self.is_heavy_load:
            return "severe"
        if self.is_short_rest:
            return "short_rest"
        if self.is_heavy_load:
            return "heavy_load"
        if self.is_fresh:
            return "fresh"
        return "normal"

    # ── Penalización ──────────────────────────────────────────────────────────

    def penalty(
        self,
        short_rest:   float = _DEFAULT_SHORT_REST,
        heavy_load:   float = _DEFAULT_HEAVY_LOAD,
        midweek_euro: float = _DEFAULT_MIDWEEK_EURO,
        max_penalty:  float = _DEFAULT_MAX_PENALTY,
    ) -> float:
        """
        Penalización en GOLES sobre la media esperada del equipo.

        Los efectos se ACUMULAN porque son mecanismos distintos: la
        fatiga por descanso corto y la degradación del once por
        rotación se suman en un equipo que juega tres partidos en ocho
        días con dos de ellos seguidos.

        El techo (`max_penalty`) evita que la acumulación produzca una
        proyección irreal. Más allá de ese punto la incertidumbre sobre
        la alineación es tan alta que el partido no debería apostarse,
        y de eso se encarga la confianza del modelo, no una
        penalización mayor.

        La penalización europea NO se aplica todavía: requiere ver los
        partidos de Champions, que están fuera del alcance actual. El
        parámetro existe para cuando se activen.
        """
        if self.is_opener:
            return 0.0

        total = 0.0
        if self.is_short_rest:
            total += short_rest
        if self.is_heavy_load:
            total += heavy_load

        return round(max(total, max_penalty), 4)

    def to_metadata(self) -> dict:
        """Metadatos para TeamFeatures.sport_metadata."""
        return {
            "days_rest":            self.days_rest,
            "congestion_category":  self.category,
            "matches_last_7":       self.matches_7,
            "matches_last_14":      self.matches_14,
            "is_short_rest":        self.is_short_rest,
            "is_heavy_load":        self.is_heavy_load,
            "congestion_incomplete": self.coverage_warning,
        }


# ── Fetcher ──────────────────────────────────────────────────────────────────

class SoccerCongestionFetcher:
    """
    Calcula la carga de calendario desde el histórico de partidos.

    Parámetros
    ----------
    schedule_fetcher -- Fuente de partidos por equipo.
    config_loader    -- ConfigLoader con soccer.yaml.
    tracked_competitions
                     -- Competiciones que el sistema observa. Se usa
                        para decidir si un perfil lleva
                        `coverage_warning`: si solo se ven ligas, la
                        congestión de los equipos que juegan en Europa
                        queda subestimada.
    """

    def __init__(
        self,
        schedule_fetcher: ScheduleSource,
        config_loader     = None,
        tracked_competitions: list[str] | None = None,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._config = config_loader
        self._tracked = set(tracked_competitions or [])

        self._short_rest   = self._cfg("soccer.congestion.short_rest_penalty",
                                       _DEFAULT_SHORT_REST)
        self._heavy_load   = self._cfg("soccer.congestion.heavy_load_penalty",
                                       _DEFAULT_HEAVY_LOAD)
        self._midweek_euro = self._cfg("soccer.congestion.midweek_european_penalty",
                                       _DEFAULT_MIDWEEK_EURO)
        self._max_penalty  = self._cfg("soccer.congestion.max_congestion_penalty",
                                       _DEFAULT_MAX_PENALTY)

        self._cache: dict[tuple[str, str, str], CongestionProfile] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def fetch(
        self,
        team:       str,
        comp:       Competition | str,
        season:     int,
        match_date: str,
    ) -> CongestionProfile:
        """
        Perfil de congestión de un equipo antes de una fecha.

        Solo considera partidos ANTERIORES a `match_date`: es la misma
        barrera que en team_stats.py, y por la misma razón — un partido
        del mismo día no está jugado cuando se apuesta.

        Nunca lanza: sin datos devuelve un perfil neutro.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return CongestionProfile(team=team, comp_id=str(comp),
                                     match_date=match_date, is_opener=True)

        canon = canonical_team(team, competition.comp_id)
        key = (canon, competition.comp_id, match_date)
        if key in self._cache:
            return self._cache[key]

        profile = self._build(canon, competition, season, match_date)
        self._cache[key] = profile
        return profile

    def fetch_for_match(
        self,
        home:       str,
        away:       str,
        comp:       Competition | str,
        season:     int,
        match_date: str,
    ) -> tuple[CongestionProfile, CongestionProfile]:
        """Perfiles de ambos equipos de un partido."""
        return (
            self.fetch(home, comp, season, match_date),
            self.fetch(away, comp, season, match_date),
        )

    def differential(
        self,
        home:       str,
        away:       str,
        comp:       Competition | str,
        season:     int,
        match_date: str,
    ) -> float:
        """
        Diferencia neta de congestión, en goles, desde el local.

        Positivo favorece al local. Es el método que consume el modelo
        de proyección, por la misma razón que en NFL: lo que mueve la
        línea no es la carga de un equipo sino la DIFERENCIA.

        Dos equipos que vienen de jugar entre semana están igualados, y
        aplicar solo la penalización del local sesgaría todas las
        jornadas de competición europea.
        """
        home_profile, away_profile = self.fetch_for_match(
            home, away, comp, season, match_date
        )

        home_penalty = home_profile.penalty(
            self._short_rest, self._heavy_load,
            self._midweek_euro, self._max_penalty,
        )
        away_penalty = away_profile.penalty(
            self._short_rest, self._heavy_load,
            self._midweek_euro, self._max_penalty,
        )

        # Ambas penalizaciones son negativas o cero. La diferencia
        # positiva significa que el visitante está más cargado.
        return round(home_penalty - away_penalty, 4)

    def penalty_for(
        self,
        team:       str,
        comp:       Competition | str,
        season:     int,
        match_date: str,
    ) -> float:
        """Penalización individual con los valores del config."""
        profile = self.fetch(team, comp, season, match_date)
        return profile.penalty(
            self._short_rest, self._heavy_load,
            self._midweek_euro, self._max_penalty,
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── Construcción ──────────────────────────────────────────────────────────

    def _build(
        self,
        team:       str,
        comp:       Competition,
        season:     int,
        match_date: str,
    ) -> CongestionProfile:
        """Construye el perfil desde el histórico del equipo."""
        try:
            history = self._schedule.get_team_matches(
                team, comp, season, before=match_date, only_final=True,
            )
        except Exception:
            history = []

        if not isinstance(history, list) or not history:
            return CongestionProfile(
                team=team, comp_id=comp.comp_id, match_date=match_date,
                is_opener=True,
                coverage_warning=self._warns(comp),
            )

        ordered = sorted(history, key=lambda m: m.date)
        previous = ordered[-1]

        days_rest = _days_between(previous.date, match_date)

        return CongestionProfile(
            team=team, comp_id=comp.comp_id, match_date=match_date,
            days_rest=days_rest,
            previous_date=previous.date,
            matches_7=_count_within(ordered, match_date, 7),
            matches_14=_count_within(ordered, match_date, 14),
            is_opener=False,
            coverage_warning=self._warns(comp),
        )

    def _warns(self, comp: Competition) -> bool:
        """
        True si el perfil puede estar subestimando la congestión real.

        Ocurre cuando el sistema solo observa la liga del equipo: sus
        partidos de Champions, Europa League o copa nacional no están
        en el calendario, así que un martes europeo seguido de un
        sábado de liga aparece como siete días de descanso.

        El aviso viaja hasta sport_metadata para que el provider pueda
        reflejarlo en data_quality.
        """
        if not self._tracked:
            # Sin lista declarada no se puede afirmar nada; se avisa
            # por prudencia, que es el supuesto conservador.
            return True
        # Si solo se observan ligas nacionales, la cobertura es parcial
        # para cualquier equipo que pueda estar en competición europea.
        return not any(c in self._tracked for c in ("ucl", "uel", "uecl"))

    # ── Config ────────────────────────────────────────────────────────────────

    def _cfg(self, key: str, default: float) -> float:
        if self._config is None:
            return default
        try:
            value = self._config.get(key, default=default)
            return float(value) if value is not None else default
        except (ValueError, TypeError, AttributeError):
            return default


# ── Utilidades ───────────────────────────────────────────────────────────────

def _days_between(earlier: str, later: str) -> int | None:
    """Días entre dos fechas ISO. None si alguna es inválida."""
    try:
        a = datetime.strptime(earlier[:10], "%Y-%m-%d")
        b = datetime.strptime(later[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    delta = (b - a).days
    return delta if delta >= 0 else None


def _count_within(
    matches: list[SoccerMatchInfo],
    reference: str,
    window_days: int,
) -> int:
    """
    Partidos jugados dentro de una ventana anterior a la referencia.

    La ventana es estricta por ambos extremos: cuenta los partidos
    posteriores a `reference - window_days` y anteriores a `reference`.
    Un partido del propio día no se cuenta porque todavía no se ha
    jugado cuando se apuesta.
    """
    count = 0
    for match in matches:
        gap = _days_between(match.date, reference)
        if gap is not None and 0 < gap <= window_days:
            count += 1
    return count