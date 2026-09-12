"""
sports/nfl/rest.py

NFLRestFetcher: días de descanso, bye weeks y semanas cortas.

Por qué el descanso es feature de primer orden en NFL
-------------------------------------------------------
En MLB los equipos juegan casi a diario y el descanso es ruido: la
diferencia entre jugar ayer o anteayer no cambia la proyección. En NFL
el intervalo entre partidos oscila entre 4 y 14 días, y esa diferencia
tiene efecto medible:

    Semana corta (4 días, Thursday Night tras domingo)
        El cuerpo técnico tiene un día de preparación en vez de tres y
        los jugadores no completan el ciclo de recuperación física.

    Descanso largo (10 días, domingo tras Thursday Night)
        Mini-bye: semana extra de preparación sin la pérdida de ritmo
        competitivo que trae el bye completo.

    Bye week (13-14 días)
        El efecto mejor documentado y más estable históricamente del
        calendario NFL. Semana completa de preparación específica más
        recuperación de lesiones menores.

Separación de responsabilidades con schedule.py
-------------------------------------------------
Este módulo no descarga datos nuevos. NFLScheduleFetcher ya expone
`home_rest`, `away_rest` y las bye weeks; aquí se interpretan.

    schedule.py  → "home_rest = 4"            (hecho del calendario)
    rest.py      → "4 días = semana corta,     (interpretación
                    penalización -1.0 puntos"   deportiva)

Mezclar ambas capas acoplaría los hechos con su lectura, y cualquier
recalibración de las penalizaciones obligaría a tocar el módulo que
construye los Event del pipeline.

El diferencial importa más que el valor absoluto
--------------------------------------------------
Lo que mueve la línea no es el descanso de un equipo sino la DIFERENCIA
entre ambos. Dos equipos jugando con 4 días cada uno están igualados; un
equipo con 10 días contra otro con 4 tiene ventaja real.

Por eso el método principal para el modelo es `differential()`, que
devuelve el ajuste neto en puntos desde la perspectiva del local. Los
ajustes individuales quedan disponibles para trazabilidad.

Lo que este módulo NO cubre
-----------------------------
El efecto de VIAJE — un equipo de la costa oeste jugando a las 13:00 ET
tiene el reloj biológico en las 10:00 — es un factor documentado pero
requiere coordenadas y husos horarios de los estadios. Corresponde a
NFLVenueFactors (tarea 10.7) y NFLContextFetcher (10.8), no aquí.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from sports.nfl.schedule import NFLGameInfo


# ── Contrato de la dependencia ───────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """
    Interfaz mínima que este módulo necesita del calendario.

    Se declara como Protocol en vez de importar NFLScheduleFetcher
    directamente por dos motivos:

    1. TIPADO. Sin anotación, `schedule_fetcher` es `Unknown` para el
       type checker y todo lo que devuelve se propaga como `Unknown`
       por los cuatro puntos donde se consume. Eso ocultaba errores
       reales: `get_bye_weeks()` podía devolver None y el checker no
       podía avisarlo porque no sabía nada del tipo de retorno.

    2. ACOPLAMIENTO. rest.py solo usa tres métodos del calendario.
       Declarar exactamente esos tres documenta la dependencia real y
       permite inyectar dobles de prueba sin arrastrar toda la
       implementación de NFLScheduleFetcher.

    Es el mismo patrón que core/pipeline/stage.py aplica para los
    contratos entre el Core y los sport plugins.
    """

    def get_game_info(self, game_id: str) -> NFLGameInfo | None:
        """Metadatos de un partido, o None si no existe."""
        ...

    def get_bye_weeks(self) -> dict[str, int]:
        """Semana de bye de cada equipo: {team_abbr: week}."""
        ...

    def get_team_schedule(self, team: str) -> list[NFLGameInfo]:
        """Partidos de un equipo ordenados por fecha."""
        ...


# ── Umbrales de categorización ───────────────────────────────────────────────
#
# Los intervalos reales del calendario NFL:
#
#     4 días   domingo → jueves (Thursday Night Football)
#     5 días   lunes   → sábado
#     6 días   lunes   → domingo
#     7 días   domingo → domingo (el caso estándar)
#     8 días   sábado  → domingo
#    10 días   jueves  → domingo (mini-bye)
#    13-14 días bye week
#
_SHORT_WEEK_MAX: int = 5    # <= 5 días: semana corta
_NORMAL_MIN:     int = 6    # 6-8 días: normal
_NORMAL_MAX:     int = 8
_LONG_REST_MIN:  int = 9    # 9-12 días: descanso largo (mini-bye)
_BYE_MIN:        int = 13   # >= 13 días: viene de bye week

# Categorías
CATEGORY_SHORT   = "short"    # semana corta — desventaja
CATEGORY_NORMAL  = "normal"   # descanso estándar
CATEGORY_LONG    = "long"     # mini-bye — ventaja moderada
CATEGORY_BYE     = "bye"      # bye week — ventaja mayor
CATEGORY_OPENER  = "opener"   # semana 1: sin partido previo
CATEGORY_UNKNOWN = "unknown"  # sin datos

# Ajustes por defecto si no hay ConfigLoader inyectado.
# Coinciden con los valores documentados en config/nfl.yaml.
_DEFAULT_BYE_BONUS:    float = 1.5
_DEFAULT_SHORT_PENALTY: float = -1.0
_DEFAULT_LONG_BONUS:   float = 0.5


@dataclass(frozen=True)
class RestProfile:
    """
    Perfil de descanso de un equipo para un partido concreto.

    Campos
    ------
    team       -- Abreviación del equipo.
    game_id    -- Partido al que corresponde este perfil.
    week       -- Semana del partido.
    rest_days  -- Días desde el partido anterior. None si no se
                  pudo determinar (semana 1 o datos ausentes).
    category   -- 'short', 'normal', 'long', 'bye', 'opener', 'unknown'.
    is_off_bye -- True si el equipo viene de su semana de descanso.
                  Se determina cruzando con el calendario de byes, no
                  solo por días transcurridos: un equipo puede tener 13
                  días por reprogramación sin haber tenido bye.
    """
    team:       str
    game_id:    str
    week:       int
    rest_days:  int | None = None
    category:   str = CATEGORY_UNKNOWN
    is_off_bye: bool = False

    @property
    def is_short_week(self) -> bool:
        """True si juega con menos descanso del habitual."""
        return self.category == CATEGORY_SHORT

    @property
    def is_long_rest(self) -> bool:
        """True si tuvo descanso extra sin llegar a bye completo."""
        return self.category == CATEGORY_LONG

    def adjustment(
        self,
        bye_bonus:      float = _DEFAULT_BYE_BONUS,
        short_penalty:  float = _DEFAULT_SHORT_PENALTY,
        long_bonus:     float = _DEFAULT_LONG_BONUS,
    ) -> float:
        """
        Ajuste en puntos por el descanso de este equipo.

        Positivo = ventaja. Los valores vienen calibrados desde
        config/nfl.yaml (sección nfl.rest).

        La semana 1 no recibe ajuste: todos los equipos llegan del
        mismo periodo de pretemporada, así que el descanso no
        discrimina entre ellos.
        """
        if self.category == CATEGORY_BYE or self.is_off_bye:
            return bye_bonus
        if self.category == CATEGORY_LONG:
            return long_bonus
        if self.category == CATEGORY_SHORT:
            return short_penalty
        return 0.0

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "rest_days":     self.rest_days,
            "rest_category": self.category,
            "is_off_bye":    self.is_off_bye,
            "is_short_week": self.is_short_week,
        }


class NFLRestFetcher:
    """
    Calcula perfiles de descanso desde el calendario.

    Parámetros
    ----------
    schedule_fetcher -- NFLScheduleFetcher. Obligatorio: este módulo
                        interpreta sus datos, no descarga los propios.
    config_loader    -- ConfigLoader con nfl.yaml, para leer los
                        ajustes calibrados de la sección nfl.rest.
    """

    def __init__(
        self,
        schedule_fetcher: ScheduleSource,
        config_loader = None,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._config   = config_loader

        self._bye_bonus     = self._cfg("nfl.rest.bye_week_bonus",      _DEFAULT_BYE_BONUS)
        self._short_penalty = self._cfg("nfl.rest.short_week_penalty",  _DEFAULT_SHORT_PENALTY)
        self._long_bonus    = self._cfg("nfl.rest.long_rest_bonus",     _DEFAULT_LONG_BONUS)

        # Caché del calendario de byes: se calcula una vez por temporada
        self._byes: dict[str, int] | None = None

    # ── API pública ───────────────────────────────────────────────────────────

    def fetch_for_game(
        self,
        game_id: str,
    ) -> tuple[RestProfile | None, RestProfile | None]:
        """
        Perfiles de descanso de ambos equipos de un partido.

        Retorna (local, visitante). Cualquiera puede ser None si el
        partido no existe en el calendario.
        """
        game = self._schedule.get_game_info(game_id)
        if game is None:
            return None, None

        home = self._build_profile(
            team      = game.home_team,
            game      = game,
            rest_days = game.home_rest,
        )
        away = self._build_profile(
            team      = game.away_team,
            game      = game,
            rest_days = game.away_rest,
        )
        return home, away

    def fetch(self, team: str, game_id: str) -> RestProfile | None:
        """Perfil de descanso de un equipo concreto en un partido."""
        game = self._schedule.get_game_info(game_id)
        if game is None:
            return None

        if team == game.home_team:
            rest = game.home_rest
        elif team == game.away_team:
            rest = game.away_rest
        else:
            return None

        return self._build_profile(team=team, game=game, rest_days=rest)

    def differential(self, game_id: str) -> float:
        """
        Ajuste NETO en puntos desde la perspectiva del equipo local.

        Este es el método que consume NFLProjectionModel: lo que mueve
        la línea es la diferencia de descanso entre los dos equipos, no
        el valor absoluto de ninguno.

        Ejemplos:
            Local sale de bye (+1.5), visitante normal (0.0)  → +1.5
            Local semana corta (-1.0), visitante normal (0.0) → -1.0
            Ambos en semana corta (-1.0 cada uno)             →  0.0
            Local bye (+1.5), visitante semana corta (-1.0)   → +2.5

        El tercer caso es el importante: dos equipos igualmente
        perjudicados no generan ventaja para ninguno. Un modelo que
        aplicara solo el ajuste absoluto del local sesgaría la
        proyección en los Thursday Night, donde AMBOS equipos llegan
        con cuatro días.
        """
        home, away = self.fetch_for_game(game_id)
        if home is None or away is None:
            return 0.0

        home_adj = home.adjustment(
            self._bye_bonus, self._short_penalty, self._long_bonus
        )
        away_adj = away.adjustment(
            self._bye_bonus, self._short_penalty, self._long_bonus
        )
        return round(home_adj - away_adj, 3)

    def adjustment_for(self, team: str, game_id: str) -> float:
        """
        Ajuste individual de un equipo, con los valores del config.

        Método de conveniencia para que el modelo de proyección no
        necesite conocer los parámetros de calibración.
        """
        profile = self.fetch(team, game_id)
        if profile is None:
            return 0.0
        return profile.adjustment(
            self._bye_bonus, self._short_penalty, self._long_bonus
        )

    def clear_cache(self) -> None:
        """Limpia el caché del calendario de byes."""
        self._byes = None

    # ── Construcción de perfiles ──────────────────────────────────────────────

    def _build_profile(
        self,
        team:      str,
        game:      NFLGameInfo,
        rest_days: int | None,
    ) -> RestProfile:
        """
        Construye el perfil interpretando los días de descanso.

        Si nflverse no trae el dato (columna ausente o NaN), lo calcula
        desde las fechas del calendario del propio equipo. Esa ruta de
        respaldo importa porque `home_rest`/`away_rest` no están
        presentes en todas las temporadas históricas, y sin ella el
        backtesting perdería esta señal en los años más antiguos.
        """
        if rest_days is None:
            rest_days = self._compute_rest_from_schedule(team, game)

        is_off_bye = self._is_off_bye(team, game.week)
        category   = self._categorize(rest_days, game.week, is_off_bye)

        return RestProfile(
            team       = team,
            game_id    = game.game_id,
            week       = game.week,
            rest_days  = rest_days,
            category   = category,
            is_off_bye = is_off_bye,
        )

    @staticmethod
    def _categorize(
        rest_days:  int | None,
        week:       int,
        is_off_bye: bool,
    ) -> str:
        """
        Clasifica el descanso en una categoría.

        La semana 1 es un caso aparte: no hay partido previo, así que
        los días transcurridos desde la pretemporada no significan
        nada. Todos los equipos llegan igual de descansados.
        """
        if week <= 1:
            return CATEGORY_OPENER
        if is_off_bye:
            return CATEGORY_BYE
        if rest_days is None:
            return CATEGORY_UNKNOWN
        if rest_days >= _BYE_MIN:
            return CATEGORY_BYE
        if rest_days >= _LONG_REST_MIN:
            return CATEGORY_LONG
        if rest_days <= _SHORT_WEEK_MAX:
            return CATEGORY_SHORT
        return CATEGORY_NORMAL

    def _is_off_bye(self, team: str, week: int) -> bool:
        """
        True si el equipo viene de su semana de descanso.

        Se cruza con el calendario de byes en vez de deducirlo solo de
        los días transcurridos. Un equipo puede acumular 13 días por
        una reprogramación (partido movido por clima o por protocolo
        sanitario) sin haber tenido bye — en ese caso descansó, pero
        no tuvo la semana de preparación específica que hace valioso
        al bye real.
        """
        byes = self._bye_weeks()
        bye_week = byes.get(team)
        return bye_week is not None and bye_week == week - 1

    def _bye_weeks(self) -> dict[str, int]:
        """
        Calendario de byes, cacheado.

        El narrowing se hace sobre una variable local y no sobre el
        atributo: el type checker no puede garantizar que `self._byes`
        siga siendo no-None entre la asignación y el return, porque un
        atributo es mutable desde fuera del método. Con una local, la
        garantía es estructural.

        El `isinstance` no es decoración para el checker: get_bye_weeks()
        viene de una dependencia inyectada, y un doble de prueba mal
        construido o una versión futura del calendario podrían devolver
        None. Validarlo aquí evita que ese None se propague hasta
        `_is_off_bye()` y reviente con AttributeError en mitad del
        pipeline.
        """
        cached = self._byes
        if cached is not None:
            return cached

        try:
            result = self._schedule.get_bye_weeks()
        except Exception:
            result = {}

        byes: dict[str, int] = result if isinstance(result, dict) else {}
        self._byes = byes
        return byes

    def _compute_rest_from_schedule(
        self,
        team: str,
        game: NFLGameInfo,
    ) -> int | None:
        """
        Calcula los días de descanso desde las fechas del calendario.

        Ruta de respaldo cuando nflverse no expone home_rest/away_rest.
        Busca el partido anterior del equipo y resta las fechas.
        """
        try:
            schedule = self._schedule.get_team_schedule(team)
        except Exception:
            return None

        if not schedule:
            return None

        previous = None
        for g in schedule:
            if g.game_id == game.game_id:
                break
            if g.gameday:
                previous = g

        if previous is None or not previous.gameday or not game.gameday:
            return None

        try:
            d_prev = datetime.strptime(previous.gameday, "%Y-%m-%d")
            d_curr = datetime.strptime(game.gameday, "%Y-%m-%d")
        except ValueError:
            return None

        delta = (d_curr - d_prev).days
        return delta if delta > 0 else None

    # ── Config ────────────────────────────────────────────────────────────────

    def _cfg(self, key: str, default: float) -> float:
        """Lee un valor del ConfigLoader con fallback documentado."""
        if self._config is None:
            return default
        try:
            val = self._config.get(key, default=default)
            return float(val) if val is not None else default
        except (ValueError, TypeError, AttributeError):
            return default