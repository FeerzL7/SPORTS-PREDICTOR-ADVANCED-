"""
sports/soccer/schedule.py

SoccerScheduleFetcher: calendario multicompetición.

Responsabilidad
----------------
Convertir los MatchRow de football-data.co.uk en objetos Event del
contrato del Core, y exponer los metadatos que el resto del plugin
necesita: cuotas de cierre, xG del partido y forma reciente.

Diferencia estructural con MLB y NFL
--------------------------------------
Aquellos plugins cubren UNA competición. `get_events(date)` consulta
una fuente y devuelve los partidos del día.

Aquí hay cinco ligas simultáneas, y un sábado de temporada puede tener
veinte partidos repartidos entre ellas. `get_events(date)` itera sobre
las competiciones activas del registro y agrega el resultado.

Esa iteración es lo que hace que añadir la Champions o Liga MX sea
cambiar un flag en competitions.py, no tocar este módulo.

Identificador de partido
--------------------------
football-data no publica IDs de partido, así que hay que construirlos.
El formato es:

    {comp}_{fecha}_{local_canónico}_{visitante_canónico}

Se usan los nombres CANÓNICOS, no los de la fuente. Eso hace que el id
sea estable aunque football-data cambie la grafía de un equipo entre
temporadas — algo que ya ha hecho: "Nott'm Forest" aparece también
como "Nottm Forest" según el año.

Un id inestable rompería la liquidación: el pick registrado en el
ledger apuntaría a un partido que el settlement no encuentra.

Todos los campos desde el principio
-------------------------------------
SoccerMatchInfo lleva las cuotas de cierre, el marcador al descanso y
el xG desde su primera versión.

En el plugin NFL, NFLGameInfo se diseñó sin los campos de mercado ni
el clima observado. Los consumidores los leían con `getattr(x, "y",
None)`, que devolvía None en silencio: el backtest produjo cero picks
en dos ejecuciones completas antes de que se localizara la causa.
Aquí el dataclass incluye todo lo que las fuentes ofrecen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from core.contracts.event import Event, EventStatus

from sports.soccer.competitions import (
    Competition, enabled_competitions, get_competition,
)
from sports.soccer.data_source import MatchRow, current_soccer_season
from sports.soccer.teams import canonical_team, display_name


# ── Dependencia inyectada ────────────────────────────────────────────────────

@runtime_checkable
class MatchSource(Protocol):
    """
    Interfaz mínima de la fuente de partidos.

    Mismo patrón que ScheduleSource en los módulos de NFL: declarar
    solo lo que se usa documenta la dependencia real y evita que el
    type checker propague `Unknown` desde una dependencia sin anotar.
    """

    def load_matches(self, comp, season: int) -> list[MatchRow]:
        """Partidos de una competición y temporada."""
        ...

    def load_match_xg(self, comp, season: int) -> list:
        """xG partido a partido, o lista vacía si no hay backend."""
        ...


# ── Metadatos del partido ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SoccerMatchInfo:
    """
    Un partido con todo lo que las fuentes ofrecen.

    Campos de identidad
    -------------------
    match_id   -- Id construido, estable entre temporadas.
    comp_id    -- Competición.
    season     -- Temporada por su año de inicio.
    date       -- Fecha ISO.
    home / away        -- Nombres canónicos, para cruzar fuentes.
    home_display / away_display  -- Grafía original, para mostrar.

    Resultado
    ---------
    home_goals / away_goals  -- Al final del tiempo reglamentario.
    ht_home / ht_away        -- Al descanso. En fútbol los mercados de
                                primera parte SÍ son liquidables, a
                                diferencia de los H1 de NFL donde no
                                teníamos el marcador parcial.

    Cuotas de cierre
    ----------------
    Las de Pinnacle cuando están; Bet365 como respaldo. `odds_source`
    registra cuál se usó.

    xG
    --
    Se rellena cruzando con Understat. None significa que no hay dato,
    no que sea cero — la distinción importa para data_quality.
    """
    match_id:     str
    comp_id:      str
    season:       int
    date:         str
    home:         str
    away:         str
    home_display: str = ""
    away_display: str = ""

    home_goals: int | None = None
    away_goals: int | None = None
    ht_home:    int | None = None
    ht_away:    int | None = None

    odds_home:  float | None = None
    odds_draw:  float | None = None
    odds_away:  float | None = None
    odds_over:  float | None = None
    odds_under: float | None = None
    ah_line:    float | None = None
    ah_home:    float | None = None
    ah_away:    float | None = None
    odds_source: str = ""

    # Cuotas de APERTURA. La diferencia con el cierre es el margen
    # accesible: el pipeline opera con precios de apertura y media
    # semana, no con el cierre.
    open_home:  float | None = None
    open_draw:  float | None = None
    open_away:  float | None = None
    open_over:  float | None = None
    open_under: float | None = None
    open_source: str = ""

    home_xg:   float | None = None
    away_xg:   float | None = None
    home_npxg: float | None = None
    away_npxg: float | None = None

    # ── Estado ────────────────────────────────────────────────────────────────

    @property
    def is_final(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> str | None:
        """'H', 'D' o 'A'."""
        if not self.is_final:
            return None
        h, a = self.home_goals or 0, self.away_goals or 0
        return "H" if h > a else ("A" if a > h else "D")

    @property
    def total_goals(self) -> int | None:
        if not self.is_final:
            return None
        return (self.home_goals or 0) + (self.away_goals or 0)

    @property
    def btts(self) -> bool | None:
        """True si ambos equipos marcaron."""
        if not self.is_final:
            return None
        return (self.home_goals or 0) > 0 and (self.away_goals or 0) > 0

    @property
    def has_opening_odds(self) -> bool:
        """True si el 1X2 de apertura está completo."""
        return all(o is not None for o in
                   (self.open_home, self.open_draw, self.open_away))

    @property
    def line_movement(self) -> float | None:
        """
        Movimiento del 1X2 local, de apertura a cierre, en porcentaje.

        Positivo: la cuota subió y el mercado se movió CONTRA el local.
        Es la magnitud que el CLV mide en producción.
        """
        if not self.open_home or not self.odds_home:
            return None
        return round((self.odds_home / self.open_home - 1.0) * 100.0, 3)

    @property
    def has_closing_odds(self) -> bool:
        """
        True si el 1X2 de cierre está completo.

        Se exigen las tres cuotas: una incompleta no permite calcular
        las probabilidades implícitas sin vig, que es lo que consume el
        blending.
        """
        return all(o is not None for o in
                   (self.odds_home, self.odds_draw, self.odds_away))

    @property
    def has_xg(self) -> bool:
        return self.home_xg is not None and self.away_xg is not None

    @property
    def total_xg(self) -> float | None:
        if not self.has_xg:
            return None
        return round((self.home_xg or 0.0) + (self.away_xg or 0.0), 3)

    def to_metadata(self) -> dict:
        """Metadatos para TeamFeatures.sport_metadata."""
        return {
            "match_id":     self.match_id,
            "competition":  self.comp_id,
            "season":       self.season,
            "closing_1x2":  [self.odds_home, self.odds_draw, self.odds_away],
            "odds_source":  self.odds_source,
            "has_xg":       self.has_xg,
            "home_xg":      self.home_xg,
            "away_xg":      self.away_xg,
            "ht_score":     [self.ht_home, self.ht_away],
        }


def build_match_id(comp_id: str, date: str, home: str, away: str) -> str:
    """
    Id estable de partido.

    Usa los nombres canónicos y sustituye los espacios por guiones, de
    forma que el id sea legible y utilizable como clave de fichero o de
    URL sin escapado.
    """
    h = canonical_team(home, comp_id).replace(" ", "-")
    a = canonical_team(away, comp_id).replace(" ", "-")
    return f"{comp_id}_{date}_{h}_{a}"


# ── Fetcher ──────────────────────────────────────────────────────────────────

class SoccerScheduleFetcher:
    """
    Calendario de las competiciones activas.

    Parámetros
    ----------
    data_source   -- Fuente de partidos. Si None, crea un
                     SoccerDataSource.
    competitions  -- Competiciones a cubrir. None usa las activas del
                     registro, que es lo que hace el pipeline.
    season        -- Temporada. None la deduce de cada fecha.
    """

    def __init__(
        self,
        data_source           = None,
        competitions:  list[Competition] | None = None,
        season:        int | None = None,
    ) -> None:
        if data_source is None:
            from sports.soccer.data_source import SoccerDataSource
            data_source = SoccerDataSource()
        self._source: MatchSource = data_source

        self._competitions = (competitions if competitions is not None
                              else enabled_competitions())
        self._season = season

        # Caché por (competición, temporada)
        self._matches: dict[tuple[str, int], list[SoccerMatchInfo]] = {}

    # ── SportDataProvider: Stage 1 ────────────────────────────────────────────

    def get_events(self, date: str) -> list[Event]:
        """
        Partidos de la fecha en TODAS las competiciones activas.

        A diferencia de MLB y NFL, aquí se consultan varias fuentes por
        llamada: un sábado de temporada puede tener veinte partidos
        repartidos entre las cinco ligas.

        El orden es el de declaración de las competiciones, no el
        horario: el pipeline no depende del orden y mantenerlo estable
        hace los logs comparables entre ejecuciones.
        """
        events: list[Event] = []
        season = self._season or _season_for_date(date)

        for comp in self._competitions:
            for match in self._competition_matches(comp, season):
                if match.date != date:
                    continue
                event = self._to_event(match, comp)
                if event is not None:
                    events.append(event)

        return events

    def get_match_info(self, match_id: str) -> SoccerMatchInfo | None:
        """
        Metadatos de un partido por su id.

        Busca en las competiciones activas. El comp_id va incrustado en
        el id, así que se extrae para no recorrerlas todas.
        """
        comp_id = match_id.split("_", 1)[0] if "_" in match_id else ""
        comp = get_competition(comp_id)
        comps = [comp] if comp else self._competitions

        season = self._season or _season_from_match_id(match_id)

        for competition in comps:
            if competition is None:
                continue
            for match in self._competition_matches(competition, season):
                if match.match_id == match_id:
                    return match
        return None

    # ── Consultas para el modelo ──────────────────────────────────────────────

    def get_team_matches(
        self,
        team:       str,
        comp:       Competition | str,
        season:     int,
        before:     str | None = None,
        only_final: bool = True,
    ) -> list[SoccerMatchInfo]:
        """
        Partidos de un equipo, ordenados por fecha.

        Parámetros
        ----------
        before     -- Si se indica, solo partidos ANTERIORES a esa
                      fecha. Es la barrera contra el look-ahead bias:
                      para proyectar un partido del 15 de marzo, el
                      modelo solo puede ver lo jugado antes.
        only_final -- Solo partidos con marcador.

        La barrera se aplica aquí y no en el llamador por la misma
        razón que `_cutoff_week` vive en NFLDataProvider: un filtro de
        fecha suelto en medio del ensamblaje es indistinguible de un
        off-by-one, y el modo de fallo no lanza excepción — produce un
        backtest inflado que no se reproduce en producción.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return []

        canon = canonical_team(team, competition.comp_id)
        if not canon:
            return []

        result = []
        for match in self._competition_matches(competition, season):
            if canon not in (match.home, match.away):
                continue
            if before and match.date >= before:
                continue
            if only_final and not match.is_final:
                continue
            result.append(match)

        return sorted(result, key=lambda m: m.date)

    def get_competition_matches(
        self,
        comp:   Competition | str,
        season: int,
    ) -> list[SoccerMatchInfo]:
        """Todos los partidos de una competición y temporada."""
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return []
        return self._competition_matches(competition, season)

    def teams_in_competition(
        self,
        comp:   Competition | str,
        season: int,
    ) -> list[str]:
        """Equipos que aparecen en el calendario, en forma canónica."""
        matches = self.get_competition_matches(comp, season)
        return sorted({m.home for m in matches} | {m.away for m in matches})

    # ── Carga y cruce de fuentes ──────────────────────────────────────────────

    def _competition_matches(
        self,
        comp:   Competition,
        season: int,
    ) -> list[SoccerMatchInfo]:
        """
        Partidos de una competición, con el xG ya cruzado.

        El cruce ocurre una sola vez por competición y temporada, y
        queda cacheado. Hacerlo en cada consulta multiplicaría el coste
        por el número de equipos.
        """
        key = (comp.comp_id, season)
        if key in self._matches:
            return self._matches[key]

        try:
            rows = self._source.load_matches(comp, season)
        except Exception:
            rows = []

        matches = [self._to_match_info(row, comp) for row in rows]
        matches = [m for m in matches if m is not None]

        matches = self._attach_xg(matches, comp, season)

        self._matches[key] = matches
        return matches

    @staticmethod
    def _to_match_info(
        row:  MatchRow,
        comp: Competition,
    ) -> SoccerMatchInfo | None:
        """Convierte un MatchRow en SoccerMatchInfo con nombres canónicos."""
        home = canonical_team(row.home_team, comp.comp_id)
        away = canonical_team(row.away_team, comp.comp_id)
        if not home or not away or not row.date:
            return None

        return SoccerMatchInfo(
            match_id     = build_match_id(comp.comp_id, row.date,
                                          row.home_team, row.away_team),
            comp_id      = comp.comp_id,
            season       = row.season,
            date         = row.date,
            home         = home,
            away         = away,
            home_display = display_name(row.home_team),
            away_display = display_name(row.away_team),
            home_goals   = row.home_goals,
            away_goals   = row.away_goals,
            ht_home      = row.ht_home,
            ht_away      = row.ht_away,
            odds_home    = row.odds_home,
            odds_draw    = row.odds_draw,
            odds_away    = row.odds_away,
            odds_over    = row.odds_over,
            odds_under   = row.odds_under,
            ah_line      = row.ah_line,
            ah_home      = row.ah_home,
            ah_away      = row.ah_away,
            odds_source  = row.odds_source,
            open_home    = row.open_home,
            open_draw    = row.open_draw,
            open_away    = row.open_away,
            open_over    = row.open_over,
            open_under   = row.open_under,
            open_source  = row.open_source,
        )

    def _attach_xg(
        self,
        matches: list[SoccerMatchInfo],
        comp:    Competition,
        season:  int,
    ) -> list[SoccerMatchInfo]:
        """
        Cruza el xG de Understat con los partidos de football-data.

        El cruce va por (fecha, equipo canónico). Sin la reconciliación
        de teams.py fallaría en silencio: "Man City" nunca encontraría
        su fila de "Manchester City", el xG quedaría vacío y el modelo
        proyectaría con la media de liga creyendo tener datos.

        Si no hay backend de xG, los partidos se devuelven sin tocar y
        con `has_xg` en False. Esa ausencia la recoge
        SoccerDataSource.effective_tier() y de ahí baja data_quality.
        """
        try:
            xg_rows = self._source.load_match_xg(comp, season)
        except Exception:
            xg_rows = []

        if not xg_rows:
            return matches

        # Índice por (fecha, equipo canónico)
        index: dict[tuple[str, str], object] = {}
        for row in xg_rows:
            team = canonical_team(getattr(row, "team", ""), comp.comp_id)
            date = str(getattr(row, "date", ""))
            if team and date:
                index[(date, team)] = row

        enriched: list[SoccerMatchInfo] = []
        for match in matches:
            home_row = index.get((match.date, match.home))
            away_row = index.get((match.date, match.away))

            if home_row is None and away_row is None:
                enriched.append(match)
                continue

            # La fila del local trae su xG y el del rival como xGA, así
            # que basta una de las dos para completar el partido. Se
            # prefiere la del local por consistencia.
            source = home_row if home_row is not None else away_row
            if home_row is not None:
                home_xg  = _num(getattr(source, "xg", None))
                away_xg  = _num(getattr(source, "xga", None))
                home_np  = _num(getattr(source, "npxg", None))
                away_np  = _num(getattr(source, "npxga", None))
            else:
                # Solo hay fila del visitante: sus campos van invertidos
                home_xg  = _num(getattr(source, "xga", None))
                away_xg  = _num(getattr(source, "xg", None))
                home_np  = _num(getattr(source, "npxga", None))
                away_np  = _num(getattr(source, "npxg", None))

            enriched.append(SoccerMatchInfo(
                **{**vars(match),
                   "home_xg": home_xg, "away_xg": away_xg,
                   "home_npxg": home_np, "away_npxg": away_np}
            ))

        return enriched

    # ── Conversión a Event ────────────────────────────────────────────────────

    @staticmethod
    def _to_event(match: SoccerMatchInfo, comp: Competition) -> Event | None:
        """
        Convierte a Event del contrato del Core.

        Sobre season_start y season_end
        --------------------------------
        Las ligas europeas cruzan el año natural: la temporada 2024 va
        de agosto de 2024 a mayo de 2025. La misma convención que
        aplicamos en NFL.

        Sobre start_time
        -----------------
        football-data publica la hora en algunas temporadas y en otras
        no, y cuando la publica es hora local del estadio sin indicar
        el huso. Ponerla en UTC sería inventar precisión, así que se usa
        mediodía UTC como marcador de fecha. El pipeline solo lo usa
        para ordenar y para saber si el partido ya empezó.
        """
        if not match.home or not match.away:
            return None

        status = EventStatus.FINAL if match.is_final else EventStatus.SCHEDULED

        return Event(
            event_id     = match.match_id,
            sport        = "soccer",
            league       = comp.name,
            season_start = match.season,
            season_end   = (match.season + 1
                            if comp.season_type == "split_year"
                            else match.season),
            date         = match.date,
            start_time   = f"{match.date}T12:00:00Z",
            home_team_id = match.home,
            away_team_id = match.away,
            home_team    = match.home_display or match.home,
            away_team    = match.away_display or match.away,
            venue_id     = match.home,
            venue_name   = "",
            status       = status,
            provider_ids = {
                "match_id":    match.match_id,
                "competition": comp.comp_id,
                "odds_api":    comp.odds_api_key,
            },
        )

    def clear_cache(self) -> None:
        self._matches.clear()


# ── Utilidades ───────────────────────────────────────────────────────────────

def _season_for_date(date: str) -> int:
    """
    Temporada europea que contiene una fecha.

    Las ligas van de agosto a mayo, así que el corte se pone en julio:
    una fecha de marzo pertenece a la temporada que empezó el agosto
    anterior.
    """
    try:
        parsed = datetime.strptime(date[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return current_soccer_season()
    return parsed.year if parsed.month >= 7 else parsed.year - 1


def _season_from_match_id(match_id: str) -> int:
    """Temporada deducida de la fecha incrustada en el id."""
    parts = match_id.split("_")
    if len(parts) >= 2:
        return _season_for_date(parts[1])
    return current_soccer_season()


def _num(value) -> float | None:
    """Convierte a float, tratando NaN como ausencia."""
    if value is None:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if result == result else None