"""
sports/soccer/context.py

SoccerContextFetcher: contexto situacional de partidos de fútbol.
Implementa SportDataProvider.get_context(event) → dict.

Por qué aquí NO hay maquinaria climática
------------------------------------------
El plugin NFL dedica 632 líneas al contexto, y la mayor parte es clima:
dos fuentes de datos, curvas de ajuste por viento y temperatura,
cortocircuito en estadios cubiertos. Está justificado — un viento de
25 mph reduce el total de un partido de NFL alrededor de un 10%, y ese
efecto es de los mayores que el modelo puede capturar.

En fútbol el clima es marginal. Se juega bajo lluvia, frío y viento sin
que la anotación cambie de forma consistente: los estudios sobre goles
y condiciones meteorológicas en ligas europeas encuentran efectos
pequeños y poco replicables entre temporadas.

Replicar aquí esa infraestructura daría precisión aparente sobre datos
que no la soportan. Este módulo se limita a registrar las condiciones
cuando la fuente las trae, sin derivar de ellas un factor de ajuste.

Lo que sí aporta señal: la clasificación
------------------------------------------
La posición en la tabla se puede calcular del propio calendario, y da
contexto que ningún otro módulo aporta:

    ETAPA DE TEMPORADA   Las últimas jornadas son distintas de las
                         primeras. Un equipo matemáticamente salvado y
                         sin opciones europeas tiene menos que jugarse
                         que uno peleando el descenso.

    DISTANCIA EN PUNTOS  Entre los dos equipos del partido. Un
                         enfrentamiento entre vecinos de tabla suele
                         ser más cerrado que uno entre extremos, más
                         allá de lo que digan los índices.

    JORNADA              Las primeras jornadas llevan más
                         incertidumbre: los índices están dominados por
                         el shrinkage y el mercado también sabe menos.

Sobre el ajuste por motivación
--------------------------------
El efecto de "no jugarse nada" es real pero modesto, y difícil de
aislar de otras causas: un equipo salvado también rota más, y la
rotación ya la captura el módulo de congestión.

Por eso este módulo CALCULA y EXPONE la situación —posición, puntos,
distancia al descenso— pero no deriva un factor de ajuste propio. Que
el modelo decida cuánto pesa, con la calibración a la vista, en vez de
enterrarlo aquí en una constante.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.contracts.event import Event

from sports.soccer.competitions import Competition, get_competition
from sports.soccer.schedule import SoccerMatchInfo
from sports.soccer.teams import canonical_team


# ── Dependencias inyectadas ──────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """Interfaz mínima del calendario."""

    def get_competition_matches(self, comp, season: int) -> list[SoccerMatchInfo]:
        ...

    def get_match_info(self, match_id: str) -> SoccerMatchInfo | None:
        ...


# ── Constantes ───────────────────────────────────────────────────────────────

# Jornadas a partir de las cuales la clasificación es informativa.
#
# Con menos de seis jornadas la tabla está dominada por el calendario:
# un equipo puede ser líder por haber enfrentado a los tres peores.
_MIN_MATCHDAYS_FOR_TABLE = 6

# Fracción de temporada a partir de la cual se considera etapa final.
# 0.80 de 38 jornadas son las siete últimas, cuando los objetivos
# empiezan a resolverse matemáticamente.
_LATE_SEASON_THRESHOLD = 0.80

# Jornadas iniciales con incertidumbre elevada.
_EARLY_SEASON_MATCHDAYS = 5

# Plazas de descenso por defecto en las cinco grandes.
_RELEGATION_SPOTS = 3


@dataclass(frozen=True)
class TablePosition:
    """
    Situación de un equipo en la clasificación.

    Campos
    ------
    position      -- Puesto, empezando en 1.
    played        -- Partidos jugados.
    points        -- Puntos acumulados.
    goal_diff     -- Diferencia de goles.
    points_from_top    -- Distancia al líder.
    points_from_relegation
                  -- Distancia a la primera plaza de descenso.
                     Negativa si el equipo está EN descenso.
    """
    team:      str
    position:  int
    played:    int
    points:    int
    goals_for: int
    goals_against: int
    points_from_top: int = 0
    points_from_relegation: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def in_relegation(self) -> bool:
        return self.points_from_relegation < 0

    @property
    def points_per_match(self) -> float:
        return round(self.points / self.played, 3) if self.played else 0.0


@dataclass(frozen=True)
class SoccerContext:
    """
    Contexto de un partido, listo para el modelo de proyección.

    Agrupa lo que aportan otros módulos —congestión, derbi— más lo que
    este calcula: clasificación, jornada y etapa de temporada.
    """
    match_id:   str
    comp_id:    str
    season:     int
    date:       str

    matchday:        int = 0
    season_progress: float = 0.0
    is_early_season: bool = False
    is_late_season:  bool = False

    home_table: TablePosition | None = None
    away_table: TablePosition | None = None

    is_derby:        bool = False
    derby_adjustment: float = 0.0
    congestion_differential: float = 0.0

    temperature:  float | None = None
    wind_speed:   float | None = None
    table_reliable: bool = False

    @property
    def points_gap(self) -> int | None:
        """
        Distancia en puntos entre ambos equipos.

        Un enfrentamiento entre vecinos de tabla suele ser más cerrado
        que uno entre extremos, más allá de lo que digan los índices de
        ataque y defensa.
        """
        if self.home_table is None or self.away_table is None:
            return None
        return abs(self.home_table.points - self.away_table.points)

    @property
    def relegation_battle(self) -> bool:
        """True si alguno de los dos pelea el descenso en etapa final."""
        if not self.is_late_season or not self.table_reliable:
            return False
        return any(
            t is not None and abs(t.points_from_relegation) <= 6
            for t in (self.home_table, self.away_table)
        )

    def to_dict(self) -> dict:
        """
        Serializa para SportDataProvider.get_context().

        Las claves de clasificación solo se emiten si la tabla es
        fiable. Con cuatro jornadas jugadas, el líder puede serlo por
        haber enfrentado a los tres peores — publicar esa posición
        invitaría al modelo a leerla como información.
        """
        base: dict = {
            "match_id":        self.match_id,
            "competition":     self.comp_id,
            "season":          self.season,
            "matchday":        self.matchday,
            "season_progress": round(self.season_progress, 3),
            "is_early_season": self.is_early_season,
            "is_late_season":  self.is_late_season,
            "is_derby":        self.is_derby,
            "derby_adjustment": self.derby_adjustment,
            "congestion_differential": self.congestion_differential,
            "table_reliable":  self.table_reliable,
        }

        if self.temperature is not None:
            base["temperature"] = self.temperature
        if self.wind_speed is not None:
            base["wind_speed"] = self.wind_speed

        if self.table_reliable:
            base.update({
                "home_position": self.home_table.position if self.home_table else None,
                "away_position": self.away_table.position if self.away_table else None,
                "home_points":   self.home_table.points if self.home_table else None,
                "away_points":   self.away_table.points if self.away_table else None,
                "points_gap":    self.points_gap,
                "relegation_battle": self.relegation_battle,
            })

        return base


# ── Fetcher ──────────────────────────────────────────────────────────────────

class SoccerContextFetcher:
    """
    Contexto situacional de partidos.

    Parámetros
    ----------
    schedule_fetcher    -- Fuente de partidos.
    congestion_fetcher  -- SoccerCongestionFetcher. Opcional: sin él,
                           el diferencial de congestión queda en 0.
    h2h_fetcher         -- SoccerH2HFetcher. Opcional: aporta la
                           detección de derbi y su ajuste.
    """

    def __init__(
        self,
        schedule_fetcher:   ScheduleSource,
        congestion_fetcher  = None,
        h2h_fetcher         = None,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._congestion = congestion_fetcher
        self._h2h = h2h_fetcher

        # Caché de clasificaciones por (competición, temporada, corte)
        self._tables: dict[tuple[str, int, str], dict[str, TablePosition]] = {}

    # ── SportDataProvider Protocol ────────────────────────────────────────────

    def get_context(self, event: Event) -> dict:
        """
        Contexto del partido.

        Nunca lanza: ante cualquier fallo devuelve el contexto que haya
        podido construir. El pipeline no debe abortar por no conocer la
        clasificación.
        """
        try:
            return self.build(event).to_dict()
        except Exception:
            return {}

    def build(self, event: Event) -> SoccerContext:
        """Contexto como objeto tipado, para consumo interno."""
        comp_id = (event.provider_ids or {}).get("competition", "")
        comp = get_competition(comp_id)
        match = self._safe_match(event.event_id)

        if comp is None or match is None:
            return SoccerContext(
                match_id=event.event_id, comp_id=comp_id or "",
                season=event.season_start, date=event.date,
            )

        table = self._standings(comp, match.season, match.date)
        matchday, progress = self._season_stage(comp, match, table)

        derby, derby_adj = self._derby(match, comp)
        congestion = self._congestion_diff(match, comp)

        reliable = matchday >= _MIN_MATCHDAYS_FOR_TABLE

        return SoccerContext(
            match_id=match.match_id,
            comp_id=comp.comp_id,
            season=match.season,
            date=match.date,
            matchday=matchday,
            season_progress=progress,
            is_early_season=matchday <= _EARLY_SEASON_MATCHDAYS,
            is_late_season=progress >= _LATE_SEASON_THRESHOLD,
            home_table=table.get(match.home),
            away_table=table.get(match.away),
            is_derby=derby,
            derby_adjustment=derby_adj,
            congestion_differential=congestion,
            table_reliable=reliable,
        )

    # ── Clasificación ─────────────────────────────────────────────────────────

    def standings(
        self,
        comp:       Competition | str,
        season:     int,
        as_of_date: str,
    ) -> dict[str, TablePosition]:
        """
        Clasificación de una competición hasta una fecha.

        Se calcula del propio calendario, sin fuente adicional. Solo
        cuenta partidos ANTERIORES a la fecha: la misma barrera que en
        team_stats.py, porque una tabla que incluya el partido a
        proyectar sería información del futuro.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return {}
        return self._standings(competition, season, as_of_date)

    def _standings(
        self,
        comp:       Competition,
        season:     int,
        as_of_date: str,
    ) -> dict[str, TablePosition]:
        """Clasificación con caché."""
        key = (comp.comp_id, season, as_of_date)
        if key in self._tables:
            return self._tables[key]

        try:
            matches = self._schedule.get_competition_matches(comp, season)
        except Exception:
            matches = []

        if not isinstance(matches, list):
            matches = []

        table = _build_table(matches, as_of_date)
        self._tables[key] = table
        return table

    def _season_stage(
        self,
        comp:  Competition,
        match: SoccerMatchInfo,
        table: dict[str, TablePosition],
    ) -> tuple[int, float]:
        """
        Jornada aproximada y progreso de temporada.

        La jornada se deduce de los partidos jugados por el equipo
        local más uno: football-data no publica el número de jornada, y
        deducirlo del calendario es más robusto que contar fechas —los
        aplazamientos desplazan las fechas pero no los partidos
        jugados.
        """
        position = table.get(match.home)
        played = position.played if position else 0
        matchday = played + 1

        total = comp.matches_per_team or 38
        progress = min(1.0, matchday / total) if total else 0.0

        return matchday, progress

    # ── Composición de otros módulos ──────────────────────────────────────────

    def _derby(
        self,
        match: SoccerMatchInfo,
        comp:  Competition,
    ) -> tuple[bool, float]:
        """
        Condición de derbi y su ajuste.

        Se delega en h2h.py, que es donde vive el catálogo. Si no hay
        fetcher inyectado se consulta la función directamente: la
        detección no necesita estado.
        """
        try:
            from sports.soccer.h2h import is_derby
            derby = is_derby(match.home, match.away, comp.comp_id)
        except Exception:
            return False, 0.0

        adjustment = 0.0
        if self._h2h is not None:
            try:
                adjustment = float(self._h2h.derby_adjustment(derby))
            except Exception:
                adjustment = 0.0

        return derby, adjustment

    def _congestion_diff(
        self,
        match: SoccerMatchInfo,
        comp:  Competition,
    ) -> float:
        """
        Diferencial de congestión desde la perspectiva del local.

        Se calcula aquí —y no en el modelo— porque requiere conocer la
        competición y la temporada del partido, que el modelo no
        recibe. Es el mismo criterio que con `rest_differential` en el
        provider de NFL.
        """
        if self._congestion is None:
            return 0.0
        try:
            return float(self._congestion.differential(
                match.home, match.away, comp, match.season, match.date,
            ))
        except Exception:
            return 0.0

    # ── Utilidad ──────────────────────────────────────────────────────────────

    def _safe_match(self, match_id: str) -> SoccerMatchInfo | None:
        try:
            result = self._schedule.get_match_info(match_id)
        except Exception:
            return None
        return result if isinstance(result, SoccerMatchInfo) else None

    def clear_cache(self) -> None:
        self._tables.clear()


# ── Cálculo de la tabla ──────────────────────────────────────────────────────

def _build_table(
    matches:    list[SoccerMatchInfo],
    as_of_date: str,
) -> dict[str, TablePosition]:
    """
    Construye la clasificación desde los resultados.

    Puntuación estándar: 3 por victoria, 1 por empate.

    El desempate sigue el criterio mayoritario en Europa —puntos,
    diferencia de goles, goles a favor— aunque algunas ligas usan el
    enfrentamiento directo. Esa diferencia no afecta a lo que el modelo
    usa de la tabla (distancia en puntos, cercanía al descenso), así
    que no justifica implementar cinco reglamentos distintos.
    """
    played: dict[str, dict] = {}

    for match in matches:
        if not match.is_final or not match.date or match.date >= as_of_date:
            continue
        hg = match.home_goals or 0
        ag = match.away_goals or 0

        for team, gf, ga in ((match.home, hg, ag), (match.away, ag, hg)):
            if not team:
                continue
            bucket = played.setdefault(team, {
                "played": 0, "points": 0, "gf": 0, "ga": 0,
            })
            bucket["played"] += 1
            bucket["gf"] += gf
            bucket["ga"] += ga
            if gf > ga:
                bucket["points"] += 3
            elif gf == ga:
                bucket["points"] += 1

    if not played:
        return {}

    ordered = sorted(
        played.items(),
        key=lambda kv: (-kv[1]["points"],
                        -(kv[1]["gf"] - kv[1]["ga"]),
                        -kv[1]["gf"]),
    )

    top_points = ordered[0][1]["points"] if ordered else 0

    # Puntos de la primera plaza de descenso. Con menos equipos que
    # plazas de descenso —imposible en liga, posible con datos
    # parciales— se usa el último clasificado.
    relegation_index = max(0, len(ordered) - _RELEGATION_SPOTS)
    relegation_points = (ordered[relegation_index][1]["points"]
                         if relegation_index < len(ordered) else 0)

    table: dict[str, TablePosition] = {}
    for index, (team, stats) in enumerate(ordered, start=1):
        table[team] = TablePosition(
            team=team,
            position=index,
            played=stats["played"],
            points=stats["points"],
            goals_for=stats["gf"],
            goals_against=stats["ga"],
            points_from_top=top_points - stats["points"],
            points_from_relegation=stats["points"] - relegation_points,
        )

    return table