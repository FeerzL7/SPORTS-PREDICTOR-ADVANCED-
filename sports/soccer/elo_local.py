"""
sports/soccer/elo_local.py

Ratings Elo calculados con los resultados que el plugin ya descarga.

Por qué un Elo propio
-----------------------
La señal de Elo dependía de clubelo.com, que lleva días respondiendo
502 (2026-10). Y aunque respondiera, ClubElo solo cubre clubes europeos
de primera y segunda: sin Elo en Liga MX, Brasil ni en buena parte de
las divisiones inferiores, que son justo donde el mercado debería ser
menos eficiente.

Los resultados ya están en los CSV de football-data, así que el rating
se puede calcular aquí, sin otra fuente, para cualquier competición que
tenga calendario.

Garantía de no look-ahead
--------------------------
`difference(home, away, date)` usa SOLO partidos jugados antes de
`date`: el rating de cada equipo se guarda como serie temporal
(fecha, rating después del partido) y se consulta el último valor con
fecha estrictamente anterior. Un partido del día no se ve a sí mismo.

El modelo
----------
Elo clásico con los ajustes estándar del Elo de fútbol:

    E_local = 1 / (1 + 10^((R_visit − R_local − HFA) / 400))
    R'      = R + K · G · (S − E)

    HFA  ventaja de campo en puntos, dentro de la EXPECTATIVA. El
         rating que se devuelve no la incluye, igual que ClubElo: el
         modelo de goles ya aplica la ventaja de campo de la liga.
    G    multiplicador por diferencia de goles: 1 con 0-1 goles,
         1.5 con 2 y (11 + d) / 8 con 3 o más. Una goleada dice más
         que un 1-0, pero no tres veces más.
    S    1, 0.5 o 0.

Entre temporadas cada rating se encoge un tercio hacia la media de la
competición: las plantillas cambian y el rating de mayo sobreestima la
diferencia de agosto. Un equipo que aparece por primera vez (ascendido)
entra 100 puntos por debajo de la media de la liga en ese momento.

Los parámetros son los habituales de la literatura (K=20, HFA=65). No
están calibrados con datos propios: el backtest decide si la señal
aporta algo, con `soccer.elo.weight`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

_DEFAULT_RATING = 1500.0


@dataclass(frozen=True)
class EloParams:
    k:                 float = 20.0
    home_advantage:    float = 65.0
    season_regression: float = 1 / 3
    promoted_offset:   float = 100.0
    seasons_back:      int   = 3


class LocalEloClient:
    """
    Elo walk-forward desde los resultados de football-data.

    Mismo contrato que ClubEloClient en lo que usa el provider:
    `difference(home, away, date, comp_id)` y `is_available()`.

    Parámetros
    ----------
    match_source  -- Objeto con load_matches(comp, season) (SoccerDataSource).
    params        -- EloParams.
    """

    def __init__(self, match_source, params: EloParams | None = None) -> None:
        self._source = match_source
        self._params = params or EloParams()
        # (comp_id, temporada objetivo) → {equipo: ([fechas], [ratings])}
        self._series: dict[tuple[str, int], dict[str, tuple[list[str], list[float]]]] = {}

    @staticmethod
    def is_available() -> bool:
        return True

    # ── API pública ───────────────────────────────────────────────────────────

    def rating_before(self, team: str, date: str, comp_id: str) -> float | None:
        """Rating del equipo con los partidos anteriores a `date`; None si no jugó."""
        from sports.soccer.teams import canonical_team

        series = self._series_for(comp_id, _season_for(date, comp_id))
        entry = series.get(canonical_team(team, comp_id))
        if entry is None:
            return None
        dates, ratings = entry
        index = bisect.bisect_left(dates, date) - 1
        return ratings[index] if index >= 0 else None

    def difference(self, home: str, away: str, date: str, comp_id: str = "") -> float | None:
        """
        Rating del local menos el del visitante, sin ventaja de campo.

        None si falta alguno de los dos (equipo sin partidos previos en
        la ventana): medio dato sería peor que ninguno.
        """
        home_rating = self.rating_before(home, date, comp_id)
        away_rating = self.rating_before(away, date, comp_id)
        if home_rating is None or away_rating is None:
            return None
        return round(home_rating - away_rating, 2)

    # ── Cálculo ───────────────────────────────────────────────────────────────

    def _series_for(self, comp_id: str, season: int):
        key = (comp_id, season)
        if key not in self._series:
            self._series[key] = self._compute(comp_id, season)
        return self._series[key]

    def _compute(self, comp_id: str, season: int):
        """
        Recorre las temporadas [season − seasons_back, season] en orden.

        Las anteriores sirven de calentamiento: sin ellas, en agosto todos
        los equipos tendrían el mismo rating.
        """
        from sports.soccer.teams import canonical_team

        p = self._params
        ratings: dict[str, float] = {}
        series: dict[str, tuple[list[str], list[float]]] = {}
        # Equipos de la temporada anterior: la media de la liga se toma
        # sobre ellos y no sobre todos los que alguna vez jugaron, que
        # incluiría a los descendidos y arrastraría la media hacia abajo.
        previous: set[str] = set()

        for year in range(season - p.seasons_back, season + 1):
            try:
                rows = self._source.load_matches(comp_id, year)
            except Exception:
                rows = []
            played = sorted((r for r in rows if r.is_final), key=lambda r: r.date)
            if not played:
                continue

            # Regresión de pretemporada hacia la media de la competición.
            if previous:
                mean = sum(ratings[t] for t in previous) / len(previous)
                for team in previous:
                    ratings[team] += (mean - ratings[team]) * p.season_regression
            current: set[str] = set()

            for row in played:
                home = canonical_team(row.home_team, comp_id)
                away = canonical_team(row.away_team, comp_id)
                if not home or not away:
                    continue
                for team in (home, away):
                    if team not in ratings:
                        ratings[team] = self._entry_rating(ratings, previous)
                    current.add(team)

                r_home, r_away = ratings[home], ratings[away]
                expected = 1.0 / (1.0 + 10 ** ((r_away - r_home - p.home_advantage) / 400.0))
                goal_diff = (row.home_goals or 0) - (row.away_goals or 0)
                score = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)
                delta = p.k * _goal_multiplier(abs(goal_diff)) * (score - expected)

                ratings[home] = r_home + delta
                ratings[away] = r_away - delta
                for team in (home, away):
                    dates, values = series.setdefault(team, ([], []))
                    dates.append(row.date)
                    values.append(ratings[team])

            previous = current

        return series

    def _entry_rating(self, ratings: dict[str, float], previous: set[str]) -> float:
        """
        Rating inicial de un equipo que aún no tenía.

        En la primera temporada de la ventana no hay temporada anterior:
        todos parten de 1500. Después, un equipo nuevo es un ascendido y
        entra `promoted_offset` puntos por debajo de la media de los
        equipos de la temporada anterior.
        """
        rated = [ratings[t] for t in previous if t in ratings]
        if not rated:
            return _DEFAULT_RATING
        return sum(rated) / len(rated) - self._params.promoted_offset


def _goal_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0


def _season_for(date: str, comp_id: str) -> int:
    """Temporada que contiene la fecha, según el calendario de la competición."""
    from sports.soccer.competitions import SEASON_CALENDAR, get_competition

    year, month = int(date[:4]), int(date[5:7])
    comp = get_competition(comp_id)
    if comp is not None and comp.season_type == SEASON_CALENDAR:
        return year
    return year if month >= 7 else year - 1
