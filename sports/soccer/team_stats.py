"""
sports/soccer/team_stats.py

SoccerTeamStatsFetcher: índices de ataque y defensa desde xG.

Por qué aquí la composición SÍ es multiplicativa
--------------------------------------------------
En el plugin NFL se rechazó la composición multiplicativa de índices y
se usó un promedio ponderado. La razón era concreta: el EPA promedia
~0.00 y puede ser negativo, así que dividir entre la media de liga
explota o cambia de signo, y con un rango de índices de 0.60 a 1.40 el
producto de dos errores del 10% arrastra un 21%.

En fútbol la situación se invierte y la composición multiplicativa es
la correcta:

    El xG es ESTRICTAMENTE POSITIVO y promedia ~1.4 por equipo y
    partido. El ratio contra la media de liga está bien definido y su
    rango es estrecho: los extremos de las cinco grandes van de ~0.55
    a ~1.75.

    Es la formulación establecida. Maher (1982) y Dixon-Coles (1997)
    modelan la media esperada de goles como

        λ_local = ataque_local × defensa_visitante × ventaja × media_liga

    Apartarse de un modelo tan estudiado sin una razón específica sería
    cambiar rigor por originalidad.

Qué compone cada índice
-------------------------
El índice no sale de una sola métrica sino de cuatro señales
ponderadas, con los pesos de config/soccer.yaml:

    npxG    0.45   El mejor predictor. Excluye penaltis, que valen
                   ~0.76 de xG cada uno y dependen mucho más del azar.
    xG      0.25   Incluye los penaltis: conseguirlos tiene un
                   componente de habilidad, solo que menor.
    Goles   0.15   Correlación ~0.40 con resultados futuros, frente al
                   ~0.60 del xG. Lleva información de finalización, que
                   es parcialmente real pero muy ruidosa.
    Forma   0.15   Ventana móvil de 8 partidos sobre xG.

Cada señal se expresa como RATIO contra la media de liga, y se
promedian los ratios. El resultado se multiplica en el modelo, no se
suma.

Separación casa/fuera
-----------------------
Un equipo no ataca igual en casa que fuera, y la diferencia no es
uniforme entre equipos: algunos explotan su campo mucho más que otros.

El modelo clásico separa los índices por localía. El coste es que
divide la muestra por dos, así que la ventana se amplía y el
shrinkage se aplica sobre la muestra parcial, no sobre la total.

Barrera contra el look-ahead
------------------------------
`fetch()` exige `as_of_date` sin valor por defecto. Para proyectar un
partido del 15 de marzo, el modelo solo puede ver lo jugado ANTES de
esa fecha.

Es el mismo criterio que `up_to_week` en NFL, por la misma razón: el
modo de fallo no lanza excepción ni deja campos vacíos. Produce un
backtest inflado que no se reproduce en producción, y el diagnóstico
llega meses después cuando el ledger real diverge del histórico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sports.soccer.competitions import Competition, get_competition
from sports.soccer.schedule import SoccerMatchInfo
from sports.soccer.teams import canonical_team


# ── Dependencia inyectada ────────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """Interfaz mínima del calendario que este módulo consume."""

    def get_competition_matches(self, comp, season: int) -> list[SoccerMatchInfo]:
        ...

    def teams_in_competition(self, comp, season: int) -> list[str]:
        ...


# ── Constantes de calibración ────────────────────────────────────────────────

# Pesos de las cuatro señales. Coinciden con config/soccer.yaml.
_W_NPXG  = 0.45
_W_XG    = 0.25
_W_GOALS = 0.15
_W_FORM  = 0.15

# Ventanas móviles, en partidos.
_FORM_WINDOW        = 8
_VENUE_SPLIT_WINDOW = 12

# Prior del shrinkage, en partidos.
#
# Seis sobre una temporada de 38 es un 16%, algo menos agresivo que el
# equivalente de NFL (3 sobre 17, un 18%). La razón: el xG estabiliza
# más rápido que el EPA — hacen falta ~8-10 partidos para que el xG por
# partido sea informativo, frente a los ~20 que necesitan los goles.
_PRIOR_MATCHES = 6.0

# Partidos mínimos para que el equipo sea proyectable.
_MIN_MATCHES = 5

# Cotas de los índices. Red de seguridad, no mecanismo de calibración:
# la protección principal es el shrinkage.
#
# Los extremos reales de las cinco grandes rondan 0.55 y 1.75 en
# ataque. Las cotas dejan margen sobre eso.
_INDEX_MIN = 0.45
_INDEX_MAX = 2.00

# Medias de liga por defecto cuando no hay datos.
_DEFAULT_GOALS_PER_TEAM = 1.40


# ── Agregados ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SoccerLeagueAverages:
    """
    Medias de una competición hasta una fecha.

    Se calculan sobre los mismos partidos que las estadísticas de
    equipo, para que la comparación sea homogénea. Usar medias
    históricas fijas introduciría un desfase: una temporada
    especialmente goleadora inflaría todos los índices por igual.
    """
    comp_id:     str
    season:      int
    as_of_date:  str
    matches:     int = 0

    home_goals:  float = _DEFAULT_GOALS_PER_TEAM
    away_goals:  float = _DEFAULT_GOALS_PER_TEAM
    home_xg:     float = _DEFAULT_GOALS_PER_TEAM
    away_xg:     float = _DEFAULT_GOALS_PER_TEAM
    home_npxg:   float = _DEFAULT_GOALS_PER_TEAM
    away_npxg:   float = _DEFAULT_GOALS_PER_TEAM
    xg_matches:  int = 0

    @property
    def goals_per_team(self) -> float:
        """Media global de goles por equipo y partido."""
        return round((self.home_goals + self.away_goals) / 2.0, 4)

    @property
    def xg_per_team(self) -> float:
        return round((self.home_xg + self.away_xg) / 2.0, 4)

    @property
    def npxg_per_team(self) -> float:
        return round((self.home_npxg + self.away_npxg) / 2.0, 4)

    @property
    def total_goals(self) -> float:
        """Media de goles por partido de la competición."""
        return round(self.home_goals + self.away_goals, 3)

    @property
    def home_advantage(self) -> float:
        """
        Ventaja de campo observada, en goles.

        Se calcula de los datos en vez de usar el valor de
        configuración porque la ventaja de campo ha cambiado de forma
        apreciable: rondaba 0.45 goles antes de 2020, cayó a ~0.15 con
        los partidos a puerta cerrada y se recuperó parcialmente. Un
        backtest que abarque 2020-21 con un valor fijo sobreestimaría
        al local en esa temporada.
        """
        return round(self.home_goals - self.away_goals, 4)

    @property
    def has_xg(self) -> bool:
        return self.xg_matches > 0

    @property
    def has_sufficient_sample(self) -> bool:
        """
        True si hay partidos suficientes para que las medias sirvan.

        Cincuenta partidos son ~5 jornadas de una liga de 20 equipos.
        Por debajo, la media de liga carga demasiado ruido para servir
        de referencia a los índices.
        """
        return self.matches >= 50


@dataclass(frozen=True)
class SoccerTeamStats:
    """
    Índices de ataque y defensa de un equipo hasta una fecha.

    Inmutable: representa el estado del equipo en un corte temporal
    concreto. Dos cortes distintos son dos objetos distintos, lo que
    hace imposible contaminar un backtest reutilizando estadísticas de
    otra jornada.

    Campos por partido
    ------------------
    Todos son MEDIAS POR PARTIDO, no acumulados: hacen comparables a
    equipos que han jugado distinto número de encuentros, algo habitual
    a mitad de temporada por aplazamientos.

    Separación casa/fuera
    ---------------------
    home_xg_for / away_xg_for y sus equivalentes defensivos. La muestra
    es la mitad, así que sus índices llevan más shrinkage.
    """
    team:       str
    comp_id:    str
    season:     int
    as_of_date: str

    matches:      int = 0
    home_matches: int = 0
    away_matches: int = 0

    # Medias de temporada
    xg_for:       float = 0.0
    xg_against:   float = 0.0
    npxg_for:     float = 0.0
    npxg_against: float = 0.0
    goals_for:    float = 0.0
    goals_against: float = 0.0

    # Separación por localía
    home_xg_for:     float = 0.0
    home_xg_against: float = 0.0
    away_xg_for:     float = 0.0
    away_xg_against: float = 0.0

    # Forma reciente, sobre xG
    form_xg_for:     float = 0.0
    form_xg_against: float = 0.0
    form_matches:    int = 0

    # Resultados recientes, del más antiguo al más reciente
    recent_results: list[str] = field(default_factory=list)
    recent_goals:   list[int] = field(default_factory=list)

    league: SoccerLeagueAverages | None = None
    xg_available: bool = False

    # ── Índices ───────────────────────────────────────────────────────────────

    def attack_index(self) -> float:
        """
        Índice de ataque. 1.0 = media de liga, >1 mejor.

        Promedio ponderado de cuatro ratios, cada uno contra la media
        de liga. El resultado se MULTIPLICA en el modelo de proyección,
        siguiendo la formulación de Maher y Dixon-Coles.

        Si falta el xG, su peso se redistribuye entre las señales
        disponibles en vez de asumir valor 1.0. Tratar una señal
        ausente como promedio sería un sesgo: un equipo sin datos de xG
        no es un equipo con xG promedio.
        """
        league = self.league
        if league is None or self.matches <= 0:
            return 1.0

        signals: list[tuple[float, float]] = []

        if self.xg_available and league.has_xg:
            if league.npxg_per_team > 0:
                signals.append((self.npxg_for / league.npxg_per_team, _W_NPXG))
            if league.xg_per_team > 0:
                signals.append((self.xg_for / league.xg_per_team, _W_XG))
            if self.form_matches > 0 and league.xg_per_team > 0:
                signals.append((self.form_xg_for / league.xg_per_team, _W_FORM))

        if league.goals_per_team > 0:
            signals.append((self.goals_for / league.goals_per_team, _W_GOALS))

        return self._blend(signals)

    def defense_index(self) -> float:
        """
        Índice de defensa. 1.0 = media de liga, >1 mejor.

        El signo se invierte respecto al ataque: en defensa conceder
        MENOS es mejor, así que el ratio se calcula al revés
        (media_liga / concedido).

        Esta convención —mayor es mejor— es la misma que usan los
        plugins de MLB y NFL. El Core depende de ella: BlendingEngine y
        los modelos de proyección asumen que un índice alto significa
        mejor rendimiento.

        Ojo: en la fórmula de Maher el factor defensivo entra como
        "goles concedidos relativos", así que el modelo de proyección
        usa el INVERSO de este índice. Se mantiene la convención del
        Core aquí y se invierte allí, en un punto explícito.
        """
        league = self.league
        if league is None or self.matches <= 0:
            return 1.0

        signals: list[tuple[float, float]] = []

        if self.xg_available and league.has_xg:
            if self.npxg_against > 0:
                signals.append((league.npxg_per_team / self.npxg_against, _W_NPXG))
            if self.xg_against > 0:
                signals.append((league.xg_per_team / self.xg_against, _W_XG))
            if self.form_matches > 0 and self.form_xg_against > 0:
                signals.append((league.xg_per_team / self.form_xg_against, _W_FORM))

        if self.goals_against > 0:
            signals.append((league.goals_per_team / self.goals_against, _W_GOALS))

        return self._blend(signals)

    def attack_index_at(self, is_home: bool) -> float:
        """
        Índice de ataque separado por localía.

        Cae al índice general cuando la muestra parcial es demasiado
        corta: con tres partidos en casa, el promedio describe tres
        rivales concretos más que al propio equipo.
        """
        league = self.league
        n = self.home_matches if is_home else self.away_matches
        value = self.home_xg_for if is_home else self.away_xg_for

        if league is None or n < 4 or not self.xg_available:
            return self.attack_index()

        reference = league.home_xg if is_home else league.away_xg
        if reference <= 0:
            return self.attack_index()

        raw = value / reference
        shrunk = _shrink(raw, n, 1.0)
        return round(_clamp(shrunk, _INDEX_MIN, _INDEX_MAX), 4)

    def defense_index_at(self, is_home: bool) -> float:
        """Índice defensivo separado por localía."""
        league = self.league
        n = self.home_matches if is_home else self.away_matches
        conceded = self.home_xg_against if is_home else self.away_xg_against

        if league is None or n < 4 or not self.xg_available or conceded <= 0:
            return self.defense_index()

        # El local concede lo que el visitante genera fuera
        reference = league.away_xg if is_home else league.home_xg
        if reference <= 0:
            return self.defense_index()

        raw = reference / conceded
        shrunk = _shrink(raw, n, 1.0)
        return round(_clamp(shrunk, _INDEX_MIN, _INDEX_MAX), 4)

    def _blend(self, signals: list[tuple[float, float]]) -> float:
        """
        Promedio ponderado de ratios, con shrinkage y cotas.

        Redistribuye el peso de las señales ausentes entre las
        presentes, en vez de rellenarlas con 1.0.
        """
        if not signals:
            return 1.0

        weight_sum = sum(w for _, w in signals)
        if weight_sum <= 0:
            return 1.0

        raw = sum(v * w for v, w in signals) / weight_sum
        shrunk = _shrink(raw, self.matches, 1.0)
        return round(_clamp(shrunk, _INDEX_MIN, _INDEX_MAX), 4)

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    @property
    def has_sufficient_sample(self) -> bool:
        return self.matches >= _MIN_MATCHES

    @property
    def xg_overperformance(self) -> float:
        """
        Goles marcados menos xG, por partido.

        Positivo indica finalización por encima de lo esperado, que
        históricamente revierte. El mercado reacciona a los goles antes
        que al xG, así que esta diferencia señala dónde puede haber
        valor.
        """
        if not self.xg_available:
            return 0.0
        return round(self.goals_for - self.xg_for, 4)

    @property
    def form_points(self) -> int:
        """Puntos en la ventana reciente."""
        return sum(3 if r == "W" else (1 if r == "D" else 0)
                   for r in self.recent_results)

    def to_metadata(self) -> dict:
        """Metadatos para TeamFeatures.sport_metadata."""
        return {
            "matches":          self.matches,
            "xg_for":           round(self.xg_for, 3),
            "xg_against":       round(self.xg_against, 3),
            "npxg_for":         round(self.npxg_for, 3),
            "npxg_against":     round(self.npxg_against, 3),
            "goals_for":        round(self.goals_for, 3),
            "goals_against":    round(self.goals_against, 3),
            "attack_index":     self.attack_index(),
            "defense_index":    self.defense_index(),
            "attack_home":      self.attack_index_at(True),
            "attack_away":      self.attack_index_at(False),
            "defense_home":     self.defense_index_at(True),
            "defense_away":     self.defense_index_at(False),
            "xg_overperformance": self.xg_overperformance,
            "form_points":      self.form_points,
            "form_matches":     self.form_matches,
            "xg_available":     self.xg_available,
            "as_of_date":       self.as_of_date,
        }


# ── Acumulador interno ───────────────────────────────────────────────────────

class _Accumulator:
    """
    Acumulador mutable usado durante la pasada sobre los partidos.

    Separado de SoccerTeamStats (inmutable) de forma deliberada: la
    fase de agregación necesita mutación, el resultado publicado no.
    Mezclar ambos roles abriría la puerta a que un consumidor
    modificara estadísticas ya calculadas.
    """

    __slots__ = ("team", "matches", "home_matches", "away_matches",
                 "xg_for", "xg_against", "npxg_for", "npxg_against",
                 "goals_for", "goals_against",
                 "home_xg_for", "home_xg_against",
                 "away_xg_for", "away_xg_against",
                 "xg_matches", "recent")

    def __init__(self, team: str) -> None:
        self.team = team
        self.matches = 0
        self.home_matches = 0
        self.away_matches = 0
        self.xg_for = 0.0
        self.xg_against = 0.0
        self.npxg_for = 0.0
        self.npxg_against = 0.0
        self.goals_for = 0
        self.goals_against = 0
        self.home_xg_for = 0.0
        self.home_xg_against = 0.0
        self.away_xg_for = 0.0
        self.away_xg_against = 0.0
        self.xg_matches = 0
        # (fecha, resultado, goles a favor, xG a favor, xG en contra)
        self.recent: list[tuple[str, str, int, float, float]] = []


# ── Fetcher ──────────────────────────────────────────────────────────────────

class SoccerTeamStatsFetcher:
    """
    Calcula índices de ataque y defensa desde el calendario.

    Parámetros
    ----------
    schedule_fetcher -- Fuente de partidos con xG ya cruzado.
    config_loader    -- ConfigLoader con soccer.yaml, para los pesos.
    """

    def __init__(
        self,
        schedule_fetcher: ScheduleSource,
        config_loader    = None,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._config = config_loader

        self._w_npxg  = self._cfg("soccer.projection.npxg_weight",  _W_NPXG)
        self._w_xg    = self._cfg("soccer.projection.xg_weight",    _W_XG)
        self._w_goals = self._cfg("soccer.projection.goals_weight", _W_GOALS)
        self._w_form  = self._cfg("soccer.projection.form_weight",  _W_FORM)
        self._form_window = int(
            self._cfg("soccer.projection.form_window", _FORM_WINDOW)
        )
        self._prior = self._cfg("soccer.shrinkage.prior_matches", _PRIOR_MATCHES)

        # Caché por (competición, temporada, fecha de corte)
        self._cache: dict[tuple[str, int, str], dict[str, SoccerTeamStats]] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def fetch(
        self,
        team:       str,
        comp:       Competition | str,
        season:     int,
        as_of_date: str,
    ) -> SoccerTeamStats:
        """
        Estadísticas de un equipo con datos ANTERIORES a `as_of_date`.

        `as_of_date` no tiene valor por defecto de forma deliberada:
        omitirlo debe ser un error de programación visible, no un fallo
        silencioso que contamina el backtest con datos del futuro.

        Nunca lanza: sin datos devuelve un objeto con índices neutros
        (1.0) y `matches=0`.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return SoccerTeamStats(team=team, comp_id=str(comp),
                                   season=season, as_of_date=as_of_date)

        canon = canonical_team(team, competition.comp_id)
        all_stats = self.fetch_all(competition, season, as_of_date)

        return all_stats.get(canon) or SoccerTeamStats(
            team=canon, comp_id=competition.comp_id,
            season=season, as_of_date=as_of_date,
            league=self.league_averages(competition, season, as_of_date),
        )

    def fetch_all(
        self,
        comp:       Competition | str,
        season:     int,
        as_of_date: str,
    ) -> dict[str, SoccerTeamStats]:
        """
        Estadísticas de todos los equipos de una competición.

        Una sola pasada sobre los partidos produce las de todos los
        equipos. `fetch()` reutiliza este resultado desde caché.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return {}

        key = (competition.comp_id, season, as_of_date)
        if key in self._cache:
            return self._cache[key]

        matches = self._matches_before(competition, season, as_of_date)
        if not matches:
            self._cache[key] = {}
            return {}

        accumulators = self._accumulate(matches)
        league = self._compute_league(matches, competition, season, as_of_date)

        stats = {
            team: self._build(acc, league, competition, season, as_of_date)
            for team, acc in accumulators.items()
        }

        self._cache[key] = stats
        return stats

    def league_averages(
        self,
        comp:       Competition | str,
        season:     int,
        as_of_date: str,
    ) -> SoccerLeagueAverages:
        """Medias de la competición hasta la fecha de corte."""
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return SoccerLeagueAverages(comp_id=str(comp), season=season,
                                        as_of_date=as_of_date)

        matches = self._matches_before(competition, season, as_of_date)
        return self._compute_league(matches, competition, season, as_of_date)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── Agregación ────────────────────────────────────────────────────────────

    def _matches_before(
        self,
        comp:       Competition,
        season:     int,
        as_of_date: str,
    ) -> list[SoccerMatchInfo]:
        """
        Partidos finalizados ANTES de la fecha de corte.

        La comparación es estricta (`<`), no `<=`: un partido del mismo
        día es el que se está proyectando, o uno simultáneo cuyo
        resultado tampoco se conoce al apostar.
        """
        try:
            matches = self._schedule.get_competition_matches(comp, season)
        except Exception:
            return []

        if not isinstance(matches, list):
            return []

        return [
            m for m in matches
            if m.is_final and m.date and m.date < as_of_date
        ]

    @staticmethod
    def _accumulate(matches: list[SoccerMatchInfo]) -> dict[str, _Accumulator]:
        """
        Una pasada acumulando por equipo.

        Se itera de forma explícita en vez de usar agrupaciones de
        pandas: mantiene el filtrado auditable y evita la frontera de
        tipado que costó erradicar en el plugin NFL.
        """
        acc: dict[str, _Accumulator] = {}

        def bucket(team: str) -> _Accumulator:
            if team not in acc:
                acc[team] = _Accumulator(team)
            return acc[team]

        for match in sorted(matches, key=lambda m: m.date):
            hg = match.home_goals or 0
            ag = match.away_goals or 0
            has_xg = match.has_xg

            for team, is_home, gf, ga in (
                (match.home, True,  hg, ag),
                (match.away, False, ag, hg),
            ):
                if not team:
                    continue
                b = bucket(team)
                b.matches += 1
                b.goals_for += gf
                b.goals_against += ga

                if is_home:
                    b.home_matches += 1
                else:
                    b.away_matches += 1

                xgf = xga = 0.0
                if has_xg:
                    xgf = (match.home_xg or 0.0) if is_home else (match.away_xg or 0.0)
                    xga = (match.away_xg or 0.0) if is_home else (match.home_xg or 0.0)
                    npf = (match.home_npxg if is_home else match.away_npxg)
                    npa = (match.away_npxg if is_home else match.home_npxg)

                    b.xg_for += xgf
                    b.xg_against += xga
                    # npxG cae al xG cuando la fuente no lo separa
                    b.npxg_for += (npf if npf is not None else xgf)
                    b.npxg_against += (npa if npa is not None else xga)
                    b.xg_matches += 1

                    if is_home:
                        b.home_xg_for += xgf
                        b.home_xg_against += xga
                    else:
                        b.away_xg_for += xgf
                        b.away_xg_against += xga

                result = "W" if gf > ga else ("D" if gf == ga else "L")
                b.recent.append((match.date, result, gf, xgf, xga))

        return acc

    @staticmethod
    def _compute_league(
        matches:    list[SoccerMatchInfo],
        comp:       Competition,
        season:     int,
        as_of_date: str,
    ) -> SoccerLeagueAverages:
        """
        Medias de la competición sobre los partidos del corte.

        Se calculan de los datos observados, no de valores históricos
        fijos: una temporada especialmente goleadora inflaría todos los
        índices por igual si la referencia fuera estática.

        Sin partidos suficientes cae a las medias históricas del
        catálogo, que es la referencia correcta al arranque de
        temporada.
        """
        if not matches:
            return SoccerLeagueAverages(
                comp_id=comp.comp_id, season=season, as_of_date=as_of_date,
                home_goals=comp.avg_home_goals, away_goals=comp.avg_away_goals,
                home_xg=comp.avg_home_goals, away_xg=comp.avg_away_goals,
                home_npxg=comp.avg_home_goals, away_npxg=comp.avg_away_goals,
            )

        n = len(matches)
        hg = sum(m.home_goals or 0 for m in matches)
        ag = sum(m.away_goals or 0 for m in matches)

        with_xg = [m for m in matches if m.has_xg]
        n_xg = len(with_xg)

        if n_xg:
            hxg = sum(m.home_xg or 0.0 for m in with_xg) / n_xg
            axg = sum(m.away_xg or 0.0 for m in with_xg) / n_xg
            hnp = sum((m.home_npxg if m.home_npxg is not None else m.home_xg) or 0.0
                      for m in with_xg) / n_xg
            anp = sum((m.away_npxg if m.away_npxg is not None else m.away_xg) or 0.0
                      for m in with_xg) / n_xg
        else:
            hxg = axg = hnp = anp = 0.0

        return SoccerLeagueAverages(
            comp_id=comp.comp_id, season=season, as_of_date=as_of_date,
            matches=n,
            home_goals=round(hg / n, 4),
            away_goals=round(ag / n, 4),
            home_xg=round(hxg, 4), away_xg=round(axg, 4),
            home_npxg=round(hnp, 4), away_npxg=round(anp, 4),
            xg_matches=n_xg,
        )

    def _build(
        self,
        acc:        _Accumulator,
        league:     SoccerLeagueAverages,
        comp:       Competition,
        season:     int,
        as_of_date: str,
    ) -> SoccerTeamStats:
        """Convierte un acumulador en estadísticas publicadas."""
        n = acc.matches or 1
        n_xg = acc.xg_matches or 1
        has_xg = acc.xg_matches > 0

        # Ventana de forma: los últimos N partidos
        window = acc.recent[-self._form_window:] if acc.recent else []
        form_n = len(window)
        form_xgf = sum(w[3] for w in window) / form_n if form_n and has_xg else 0.0
        form_xga = sum(w[4] for w in window) / form_n if form_n and has_xg else 0.0

        return SoccerTeamStats(
            team=acc.team, comp_id=comp.comp_id, season=season,
            as_of_date=as_of_date,
            matches=acc.matches,
            home_matches=acc.home_matches,
            away_matches=acc.away_matches,
            xg_for=acc.xg_for / n_xg if has_xg else 0.0,
            xg_against=acc.xg_against / n_xg if has_xg else 0.0,
            npxg_for=acc.npxg_for / n_xg if has_xg else 0.0,
            npxg_against=acc.npxg_against / n_xg if has_xg else 0.0,
            goals_for=acc.goals_for / n,
            goals_against=acc.goals_against / n,
            home_xg_for=(acc.home_xg_for / acc.home_matches
                         if acc.home_matches and has_xg else 0.0),
            home_xg_against=(acc.home_xg_against / acc.home_matches
                             if acc.home_matches and has_xg else 0.0),
            away_xg_for=(acc.away_xg_for / acc.away_matches
                         if acc.away_matches and has_xg else 0.0),
            away_xg_against=(acc.away_xg_against / acc.away_matches
                             if acc.away_matches and has_xg else 0.0),
            form_xg_for=round(form_xgf, 4),
            form_xg_against=round(form_xga, 4),
            form_matches=form_n,
            recent_results=[w[1] for w in window],
            recent_goals=[w[2] for w in window],
            league=league,
            xg_available=has_xg,
        )

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

def _shrink(observed: float, n: int, prior: float,
            k: float = _PRIOR_MATCHES) -> float:
    """
    Regresión a la media.

        resultado = (n × observado + k × prior) / (n + k)

    Con k = 6 partidos sobre una temporada de 38:
        Jornada 3   → 33% el equipo, 67% la media de liga
        Jornada 10  → 62%
        Jornada 20  → 77%
        Jornada 38  → 86%

    Al arrancar la temporada el modelo apenas se despega del mercado, y
    gana confianza conforme la muestra crece.
    """
    if n <= 0:
        return prior
    return (n * observed + k * prior) / (n + k)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))