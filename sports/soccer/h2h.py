"""
sports/soccer/h2h.py

SoccerH2HFetcher: historial de enfrentamientos directos.

Fuerza real de esta señal
---------------------------
Algo mayor que en NFL, pero sigue siendo débil:

                                    NFL          Fútbol
    Enfrentamientos por temporada   2 (división)  2 (siempre)
    Rotación de plantilla anual     ~25%          ~15-20%
    Cambios de entrenador           poco          muy frecuentes

El menor recambio de jugadores juega a favor; la volatilidad de los
banquillos, en contra. En conjunto, un historial de tres temporadas
describe a dos equipos parcialmente distintos de los actuales, igual
que allí.

Por eso se mantienen los mismos mecanismos de prudencia que en el
plugin NFL: decaimiento por antigüedad, ventana corta y un indicador
de fiabilidad que impide que el modelo tome como señal lo que es
anécdota.

Dos cosas que NFL no tenía
----------------------------

1. SEPARACIÓN CASA / FUERA
   Con una ventaja de campo de ~0.35 goles, un historial de 3-0 en
   casa y 0-3 fuera describe algo muy distinto de un 3-3 agregado: el
   primero indica que la localía decide la serie, el segundo que los
   equipos están igualados.

   El modelo proyecta un partido en un estadio concreto, así que el
   historial EN ESE ESTADIO es más informativo que el global. Pero la
   muestra se reduce a la mitad, y de ahí que su umbral de fiabilidad
   sea distinto.

2. DERBIS
   Los enfrentamientos entre equipos de la misma ciudad tienen dos
   efectos documentados y estables:

       Menos ventaja de campo. El visitante lleva afición, el
       desplazamiento es nulo y la carga emocional iguala.

       Más empates de lo que sugieren los ratings, por el mismo
       motivo por el que los partidos divisionales de NFL quedan más
       cerrados: los equipos se conocen y preparan específicamente.

   El efecto NO viene del historial concreto sino de la condición de
   derbi, igual que la compresión divisional en NFL. Se expone en
   `derby_adjustment()` porque quien pregunta por el historial entre
   dos equipos es quien necesita saber si son vecinos.

Reutilización del Core
------------------------
El cálculo estadístico genérico vive en core/utils/h2h_base.py y lo
comparten todos los deportes. Este módulo aporta lo específico:
obtener los encuentros del calendario, la separación por localía y la
lectura de derbi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.utils.h2h_base import H2HMetrics, compute_h2h

from sports.soccer.competitions import Competition, get_competition
from sports.soccer.schedule import SoccerMatchInfo
from sports.soccer.teams import canonical_team


# ── Dependencia inyectada ────────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """Interfaz mínima del calendario que este módulo consume."""

    def get_competition_matches(self, comp, season: int) -> list[SoccerMatchInfo]:
        ...


# ── Constantes de calibración ────────────────────────────────────────────────

# Ventana histórica, en temporadas.
#
# Cuatro y no tres como en NFL: la rotación de plantilla es menor
# (~15-20% anual frente al ~25%) y el calendario garantiza dos
# enfrentamientos por temporada, así que cuatro años dan ocho
# encuentros — muestra comparable a la de seis de NFL en tres años.
_DEFAULT_SEASONS_BACK = 4

# Decaimiento por temporada de antigüedad.
#
# 0.75 frente al 0.65 de NFL, por la misma razón: con menos recambio de
# jugadores, un partido de hace dos años conserva más información sobre
# los equipos actuales.
_SEASON_DECAY = 0.75

# Encuentros mínimos para considerar el historial informativo.
_MIN_MEETINGS       = 4   # historial global
_MIN_MEETINGS_VENUE = 3   # en el mismo estadio, donde la muestra es la mitad

# Ajuste de derbi, en goles sobre la ventaja de campo.
#
# El visitante lleva afición, el desplazamiento es nulo y la carga
# emocional iguala. El efecto documentado ronda los 0.10-0.15 goles de
# reducción sobre la ventaja de campo habitual.
_DEFAULT_DERBY_ADJUSTMENT = -0.12


# ── Derbis ───────────────────────────────────────────────────────────────────
#
# Pares de equipos de la misma ciudad o área metropolitana, en forma
# canónica. Se usan pares ordenados alfabéticamente para que la
# comparación sea simétrica sin duplicar entradas.
#
# Se limita a derbis URBANOS, no a rivalidades históricas: el efecto
# modelable —desplazamiento nulo, afición visitante, ausencia de
# ventaja logística— viene de la proximidad geográfica. El
# Madrid-Barcelona es la mayor rivalidad de España y no es un derbi en
# este sentido: hay 600 km de por medio.

_DERBIES: dict[str, frozenset[tuple[str, str]]] = {
    "epl": frozenset({
        ("arsenal", "tottenham"),                    # norte de Londres
        ("chelsea", "fulham"),
        ("arsenal", "chelsea"),
        ("chelsea", "tottenham"),
        ("crystal palace", "west ham united"),
        ("everton", "liverpool"),                    # Merseyside
        ("manchester city", "manchester united"),
        ("aston villa", "birmingham city"),
        ("newcastle united", "sunderland"),
    }),
    "laliga": frozenset({
        ("atletico madrid", "real madrid"),
        ("getafe", "real madrid"),
        ("atletico madrid", "getafe"),
        ("leganes", "real madrid"),
        ("barcelona", "espanyol"),                   # derbi barcelonés
        ("betis", "sevilla"),                        # derbi sevillano
        ("real betis", "sevilla"),
        ("athletic club", "real sociedad"),          # derbi vasco
        ("deportivo la coruna", "celta vigo"),       # derbi gallego
    }),
    "seriea": frozenset({
        ("ac milan", "inter"),                       # derbi della Madonnina
        ("lazio", "roma"),                           # derbi della Capitale
        ("juventus", "torino"),                      # derbi della Mole
        ("genoa", "sampdoria"),                      # derbi della Lanterna
        ("empoli", "fiorentina"),
    }),
    "bundesliga": frozenset({
        ("bayern munich", "munich 1860"),
        ("borussia dortmund", "schalke 04"),         # Revierderby
        ("hamburger sv", "st pauli"),                # derbi de Hamburgo
        ("hertha berlin", "union berlin"),           # derbi de Berlín
        ("cologne", "bayer leverkusen"),
        ("borussia m gladbach", "cologne"),
    }),
    "ligue1": frozenset({
        ("lyon", "saint etienne"),                   # derbi del Ródano
        ("marseille", "nice"),
        ("lille", "lens"),                           # derbi del norte
        ("nantes", "rennes"),
        ("paris saint germain", "paris fc"),
    }),
}


def is_derby(team_a: str, team_b: str, comp_id: str) -> bool:
    """
    True si ambos equipos son de la misma ciudad o área.

    La comparación es simétrica: el orden de los argumentos no importa.
    """
    canon_a = canonical_team(team_a, comp_id)
    canon_b = canonical_team(team_b, comp_id)
    if not canon_a or not canon_b or canon_a == canon_b:
        return False

    pair = tuple(sorted((canon_a, canon_b)))
    return pair in _DERBIES.get(str(comp_id).strip().lower(), frozenset())


@dataclass(frozen=True)
class SoccerH2HResult:
    """
    Historial entre dos equipos, con separación por localía.

    Campos
    ------
    home_team / away_team -- Equipos del partido a proyectar, en forma
                             canónica.
    metrics               -- Métricas genéricas del Core.
    n_meetings            -- Encuentros en la ventana.
    n_at_venue            -- De esos, los jugados en el mismo estadio.
    is_derby              -- Si son equipos de la misma ciudad.
    is_reliable           -- Si la muestra global alcanza el mínimo.
    venue_reliable        -- Si la muestra en el mismo estadio lo hace.
                             Umbral distinto porque es la mitad.

    Agregados ponderados por antigüedad
    -----------------------------------
    weighted_margin  -- Margen medio desde la perspectiva del local del
                        partido a proyectar.
    weighted_total   -- Goles totales medios.
    venue_margin     -- Margen medio SOLO en el mismo estadio. Más
                        informativo cuando hay muestra, porque el
                        modelo proyecta un partido en un sitio
                        concreto.
    """
    home_team:  str
    away_team:  str
    comp_id:    str
    metrics:    H2HMetrics

    n_meetings:     int = 0
    n_at_venue:     int = 0
    seasons_covered: int = 0
    is_derby:       bool = False
    is_reliable:    bool = False
    venue_reliable: bool = False

    weighted_margin: float | None = None
    weighted_total:  float | None = None
    venue_margin:    float | None = None
    venue_total:     float | None = None

    effective_sample: float = 0.0
    draws: int = 0

    @property
    def has_data(self) -> bool:
        return self.n_meetings > 0

    @property
    def draw_rate(self) -> float | None:
        """
        Proporción de empates en el historial.

        Solo se expone si la muestra es fiable: en fútbol el empate
        ronda el 25% de los partidos, así que dos empates en tres
        encuentros no significa nada.
        """
        if not self.is_reliable or self.n_meetings == 0:
            return None
        return round(self.draws / self.n_meetings, 4)

    def to_metadata(self) -> dict:
        """
        Metadatos para TeamFeatures.sport_metadata.

        Las métricas numéricas se emiten SOLO si la muestra es fiable.
        Publicar un margen calculado sobre dos encuentros invitaría al
        modelo a usarlo como señal, que es justo lo que `is_reliable`
        existe para impedir.
        """
        base = {
            "h2h_n_meetings":      self.n_meetings,
            "h2h_n_at_venue":      self.n_at_venue,
            "h2h_seasons":         self.seasons_covered,
            "h2h_is_derby":        self.is_derby,
            "h2h_is_reliable":     self.is_reliable,
            "h2h_venue_reliable":  self.venue_reliable,
            "h2h_effective_sample": round(self.effective_sample, 3),
        }
        if self.is_reliable:
            base.update({
                "h2h_home_win_rate":   self.metrics.win_rate_a,
                "h2h_draw_rate":       self.draw_rate,
                "h2h_weighted_margin": self.weighted_margin,
                "h2h_weighted_total":  self.weighted_total,
            })
        if self.venue_reliable:
            base.update({
                "h2h_venue_margin": self.venue_margin,
                "h2h_venue_total":  self.venue_total,
            })
        return base


class SoccerH2HFetcher:
    """
    Historial de enfrentamientos entre equipos.

    Parámetros
    ----------
    schedule_fetcher -- Fuente de partidos.
    config_loader    -- ConfigLoader con soccer.yaml.
    seasons_back     -- Ventana histórica en temporadas. Default 4.
    """

    def __init__(
        self,
        schedule_fetcher: ScheduleSource,
        config_loader     = None,
        seasons_back: int = _DEFAULT_SEASONS_BACK,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._config = config_loader
        self._seasons_back = seasons_back

        self._derby_adjustment = self._cfg(
            "soccer.derby_home_advantage_adjustment", _DEFAULT_DERBY_ADJUSTMENT
        )

        # Caché: {(local, visitante, comp, temporada, corte): resultado}
        self._cache: dict[tuple, SoccerH2HResult] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def get_h2h(
        self,
        home_team:  str,
        away_team:  str,
        comp:       Competition | str,
        season:     int,
        as_of_date: str,
    ) -> SoccerH2HResult:
        """
        Historial entre dos equipos anterior a la fecha de corte.

        `as_of_date` aplica la misma barrera que en team_stats.py: un
        encuentro del propio día es el que se está proyectando.

        Nunca lanza: sin datos devuelve un resultado con
        `has_data=False` e `is_reliable=False`.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return self._empty(home_team, away_team, str(comp))

        canon_home = canonical_team(home_team, competition.comp_id)
        canon_away = canonical_team(away_team, competition.comp_id)
        if not canon_home or not canon_away:
            return self._empty(home_team, away_team, competition.comp_id)

        key = (canon_home, canon_away, competition.comp_id, season, as_of_date)
        if key in self._cache:
            return self._cache[key]

        result = self._compute(
            canon_home, canon_away, competition, season, as_of_date
        )
        self._cache[key] = result
        return result

    def derby_adjustment(self, is_derby_match: bool) -> float:
        """
        Ajuste de la ventaja de campo en derbis, en goles.

        Negativo: el derbi REDUCE la ventaja del local. El visitante
        lleva afición, el desplazamiento es nulo y la carga emocional
        iguala.

        El efecto no depende del historial concreto sino de la
        condición de derbi, igual que la compresión divisional de NFL.
        Devuelve 0.0 para partidos normales.
        """
        return self._derby_adjustment if is_derby_match else 0.0

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── Cálculo ───────────────────────────────────────────────────────────────

    def _compute(
        self,
        home:       str,
        away:       str,
        comp:       Competition,
        season:     int,
        as_of_date: str,
    ) -> SoccerH2HResult:
        meetings = self._find_meetings(home, away, comp, season, as_of_date)
        derby = is_derby(home, away, comp.comp_id)

        if not meetings:
            return SoccerH2HResult(
                home_team=home, away_team=away, comp_id=comp.comp_id,
                metrics=compute_h2h(home, away, []), is_derby=derby,
            )

        metrics = compute_h2h(
            team_a_id=home, team_b_id=away,
            meetings=[m["record"] for m in meetings],
        )

        # Encuentros en el MISMO estadio: aquellos donde el local del
        # partido a proyectar también era local.
        at_venue = [m for m in meetings if m["record"]["home_id"] == home]

        seasons = {m["season"] for m in meetings}
        draws = sum(1 for m in meetings
                    if m["record"]["home_score"] == m["record"]["away_score"])

        overall = self._weighted(meetings, reference=home)
        venue = self._weighted(at_venue, reference=home) if at_venue else {}

        return SoccerH2HResult(
            home_team=home, away_team=away, comp_id=comp.comp_id,
            metrics=metrics,
            n_meetings=len(meetings),
            n_at_venue=len(at_venue),
            seasons_covered=len(seasons),
            is_derby=derby,
            is_reliable=len(meetings) >= _MIN_MEETINGS,
            venue_reliable=len(at_venue) >= _MIN_MEETINGS_VENUE,
            weighted_margin=overall.get("margin"),
            weighted_total=overall.get("total"),
            venue_margin=venue.get("margin"),
            venue_total=venue.get("total"),
            effective_sample=overall.get("weight_sum", 0.0),
            draws=draws,
        )

    def _find_meetings(
        self,
        home:       str,
        away:       str,
        comp:       Competition,
        season:     int,
        as_of_date: str,
    ) -> list[dict]:
        """
        Encuentros directos en la ventana histórica.

        Recorre las temporadas de la ventana y filtra los partidos
        finalizados entre ambos equipos, anteriores a la fecha de
        corte.

        Cada entrada lleva su peso de decaimiento ya calculado, para
        que los agregados no tengan que recalcularlo.
        """
        meetings: list[dict] = []
        pair = {home, away}

        for offset in range(self._seasons_back + 1):
            target = season - offset
            try:
                matches = self._schedule.get_competition_matches(comp, target)
            except Exception:
                continue

            if not isinstance(matches, list):
                continue

            for match in matches:
                if not match.is_final or not match.date:
                    continue
                if match.date >= as_of_date:
                    continue
                if {match.home, match.away} != pair:
                    continue
                if match.home_goals is None or match.away_goals is None:
                    continue

                meetings.append({
                    "season": target,
                    "weight": _SEASON_DECAY ** offset,
                    "record": {
                        "home_id":    match.home,
                        "away_id":    match.away,
                        "home_score": match.home_goals,
                        "away_score": match.away_goals,
                    },
                })

        return meetings

    @staticmethod
    def _weighted(meetings: list[dict], reference: str) -> dict:
        """
        Margen y total medios ponderados por antigüedad.

        El margen se normaliza a la perspectiva de `reference` —el
        local del partido a proyectar— y NO del que fue local en cada
        encuentro histórico.

        Sin esa normalización, una serie donde un equipo domina daría
        un promedio cercano a cero por cancelación de signos: los
        equipos juegan uno en cada estadio, así que las victorias
        fuera restarían en vez de sumar.

        Es exactamente el bug que apareció en el plugin NFL: un equipo
        ganaba 5 de 6 enfrentamientos y el margen ponderado salía
        NEGATIVO.
        """
        weight_sum = sum(m["weight"] for m in meetings)
        if weight_sum <= 0:
            return {"margin": None, "total": None, "weight_sum": 0.0}

        margin_sum = 0.0
        total_sum = 0.0

        for m in meetings:
            record = m["record"]
            w = m["weight"]
            hs = float(record["home_score"])
            aws = float(record["away_score"])

            if record["home_id"] == reference:
                margin_sum += w * (hs - aws)
            else:
                margin_sum += w * (aws - hs)

            total_sum += w * (hs + aws)

        return {
            "margin":     round(margin_sum / weight_sum, 3),
            "total":      round(total_sum / weight_sum, 3),
            "weight_sum": round(weight_sum, 4),
        }

    @staticmethod
    def _empty(home: str, away: str, comp_id: str) -> SoccerH2HResult:
        return SoccerH2HResult(
            home_team=home, away_team=away, comp_id=comp_id,
            metrics=compute_h2h(home, away, []),
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


# ── Adaptador ────────────────────────────────────────────────────────────────

def h2h_metadata(result: SoccerH2HResult) -> dict:
    """
    Adaptador para TeamFeatures.sport_metadata.

    Equivalente a las funciones del mismo nombre en los plugins de MLB
    y NFL, para que los providers las consuman con idéntica forma.
    """
    return result.to_metadata()


def derbies_in(comp_id: str) -> list[tuple[str, str]]:
    """Derbis declarados de una competición, para diagnóstico."""
    return sorted(_DERBIES.get(str(comp_id).strip().lower(), frozenset()))