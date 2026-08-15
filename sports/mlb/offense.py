"""
sports/mlb/offense.py

OffenseFetcher: estadísticas ofensivas y carreras recientes para MLB.

Migrado de analysis/offense.py del sistema MLB con dos bugs críticos
corregidos documentados en CRITICAL_FINDINGS_VALIDATION.md:

BUG F1 — runs_recientes_lista SIEMPRE VACÍO
-------------------------------------------
El sistema MLB obtenía el schedule del equipo (últimos 30 días) pero
nunca extraía las carreras de cada partido. El resultado:

    runs_recientes_lista = []  # SIEMPRE — el loop nunca llenaba la lista

Esto hacía que TeamFeatures.recent_scores=[] en producción, lo cual:
- Desactivaba EnsembleModel (has_sufficient_sample=False)
- Hacía recent_avg=0.0 (promedio de lista vacía)
- Hacía has_sufficient_sample=False siempre

Corrección: el loop ahora extrae correctamente las carreras del equipo
desde la respuesta del schedule de MLB Stats API:
    game['teams']['home']['score'] o game['teams']['away']['score']
según si el equipo jugó de local o visitante.

BUG F2 — offense_index con fórmula INVERTIDA
----------------------------------------------
El sistema MLB calculaba:
    offense_index = OPS_LIGA / OPS_equipo  ← INCORRECTO

Esto hacía que equipos con OPS ALTO (mejor ofensiva) tuvieran un
offense_index BAJO, y vice versa — el modelo penalizaba a los mejores
equipos ofensivos.

La fórmula correcta (misma que defense_index pero sin invertir):
    offense_index = OPS_equipo / OPS_LIGA

Con OPS_LIGA = 0.720 (promedio MLB 2024):
    OPS=0.800 → offense_index=1.111 (11% mejor que liga) ✓
    OPS=0.650 → offense_index=0.903 (10% peor que liga)  ✓

La corrección fue validada en los tests del Bloque 0 donde se
verificó que `defense_index` también usaba ERA/LEAGUE_ERA (no
invertido) — consistencia entre ambas métricas normalización.

Posición en el pipeline
------------------------
MLBDataProvider.enrich_event(event)
    ↓
OffenseFetcher.fetch(team_id) → OffenseStats
    ↓
OffenseFetcher.recent_scores(team_id, game_pks) → list[float]
    ↓
TeamFeatures(
    offense_index = ops / OPS_LIGA,          # CORREGIDO F2
    recent_scores = [3, 7, 4, 2, 8, ...],    # CORREGIDO F1
    recent_avg    = mean(recent_scores),
)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sports.mlb.statcast import _LEAGUE_OPS, _safe_float, _current_season

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

# Ventana de partidos recientes para recent_scores
_DEFAULT_RECENT_DAYS: int = 30
_DEFAULT_RECENT_MAX:  int = 15   # Máximo de partidos en la ventana


# ── Dataclass de estadísticas ofensivas ───────────────────────────────────────

@dataclass(frozen=True)
class OffenseStats:
    """
    Estadísticas ofensivas de un equipo en una temporada.

    Campos
    ------
    team_id     -- ID del equipo en MLB Stats API.
    season      -- Temporada (año).
    ops         -- On-base Plus Slugging (temporada completa).
    obp         -- On-base Percentage.
    slg         -- Slugging Percentage.
    avg         -- Batting Average.
    ops_vs_rhp  -- OPS específico vs pitchers derechos.
    ops_vs_lhp  -- OPS específico vs pitchers zurdos.
    runs_per_game -- Carreras por partido en la temporada.
    """
    team_id:       int
    season:        int
    ops:           float | None = None
    obp:           float | None = None
    slg:           float | None = None
    avg:           float | None = None
    ops_vs_rhp:    float | None = None
    ops_vs_lhp:    float | None = None
    runs_per_game: float | None = None

    def offense_index(self) -> float:
        """
        Índice de ofensiva normalizado a la liga.

        CORRECCIÓN BUG F2: ops / OPS_LIGA (no invertido).
            ops=0.800 → 0.800/0.720 = 1.111 (mejor que liga) ✓
            ops=0.650 → 0.650/0.720 = 0.903 (peor que liga)  ✓

        El sistema MLB original usaba OPS_LIGA/ops que producía el
        resultado contrario — equipos buenos parecían tener bajo índice.
        """
        if self.ops is None or self.ops <= 0:
            return 1.0
        return round(self.ops / _LEAGUE_OPS, 4)

    def ops_vs_hand(self, hand: str | None) -> float | None:
        """
        OPS específico contra la mano del abridor rival.

        Retorna el split apropiado si está disponible,
        o el OPS general como fallback.
        """
        if hand == 'R' and self.ops_vs_rhp is not None:
            return self.ops_vs_rhp
        if hand == 'L' and self.ops_vs_lhp is not None:
            return self.ops_vs_lhp
        return self.ops

    def offense_index_vs_hand(self, hand: str | None) -> float:
        """
        offense_index usando el OPS específico vs la mano del abridor.

        Más preciso que offense_index() genérico cuando se conoce
        la mano del pitcher rival (disponible desde ProbablePitcher.hand).
        """
        ops_split = self.ops_vs_hand(hand)
        if ops_split is None or ops_split <= 0:
            return self.offense_index()
        return round(ops_split / _LEAGUE_OPS, 4)

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "ops":            self.ops,
            "obp":            self.obp,
            "slg":            self.slg,
            "avg":            self.avg,
            "ops_vs_rhp":     self.ops_vs_rhp,
            "ops_vs_lhp":     self.ops_vs_lhp,
            "runs_per_game":  self.runs_per_game,
        }


# ── Fetcher principal ─────────────────────────────────────────────────────────

class OffenseFetcher:
    """
    Obtiene estadísticas ofensivas y carreras recientes desde MLB Stats API.

    Parámetros
    ----------
    recent_days  -- Ventana de días para recent_scores. Default 30.
    recent_max   -- Máximo de partidos en recent_scores. Default 15.
    timeout      -- Timeout HTTP en segundos. Default 10.
    """

    def __init__(
        self,
        recent_days: int = _DEFAULT_RECENT_DAYS,
        recent_max:  int = _DEFAULT_RECENT_MAX,
        timeout:     int = 10,
    ) -> None:
        self._recent_days = recent_days
        self._recent_max  = recent_max
        self._timeout     = timeout

    def fetch(
        self,
        team_id: int,
        season:  int | None = None,
    ) -> OffenseStats:
        """
        Obtiene estadísticas ofensivas de temporada del equipo.

        Incluye splits vsRHP/vsLHP si están disponibles en la API.

        Parámetros
        ----------
        team_id  -- ID del equipo en MLB Stats API.
        season   -- Temporada. Default: actual.

        Retorna
        -------
        OffenseStats con todos los campos disponibles.
        Si la API falla, retorna OffenseStats con campos None.
        """
        season = season or _current_season()
        raw    = self._fetch_team_hitting(team_id, season)

        if raw is None:
            return OffenseStats(team_id=team_id, season=season)

        return self._parse_offense(raw, team_id, season)

    def fetch_recent_scores(
        self,
        team_id:    int,
        date_to:    str | None = None,
    ) -> list[float]:
        """
        Obtiene las carreras anotadas por el equipo en los últimos
        N partidos finalizados.

        CORRECCIÓN BUG F1: el sistema MLB obtenía el schedule pero
        nunca extraía las carreras de cada partido, retornando siempre
        una lista vacía. Aquí se extrae correctamente el score del
        equipo (home o away) de cada partido finalizado.

        Parámetros
        ----------
        team_id   -- ID del equipo en MLB Stats API.
        date_to   -- Fecha fin en 'YYYY-MM-DD'. Default: hoy UTC.

        Retorna
        -------
        list[float] con las carreras anotadas en los últimos partidos
        finalizados, ordenados del más antiguo al más reciente.
        Lista vacía si la API falla o no hay partidos finalizados.
        """
        if not _REQUESTS_AVAILABLE or _requests is None:
            return []

        # Calcular rango de fechas
        today    = datetime.now(timezone.utc)
        date_end = date_to or today.strftime("%Y-%m-%d")
        date_start = (
            datetime.strptime(date_end, "%Y-%m-%d")
            - timedelta(days=self._recent_days)
        ).strftime("%Y-%m-%d")

        url    = f"{_MLB_API_BASE}/schedule"
        params = {
            "sportId":    1,
            "teamId":     team_id,
            "startDate":  date_start,
            "endDate":    date_end,
            "gameType":   "R",          # Regular season únicamente
            "hydrate":    "linescore",  # Necesario para obtener scores
        }

        try:
            resp = _requests.get(url, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        scores: list[float] = []

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):

                # Solo partidos finalizados
                status = game.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    continue

                # ── CORRECCIÓN BUG F1 ────────────────────────────────────
                # El sistema original hacía:
                #   for game in games:
                #       pass  # nunca extraía el score
                #   runs_recientes_lista = []  # siempre vacío
                #
                # Corrección: extraer score del equipo (home o away)
                teams     = game.get("teams", {})
                home_data = teams.get("home", {})
                away_data = teams.get("away", {})

                home_id = home_data.get("team", {}).get("id")
                away_id = away_data.get("team", {}).get("id")

                # Determinar si el equipo jugó como local o visitante
                if home_id == team_id:
                    score = home_data.get("score")
                elif away_id == team_id:
                    score = away_data.get("score")
                else:
                    continue  # partido sin relación con el equipo

                if score is not None:
                    try:
                        scores.append(float(score))
                    except (ValueError, TypeError):
                        continue

        # Ordenar del más antiguo al más reciente y limitar
        # (las fechas ya vienen ordenadas de la API pero por seguridad)
        return scores[-self._recent_max:]

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _fetch_team_hitting(self, team_id: int, season: int) -> dict | None:
        """Fetch de estadísticas de bateo del equipo."""
        if not _REQUESTS_AVAILABLE or _requests is None:
            return None

        url    = f"{_MLB_API_BASE}/teams/{team_id}/stats"
        params = {
            "stats":  "season",
            "group":  "hitting",
            "season": season,
        }
        try:
            resp = _requests.get(url, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def _parse_offense(
        self,
        raw:     dict,
        team_id: int,
        season:  int,
    ) -> OffenseStats:
        """Parsea la respuesta de MLB Stats API a OffenseStats."""
        stats_groups = raw.get("stats", [])
        if not stats_groups:
            return OffenseStats(team_id=team_id, season=season)

        splits = stats_groups[0].get("splits", [])
        if not splits:
            return OffenseStats(team_id=team_id, season=season)

        stat = splits[0].get("stat", {})

        # OPS como suma de OBP + SLG si no viene precalculado
        obp   = _safe_float(stat.get("obp"))
        slg   = _safe_float(stat.get("slg"))
        ops   = _safe_float(stat.get("ops"))
        if ops is None and obp is not None and slg is not None:
            ops = round(obp + slg, 4)

        # Carreras por partido
        runs  = _safe_float(stat.get("runs"))
        games = _safe_float(stat.get("gamesPlayed"))
        rpg   = round(runs / games, 3) if runs and games else None

        return OffenseStats(
            team_id       = team_id,
            season        = season,
            ops           = ops,
            obp           = obp,
            slg           = slg,
            avg           = _safe_float(stat.get("avg")),
            runs_per_game = rpg,
        )