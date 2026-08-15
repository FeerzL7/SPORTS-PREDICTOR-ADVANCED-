"""
sports/mlb/defense.py

DefenseFetcher: estadísticas de fielding (defensa de guante) para MLB.

MÓDULO AÑADIDO EN LA AUDITORÍA 2026-08 — este archivo no existía en el
repositorio. `sports/mlb/provider.py` lo importaba
(`from sports.mlb.defense import DefenseFetcher, FieldingStats`) desde
el momento en que se escribió `provider.py`, pero el archivo nunca se
creó. Era el bloqueador más severo de los cuatro encontrados en la
auditoría: al ser un `ModuleNotFoundError` en tiempo de import, tumbaba
la carga de `sports.mlb.provider` completo — y por extensión,
`sports.mlb.plugin.MLBPlugin` — antes de que cualquier otro código del
plugin pudiera ejecutarse.

Por qué "fielding" es un ajuste menor, no la señal principal
----------------------------------------------------------------
`BullpenFetcher.defense_index()` (ERA del abridor + bullpen) ya captura
la mayor parte de la variación en carreras permitidas — el pitcheo
domina sobre el fielding en el impacto real sobre el resultado de un
partido de MLB. Este módulo NO reemplaza esa señal: la ajusta
marginalmente. `combined_defense_index()` aplica un ajuste multiplicativo
pequeño (`_FIELDING_WEIGHT`) sobre el `defense_index` ya calculado desde
ERA, en la misma dirección de signo que usa `bullpen.py`
(`defense_index_from_combined_era`): valores más altos de
`defense_index` = mejor defensa. Esto es consistente con la fórmula real
de `bullpen.py` (`LEAGUE_ERA / ERA_combinada` — matemáticamente, ratio >
1.0 cuando el equipo permite MENOS carreras que la liga), aunque el
docstring de esa función en el repo actual describe el signo al revés;
no se corrige aquí para mantener acotado el alcance de este fix a
"crear el módulo faltante", pero queda documentado como hallazgo
adicional para una tarea de calibración aparte.

Posición en el pipeline
------------------------
MLBDataProvider.enrich_event(event)
    ↓
DefenseFetcher.fetch(team_id, season) → FieldingStats
    ↓
DefenseFetcher.combined_defense_index(defense_index_de_pitcheo, fielding)
    → float (defense_index ajustado)
    ↓
TeamFeatures(
    defense_index  = <ajustado>,
    sport_metadata = {..., **fielding.to_metadata()},
)

Endpoint MLB Stats API
-----------------------
/api/v1/teams/{teamId}/stats?stats=season&group=fielding&season={season}
"""

from __future__ import annotations

from dataclasses import dataclass

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

# Fielding percentage promedio de liga MLB (referencia histórica estable,
# ronda .983-.985 temporada a temporada). Se usa como base de comparación
# para el ajuste multiplicativo, igual que _LEAGUE_ERA/_LEAGUE_OPS en
# statcast.py.
_LEAGUE_FIELDING_PCT: float = 0.984

# Peso del ajuste de fielding sobre el defense_index ya calculado desde
# ERA. Deliberadamente pequeño: el fielding es la señal secundaria, no la
# principal (ver docstring del módulo). Un equipo con fielding_pct 1%
# por encima de liga solo mueve el defense_index final ~0.1%
# (_FIELDING_WEIGHT × 1%), no 1% completo.
_FIELDING_WEIGHT: float = 0.10

# Mínimo de "chances" (oportunidades de fildeo) para considerar el
# fielding_pct de la temporada una muestra confiable. Por debajo de esto
# (ej. inicio de temporada), se prefiere no aplicar el ajuste.
_MIN_CHANCES_RELIABLE: int = 200


# ── Dataclass de estadísticas de fielding ─────────────────────────────────────

@dataclass(frozen=True)
class FieldingStats:
    """
    Estadísticas de fielding de un equipo en una temporada.

    Todos los campos numéricos son opcionales (None) si la API falla o
    no hay datos — el mismo patrón de fallback documentado que usan
    `BullpenStats`, `OffenseStats` y `PitcherStatcast` en este plugin.

    Campos
    ------
    team_id       -- ID del equipo en MLB Stats API.
    season        -- Temporada (año).
    fielding_pct  -- Porcentaje de fildeo del equipo (0.0-1.0).
    errors        -- Errores cometidos en la temporada.
    double_plays  -- Doble plays completados.
    assists       -- Asistencias defensivas.
    putouts       -- Outs registrados por fildeo.
    chances       -- Oportunidades totales de fildeo (assists + putouts +
                    errors). Usado para juzgar si la muestra es confiable
                    vía `has_sufficient_sample`.
    """
    team_id:      int
    season:       int
    fielding_pct: float | None = None
    errors:       int   | None = None
    double_plays: int   | None = None
    assists:      int   | None = None
    putouts:      int   | None = None
    chances:      int   | None = None

    @property
    def has_sufficient_sample(self) -> bool:
        """True si hay suficientes oportunidades de fildeo para confiar en fielding_pct."""
        return self.chances is not None and self.chances >= _MIN_CHANCES_RELIABLE

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "fielding_pct": self.fielding_pct,
            "errors":       self.errors,
            "double_plays": self.double_plays,
        }


# ── Fetcher principal ─────────────────────────────────────────────────────────

class DefenseFetcher:
    """
    Obtiene estadísticas de fielding desde MLB Stats API y las combina
    con el defense_index calculado desde ERA (BullpenFetcher).

    Parámetros
    ----------
    timeout  -- Timeout HTTP en segundos. Default 10.
    """

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout
        # Caché en memoria por (team_id, season) — mismo patrón que
        # BullpenFetcher/StatcastFetcher: evita refetch dentro de la
        # misma ejecución del pipeline si dos partidos comparten equipo.
        self._cache: dict[tuple[int, int], FieldingStats] = {}

    def fetch(self, team_id: int, season: int) -> FieldingStats:
        """
        Obtiene las estadísticas de fielding del equipo para la temporada.

        Retorna FieldingStats con todos los campos numéricos en None si
        la API falla — nunca lanza excepción (el caller en provider.py
        ya envuelve esto en `_safe_fetch`, pero el fetcher es
        defensivo por sí mismo, igual que sus pares).
        """
        key = (team_id, season)
        if key in self._cache:
            return self._cache[key]

        raw = self._fetch_raw(team_id, season)
        result = (
            self._parse(raw, team_id, season)
            if raw is not None
            else FieldingStats(team_id=team_id, season=season)
        )
        self._cache[key] = result
        return result

    def combined_defense_index(
        self,
        defense_index: float,
        fielding:      FieldingStats,
    ) -> float:
        """
        Ajusta un defense_index ya calculado desde ERA con el
        fielding_pct del equipo.

        Si la muestra de fielding no es confiable (`has_sufficient_sample`
        False) o falta el dato, retorna `defense_index` sin modificar —
        el caller en provider.py ya solo llama a este método cuando
        `fielding.fielding_pct is not None`, pero esta función es
        defensiva por sí misma para poder usarse de forma independiente.

        Fórmula: multiplica defense_index por un factor que se acerca a
        1.0 cuanto más fielding_pct se aproxime al promedio de liga, y se
        aleja (hacia arriba o abajo) proporcionalmente a
        `_FIELDING_WEIGHT` cuando el equipo fildea mejor o peor que la
        liga.
        """
        if fielding.fielding_pct is None or fielding.fielding_pct <= 0:
            return defense_index
        if not fielding.has_sufficient_sample:
            return defense_index

        fielding_factor = fielding.fielding_pct / _LEAGUE_FIELDING_PCT
        adjustment = 1.0 + _FIELDING_WEIGHT * (fielding_factor - 1.0)

        return round(defense_index * adjustment, 4)

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _fetch_raw(self, team_id: int, season: int) -> dict | None:
        """Fetch crudo de estadísticas de fielding desde MLB Stats API."""
        if not _REQUESTS_AVAILABLE or _requests is None:
            return None

        url    = f"{_MLB_API_BASE}/teams/{team_id}/stats"
        params = {
            "stats":  "season",
            "group":  "fielding",
            "season": season,
        }
        try:
            resp = _requests.get(url, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _parse(raw: dict, team_id: int, season: int) -> FieldingStats:
        """Parsea la respuesta de MLB Stats API a FieldingStats."""
        stats_groups = raw.get("stats", [])
        if not stats_groups:
            return FieldingStats(team_id=team_id, season=season)

        splits = stats_groups[0].get("splits", [])
        if not splits:
            return FieldingStats(team_id=team_id, season=season)

        stat = splits[0].get("stat", {})

        def _safe_float(value):
            try:
                return float(value) if value is not None else None
            except (ValueError, TypeError):
                return None

        def _safe_int(value):
            try:
                return int(value) if value is not None else None
            except (ValueError, TypeError):
                return None

        assists = _safe_int(stat.get("assists"))
        putouts = _safe_int(stat.get("putOuts"))
        errors  = _safe_int(stat.get("errors"))

        chances = None
        if assists is not None and putouts is not None and errors is not None:
            chances = assists + putouts + errors

        return FieldingStats(
            team_id      = team_id,
            season       = season,
            fielding_pct = _safe_float(stat.get("fielding")),
            errors       = errors,
            double_plays = _safe_int(stat.get("doublePlays")),
            assists      = assists,
            putouts      = putouts,
            chances      = chances,
        )
