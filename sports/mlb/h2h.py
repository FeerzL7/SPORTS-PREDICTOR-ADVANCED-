"""
sports/mlb/h2h.py

MLBH2HFetcher: historial de enfrentamientos directos para MLB.

Migrado de analysis/h2h.py del sistema MLB.

CORRECCIÓN DE CONTRATO (auditoría 2026-08): este módulo originalmente
importaba `H2HRecord`, `H2HStats`, `compute_h2h_stats` y `filter_recent`
desde `core/utils/h2h_base.py` — pero ese módulo del Core nunca definió
esos nombres (define `H2HMetrics` y `compute_h2h`, una API más simple).
Los dos archivos se diseñaron para trabajar juntos y nunca se
integraron: el import fallaba en el 100% de los casos (ImportError al
cargar el módulo, antes de ejecutar una sola línea de negocio).

Decisión de diseño (Opción A del roadmap de remediación): el Core se
queda simple y genérico (`H2HMetrics`/`compute_h2h`, sin conocer nada de
MLB). `H2HRecord` — el partido individual crudo, con `game_id`,
`season`, IDs de equipo — es un concepto de datos crudos específico de
cómo MLB expone su historial, así que pasa a vivir aquí, en el plugin,
no en el Core. La función `h2h_metadata()` de este módulo es el
adaptador que traduce `H2HMetrics` (genérico) a las claves de
`sport_metadata` que `MLBProjectionModel` puede llegar a consumir.

Posición en el pipeline
------------------------
MLBDataProvider.enrich_event(event)
    ↓
MLBH2HFetcher.fetch(home_team_id, away_team_id)
    → list[H2HRecord]                          (crudo, plugin-local)
    ↓
MLBH2HFetcher.get_stats(home_team_id, away_team_id)
    → core.utils.h2h_base.H2HMetrics           (agregado, genérico)
    ↓
h2h_metadata(metrics) → dict apto para TeamFeatures.sport_metadata:
    {
        'h2h_n_meetings':     8,
        'h2h_home_win_rate':  0.55,
        'h2h_avg_total':      9.2,
        'h2h_weight':         0.85,
    }

`h2h_weight` es 0.0 si hay menos de `_MIN_MEETINGS_FOR_WEIGHT` encuentros
en la muestra (historial insuficiente para confiar en él) y escala hasta
`_MAX_H2H_WEIGHT` conforme crece la muestra. `MLBProjectionModel` puede
usar ese peso para decidir cuánto ponderar el H2H frente a las métricas
de temporada actuales — hoy no lo consume (ver Fase 4.3 del roadmap de
remediación: decidir si se integra a la fórmula de proyección o se
retira el fetch para no pagar llamadas a la API sin usarlas).

Endpoint MLB Stats API
-----------------------
/api/v1/schedule?sportId=1&teamId={teamId}&opponentId={opponentId}
    &startDate={start}&endDate={end}&gameType=R&hydrate=linescore

Retorna partidos de temporada regular entre los dos equipos en el
rango de fechas especificado, incluyendo scores finales.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from core.utils.h2h_base import H2HMetrics, compute_h2h

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

# Ventana histórica por defecto: últimas 3 temporadas
_DEFAULT_SEASONS_BACK: int = 3
_DEFAULT_LAST_N:       int = 20

# Mínimo de encuentros para que h2h_weight deje de ser 0.0 — con menos
# muestra que esto, el historial no es confiable y se ignora.
_MIN_MEETINGS_FOR_WEIGHT: int = 5

# Techo del peso h2h_weight. Nunca debe dominar por completo sobre las
# métricas de temporada actual, sin importar cuántos encuentros haya.
_MAX_H2H_WEIGHT: float = 0.85

# Encuentros necesarios para que h2h_weight sature en _MAX_H2H_WEIGHT.
_SATURATION_MEETINGS: int = _MIN_MEETINGS_FOR_WEIGHT * 3


# ── Registro crudo de un encuentro (plugin-local) ─────────────────────────────

@dataclass(frozen=True)
class H2HRecord:
    """
    Un encuentro directo individual entre dos equipos MLB, tal como lo
    expone MLB Stats API.

    Plugin-local a propósito: `game_id`, `season` y la resolución de
    `winner_id` son detalles de cómo MLB estructura su schedule, no
    conceptos que el Core deba conocer. El Core (`core/utils/h2h_base.py`)
    solo entiende `dict`s planos con `home_id/away_id/home_score/
    away_score` — ver `to_meeting_dict()`.

    Campos
    ------
    date          -- Fecha del partido, 'YYYY-MM-DD'.
    home_team_id  -- ID del equipo que jugó de local ESE partido
                    (no necesariamente el home_team_id del partido a
                    proyectar — dos equipos que se enfrentan varias
                    veces alternan localía).
    away_team_id  -- ID del equipo que jugó de visitante ESE partido.
    home_score    -- Carreras del local.
    away_score    -- Carreras del visitante.
    winner_id     -- ID del equipo ganador. None si no se pudo resolver.
    season        -- Año de la temporada del encuentro.
    game_id       -- game_pk de MLB Stats API como string. None si
                    faltaba en la respuesta.
    """
    date:         str
    home_team_id: int
    away_team_id: int
    home_score:   float
    away_score:   float
    winner_id:    int | None
    season:       int
    game_id:      str | None

    def to_meeting_dict(self) -> dict:
        """
        Convierte a la forma de dict plano que
        `core.utils.h2h_base.compute_h2h()` espera en su parámetro
        `meetings`. Es la frontera exacta entre el modelo de datos del
        plugin (este dataclass) y la función genérica del Core.
        """
        return {
            "home_id":    self.home_team_id,
            "away_id":    self.away_team_id,
            "home_score": self.home_score,
            "away_score": self.away_score,
        }


def h2h_metadata(metrics: H2HMetrics) -> dict:
    """
    Adapta `H2HMetrics` (genérico, del Core) a un dict apto para
    `TeamFeatures.sport_metadata`.

    Este es el punto único donde vive la traducción "métrica genérica →
    clave específica que el modelo de proyección de MLB podría leer".
    Si mañana se decide (Fase 4.3 del roadmap) incorporar H2H a la
    fórmula de `MLBProjectionModel`, este es el dict del que debe leer.

    `h2h_weight` escala linealmente de 0.0 (menos de
    `_MIN_MEETINGS_FOR_WEIGHT` encuentros — muestra insuficiente para
    confiar en ella) hasta `_MAX_H2H_WEIGHT` conforme crece la muestra,
    saturando en `_SATURATION_MEETINGS` encuentros.
    """
    if not metrics.has_data or metrics.n_meetings < _MIN_MEETINGS_FOR_WEIGHT:
        weight = 0.0
    else:
        weight = min(
            _MAX_H2H_WEIGHT,
            _MAX_H2H_WEIGHT * metrics.n_meetings / _SATURATION_MEETINGS,
        )

    return {
        "h2h_n_meetings":    metrics.n_meetings,
        "h2h_home_win_rate": metrics.win_rate_a,
        "h2h_avg_total":     metrics.avg_total,
        "h2h_weight":        round(weight, 4),
    }


class MLBH2HFetcher:
    """
    Obtiene y analiza el historial H2H entre dos equipos MLB.

    Parámetros
    ----------
    seasons_back  -- Temporadas históricas a consultar. Default 3.
                    Con más temporadas hay más muestra pero los datos
                    son menos representativos del equipo actual.
    last_n        -- Máximo de partidos más recientes a usar. Default 20.
    timeout       -- Timeout HTTP. Default 10s.
    """

    def __init__(
        self,
        seasons_back: int = _DEFAULT_SEASONS_BACK,
        last_n:       int = _DEFAULT_LAST_N,
        timeout:      int = 10,
    ) -> None:
        self._seasons_back = seasons_back
        self._last_n       = last_n
        self._timeout      = timeout

    def fetch(
        self,
        home_team_id: int,
        away_team_id: int,
        reference_date: str | None = None,
    ) -> list[H2HRecord]:
        """
        Obtiene los partidos históricos entre dos equipos MLB.

        Consulta los últimos seasons_back años de temporada regular.

        Parámetros
        ----------
        home_team_id    -- ID del equipo local (partido actual).
        away_team_id    -- ID del equipo visitante (partido actual).
        reference_date  -- Fecha de referencia 'YYYY-MM-DD'.
                          Default: hoy UTC. Los partidos se obtienen
                          desde (reference_date - seasons_back años).

        Retorna
        -------
        list[H2HRecord] ordenada del más antiguo al más reciente.
        Lista vacía si la API falla o no hay historial.
        """
        today    = reference_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ref_date = datetime.strptime(today, "%Y-%m-%d")

        # Calcular rango de fechas: desde N temporadas atrás hasta hoy
        start_year = ref_date.year - self._seasons_back
        start_date = f"{start_year}-03-01"   # Inicio típico de temporada MLB
        end_date   = today

        raw_games = self._fetch_schedule(
            home_team_id = home_team_id,
            away_team_id = away_team_id,
            start_date   = start_date,
            end_date     = end_date,
        )

        records = [
            self._parse_game(game, home_team_id, away_team_id)
            for game in raw_games
        ]
        records = [r for r in records if r is not None]

        # Ordenar del más antiguo al más reciente y quedarnos con los
        # últimos `last_n` — reemplaza al `filter_recent()` que se
        # esperaba importar del Core y que nunca existió ahí.
        records.sort(key=lambda r: r.date)
        if self._last_n > 0:
            records = records[-self._last_n:]

        return records

    def get_stats(
        self,
        home_team_id:   int,
        away_team_id:   int,
        reference_date: str | None = None,
    ) -> H2HMetrics:
        """
        Retorna las estadísticas H2H calculadas directamente.

        Convenience method: fetch() + compute_h2h() en uno. Traduce cada
        `H2HRecord` crudo a dict plano vía `to_meeting_dict()` antes de
        pasarlo a la función genérica del Core.

        Parámetros
        ----------
        home_team_id    -- ID del equipo local.
        away_team_id    -- ID del equipo visitante.
        reference_date  -- Fecha de referencia. Default: hoy.

        Retorna
        -------
        H2HMetrics — con n_meetings=0 si no hay historial disponible.
        """
        records  = self.fetch(
            home_team_id   = home_team_id,
            away_team_id   = away_team_id,
            reference_date = reference_date,
        )
        meetings = [r.to_meeting_dict() for r in records]
        return compute_h2h(
            team_a_id = home_team_id,
            team_b_id = away_team_id,
            meetings  = meetings,
        )

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _fetch_schedule(
        self,
        home_team_id: int,
        away_team_id: int,
        start_date:   str,
        end_date:     str,
    ) -> list[dict]:
        """
        Fetch del schedule de enfrentamientos desde MLB Stats API.

        Usa teamId + opponentId para filtrar solo los partidos entre
        estos dos equipos específicos.
        """
        if not _REQUESTS_AVAILABLE or _requests is None:
            return []

        url    = f"{_MLB_API_BASE}/schedule"
        params = {
            "sportId":    1,
            "teamId":     home_team_id,
            "opponentId": away_team_id,
            "startDate":  start_date,
            "endDate":    end_date,
            "gameType":   "R",           # Regular season únicamente
            "hydrate":    "linescore",   # Necesario para scores
        }

        try:
            resp = _requests.get(url, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        games: list[dict] = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                # Solo partidos finalizados
                status = game.get("status", {}).get("abstractGameState", "")
                if status == "Final":
                    games.append(game)

        return games

    @staticmethod
    def _parse_game(
        game:         dict,
        home_team_id: int,
        away_team_id: int,
    ) -> H2HRecord | None:
        """
        Convierte un partido del schedule a H2HRecord.

        Extrae los scores finales y determina el ganador.
        Retorna None si el partido no tiene datos de score completos.
        """
        teams    = game.get("teams", {})
        home     = teams.get("home", {})
        away     = teams.get("away", {})

        home_id  = home.get("team", {}).get("id")
        away_id  = away.get("team", {}).get("id")

        home_score = home.get("score")
        away_score = away.get("score")

        # Validar que tenemos scores
        if home_score is None or away_score is None:
            return None

        try:
            home_score = float(home_score)
            away_score = float(away_score)
        except (ValueError, TypeError):
            return None

        # Determinar ganador
        if home_score > away_score:
            winner_id = int(home_id) if home_id else None
        elif away_score > home_score:
            winner_id = int(away_id) if away_id else None
        else:
            winner_id = None  # Empate (no ocurre en MLB, pero por completitud)

        # Fecha y temporada
        game_date = game.get("gameDate", "")[:10]  # YYYY-MM-DD
        season    = int(game_date[:4]) if game_date else 0

        # game_id
        game_pk = game.get("gamePk")
        game_id = str(game_pk) if game_pk else None

        return H2HRecord(
            date         = game_date,
            home_team_id = int(home_id) if home_id else home_team_id,
            away_team_id = int(away_id) if away_id else away_team_id,
            home_score   = home_score,
            away_score   = away_score,
            winner_id    = winner_id,
            season       = season,
            game_id      = game_id,
        )