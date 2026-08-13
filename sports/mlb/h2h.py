"""
sports/mlb/h2h.py

MLBH2HFetcher: historial de enfrentamientos directos para MLB.

Migrado de analysis/h2h.py del sistema MLB. La lógica estadística
genérica (compute_h2h_stats, filter_recent) vive en
core/utils/h2h_base.py — este módulo solo provee los datos desde
MLB Stats API y los convierte a H2HRecord.

Posición en el pipeline
------------------------
MLBDataProvider.enrich_event(event)
    ↓
MLBH2HFetcher.fetch(home_team_id, away_team_id, last_n=20)
    → list[H2HRecord]
    ↓
compute_h2h_stats(records, home_team_id) → H2HStats
    ↓
TeamFeatures(sport_metadata={
    'h2h_home_win_rate': 0.55,
    'h2h_avg_total':     9.2,
    'h2h_weight':        0.85,
})

El MLBProjectionModel usa h2h_weight para determinar cuánto peso
dar al historial H2H vs las métricas de temporada actuales.
Con h2h_weight=0.0 (< 5 partidos en el historial), el H2H se ignora.

Endpoint MLB Stats API
-----------------------
/api/v1/schedule?sportId=1&teamId={teamId}&opponentId={opponentId}
    &startDate={start}&endDate={end}&gameType=R&hydrate=linescore

Retorna partidos de temporada regular entre los dos equipos en el
rango de fechas especificado, incluyendo scores finales.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.utils.h2h_base import (
    H2HRecord,
    H2HStats,
    compute_h2h_stats,
    filter_recent,
)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Ventana histórica por defecto: últimas 3 temporadas
_DEFAULT_SEASONS_BACK: int = 3
_DEFAULT_LAST_N:       int = 20


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

        return filter_recent(records, last_n=self._last_n)

    def get_stats(
        self,
        home_team_id:   int,
        away_team_id:   int,
        reference_date: str | None = None,
    ) -> H2HStats:
        """
        Retorna las estadísticas H2H calculadas directamente.

        Convenience method: fetch() + compute_h2h_stats() en uno.

        Parámetros
        ----------
        home_team_id    -- ID del equipo local.
        away_team_id    -- ID del equipo visitante.
        reference_date  -- Fecha de referencia. Default: hoy.

        Retorna
        -------
        H2HStats — con n_games=0 si no hay historial disponible.
        """
        records = self.fetch(
            home_team_id   = home_team_id,
            away_team_id   = away_team_id,
            reference_date = reference_date,
        )
        return compute_h2h_stats(
            records      = records,
            home_team_id = home_team_id,
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
        if not _REQUESTS_AVAILABLE:
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