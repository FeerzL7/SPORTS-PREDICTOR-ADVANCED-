"""
core/utils/h2h_base.py

Cálculo genérico de métricas head-to-head (H2H) entre dos equipos.

Este módulo es parte del core — no contiene conceptos deportivos.
Opera sobre listas de resultados de partidos anteriores entre dos
equipos, independientemente del deporte.

La implementación específica de cómo obtener esos resultados desde
cada API deportiva vive en el plugin correspondiente:
    sports/mlb/h2h.py  → MLB Stats API
    sports/nba/h2h.py  → NBA API  (futuro)
    sports/soccer/h2h.py → football-data API (futuro)

Uso típico
-----------
    from core.utils.h2h_base import H2HMetrics, compute_h2h

    meetings = [
        {'home_score': 5, 'away_score': 3, 'home_id': 147, 'away_id': 111},
        {'home_score': 2, 'away_score': 4, 'home_id': 111, 'away_id': 147},
    ]
    metrics = compute_h2h(
        team_a_id=147,  # NYY
        team_b_id=111,  # BOS
        meetings=meetings,
    )
    # metrics.win_rate_a = 0.5 (ganó 1 de 2)
    # metrics.avg_total  = 7.0 ((8 + 6) / 2)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class H2HMetrics:
    """
    Métricas H2H calculadas sobre los encuentros directos.

    Inmutable: representa un snapshot calculado en un momento.

    Campos
    ------
    n_meetings       -- Total de encuentros directos en la ventana.
    wins_a           -- Victorias del equipo A.
    wins_b           -- Victorias del equipo B.
    draws            -- Empates (relevante en soccer).
    win_rate_a       -- wins_a / n_meetings. None si n_meetings=0.
    avg_score_a      -- Media de carreras/puntos/goles del equipo A.
    avg_score_b      -- Media del equipo B.
    avg_total        -- Media de la suma de ambos equipos.
    home_win_rate    -- % de victorias del local. None si n=0.
    """
    n_meetings:   int
    wins_a:       int
    wins_b:       int
    draws:        int
    win_rate_a:   float | None
    avg_score_a:  float | None
    avg_score_b:  float | None
    avg_total:    float | None
    home_win_rate: float | None

    @property
    def has_data(self) -> bool:
        return self.n_meetings > 0


def compute_h2h(
    team_a_id: int,
    team_b_id: int,
    meetings:  list[dict],
) -> H2HMetrics:
    """
    Calcula métricas H2H sobre una lista de encuentros directos.

    Parámetros
    ----------
    team_a_id  -- ID del equipo A (home en el partido a proyectar).
    team_b_id  -- ID del equipo B (away en el partido a proyectar).
    meetings   -- Lista de dicts con los encuentros pasados.
                  Cada dict debe tener:
                      home_id:    int — ID del equipo local
                      away_id:    int — ID del equipo visitante
                      home_score: float — puntos del local
                      away_score: float — puntos del visitante

    Retorna
    -------
    H2HMetrics con todas las métricas calculadas.
    """
    if not meetings:
        return H2HMetrics(
            n_meetings=0, wins_a=0, wins_b=0, draws=0,
            win_rate_a=None, avg_score_a=None, avg_score_b=None,
            avg_total=None, home_win_rate=None,
        )

    wins_a = wins_b = draws = 0
    scores_a: list[float] = []
    scores_b: list[float] = []
    home_wins = home_games = 0

    for m in meetings:
        home_id    = m.get("home_id")
        away_id    = m.get("away_id")
        home_score = m.get("home_score")
        away_score = m.get("away_score")

        if home_score is None or away_score is None:
            continue

        home_score = float(home_score)
        away_score = float(away_score)

        # Determinar score de A y B según qué rol jugó cada equipo
        if home_id == team_a_id and away_id == team_b_id:
            score_a, score_b = home_score, away_score
            home_games += 1
        elif home_id == team_b_id and away_id == team_a_id:
            score_a, score_b = away_score, home_score
            home_games += 1
            # home_id = team_b → si gana B, es victoria del local
        else:
            continue  # partido sin relación con los equipos solicitados

        scores_a.append(score_a)
        scores_b.append(score_b)

        if score_a > score_b:
            wins_a += 1
            if home_id == team_a_id:
                home_wins += 1
        elif score_b > score_a:
            wins_b += 1
            if home_id == team_b_id:
                home_wins += 1
        else:
            draws += 1

    n = len(scores_a)
    if n == 0:
        return H2HMetrics(
            n_meetings=0, wins_a=0, wins_b=0, draws=0,
            win_rate_a=None, avg_score_a=None, avg_score_b=None,
            avg_total=None, home_win_rate=None,
        )

    totals = [a + b for a, b in zip(scores_a, scores_b)]

    return H2HMetrics(
        n_meetings    = n,
        wins_a        = wins_a,
        wins_b        = wins_b,
        draws         = draws,
        win_rate_a    = round(wins_a / n, 4),
        avg_score_a   = round(sum(scores_a) / n, 3),
        avg_score_b   = round(sum(scores_b) / n, 3),
        avg_total     = round(sum(totals) / n, 3),
        home_win_rate = round(home_wins / home_games, 4) if home_games > 0 else None,
    )