"""
core/simulation/bradley_terry.py

Modelo de probabilidad Bradley-Terry para Tennis.

El modelo Bradley-Terry es el estándar para deportes de enfrentamiento
directo entre dos jugadores donde no existe "scoring" acumulable —
solo importa quién gana el enfrentamiento.

Fórmula central (Bradley & Terry, 1952):
    P(i beats j) = strength_i / (strength_i + strength_j)

Donde strength_i y strength_j son los ratings de "fuerza" de cada
jugador, derivados de sus resultados históricos ponderados por
superficie y recency.

Por qué Bradley-Terry para Tennis y NO Poisson/Normal
------------------------------------------------------
Poisson y Normal modelan scoring acumulable (carreras, puntos). En
Tennis el resultado final es 6-3, 7-5, 6-4 — no importan los games
exactos, solo quién gana el match. El modelo de "fuerza relativa"
es más apropiado porque:

1. El scoring en Tennis tiene estructura fija (best of 3/5 sets,
   cada set hasta 6 games, tiebreak a 7 puntos) — no es un proceso
   de conteo libre como MLB/NBA.
2. La probabilidad de ganar un match en Tennis se puede calcular
   analíticamente desde la probabilidad de ganar un punto/game/set
   usando el modelo de "punto-game-set-match" de Kemeny & Snell,
   pero a nivel de mercado la aproximación Bradley-Terry es suficiente
   y más estable estadísticamente.
3. La superficie (clay, hard, grass) y el recency de resultados
   son los dos factores más importantes para ajustar los ratings.

Parámetros de strength desde Projection.distribution_params
-------------------------------------------------------------
El sport plugin de Tennis (sports/tennis/, Fase 4 del roadmap) provee:

    distribution_params = {
        'strength_home': 1.85,   # rating BT del jugador home
        'strength_away': 1.20,   # rating BT del jugador away
    }

Si no están disponibles, el modelo usa strength=1.0 para ambos
(partido 50/50), lo que produce P(home)=0.5 — fallback seguro y
documentado.

Conversión de strength a probabilidades de mercado
----------------------------------------------------
win_probabilities:
    P(home) = strength_h / (strength_h + strength_a)
    P(away) = 1 - P(home)
    draw = 0.0 (Tennis no tiene empate)

spread_probability:
    Tennis no tiene spread en el sentido tradicional. El mercado
    de spread más común es "games handicap" (ej. home -2.5 games).
    Se aproxima usando la distribución Binomial sobre games esperados,
    derivada de la probabilidad de ganar un game individual.

total_probability:
    "Total games" es el mercado de totales en Tennis (ej. Over 21.5
    games en un partido best-of-3). Se aproxima desde la probabilidad
    de ganar un game y la estructura del scoring.
"""

from __future__ import annotations

import math

from core.contracts import Projection
from core.simulation.protocols import SimulationResult

# Mínimo de strength para evitar divisiones por cero o resultados degenerados
_MIN_STRENGTH = 1e-6

# Número de sets máximo por defecto (best of 3 en ATP/WTA estándar)
# El plugin de Tennis puede sobreescribir via distribution_params['best_of']
_DEFAULT_BEST_OF = 3


def _win_prob_from_strengths(strength_h: float, strength_a: float) -> float:
    """
    P(home beats away) = strength_h / (strength_h + strength_a).

    Fórmula exacta del modelo Bradley-Terry, documentada en
    SPORTS_PREDICTOR_ARCHITECTURE.md §8.1.
    """
    total = strength_h + strength_a
    if total <= 0:
        return 0.5
    return strength_h / total


def _match_win_prob_from_game_prob(p_game: float, best_of: int) -> float:
    """
    Calcula P(ganar el match) desde P(ganar un game individual) usando
    la estructura recursiva del scoring de Tennis.

    Para best_of=3: ganar 2 sets, cada set se gana con 6 games (+ tiebreak)
    Para best_of=5: ganar 3 sets

    Aproximación: modela el match como best-of-N sets, cada set como
    una variable Bernoulli con probabilidad p_set derivada de p_game.

    p_set se calcula como P(ganar un set) asumiendo que la probabilidad
    de ganar un game es constante e igual a p_game (simplificación que
    ignora el efecto del servicio, suficiente para nivel de mercado).

    Para un set a 6 games (ignorando tiebreak):
        P(ganar set) ≈ P(ganar al menos 6 de los primeros 10 games)
        usando distribución Binomial(10, p_game) con ajuste.
    """
    if p_game <= 0:
        return 0.0
    if p_game >= 1:
        return 1.0

    # Aproximar P(ganar set) con Binomial simplificada
    # Un set se gana al llegar a 6 antes que el rival (con diferencia de 2)
    # Aproximación práctica: P(set) usando Binomial hasta 12 games
    p_set = 0.0
    # P(ganar 6-k para k=0..5) = B(6+k-1, k, p_game) × p_game para cada valor de k
    # Simplificación: P(set) ≈ sum de P(ganar x de 11 games) para x>=6
    q_game = 1 - p_game
    for x in range(6, 12):
        # P(exactamente x wins en 11 intentos) — esto es una aproximación
        binom_coef = math.comb(11, x)
        p_set += binom_coef * (p_game ** x) * (q_game ** (11 - x))

    # Match: best_of=3 → ganar 2 sets, best_of=5 → ganar 3 sets
    sets_needed = (best_of + 1) // 2
    q_set = 1 - p_set

    # P(ganar match best-of-N) con probabilidad p_set por set
    # Suma sobre todos los escenarios donde se ganan exactamente sets_needed
    # antes que el rival
    p_match = 0.0
    for losses in range(sets_needed):
        # Ganar en (sets_needed + losses) sets: últimos set es victoria
        # C(sets_needed-1+losses, losses) × p_set^sets_needed × q_set^losses
        ways = math.comb(sets_needed - 1 + losses, losses)
        p_match += ways * (p_set ** sets_needed) * (q_set ** losses)

    return p_match


class BradleyTerryModel:
    """
    Modelo Bradley-Terry para Tennis.

    Implementa ProbabilityModel via duck typing estructural.

    Lee los ratings de strength desde Projection.distribution_params:
        'strength_home': float  — rating del jugador home
        'strength_away': float  — rating del jugador away
        'best_of':       int    — best of 3 o best of 5 (opcional)

    Si los ratings no están disponibles, usa strength=1.0 para ambos
    (50/50) — fallback explícito y documentado, no error silencioso.
    """

    def __init__(self) -> None:
        pass  # Sin parámetros: BT es puramente relativo (strength_h/strength_a)

    def _get_strengths(
        self,
        projection: Projection,
    ) -> tuple[float, float, int]:
        """
        Resuelve strength_home, strength_away y best_of desde
        Projection.distribution_params.

        Retorna (strength_h, strength_a, best_of).
        Fallback a (1.0, 1.0, 3) si los parámetros no existen.
        """
        params = projection.distribution_params or {}

        strength_h = float(params.get('strength_home', 1.0))
        strength_a = float(params.get('strength_away', 1.0))
        best_of    = int(params.get('best_of', _DEFAULT_BEST_OF))

        # Proteger contra valores inválidos
        strength_h = max(strength_h, _MIN_STRENGTH)
        strength_a = max(strength_a, _MIN_STRENGTH)
        best_of    = best_of if best_of in (3, 5) else _DEFAULT_BEST_OF

        return strength_h, strength_a, best_of

    # ── ProbabilityModel interface ─────────────────────────────────────────────

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        P(home wins match) usando la fórmula Bradley-Terry:
            P(home) = strength_h / (strength_h + strength_a)

        draw=0.0: Tennis no tiene empate (siempre hay un ganador,
        incluso en 5 sets — no existe el OT).
        """
        strength_h, strength_a, _ = self._get_strengths(projection)
        p_home = _win_prob_from_strengths(strength_h, strength_a)

        return {
            'home': round(p_home, 4),
            'away': round(1.0 - p_home, 4),
            'draw': 0.0,
        }

    def spread_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        Aproximación de P(equipo cubre el handicap de games) usando
        la probabilidad derivada de los ratings Bradley-Terry.

        En Tennis, el spread más común es "games handicap":
            home -2.5 games: home debe ganar por 3+ games en total

        Aproximación: deriva p_game desde los ratings BT y modela
        el total de games jugados con distribución Binomial.

        Nota: esta es una aproximación de nivel de mercado. El sport
        plugin de Tennis (Fase 4) puede proveer expected_home/away
        como games esperados para un modelo más preciso.
        """
        strength_h, strength_a, best_of = self._get_strengths(projection)
        p_game_home = _win_prob_from_strengths(strength_h, strength_a)

        # Usar expected_home/away como games esperados si el plugin los provee
        # (cuando el plugin calcula la proyección en games, no en puntos)
        mu_h = projection.expected_home
        mu_a = projection.expected_away

        threshold = -line  # = 2.5 cuando line=-2.5

        if side == 'home':
            # P(home gana por más de threshold games en total)
            # Aproximar con Normal sobre la diferencia de games esperados
            mu_diff = mu_h - mu_a
            # sigma estimada para diferencia de games en Tennis
            sigma_diff = math.sqrt(mu_h + mu_a) * 0.5  # heurística calibrable
            if sigma_diff <= 0:
                return 0.5
            z = (threshold - mu_diff) / sigma_diff
            from core.simulation.normal import _norm_sf
            return round(float(_norm_sf(z)), 4)
        else:
            mu_diff = mu_h - mu_a
            sigma_diff = math.sqrt(mu_h + mu_a) * 0.5
            if sigma_diff <= 0:
                return 0.5
            z = (threshold - mu_diff) / sigma_diff
            from core.simulation.normal import _norm_cdf
            return round(float(_norm_cdf(z)), 4)

    def total_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(total games > line) o P(total games < line).

        Usa expected_home + expected_away como total de games esperados
        (el sport plugin de Tennis provee la proyección en games totales),
        con distribución Normal sobre la suma — suficiente precisión
        para mercados de totales de Tennis.

        Para un match best-of-3, el rango de games posible es [12, 23].
        Para best-of-5: [18, 35].
        """
        mu_total = projection.expected_home + projection.expected_away

        # Sigma estimada para total de games en Tennis
        # Best-of-3: la varianza del total es mayor en partidos cerrados
        _, _, best_of = self._get_strengths(projection)
        sigma_total = math.sqrt(mu_total) * (0.8 if best_of == 3 else 0.9)

        if sigma_total <= 0:
            return 0.5 if side == 'over' else 0.5

        z = (line - mu_total) / sigma_total

        from core.simulation.normal import _norm_sf, _norm_cdf
        if side == 'over':
            return round(float(_norm_sf(z)), 4)
        else:
            return round(float(_norm_cdf(z)), 4)

    def model_version(self) -> str:
        return 'bradley_terry-v1.0'

    def simulate(
        self,
        projection: Projection,
        spread_line: float | None = None,
        spread_side: str | None = None,
        total_line: float | None = None,
    ) -> SimulationResult:
        """Calcula todos los mercados en una sola llamada."""
        win_probs = self.win_probabilities(projection)

        spread_home = (
            self.spread_probability(projection, spread_line, 'home')
            if spread_line is not None else None
        )
        spread_away = (
            self.spread_probability(projection, spread_line, 'away')
            if spread_line is not None else None
        )
        over_prob = (
            self.total_probability(projection, total_line, 'over')
            if total_line is not None else None
        )
        under_prob = (
            self.total_probability(projection, total_line, 'under')
            if total_line is not None else None
        )

        return SimulationResult(
            home_win_prob=win_probs['home'],
            away_win_prob=win_probs['away'],
            draw_prob=0.0,
            spread_home_prob=spread_home,
            spread_away_prob=spread_away,
            over_prob=over_prob,
            under_prob=under_prob,
            model_name=self.model_version(),
            projection=projection,
        )