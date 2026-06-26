"""
core/simulation/poisson.py

Modelo de probabilidad basado en distribución de Poisson.

Implementa ProbabilityModel (core/simulation/protocols.py) usando
la matemática de core/utils/math/poisson_math.py. Adecuado para
deportes de baja anotación con scoring discreto independiente:
MLB, NHL, Soccer (como alternativa al modelo bivariado).

Migrado desde analysis/simulation.py del sistema MLB-PREDICTOR-ADVANCED.
Diferencias respecto al original:

1. max_runs=15 → max_score=20 (configurable, default 20 per architecture).
   El audit MLB identificó que max_runs=15 podía perder masa de
   probabilidad para proyecciones altas — el nuevo default elimina
   ese riesgo sin costo de rendimiento apreciable.

2. Funciones sueltas (simular_probabilidades, simular_runline) →
   métodos de clase PoissonModel. La lógica matemática es idéntica.

3. spread_probability() y total_probability() usan los métodos
   cdf()/sf() de poisson_math en vez de iterar sobre scores para
   totales — más eficiente y exacto para mercados Over/Under.

4. Retorna SimulationResult en vez de mutar un dict de partido.

Limitaciones del modelo documentadas en poisson_math.py (sección
"Limitaciones conocidas"):
- Equidispersión (Var = Media): sobreestima probabilidades de
  victoria para equipos con scoring muy variable (overdispersión).
  NHL y algunos equipos MLB pueden requerir NegBinomialModel.
- No modela correlación entre scores de ambos equipos. Soccer
  tiene correlación negativa conocida (un gol cambia el juego) →
  usar BivariatePoissonModel para mayor precisión.
"""

from __future__ import annotations

from core.contracts import Projection
from core.simulation.protocols import ProbabilityModel, SimulationResult
from core.utils.math.poisson_math import cdf, pmf, sf


class PoissonModel:
    """
    Modelo Poisson para deportes de scoring discreto.

    Implementa ProbabilityModel via duck typing estructural — no hereda
    de ninguna clase base, satisface el Protocol por tener los métodos
    con las firmas correctas.

    Parámetros
    ----------
    max_score   -- Límite superior de iteración para win_probabilities()
                  y spread_probability(). Valores más altos son más
                  exactos pero más lentos. Default 20 per arquitectura
                  (MLB: range 0–20 cubre >99.9% de la masa de probabilidad
                  para proyecciones típicas de 3–6 carreras/goles).
    """

    def __init__(self, max_score: int = 20) -> None:
        self.max_score = max_score

    # ── ProbabilityModel interface ─────────────────────────────────────────────

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        Calcula P(home wins), P(away wins), P(draw) via simulación
        Poisson de todos los pares de scores posibles hasta max_score.

        Mismo algoritmo que simular_probabilidades() del sistema MLB
        original, con max_runs=15 → max_score=20.

        draw=0.0 para deportes sin empate (MLB, NHL). Para Soccer con
        Poisson simple (no bivariado), draw sí puede ser > 0 — aunque
        BivariatePoissonModel es más preciso para ese caso.

        Normaliza excluyendo empates para deportes sin draw
        (draw_prob == 0.0 en Projection), preservando la semántica de
        "probabilidad condicional de ganar el partido decidido".
        """
        mu_home = max(projection.expected_home, 0.1)
        mu_away = max(projection.expected_away, 0.1)

        prob_home = 0.0
        prob_away = 0.0
        prob_draw = 0.0

        for h in range(self.max_score):
            ph = pmf(h, mu_home)
            for a in range(self.max_score):
                p = ph * pmf(a, mu_away)
                if h > a:
                    prob_home += p
                elif a > h:
                    prob_away += p
                else:
                    prob_draw += p

        # Para deportes sin empate (draw_prob=0.0 en Projection):
        # normalizar sobre total decidido, igual que el sistema MLB original.
        # Para Soccer con draw permitido: retornar las tres crudas.
        if projection.draw_prob == 0.0:
            total_decidido = prob_home + prob_away
            if total_decidido <= 0:
                return {'home': 0.5, 'away': 0.5, 'draw': 0.0}
            return {
                'home': round(prob_home / total_decidido, 4),
                'away': round(prob_away / total_decidido, 4),
                'draw': 0.0,
            }

        total = prob_home + prob_away + prob_draw
        if total <= 0:
            return {'home': 0.33, 'away': 0.33, 'draw': 0.34}
        return {
            'home': round(prob_home / total, 4),
            'away': round(prob_away / total, 4),
            'draw': round(prob_draw / total, 4),
        }

    def spread_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(equipo cubre el handicap line) via iteración Poisson.

        Mismo algoritmo que simular_runline() del sistema MLB original.
        side='home': P(home_score + line > away_score)
        side='away': P(away_score + line > home_score)

        line es el handicap publicado tal como viene de MarketOdds.line:
        negativo para el favorito (-1.5 en MLB runline), positivo para
        el underdog (+1.5).
        """
        if side == 'home':
            mu_team = max(projection.expected_home, 0.1)
            mu_opp  = max(projection.expected_away, 0.1)
        else:
            mu_team = max(projection.expected_away, 0.1)
            mu_opp  = max(projection.expected_home, 0.1)

        cover_prob = 0.0
        for t in range(self.max_score):
            pt = pmf(t, mu_team)
            for o in range(self.max_score):
                if t + line > o:
                    cover_prob += pt * pmf(o, mu_opp)

        return round(cover_prob, 4)

    def total_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(total > line) u P(total < line) usando CDF/SF de Poisson
        sobre la suma de proyecciones.

        Usa poisson_math.sf()/cdf() en vez de iterar pares de scores
        — más eficiente y exacto que la aproximación bivariada iterada.
        La suma de dos Poisson independientes es Poisson(mu_home + mu_away).

        side='over':  P(X > line)     = sf(line, mu_total)
        side='under': P(X < line)     — excluye push cuando line es
                      entera: cdf(floor(line)-1, mu_total). Para líneas
                      no enteras (8.5): cdf(floor(line), mu_total).

        Documentado en el Test 4b de poisson_math_test: Over + Push +
        Under = 1.0 cuando line es entera, Over + Under = 1.0 cuando
        line tiene decimales (.5).
        """
        mu_total = max(projection.expected_home + projection.expected_away, 0.1)

        if side == 'over':
            return round(float(sf(line, mu_total)), 4)

        # Under: excluir push si la línea es entera
        import math as _math
        if line == _math.floor(line):
            # Línea entera: Under = P(X <= line-1) = cdf(line-1)
            return round(float(cdf(line - 1, mu_total)), 4)
        else:
            # Línea .5: no hay push posible, Under = P(X <= floor(line))
            return round(float(cdf(_math.floor(line), mu_total)), 4)

    def model_version(self) -> str:
        return f'poisson-v1.0-max{self.max_score}'

    # ── Método de conveniencia: SimulationResult completo ─────────────────────

    def simulate(
        self,
        projection: Projection,
        spread_line: float | None = None,
        spread_side: str | None = None,
        total_line: float | None = None,
    ) -> SimulationResult:
        """
        Calcula todos los mercados en una sola llamada y retorna
        SimulationResult. Método de conveniencia para el PipelineRunner
        que necesita los 3 mercados simultáneamente.

        spread_line/spread_side y total_line son opcionales — si no se
        proveen, los campos correspondientes en SimulationResult serán
        None.
        """
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
            draw_prob=win_probs['draw'],
            spread_home_prob=spread_home,
            spread_away_prob=spread_away,
            over_prob=over_prob,
            under_prob=under_prob,
            model_name=self.model_version(),
            projection=projection,
        )