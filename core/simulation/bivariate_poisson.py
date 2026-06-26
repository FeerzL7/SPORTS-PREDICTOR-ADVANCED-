"""
core/simulation/bivariate_poisson.py

Modelo de probabilidad Poisson bivariado con corrección Dixon-Coles.

Diseñado específicamente para soccer, donde los goles de ambos equipos
NO son independientes — a diferencia de MLB donde carreras home y away
son razonablemente independientes (Poisson simple funciona bien).

Por qué Soccer necesita un modelo distinto
------------------------------------------
En soccer, los resultados bajos (0-0, 1-0, 0-1, 1-1) ocurren con
más frecuencia de lo que predice Poisson independiente. Esto se debe
a la dependencia entre los scoring processes de ambos equipos: cuando
un equipo marca un gol, el otro equipo cambia su estrategia (más
presión → más espacios atrás → mayor probabilidad de gol en contra).
Esta correlación negativa entre scores hace que los empates bajos y
las victorias por un gol sean más frecuentes de lo esperado.

Poisson independiente subestima P(0-0), P(1-0), P(0-1), P(1-1) y
sobreestima P(goles altos), lo que produce líneas de totales y
probabilidades de moneyline sistemáticamente incorrectas.

Corrección Dixon-Coles (Dixon & Coles, 1997)
---------------------------------------------
Aplica un factor de corrección τ (tau) a los cuatro resultados bajos:

    P_corr(h, a) = P_poisson(h) × P_poisson(a) × τ(h, a, ρ)

Donde τ se define como:

    τ(0,0) = 1 - μ_h × μ_a × ρ
    τ(1,0) = 1 + μ_a × ρ
    τ(0,1) = 1 + μ_h × ρ
    τ(1,1) = 1 - ρ
    τ(h,a) = 1  para h+a >= 3

El parámetro ρ (rho) controla la fuerza de la correlación:
    ρ = 0.0  → sin corrección (equivale a Poisson independiente)
    ρ < 0.0  → correlación negativa (más empates bajos) — soccer típico
    ρ ∈ [-0.15, -0.05] cubre la mayoría de ligas europeas según literatura

Default: ρ = -0.10 (valor conservador bien documentado para EPL/LaLiga/Serie A).

Uso típico:
    from core.simulation.bivariate_poisson import BivariatePoissonModel

    model = BivariatePoissonModel(rho=-0.10, max_score=10)
    projection = Projection(
        event_id='e1', sport='soccer',
        expected_home=1.6, expected_away=1.1,
        home_win_prob=0.45, away_win_prob=0.28, draw_prob=0.27,
        distribution='bivariate_poisson',
    )
    result = model.simulate(projection, total_line=2.5)
"""

from __future__ import annotations

import math

from core.contracts import Projection
from core.simulation.protocols import SimulationResult
from core.utils.math.poisson_math import cdf, pmf, sf


def _tau(h: int, a: int, mu_h: float, mu_a: float, rho: float) -> float:
    """
    Factor de corrección Dixon-Coles para resultados bajos.

    Solo modifica los cuatro resultados críticos (0-0, 1-0, 0-1, 1-1).
    Para todos los demás retorna 1.0 (sin corrección).

    La corrección puede producir valores ligeramente negativos si rho
    es muy extremo para las proyecciones dadas — se acota a 0.0 por
    seguridad, aunque en práctica con rho ∈ [-0.30, 0.0] y proyecciones
    de soccer (1.0–2.5 goles), esto no debería ocurrir.
    """
    if h == 0 and a == 0:
        return max(1 - mu_h * mu_a * rho, 0.0)
    if h == 1 and a == 0:
        return max(1 + mu_a * rho, 0.0)
    if h == 0 and a == 1:
        return max(1 + mu_h * rho, 0.0)
    if h == 1 and a == 1:
        return max(1 - rho, 0.0)
    return 1.0


def _build_joint_matrix(
    mu_h: float,
    mu_a: float,
    rho: float,
    max_score: int,
) -> list[list[float]]:
    """
    Construye la matriz de probabilidades conjuntas P(home=h, away=a)
    con corrección Dixon-Coles, normalizada para que sume 1.0.

    Retorna matrix[h][a] = P(home scores h, away scores a).
    """
    matrix = []
    total = 0.0

    for h in range(max_score):
        row = []
        ph = pmf(h, mu_h)
        for a in range(max_score):
            p = ph * pmf(a, mu_a) * _tau(h, a, mu_h, mu_a, rho)
            p = max(p, 0.0)  # por seguridad ante tau negativo
            row.append(p)
            total += p
        matrix.append(row)

    # Normalizar para compensar la masa truncada fuera de max_score
    # y cualquier distorsión de la corrección tau
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]

    return matrix


class BivariatePoissonModel:
    """
    Modelo Poisson bivariado con corrección Dixon-Coles para soccer.

    Implementa ProbabilityModel via duck typing estructural.

    Parámetros
    ----------
    rho        -- Parámetro de correlación Dixon-Coles. Debe ser <= 0
                  para soccer (correlación negativa entre scores).
                  Default -0.10, conservador y bien documentado para
                  ligas europeas de primer nivel.
                  El sport plugin puede sobreescribir este valor via
                  config/soccer.yaml (key: simulation.bivariate.rho).
    max_score   -- Límite de goles por equipo para la matriz conjunta.
                  Default 10 para soccer — suficiente para cubrir
                  >99.9% de la masa de probabilidad en cualquier
                  partido de soccer profesional.
    """

    def __init__(
        self,
        rho: float = -0.10,
        max_score: int = 10,
    ) -> None:
        if rho > 0.0:
            raise ValueError(
                f"rho={rho} debe ser <= 0.0 para BivariatePoissonModel "
                f"(correlación negativa entre scores en soccer). "
                f"Para rho=0.0 sin corrección, usar PoissonModel."
            )
        self.rho = rho
        self.max_score = max_score

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        P(home wins), P(away wins), P(draw) desde la matriz conjunta
        Dixon-Coles. Los tres suman 1.0 (la normalización de la matriz
        garantiza esto).
        """
        mu_h = max(projection.expected_home, 0.1)
        mu_a = max(projection.expected_away, 0.1)
        matrix = _build_joint_matrix(mu_h, mu_a, self.rho, self.max_score)

        home_win = away_win = draw = 0.0
        for h in range(self.max_score):
            for a in range(self.max_score):
                p = matrix[h][a]
                if h > a:
                    home_win += p
                elif a > h:
                    away_win += p
                else:
                    draw += p

        return {
            'home': round(home_win, 4),
            'away': round(away_win, 4),
            'draw': round(draw, 4),
        }

    def spread_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(equipo cubre el handicap) desde la matriz conjunta.

        Para soccer, el handicap asiático (AH) es el más común —
        line puede ser entero (.0), medio (.5) o cuarto (.25/.75).
        La iteración sobre la matriz conjunta maneja todos los casos.
        """
        mu_h = max(projection.expected_home, 0.1)
        mu_a = max(projection.expected_away, 0.1)
        matrix = _build_joint_matrix(mu_h, mu_a, self.rho, self.max_score)

        cover_prob = 0.0
        for h in range(self.max_score):
            for a in range(self.max_score):
                team_score = h if side == 'home' else a
                opp_score  = a if side == 'home' else h
                if team_score + line > opp_score:
                    cover_prob += matrix[h][a]

        return round(cover_prob, 4)

    def total_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(total_goles > line) o P(total_goles < line) desde la
        matriz conjunta.

        A diferencia de PoissonModel (que usa sf/cdf sobre Poisson
        de la suma, válido solo cuando los scores son independientes),
        aquí sumamos sobre la matriz conjunta — necesario porque la
        corrección Dixon-Coles hace que la distribución de la suma
        NO sea Poisson simple.
        """
        mu_h = max(projection.expected_home, 0.1)
        mu_a = max(projection.expected_away, 0.1)
        matrix = _build_joint_matrix(mu_h, mu_a, self.rho, self.max_score)

        prob = 0.0
        for h in range(self.max_score):
            for a in range(self.max_score):
                total = h + a
                if side == 'over' and total > line:
                    prob += matrix[h][a]
                elif side == 'under' and total < line:
                    prob += matrix[h][a]

        return round(prob, 4)

    def model_version(self) -> str:
        return f'bivariate_poisson-v1.0-rho{self.rho}'

    def simulate(
        self,
        projection: Projection,
        spread_line: float | None = None,
        spread_side: str | None = None,
        total_line: float | None = None,
    ) -> SimulationResult:
        """
        Calcula todos los mercados desde la matriz conjunta Dixon-Coles
        en una sola llamada. La matriz se construye una vez y se reutiliza
        para los tres mercados — no se recalcula tres veces.
        """
        mu_h = max(projection.expected_home, 0.1)
        mu_a = max(projection.expected_away, 0.1)
        matrix = _build_joint_matrix(mu_h, mu_a, self.rho, self.max_score)

        # Win probabilities
        home_win = away_win = draw = 0.0
        for h in range(self.max_score):
            for a in range(self.max_score):
                p = matrix[h][a]
                if h > a:   home_win += p
                elif a > h: away_win += p
                else:       draw     += p

        # Spread
        spread_home = spread_away = None
        if spread_line is not None:
            sh = sa = 0.0
            for h in range(self.max_score):
                for a in range(self.max_score):
                    if h + spread_line > a: sh += matrix[h][a]
                    if a + spread_line > h: sa += matrix[h][a]
            spread_home = round(sh, 4)
            spread_away = round(sa, 4)

        # Total
        over_prob = under_prob = None
        if total_line is not None:
            op = up = 0.0
            for h in range(self.max_score):
                for a in range(self.max_score):
                    t = h + a
                    if t > total_line: op += matrix[h][a]
                    elif t < total_line: up += matrix[h][a]
            over_prob  = round(op, 4)
            under_prob = round(up, 4)

        return SimulationResult(
            home_win_prob=round(home_win, 4),
            away_win_prob=round(away_win, 4),
            draw_prob=round(draw, 4),
            spread_home_prob=spread_home,
            spread_away_prob=spread_away,
            over_prob=over_prob,
            under_prob=under_prob,
            model_name=self.model_version(),
            projection=projection,
        )