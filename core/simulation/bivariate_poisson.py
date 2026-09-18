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


# API pública del módulo.
#
# Se declara explícitamente porque build_joint_matrix es matemática
# compartida con el plugin de fútbol: dejar su condición de pública al
# criterio de quien lea el código invita a que alguien la vuelva
# privada y rompa esa dependencia sin darse cuenta.
__all__ = [
    "BivariatePoissonModel",
    "build_joint_matrix",
]
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


def _means(projection: Projection) -> tuple[float, float]:
    """
    Medias esperadas de la proyección, con prioridad a las explícitas.

    distribution_params puede traer lambda_home/lambda_away, que son
    las que el modelo de proyección usó realmente. expected_home y
    expected_away son el mismo valor redondeado a tres decimales, así
    que preferir las explícitas evita una divergencia de milésimas
    entre la matriz de la proyección y la de aquí.
    """
    params = projection.distribution_params or {}
    mu_h = _param(params.get("lambda_home"), projection.expected_home)
    mu_a = _param(params.get("lambda_away"), projection.expected_away)
    return max(mu_h, 0.1), max(mu_a, 0.1)


def _param(value, fallback: float) -> float:
    """
    Convierte a float, con respaldo cuando el valor está AUSENTE.

    No se usa `value or fallback` porque el cero es legítimo aquí:
    rho = 0.0 desactiva la corrección Dixon-Coles y lambda_3 = 0.0 el
    componente bivariado. Con `or`, una proyección calculada sin
    corrección se recalcularía CON ella.
    """
    if value is None:
        return fallback
    try:
        result = float(value)
    except (ValueError, TypeError):
        return fallback
    return result if result == result else fallback


def build_joint_matrix(
    mu_h: float,
    mu_a: float,
    rho: float,
    max_score: int,
    lambda_3: float = 0.0,
) -> list[list[float]]:
    """
    Matriz de probabilidades conjuntas P(home=h, away=a).

    PÚBLICA de forma deliberada. La versión anterior se llamaba
    `_build_joint_matrix`, y sports/soccer/dixon_coles.py la importaba
    con el guion bajo incluido — una contradicción: el guion declara
    "interno a este módulo" y el import venía de fuera. El type checker
    lo señaló con razón.

    Un símbolo del que depende otro módulo es parte de la API pública
    por definición. Renombrarla es reconocer lo que ya era.

    Fuente ÚNICA de esta matemática en todo el proyecto. El módulo
    sports/soccer/dixon_coles.py delega aquí en vez de reimplementarla:
    tener dos versiones significaría que cualquier recalibración habría
    que aplicarla en dos sitios, y divergirían en silencio.

    La dirección de la dependencia es sports → core, nunca al revés.

    Parámetros
    ----------
    mu_h / mu_a -- Medias esperadas de cada equipo. Son las MARGINALES:
                   con lambda_3 > 0 los parámetros del proceso interno
                   se ajustan para que la media total siga siendo la
                   pedida.
    rho         -- Corrección Dixon-Coles. Negativo aumenta la masa en
                   empates de marcador bajo. Cero la desactiva.
    max_score   -- Máximo de goles por equipo. La matriz resultante es
                   de (max_score + 1) × (max_score + 1), cubriendo los
                   marcadores de 0 a max_score inclusive.

                   CORRECCIÓN: la versión anterior usaba
                   range(max_score), así que max_score=10 producía
                   marcadores 0-9 y truncaba el 10. La masa perdida la
                   redistribuía la normalización, desplazando
                   ligeramente todas las probabilidades.
    lambda_3    -- Componente compartido del Poisson bivariado. Cero lo
                   reduce a dos Poisson independientes.

                   AÑADIDO: pese a llamarse BivariatePoissonModel, el
                   módulo no lo soportaba — era un Poisson
                   INDEPENDIENTE con corrección Dixon-Coles. Ahora
                   ofrece ambas vías.

                   Nota importante: Dixon-Coles y el componente
                   bivariado modelan la MISMA dependencia. Activar los
                   dos la cuenta dos veces y empuja el empate fuera del
                   rango observado. Usar uno u otro, no ambos.
    """
    size = max(2, int(max_score))
    lam_3 = max(0.0, min(lambda_3, min(mu_h, mu_a) * 0.9))

    if lam_3 > 0:
        matrix = _bivariate_base(mu_h, mu_a, lam_3, size)
    else:
        matrix = _independent_base(mu_h, mu_a, size)

    # Corrección tau sobre las cuatro celdas de marcador bajo
    for h in range(min(2, size + 1)):
        for a in range(min(2, size + 1)):
            matrix[h][a] = max(0.0, matrix[h][a] * _tau(h, a, mu_h, mu_a, rho))

    total = sum(sum(row) for row in matrix)
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]

    return matrix


def _independent_base(
    mu_h: float,
    mu_a: float,
    size: int,
) -> list[list[float]]:
    """Producto exterior de dos Poisson independientes."""
    p_home = [pmf(k, mu_h) for k in range(size + 1)]
    p_away = [pmf(k, mu_a) for k in range(size + 1)]
    return [[ph * pa for pa in p_away] for ph in p_home]


def _bivariate_base(
    mu_h:  float,
    mu_a:  float,
    lam_3: float,
    size:  int,
) -> list[list[float]]:
    """
    Poisson bivariado con componente compartido.

        Goles_local     = X₁ + X₃
        Goles_visitante = X₂ + X₃

    con X₁, X₂, X₃ Poisson independientes. Para conservar las medias
    marginales: λ₁ = mu_h - λ₃ y λ₂ = mu_a - λ₃.

    La probabilidad conjunta suma sobre los valores del componente
    compartido:

        P(h, a) = Σ_k P(X₁ = h-k) · P(X₂ = a-k) · P(X₃ = k)
    """
    lam_1 = max(mu_h - lam_3, 0.01)
    lam_2 = max(mu_a - lam_3, 0.01)

    p1 = [pmf(k, lam_1) for k in range(size + 1)]
    p2 = [pmf(k, lam_2) for k in range(size + 1)]
    p3 = [pmf(k, lam_3) for k in range(size + 1)]

    matrix = [[0.0] * (size + 1) for _ in range(size + 1)]
    for h in range(size + 1):
        for a in range(size + 1):
            acc = 0.0
            for k in range(min(h, a) + 1):
                acc += p1[h - k] * p2[a - k] * p3[k]
            matrix[h][a] = acc

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

    # ── Resolución de parámetros ──────────────────────────────────────────────

    def _get_params(self, projection: Projection) -> tuple[float, float, int]:
        """
        Parámetros de la matriz, con prioridad al valor de la proyección.

        Retorna (rho, lambda_3, max_score).

        Por qué la proyección manda sobre el constructor
        ------------------------------------------------
        El modelo de proyección de fútbol calibra rho contra las medias
        de su liga y lo deja en distribution_params. La versión
        anterior de este módulo NO los leía: usaba el rho fijado al
        construir, así que las probabilidades del pick diferían de las
        que produjeron la proyección.

        La diferencia medida era del 4.5% en el empate — precisamente
        el mercado donde los books aplican más margen y donde el
        modelo busca su valor. Un pick de empate calculado con un rho
        distinto al proyectado no corresponde a nada.

        Es el mismo defecto que apareció en NormalModel para NFL, donde
        la sigma adaptativa del modelo se descartaba en silencio.
        """
        params = projection.distribution_params or {}

        rho = _param(params.get("rho"), self.rho)
        # rho positivo no tiene sentido en fútbol; se acota en vez de
        # lanzar, porque un parámetro corrupto en una proyección no
        # debería tumbar el pipeline entero.
        rho = min(0.0, rho)

        lambda_3 = max(0.0, _param(params.get("lambda_3"), 0.0))
        max_score = int(_param(params.get("max_goals"), self.max_score))

        return rho, lambda_3, max(2, max_score)

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        P(home wins), P(away wins), P(draw) desde la matriz conjunta
        Dixon-Coles. Los tres suman 1.0 (la normalización de la matriz
        garantiza esto).
        """
        mu_h, mu_a = _means(projection)
        rho, lambda_3, size = self._get_params(projection)
        matrix = build_joint_matrix(mu_h, mu_a, rho, size, lambda_3)

        home_win = away_win = draw = 0.0
        for h in range(len(matrix)):
            for a in range(len(matrix[h])):
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
        mu_h, mu_a = _means(projection)
        rho, lambda_3, size = self._get_params(projection)
        matrix = build_joint_matrix(mu_h, mu_a, rho, size, lambda_3)

        cover_prob = 0.0
        for h in range(len(matrix)):
            for a in range(len(matrix[h])):
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
        mu_h, mu_a = _means(projection)
        rho, lambda_3, size = self._get_params(projection)
        matrix = build_joint_matrix(mu_h, mu_a, rho, size, lambda_3)

        prob = 0.0
        for h in range(len(matrix)):
            for a in range(len(matrix[h])):
                total = h + a
                if side == 'over' and total > line:
                    prob += matrix[h][a]
                elif side == 'under' and total < line:
                    prob += matrix[h][a]

        return round(prob, 4)

    def model_version(self) -> str:
        return f'bivariate_poisson-v1.1-rho{self.rho}'

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
        mu_h, mu_a = _means(projection)
        rho, lambda_3, size = self._get_params(projection)
        matrix = build_joint_matrix(mu_h, mu_a, rho, size, lambda_3)
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0

        # Win probabilities
        home_win = away_win = draw = 0.0
        for h in range(rows):
            for a in range(cols):
                p = matrix[h][a]
                if h > a:   home_win += p
                elif a > h: away_win += p
                else:       draw     += p

        # Spread
        #
        # CORRECCIÓN: la versión anterior aplicaba la MISMA línea a
        # ambos lados:
        #
        #     if h + spread_line > a: sh += ...
        #     if a + spread_line > h: sa += ...
        #
        # Eso calcula dos veces el mismo lado del mercado. Si el local
        # es -1.0, el visitante es +1.0: `spread_away` debe usar la
        # línea NEGADA. Sin ello las dos probabilidades no suman 1.0 y
        # no corresponden a ningún mercado real.
        #
        # Es el mismo defecto que se corrigió en NormalModel.simulate()
        # en la tarea 10.11. SkellamModel lo sigue arrastrando.
        spread_home = spread_away = None
        if spread_line is not None:
            sh = sa = 0.0
            for h in range(rows):
                for a in range(cols):
                    if h + spread_line > a:
                        sh += matrix[h][a]
                    if a - spread_line > h:
                        sa += matrix[h][a]
            spread_home = round(sh, 4)
            spread_away = round(sa, 4)

        # Total
        over_prob = under_prob = None
        if total_line is not None:
            op = up = 0.0
            for h in range(rows):
                for a in range(cols):
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