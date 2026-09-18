"""
sports/soccer/dixon_coles.py

Matriz de resultados con corrección Dixon-Coles.

El problema que resuelve
-------------------------
Modelar los goles de cada equipo como dos Poisson independientes es la
aproximación clásica, y funciona razonablemente bien salvo en un punto
concreto: SUBESTIMA los marcadores bajos.

Los cuatro resultados 0-0, 1-0, 0-1 y 1-1 concentran alrededor del 30%
de los partidos reales en ligas europeas. El Poisson independiente les
asigna menos, y el déficit no se reparte de forma uniforme — se
concentra justo donde está la mayor masa de probabilidad.

Dixon y Coles (1997) lo corrigen con un parámetro `rho` que ajusta
únicamente esas cuatro celdas, dejando el resto de la matriz intacto:

    tau(0,0) = 1 - λ·μ·rho
    tau(0,1) = 1 + λ·rho
    tau(1,0) = 1 + μ·rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1            para el resto

con rho negativo, lo que AUMENTA la probabilidad de 0-0 y 1-1 y la
reduce en 1-0 y 0-1. El resultado neto es más masa en marcadores bajos
y más empates.

Por qué importa para el negocio, no solo para la teoría
---------------------------------------------------------
El empate es el 25% de los partidos y el mercado donde los books
aplican más margen — justamente porque es el que peor modela la mayoría
de sistemas. Un modelo que infravalore el empate no encontrará valor en
el único mercado donde estructuralmente más lo hay.

La corrección es la diferencia entre proyectar el 1X2 y proyectar solo
"quién gana".

Correlación entre las anotaciones
-----------------------------------
Además de Dixon-Coles, el módulo soporta el Poisson BIVARIADO, que
modela la correlación entre los goles de ambos equipos mediante un
componente compartido λ₃:

    Goles_local    = X₁ + X₃
    Goles_visitante = X₂ + X₃

con X₁, X₂, X₃ Poisson independientes. λ₃ captura lo que afecta a ambos
por igual: un partido abierto donde los dos atacan, o uno cerrado donde
ninguno arriesga.

En fútbol esa correlación es POSITIVA y modesta (~0.12). Es la
situación opuesta a la de NFL, donde el game script la vuelve negativa:
el equipo que gana corre el reloj y suprime la anotación de ambos.

NO son complementarios: son ALTERNATIVAS
-----------------------------------------
Una primera versión de este módulo afirmaba que ambos mecanismos se
complementaban y traía los dos activos por defecto. Es incorrecto, y la
validación lo expuso: modelan la MISMA dependencia por vías distintas.

    rho = -0.13 procede de estudios SIN componente bivariado.
    λ₃  =  0.12 procede de estudios SIN Dixon-Coles.

Aplicar ambos cuenta el efecto dos veces. Con λ 1.50/1.20:

    ninguno          empate 25.5%
    solo Dixon-Coles empate 28.6%
    solo λ₃          empate 26.9%
    AMBOS            empate 30.2%   ← fuera del rango real (22-28%)

Por eso λ₃ viene desactivado por defecto: Dixon-Coles hace el trabajo
con menos parámetros. Queda disponible para quien prefiera la vía
bivariada, pero entonces debe ponerse rho a cero.

Qué hace exactamente la corrección
------------------------------------
Otro punto que la validación aclaró: Dixon-Coles CONSERVA la masa total
de marcadores bajos y la redistribuye dentro.

    0-0   +0.0157      1-0   -0.0157
    1-1   +0.0157      0-1   -0.0157

El efecto neto sobre los cuatro es cero; lo que cambia es el reparto
entre empates (0-0, 1-1) y victorias mínimas (1-0, 0-1). Por eso el
indicador correcto de que la corrección funciona es la probabilidad de
EMPATE, no la masa de marcadores bajos.

Sobre el valor de rho
-----------------------
El default es -0.05, no el -0.13 que aparece en buena parte de la
literatura. La razón: el efecto neto de tau depende del producto λ_h·λ_a
del modelo concreto, y los valores publicados se calibraron junto con
sus propias estimaciones de ataque y defensa.

Con las medias de las cinco grandes, rho=-0.13 empuja el empate al
28.6%, por encima del rango observado. -0.05 lo deja en 26.7%, dentro
de rango en escenarios igualados y desiguales por igual.

`calibrate_rho()` permite estimarlo con resultados reales, que es lo
que debería hacerse antes de operar con dinero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# API pública del módulo.
#
# Se declara explícitamente por lo mismo que en el Core: sin una lista
# de exportación, la condición de pública de cada símbolo queda al
# criterio de quien lea el código, y el type checker no tiene nada
# contra lo que validar un import.
#
# Las constantes _DEFAULT_* quedan FUERA a propósito: son los valores
# de respaldo de este módulo, no configuración compartida. Un
# consumidor que los necesite debería declarar los suyos —igual que
# hace SoccerProjectionModel— en vez de acoplarse a los internos de
# aquí.
__all__ = [
    "ScoreMatrix",
    "build_score_matrix",
    "calibrate_rho",
]


# ── Constantes ───────────────────────────────────────────────────────────────

# Máximo de goles por equipo en la matriz.
#
# Diez cubre más del 99.9% de los marcadores reales: un 10-0 ocurre
# cada varios miles de partidos. Ampliarlo solo añade coste
# computacional sin cambiar ninguna probabilidad de mercado.
_DEFAULT_MAX_GOALS = 10

# Parámetro rho de Dixon-Coles.
#
# Negativo por construcción: indica que los marcadores bajos son MÁS
# frecuentes de lo que predice la independencia. Los valores publicados
# para ligas europeas van de -0.10 a -0.18.
_DEFAULT_RHO = -0.05

# Componente compartido del Poisson bivariado.
#
# DESACTIVADO por defecto. Dixon-Coles y el bivariado modelan la misma
# dependencia; activar ambos la cuenta dos veces y empuja el empate
# fuera del rango observado. Ver la nota del módulo.
#
# Para usar la vía bivariada en su lugar: lambda_3 ≈ 0.12 con rho = 0.
_DEFAULT_LAMBDA_3 = 0.0

# Cotas de las medias esperadas. Red de seguridad ante datos corruptos:
# ningún equipo de las cinco grandes proyecta fuera de este rango.
_LAMBDA_MIN = 0.05
_LAMBDA_MAX = 6.00


@dataclass(frozen=True)
class ScoreMatrix:
    """
    Distribución de probabilidad sobre los marcadores posibles.

    `grid[h][a]` es la probabilidad del marcador h-a. La matriz está
    normalizada: sus celdas suman 1.0 salvo la masa residual por
    encima de `max_goals`, que se redistribuye proporcionalmente.

    Todos los mercados de fútbol se derivan de aquí: 1X2 sumando las
    celdas de cada región, over/under sumando por diagonal, BTTS
    sumando el cuadrante donde ambos anotan. Calcular una sola matriz
    y derivar todo de ella garantiza que los mercados sean coherentes
    entre sí — una condición que un cálculo independiente por mercado
    no asegura.
    """
    grid:      tuple[tuple[float, ...], ...]
    lambda_home: float
    lambda_away: float
    rho:       float
    lambda_3:  float
    max_goals: int

    # ── Mercados principales ──────────────────────────────────────────────────

    def outcome_probabilities(self) -> dict[str, float]:
        """
        Probabilidades de 1X2.

        Retorna {'home': p, 'draw': p, 'away': p}, que suman 1.0.
        """
        home = draw = away = 0.0
        for h, row in enumerate(self.grid):
            for a, p in enumerate(row):
                if h > a:
                    home += p
                elif h == a:
                    draw += p
                else:
                    away += p
        return {
            "home": round(home, 6),
            "draw": round(draw, 6),
            "away": round(away, 6),
        }

    def total_over(self, line: float) -> float:
        """
        Probabilidad de que el total supere la línea.

        Con línea entera (2.0, 3.0) el marcador exacto es PUSH y no
        cuenta para ninguno de los dos lados. Esta función devuelve
        solo la masa estrictamente superior; `total_push` da la del
        empate.
        """
        return round(
            sum(p for h, row in enumerate(self.grid)
                for a, p in enumerate(row) if h + a > line),
            6,
        )

    def total_under(self, line: float) -> float:
        """Probabilidad de que el total quede por debajo de la línea."""
        return round(
            sum(p for h, row in enumerate(self.grid)
                for a, p in enumerate(row) if h + a < line),
            6,
        )

    def total_push(self, line: float) -> float:
        """
        Probabilidad de push en el total.

        Cero con líneas .5, que es el caso mayoritario en fútbol. Con
        línea entera es la masa del marcador exacto, y en fútbol no es
        despreciable: un total de 2.0 tiene ~22% de push porque los
        partidos de dos goles son muy frecuentes.
        """
        if line != int(line):
            return 0.0
        target = int(line)
        return round(
            sum(p for h, row in enumerate(self.grid)
                for a, p in enumerate(row) if h + a == target),
            6,
        )

    def btts(self) -> float:
        """Probabilidad de que ambos equipos marquen."""
        return round(
            sum(p for h, row in enumerate(self.grid)
                for a, p in enumerate(row) if h >= 1 and a >= 1),
            6,
        )

    # ── Mercados derivados ────────────────────────────────────────────────────

    def handicap(self, line: float, side: str = "home") -> float:
        """
        Probabilidad de cubrir un hándicap europeo.

        `line` es el handicap de la selección: negativo para el
        favorito. Un local con -1.0 cubre si gana por dos o más.

        Con línea entera existe el push, igual que en el total.
        """
        total = 0.0
        for h, row in enumerate(self.grid):
            for a, p in enumerate(row):
                margin = (h - a) if side == "home" else (a - h)
                if margin + line > 0:
                    total += p
        return round(total, 6)

    def exact_score(self, home_goals: int, away_goals: int) -> float:
        """Probabilidad de un marcador concreto."""
        if not (0 <= home_goals <= self.max_goals):
            return 0.0
        if not (0 <= away_goals <= self.max_goals):
            return 0.0
        return round(self.grid[home_goals][away_goals], 6)

    def most_likely_score(self) -> tuple[int, int, float]:
        """Marcador más probable y su probabilidad."""
        best = (0, 0, 0.0)
        for h, row in enumerate(self.grid):
            for a, p in enumerate(row):
                if p > best[2]:
                    best = (h, a, p)
        return best

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    @property
    def expected_total(self) -> float:
        """Media de goles de la distribución, no de los parámetros."""
        return round(
            sum((h + a) * p for h, row in enumerate(self.grid)
                for a, p in enumerate(row)),
            4,
        )

    @property
    def expected_margin(self) -> float:
        """Margen medio desde la perspectiva del local."""
        return round(
            sum((h - a) * p for h, row in enumerate(self.grid)
                for a, p in enumerate(row)),
            4,
        )

    @property
    def low_score_mass(self) -> float:
        """
        Masa de probabilidad en los cuatro marcadores bajos.

        Es la que Dixon-Coles corrige. En ligas europeas ronda el 30%
        real; sin corrección el Poisson independiente se queda corto.
        """
        return round(
            self.grid[0][0] + self.grid[1][0]
            + self.grid[0][1] + self.grid[1][1],
            6,
        )


# ── Construcción de la matriz ────────────────────────────────────────────────

def build_score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho:         float = _DEFAULT_RHO,
    lambda_3:    float = _DEFAULT_LAMBDA_3,
    max_goals:   int = _DEFAULT_MAX_GOALS,
) -> ScoreMatrix:
    """
    Construye la matriz de resultados.

    Parámetros
    ----------
    lambda_home / lambda_away
        Medias esperadas de goles de cada equipo. Son los parámetros
        MARGINALES: si lambda_3 > 0, las medias del proceso bivariado
        se ajustan internamente para que la media total siga siendo la
        pedida.
    rho
        Parámetro de Dixon-Coles. Negativo aumenta la masa en
        marcadores bajos. Cero desactiva la corrección.
    lambda_3
        Componente compartido del Poisson bivariado. Cero lo reduce a
        dos Poisson independientes.
    max_goals
        Tamaño de la matriz.

    Nunca lanza: medias fuera de rango se acotan, y valores inválidos
    caen a los parámetros por defecto. Una matriz mal formada rompería
    todos los mercados a la vez, así que la degradación tiene que ser
    silenciosa y segura.
    """
    from core.simulation.bivariate_poisson import build_joint_matrix

    lam_h = _clamp(_safe(lambda_home, 1.35), _LAMBDA_MIN, _LAMBDA_MAX)
    lam_a = _clamp(_safe(lambda_away, 1.15), _LAMBDA_MIN, _LAMBDA_MAX)
    rho_v = _clamp(_safe(rho, _DEFAULT_RHO), -0.50, 0.0)
    lam_3 = _clamp(_safe(lambda_3, _DEFAULT_LAMBDA_3), 0.0,
                   min(lam_h, lam_a) * 0.9)
    size = max(2, int(max_goals))

    grid = build_joint_matrix(lam_h, lam_a, rho_v, size, lam_3)

    return ScoreMatrix(
        grid=tuple(tuple(row) for row in grid),
        lambda_home=round(lam_h, 4),
        lambda_away=round(lam_a, 4),
        rho=round(rho_v, 4),
        lambda_3=round(lam_3, 4),
        max_goals=size,
    )


# ── Delegación al Core ───────────────────────────────────────────────────────
#
# La construcción de la matriz vive en core/simulation/bivariate_poisson.py
# y este módulo la consume mediante `build_joint_matrix`, que es parte
# de la API pública del Core.
#
# Una primera versión importaba `_build_joint_matrix` con el guion bajo:
# un símbolo privado usado desde otro módulo. El guion declara "interno"
# y el import venía de fuera, así que una de las dos cosas estaba mal.
# Se resolvió haciendo pública la función en el Core, que es lo que ya
# era de hecho. Una primera versión la reimplementaba aquí,
# lo que dejaba DOS copias de la misma matemática: cualquier
# recalibración habría que aplicarla en dos sitios y divergirían en
# silencio.
#
# La dirección de la dependencia importa: sports → core, nunca al
# revés. El Core no puede importar de un plugin, así que la matemática
# compartida tiene que estar allí.
#
# Lo que este módulo aporta y el Core no: ScoreMatrix con los mercados
# de fútbol ya derivados (1X2, over/under con push, BTTS, hándicap) y
# calibrate_rho(). Eso sí es específico del deporte.


# ── Calibración ──────────────────────────────────────────────────────────────

def calibrate_rho(
    matches:    list[tuple[float, float, int, int]],
    candidates: tuple[float, ...] = (0.0, -0.02, -0.04, -0.06, -0.08, -0.10, -0.13),
    lambda_3:   float = 0.0,
) -> dict:
    """
    Estima rho maximizando la verosimilitud sobre partidos reales.

    Es lo que debería hacerse antes de operar con dinero: el default
    de -0.05 está elegido para que las tasas de empate caigan en el
    rango observado, pero eso no es lo mismo que ajustarlo a los datos.

    Parámetros
    ----------
    matches -- Lista de (λ_local, λ_visitante, goles_local,
               goles_visitante). Las medias son las que el modelo
               proyectó ANTES del partido; los goles, el resultado.
    candidates -- Valores de rho a probar.

    Retorna
    -------
    dict con `best_rho`, la log-verosimilitud de cada candidato y la
    tasa de empate observada frente a la predicha. Esto último importa
    tanto como la verosimilitud: un rho que maximice la verosimilitud
    pero desvíe la tasa de empate indica que el problema está en las
    medias, no en la corrección.
    """
    if not matches:
        return {"best_rho": _DEFAULT_RHO, "n_matches": 0, "scores": {}}

    observed_draws = sum(1 for _, _, h, a in matches if h == a)
    observed_rate = observed_draws / len(matches)

    scores: dict[float, float] = {}
    predicted: dict[float, float] = {}

    for rho in candidates:
        log_likelihood = 0.0
        draw_sum = 0.0
        for lam_h, lam_a, goals_h, goals_a in matches:
            matrix = build_score_matrix(lam_h, lam_a, rho=rho,
                                        lambda_3=lambda_3)
            p = matrix.exact_score(int(goals_h), int(goals_a))
            log_likelihood += math.log(max(p, 1e-12))
            draw_sum += matrix.outcome_probabilities()["draw"]

        scores[rho] = round(log_likelihood, 4)
        predicted[rho] = round(draw_sum / len(matches), 4)

    best = max(scores, key=lambda r: scores[r])

    return {
        "best_rho":        best,
        "n_matches":       len(matches),
        "scores":          scores,
        "observed_draw_rate":  round(observed_rate, 4),
        "predicted_draw_rate": predicted,
        "draw_rate_error": round(predicted[best] - observed_rate, 4),
    }


# ── Utilidades ───────────────────────────────────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    """
    Función de masa de Poisson.

    Se calcula en espacio logarítmico y se exponencia al final. Con k
    hasta 10 el cálculo directo no desbordaría, pero el logarítmico
    evita el factorial explícito y es numéricamente más estable si
    algún día se amplía la matriz.
    """
    if k < 0 or lam <= 0:
        return 0.0
    try:
        log_p = -lam + k * math.log(lam) - math.lgamma(k + 1)
        return math.exp(log_p)
    except (ValueError, OverflowError):
        return 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe(value, default: float) -> float:
    """Convierte a float tratando None y NaN como ausencia."""
    if value is None:
        return default
    try:
        result = float(value)
    except (ValueError, TypeError):
        return default
    return result if result == result else default