"""
core/evaluation/blend_fit.py

Ajuste del peso del modelo frente al mercado, con validación fuera de
muestra.

OBJETIVO
--------
`blending.{MERCADO}.model_weight` decide cuánto pesa la opinión del
modelo frente a la probabilidad sin vig del mercado:

    blended = w · modelo + (1 − w) · mercado

Hasta la Fase 0 (2026-10) ese peso se fijaba a mano (0.35, 0.45,
0.62...). Este módulo lo estima con datos y, sobre todo, comprueba si
el valor estimado sobrevive a una temporada que el ajuste no vio.

La pregunta que responde
--------------------------
¿Añade el modelo información que el mercado de cierre no tenga ya?

Si la respuesta es no, el peso óptimo es 0 y cualquier pick que el
modelo genere sale de sus discrepancias con el mercado, que son ruido.
Es exactamente lo que se midió en fútbol: peso óptimo 0.00 en 1X2 y
TOTAL, con la config usando 0.45.

MÉTODO
------
    1. Separar las observaciones en ENTRENAMIENTO y PRUEBA por
       temporada. Nunca al azar: dentro de una temporada los partidos
       comparten equipos y forma, y mezclarlos filtra información.

    2. En entrenamiento, buscar el w ∈ [0, 1] que minimiza la log-loss
       de la mezcla. Es una búsqueda en rejilla, no una optimización:
       con un solo parámetro acotado es exacta y no tiene mínimos
       locales que vigilar.

    3. En prueba, medir Δlog-loss = LL(mezcla con ese w) − LL(mercado).
       Negativo significa que el modelo mejora al mercado.

    4. Intervalo de confianza del 95% por bootstrap AGRUPADO por
       partido. Las selecciones de un mismo partido (local, empate,
       visitante) no son independientes: remuestrearlas por separado
       estrecharía el intervalo artificialmente.

Diagnóstico adicional: combinación logística
----------------------------------------------
    logit(p) = a + b · logit(modelo) + c · logit(mercado)

El signo de b dice algo que w no puede decir, porque w está acotado
en [0, 1]: si b < 0, apostar contra las discrepancias del modelo
sería MEJOR que apostar a favor. Es el caso de fútbol (b ≈ −0.13).

VEREDICTO
---------
    INFORMATIVO   w > 0 en entrenamiento Y el intervalo de Δlog-loss
                  en prueba queda entero por debajo de 0.
    SIN SEÑAL     cualquier otro caso → el mercado va a solo-papel.

INFORMATIVO no autoriza a operar: es la condición NECESARIA del
backtest. La suficiente es CLV > 0 en el ledger de papel con n ≥ 100.

Dependencias: solo la biblioteca estándar, como el resto del core.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

# Límite para que log(0) no aparezca con probabilidades de 0 o 1.
_EPS = 1e-6

VERDICT_INFORMATIVE = "INFORMATIVO"
VERDICT_NO_SIGNAL   = "SIN SEÑAL"


# ── Datos de entrada ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    """
    Una selección evaluada: lo que dijo el modelo, lo que dijo el
    mercado y lo que pasó.

    model_prob   -- Probabilidad cruda del modelo para la selección.
    market_prob  -- Probabilidad sin vig del mercado de referencia.
    outcome      -- 1 si la selección ganó, 0 si perdió. Los push y
                   anulados se excluyen antes de construir esto.
    season       -- Temporada, para la separación entrenamiento/prueba.
    group        -- Identificador del partido para el bootstrap
                   agrupado.
    """
    model_prob:  float
    market_prob: float
    outcome:     int
    season:      int
    group:       str


# ── Funciones de pérdida ──────────────────────────────────────────────────────

def _clip(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def point_log_loss(p: float, outcome: int) -> float:
    """Log-loss de una predicción binaria."""
    p = _clip(p)
    return -math.log(p) if outcome else -math.log(1.0 - p)


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Log-loss media. Menor es mejor."""
    if not probs:
        raise ValueError("log_loss() sin observaciones")
    return sum(point_log_loss(p, y) for p, y in zip(probs, outcomes)) / len(probs)


def blend(model_prob: float, market_prob: float, weight: float) -> float:
    """Mezcla convexa, idéntica a la de BlendingEngine."""
    return weight * model_prob + (1.0 - weight) * market_prob


# ── Ajuste ────────────────────────────────────────────────────────────────────

def fit_linear_weight(obs: Sequence[Observation], step: float = 0.01) -> float:
    """
    Peso w ∈ [0, 1] que minimiza la log-loss de la mezcla.

    En caso de empate se queda con el menor: ante la duda, más peso
    al mercado.
    """
    if not obs:
        raise ValueError("fit_linear_weight() sin observaciones")
    n_steps = int(round(1.0 / step))
    best_w, best_ll = 0.0, math.inf
    for i in range(n_steps + 1):
        w = i / n_steps
        ll = sum(point_log_loss(blend(o.model_prob, o.market_prob, w), o.outcome)
                 for o in obs)
        if ll < best_ll - 1e-12:
            best_w, best_ll = w, ll
    return best_w


def fit_logistic_stack(
    obs: Sequence[Observation],
    iters: int = 50,
    l2: float = 1e-4,
) -> tuple[float, float, float]:
    """
    Regresión logística (a, b, c) sobre (1, logit(modelo), logit(mercado)).

    Newton-Raphson con una pequeña penalización L2 para que la matriz
    sea invertible aunque modelo y mercado estén muy correlacionados.
    """
    if not obs:
        raise ValueError("fit_logistic_stack() sin observaciones")
    xs = [(1.0, _logit(o.model_prob), _logit(o.market_prob)) for o in obs]
    ys = [o.outcome for o in obs]
    beta = [0.0, 0.0, 1.0]  # arranque: el mercado tal cual

    for _ in range(iters):
        grad = [0.0, 0.0, 0.0]
        hess = [[0.0] * 3 for _ in range(3)]
        for x, y in zip(xs, ys):
            z = beta[0] * x[0] + beta[1] * x[1] + beta[2] * x[2]
            p = 1.0 / (1.0 + math.exp(-max(min(z, 35.0), -35.0)))
            w = p * (1.0 - p)
            for i in range(3):
                grad[i] += (y - p) * x[i]
                for j in range(3):
                    hess[i][j] += w * x[i] * x[j]
        for i in range(3):
            grad[i] -= l2 * beta[i]
            hess[i][i] += l2
        delta = _solve3(hess, grad)
        beta = [b + d for b, d in zip(beta, delta)]
        if max(abs(d) for d in delta) < 1e-9:
            break
    return beta[0], beta[1], beta[2]


def _solve3(a: list[list[float]], b: list[float]) -> list[float]:
    """Resuelve un sistema 3×3 por eliminación gaussiana con pivoteo."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-15:
            return [0.0, 0.0, 0.0]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        x[r] = (m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))) / m[r][r]
    return x


# ── Validación fuera de muestra ───────────────────────────────────────────────

@dataclass(frozen=True)
class MarketFit:
    """
    Resultado del ajuste de un mercado.

    weight          -- w estimado SOLO con las temporadas de entrenamiento.
    delta_train     -- Δlog-loss en entrenamiento (optimista por
                      construcción: el peso se eligió ahí).
    delta_test      -- Δlog-loss en prueba. La cifra que cuenta.
    ci_low/ci_high  -- Intervalo del 95% de delta_test (bootstrap
                      agrupado por partido).
    stack_model_coef -- Coeficiente b del modelo en la combinación
                       logística (ajustada en entrenamiento).
    verdict         -- VERDICT_INFORMATIVE o VERDICT_NO_SIGNAL.
    """
    market:           str
    train_seasons:    tuple[int, ...]
    test_seasons:     tuple[int, ...]
    n_train:          int
    n_test:           int
    weight:           float
    delta_train:      float
    delta_test:       float
    ci_low:           float
    ci_high:          float
    stack_model_coef: float
    stack_market_coef: float
    verdict:          str


def evaluate_market(
    market:       str,
    obs:          Sequence[Observation],
    test_seasons: Iterable[int],
    n_boot:       int = 1000,
    seed:         int = 7,
) -> MarketFit:
    """
    Ajusta w en las temporadas que no están en `test_seasons` y lo
    evalúa en las que sí.

    Raises
    ------
    ValueError -- Si entrenamiento o prueba quedan vacíos: sin una de
                 las dos mitades no hay validación fuera de muestra.
    """
    test_set = frozenset(test_seasons)
    train = [o for o in obs if o.season not in test_set]
    test  = [o for o in obs if o.season in test_set]
    if not train or not test:
        raise ValueError(
            f"{market}: entrenamiento={len(train)} prueba={len(test)}. "
            f"Hacen falta observaciones en ambas mitades."
        )

    w = fit_linear_weight(train)
    _, coef_model, coef_market = fit_logistic_stack(train)

    def delta(sample: Sequence[Observation]) -> float:
        return (
            sum(point_log_loss(blend(o.model_prob, o.market_prob, w), o.outcome)
                - point_log_loss(o.market_prob, o.outcome) for o in sample)
            / len(sample)
        )

    ci_low, ci_high = _cluster_bootstrap_ci(test, w, n_boot, seed)
    d_test = delta(test)

    verdict = (VERDICT_INFORMATIVE if w > 0 and ci_high < 0
               else VERDICT_NO_SIGNAL)

    return MarketFit(
        market            = market,
        train_seasons     = tuple(sorted({o.season for o in train})),
        test_seasons      = tuple(sorted({o.season for o in test})),
        n_train           = len(train),
        n_test            = len(test),
        weight            = w,
        delta_train       = delta(train),
        delta_test        = d_test,
        ci_low            = ci_low,
        ci_high           = ci_high,
        stack_model_coef  = coef_model,
        stack_market_coef = coef_market,
        verdict           = verdict,
    )


def _cluster_bootstrap_ci(
    test:   Sequence[Observation],
    weight: float,
    n_boot: int,
    seed:   int,
) -> tuple[float, float]:
    """
    Intervalo del 95% de Δlog-loss remuestreando PARTIDOS, no filas.

    Con w = 0 la mezcla es el mercado y Δ es exactamente 0 en cada
    remuestreo: el intervalo sale [0, 0], que es la respuesta correcta.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for o in test:
        d = (point_log_loss(blend(o.model_prob, o.market_prob, weight), o.outcome)
             - point_log_loss(o.market_prob, o.outcome))
        sums[o.group] = sums.get(o.group, 0.0) + d
        counts[o.group] = counts.get(o.group, 0) + 1

    groups = list(sums)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        s = c = 0
        for _g in range(len(groups)):
            g = groups[rng.randrange(len(groups))]
            s += sums[g]
            c += counts[g]
        stats.append(s / c)
    stats.sort()
    lo = stats[int(0.025 * (n_boot - 1))]
    hi = stats[int(0.975 * (n_boot - 1))]
    return lo, hi
