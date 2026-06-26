"""
core/utils/math/poisson_math.py

Matemática de la distribución de Poisson, sin dependencias del
stdlib más allá de math.

Migrado desde utils/poisson_math.py del sistema MLB-PREDICTOR-ADVANCED
sin cambios funcionales (ver SPORTS_PREDICTOR_ROADMAP.md, tarea
C1.2.1). Las tres funciones (pmf, cdf, sf) son idénticas byte a byte
en su lógica al sistema original — es matemática pura sin ningún
acoplamiento a MLB ni a ningún otro deporte, por lo que no requirió
ninguna decisión de diseño ni generalización paramétrica, a
diferencia de core/utils/logger.py.

Es la base aritmética consumida por core/simulation/poisson.py
(PoissonModel, Bloque 2 del roadmap) para MLB, NHL y cualquier otro
deporte de baja anotación modelable como conteo discreto.

Limitaciones conocidas del modelo Poisson — documentadas aquí porque
afectan directamente cómo se interpretan los resultados de cdf()/sf(),
identificadas en MLB_EDGE_AUDIT.md:

1. Equidispersión (varianza = media): Poisson asume que la varianza
   de las anotaciones es igual a su media. Los datos reales de MLB
   muestran overdispersión (varianza > media), lo que produce
   probabilidades de victoria sistemáticamente más altas que las
   reales cuando se usa Poisson puro. Esta limitación NO se corrige
   en este módulo — es responsabilidad de capas superiores
   (EnsembleModel, NegBinomialModel) compensarla cuando el
   ProjectionModel de un deporte específico lo requiera.

2. Masa de probabilidad truncada con max_score finito: cualquier
   simulación que itere k en range(max_score) para aproximar
   probabilidades de mercado pierde la masa de probabilidad residual
   más allá de ese límite. Para mu altos relativos a max_score, esa
   masa truncada puede ser no despreciable. cdf()/sf() en este módulo
   no truncan — son exactas vía suma directa de pmf() hasta k. El
   truncamiento ocurre en capas superiores que iteran un rango finito
   (ej. PoissonModel.win_probabilities con max_score configurable),
   no aquí.

Uso típico:
    from core.utils.math.poisson_math import pmf, cdf, sf

    # Probabilidad de exactamente 4 anotaciones con media 4.5
    pmf(4, 4.5)

    # Probabilidad de 8 o menos anotaciones (para mercado Under 8.5)
    cdf(8, 4.5)

    # Probabilidad de más de 8 anotaciones (para mercado Over 8.5)
    sf(8, 4.5)
"""

import math


def pmf(k: int, mu: float) -> float:
    """
    Probability Mass Function: P(X = k) para X ~ Poisson(mu).

    Calculada en espacio logarítmico (k*log(mu) - mu - lgamma(k+1))
    para evitar overflow/underflow con factoriales grandes — idéntico
    al sistema MLB original.

    Parámetros
    ----------
    k   -- Número de eventos (anotaciones). k < 0 retorna 0.0.
    mu  -- Media de la distribución (debe ser > 0; se fuerza a un
          mínimo de 0.0001 para evitar log(0) si mu llega como 0
          desde una proyección degenerada).
    """
    if k < 0:
        return 0.0
    mu = max(float(mu), 0.0001)
    return math.exp(k * math.log(mu) - mu - math.lgamma(k + 1))


def cdf(k: float, mu: float) -> float:
    """
    Cumulative Distribution Function: P(X <= k) para X ~ Poisson(mu).

    Suma exacta de pmf(i, mu) para i en [0, floor(k)]. No trunca masa
    de probabilidad — es la suma completa hasta el entero floor(k).

    Usado para mercados Under: P(total < linea) = cdf(linea - 1, mu).

    Parámetros
    ----------
    k   -- Límite superior (inclusive tras floor). k < 0 retorna 0.0.
    mu  -- Media de la distribución.
    """
    upper = math.floor(k)
    if upper < 0:
        return 0.0
    return sum(pmf(i, mu) for i in range(upper + 1))


def sf(k: float, mu: float) -> float:
    """
    Survival Function: P(X > k) para X ~ Poisson(mu).

    Complemento de cdf(): sf(k, mu) = 1 - cdf(k, mu), acotado en 0.0
    por seguridad numérica ante errores de redondeo acumulados en la
    suma de cdf().

    Usado para mercados Over: P(total > linea) = sf(linea, mu).
    """
    return max(0.0, 1.0 - cdf(k, mu))