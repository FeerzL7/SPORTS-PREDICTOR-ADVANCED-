"""
core/simulation/skellam.py

Modelo de probabilidad basado en la distribución Skellam.

La distribución Skellam modela la diferencia de dos variables Poisson
independientes:

    Si X ~ Poisson(μ1) y Y ~ Poisson(μ2), independientes,
    entonces D = X - Y ~ Skellam(μ1, μ2)

Uso en NBA
----------
En baloncesto, cada posesión puede modelarse como un evento Poisson:
P(score en esta posesión) ≈ constante por equipo. Por tanto:

    Home_points ~ Poisson(μ_h)
    Away_points ~ Poisson(μ_a)
    Diff = Home - Away ~ Skellam(μ_h, μ_a)

Esto hace que SkellamModel sea más apropiado que NormalModel para
spreads NBA porque captura la naturaleza discreta del scoring de
puntos (cada canasta vale 1, 2 o 3 puntos — los puntos no son
continuos). Para totales, NormalModel sigue siendo preferible por
eficiencia numérica.

El uso combinado documentado en la arquitectura (§8.1):
    sport='nba' → SkellamModel() para spread
                  NormalModel(default_sigma=11.0) para totales

PMF de la distribución Skellam
--------------------------------
La PMF exacta usa la función de Bessel modificada de primer tipo I_k:

    P(D = k) = exp(-(μ1 + μ2)) × (μ1/μ2)^(k/2) × I_|k|(2√(μ1×μ2))

donde I_k es la función de Bessel modificada de primer tipo de orden k.

Implementación: math.gamma y la serie de potencias de I_k evitando
scipy. La serie converge rápidamente para los valores de μ típicos
de NBA (80–130 puntos por equipo).

Rango de diferencias
---------------------
En NBA, la diferencia máxima realista es ≈ ±50 puntos. Iteramos
d ∈ [-max_diff, +max_diff] con max_diff=60 por defecto — suficiente
para cubrir >99.9% de la masa de probabilidad para cualquier partido
de NBA.
"""

from __future__ import annotations

import math

from core.contracts import Projection
from core.simulation.protocols import SimulationResult


# ── Función de Bessel modificada I_k vía serie de potencias ──────────────────

# Umbral a partir del cual se usa la expansión asintótica en vez de la
# serie de potencias. Para x > _BESSEL_ASYMPTOTIC_THRESHOLD, la serie de
# potencias requiere cientos de términos para converger y acumula error
# de cancelación catastrófica en valores intermedios del orden de 10^90+
# antes de combinarse con exp(-(mu1+mu2)) en _skellam_pmf — el caso real
# que produjo masa de probabilidad incompleta (0.25 en vez de 1.0) para
# proyecciones NBA con mu≈110 por equipo (x=2*sqrt(mu1*mu2)≈220).
_BESSEL_ASYMPTOTIC_THRESHOLD = 50.0


def _bessel_i(k: int, x: float) -> float:
    """
    Función de Bessel modificada de primer tipo I_k(x) evaluada en x.

    Dos regímenes, seleccionados por la relación entre k Y x —no solo
    por x absoluto:

    1. Serie de potencias exacta — usada cuando x <= umbral, O cuando
       k es comparable/mayor a x (la expansión asintótica de Bessel
       solo es válida para x >> k; si k se acerca a x, sus términos
       de corrección (4k²-1)/(8x) superan 1.0 y la serie asintótica
       DIVERGE en vez de converger — bug real detectado en validación:
       para x≈222 (NBA) y k≥22, el ratio de corrección ya excedía 1.0,
       produciendo log_bessel_i incorrecto y una caída artificial de
       la PMF de ~0.014 a ~10^-302 entre k=20 y k=30).

           I_k(x) = Σ_{m=0}^{∞} (x/2)^(2m+k) / (m! × Γ(m+k+1))

    2. Expansión asintótica — solo cuando x > umbral Y x >> k
       (heurística: x > 4*k, margen conservador donde el primer
       término de corrección (4k²-1)/(8x) se mantiene < 0.5).

           I_k(x) ≈ exp(x) / √(2πx) × [1 - (4k²-1)/(8x) + ...]

    k se usa como |k| (la función es simétrica: I_{-k}(x) = I_k(x)).
    """
    k = abs(k)

    # Criterio correcto de validez de la expansión asintótica: el
    # primer término de corrección (4k²-1)/(8x) debe ser pequeño
    # (< 0.5 como margen seguro) para que la serie asintótica converja
    # en vez de diverger. La heurística anterior (x > 4*k) era
    # insuficiente: para x≈222, ya en k=24 el ratio de corrección
    # alcanzaba 1.297 (>1.0), muy por debajo del umbral x>4*k=96 que
    # seguía clasificando el caso como "asintótico válido" — esto
    # producía un salto artificial de la PMF de ~10^-3 a ~10^-302
    # entre k=26 y k=28 (bug detectado en validación contra scipy).
    correction_ratio = (4 * k * k - 1) / (8 * x) if x > 0 else float('inf')

    use_asymptotic = (
        x > _BESSEL_ASYMPTOTIC_THRESHOLD
        and correction_ratio < 0.5
    )

    if not use_asymptotic:
        return _bessel_i_series(k, x)

    log_i_k = _log_bessel_i_asymptotic(k, x)
    return math.exp(log_i_k)


def _bessel_i_series(k: int, x: float) -> float:
    """
    Serie de potencias exacta de I_k(x). Itera hasta 500 términos —
    suficiente incluso para x moderadamente grande cuando k también
    es grande (caso donde la asintótica no es aplicable pero la
    convergencia de la serie es más lenta que para k pequeño).
    """
    half_x = x / 2.0
    result = 0.0
    term = (half_x ** k) / math.gamma(k + 1)  # m=0

    for m in range(500):
        result += term
        next_term = term * (half_x ** 2) / ((m + 1) * (m + k + 1))
        if next_term < 1e-15 * result and m > 0:
            break
        term = next_term
    else:
        # No convergió en 500 iteraciones — caso extremo no esperado
        # para los rangos de mu/k de los deportes soportados (MLB,
        # NHL, soccer, NBA). Se retorna el mejor resultado disponible
        # en vez de fallar — _skellam_pmf truncará a 0.0 si el valor
        # resultante es despreciable en el contexto de la PMF completa.
        pass

    return result


def _log_bessel_i_asymptotic(k: int, x: float) -> float:
    """
    log(I_k(x)) vía expansión asintótica para x grande:

        I_k(x) ≈ exp(x) / √(2πx) × [1 - (4k²-1)/(8x) + (4k²-1)(4k²-9)/(2!(8x)²) - ...]

        log(I_k(x)) ≈ x - 0.5*log(2πx) + log(corrección)

    Se usan los primeros 3 términos de la serie de corrección —
    suficiente precisión (error relativo < 1e-6) para x > 50, que es
    el régimen en el que esta función se invoca.
    """
    mu = 4 * k * k
    correction = (
        1
        - (mu - 1) / (8 * x)
        + (mu - 1) * (mu - 9) / (2 * (8 * x) ** 2)
        - (mu - 1) * (mu - 9) * (mu - 25) / (6 * (8 * x) ** 3)
    )
    # correction puede ser ligeramente negativo en el límite numérico
    # para k muy grandes relativos a x — se acota para evitar log de
    # un número no positivo, lo cual no debería ocurrir para los
    # valores de k (diferencias de puntos NBA, |k|<200) y x (>50)
    # relevantes en este modelo.
    correction = max(correction, 1e-300)

    return x - 0.5 * math.log(2 * math.pi * x) + math.log(correction)


def _skellam_pmf(k: int, mu1: float, mu2: float) -> float:
    """
    PMF de la distribución Skellam: P(X1 - X2 = k) donde
    X1 ~ Poisson(mu1), X2 ~ Poisson(mu2).

    P(D=k) = exp(-(μ1+μ2)) × (μ1/μ2)^(k/2) × I_|k|(2√(μ1×μ2))

    Para μ moderados (MLB, soccer, NHL: x=2√(μ1μ2) < 50), se calcula
    directamente vía _bessel_i() y luego a log-espacio.

    Para μ altos (NBA: x≈200+), TODO el cálculo permanece en espacio
    logarítmico hasta el paso final — nunca se materializa I_k(x)
    como número en espacio normal, porque exp(x) con x≈200 ya excede
    holgadamente el rango de float64 (~1.8e308) y produciría overflow
    o pérdida de precisión antes de combinarse con exp(-(μ1+μ2)).

    Este es el bug raíz que producía masa de probabilidad incompleta
    (0.25 en vez de 1.0) para proyecciones NBA: _bessel_i() truncaba
    su serie de potencias a 100 términos sin haber convergido para
    x≈220, y el valor resultante (aunque finito) ya había perdido
    la precisión necesaria.
    """
    if mu1 <= 0 or mu2 <= 0:
        return 1.0 if k == 0 else 0.0

    x = 2 * math.sqrt(mu1 * mu2)
    k_abs = abs(k)

    correction_ratio = (4 * k_abs * k_abs - 1) / (8 * x) if x > 0 else float('inf')
    use_asymptotic = (
        x > _BESSEL_ASYMPTOTIC_THRESHOLD
        and correction_ratio < 0.5
    )

    if not use_asymptotic:
        bessel = _bessel_i_series(k_abs, x)
        if bessel <= 0:
            return 0.0
        log_bessel = math.log(bessel)
    else:
        # Permanecer en espacio log de principio a fin para x grande
        log_bessel = _log_bessel_i_asymptotic(k_abs, x)

    log_pmf = (
        -(mu1 + mu2)
        + (k / 2) * math.log(mu1 / mu2)
        + log_bessel
    )

    # Protección contra underflow de exp() para log_pmf muy negativo
    # (k muy alejado de 0 relativo a mu1, mu2) — resultado correctamente 0.0
    if log_pmf < -700:  # exp(-700) ya es menor que el float mínimo normal
        return 0.0

    result = math.exp(log_pmf)
    return max(result, 0.0)


class SkellamModel:
    """
    Modelo Skellam para spreads NBA (diferencia de puntos discreta).

    Implementa ProbabilityModel via duck typing estructural.

    Usa SkellamModel para spread_probability() y win_probabilities().
    Para total_probability() en NBA, usar NormalModel — la arquitectura
    documenta explícitamente esta combinación:
        sport='nba' → SkellamModel() para spread
                      NormalModel(default_sigma=11.0) para totales

    Parámetros
    ----------
    max_diff    -- Rango base de diferencias. El rango real se amplía
                  automáticamente si las proyecciones tienen sigma
                  Skellam grande (ej. NBA con μ≈110 → σ≈15).
                  El rango efectivo es max(max_diff, ceil(6*sigma))
                  para garantizar cobertura de >99.99% de la masa.
                  Default 60 suficiente para deportes de bajo scoring;
                  se amplía automáticamente para NBA.
    """

    def __init__(self, max_diff: int = 60) -> None:
        if max_diff <= 0:
            raise ValueError(f"max_diff={max_diff} debe ser > 0.")
        self.max_diff = max_diff

    def _effective_max_diff(self, mu_h: float, mu_a: float) -> int:
        """
        Calcula el rango efectivo de iteración garantizando cobertura
        suficiente de la masa de probabilidad.

        sigma_skellam = sqrt(mu_h + mu_a)
        rango = max(self.max_diff, ceil(6 * sigma))

        Para NBA (mu≈110): sigma≈14.8 → rango=ceil(6*14.8)=89
        Para MLB (mu≈4.5): sigma≈3.0  → rango=max(60, 18)=60
        """
        import math as _math
        sigma = _math.sqrt(mu_h + mu_a)
        return max(self.max_diff, _math.ceil(6 * sigma))

    # ── ProbabilityModel interface ─────────────────────────────────────────────

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        P(home wins), P(away wins), P(draw) via Skellam(μ_h, μ_a).

        draw > 0 en teoría (P(Diff=0) es la probabilidad de empate
        exacto en tiempo reglamentario). En NBA existe overtime para
        resolver empates, así que 'draw' aquí significa "necesita OT"
        — el sport plugin puede manejar esto con lógica adicional si
        necesita modelar OT explícitamente.
        """
        mu_h = max(projection.expected_home, 0.1)
        mu_a = max(projection.expected_away, 0.1)
        eff_max = self._effective_max_diff(mu_h, mu_a)

        p_home = p_away = p_draw = 0.0

        for d in range(-eff_max, eff_max + 1):
            p = _skellam_pmf(d, mu_h, mu_a)
            if d > 0:
                p_home += p
            elif d < 0:
                p_away += p
            else:
                p_draw += p

        total = p_home + p_away + p_draw
        if total <= 0:
            return {'home': 0.5, 'away': 0.5, 'draw': 0.0}

        return {
            'home': round(p_home / total, 4),
            'away': round(p_away / total, 4),
            'draw': round(p_draw / total, 4),
        }

    def spread_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(equipo cubre el spread) sumando la PMF Skellam sobre el
        rango de diferencias que satisfacen la condición.

        Convención de line idéntica a NormalModel: line es el handicap
        del equipo seleccionado en side, igual que MarketOdds.line.
        Para consultar el lado 'away' de un mercado, el caller pasa
        la línea CON SIGNO OPUESTO a la del lado 'home' del mismo
        mercado (ej. home=-3.5 → away=+3.5), igual que en cualquier
        mercado real de spread.

        Ejemplo: home favorito -3.5 (home debe ganar por 4+):
            side='home', line=-3.5 → P(home - away > 3.5) = P(D > 3.5)
            side='away', line=+3.5 → P(away cubre +3.5)
                                    = P(home NO gana por más de 3.5)
                                    = P(D <= 3.5) = P(D <= 3) para D entero

        Cálculo del threshold (deliberadamente distinto por side, no
        una única fórmula simétrica — error real cometido en una
        iteración previa de este método, donde usar threshold=-line
        para ambos sides producía spread_home + spread_away ≈ 0.82 en
        vez de 1.0 para proyecciones NBA):

            side='home', line=-3.5: threshold = -line = 3.5
                                     prob = P(D > threshold)
            side='away', line=+3.5: threshold = +line = 3.5
                                     prob = P(D <= threshold)

        Propiedad garantizada (verificada en tests): para líneas
        opuestas correctas del mismo mercado,
            spread_probability(line, 'home') +
            spread_probability(-line, 'away') = 1.0
        porque D es una variable discreta y los dos eventos son
        complementarios exactos: o bien home gana por más del margen,
        o bien no lo hace (lo cual equivale a que away cubra +margen).
        """
        mu_h = max(projection.expected_home, 0.1)
        mu_a = max(projection.expected_away, 0.1)
        eff_max = self._effective_max_diff(mu_h, mu_a)

        prob = 0.0
        if side == 'home':
            threshold = -line
            for d in range(-eff_max, eff_max + 1):
                if d > threshold:
                    prob += _skellam_pmf(d, mu_h, mu_a)
        else:
            threshold = line
            for d in range(-eff_max, eff_max + 1):
                if d <= threshold:
                    prob += _skellam_pmf(d, mu_h, mu_a)

        return round(prob, 4)

    def total_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        Nota: para NBA, la arquitectura recomienda NormalModel para
        total_probability(). Este método se provee por completitud del
        Protocol, pero su precisión es menor que NormalModel para totales
        — la distribución de la suma de dos Poisson no es Skellam, es
        Poisson(μ_h + μ_a), y para valores de μ≈113, la aproximación
        Normal es más eficiente y prácticamente equivalente.

        Implementación: itera sobre todos los pares (h, a) donde h+a
        cruza la línea — computacionalmente costoso para μ altos.
        Para producción NBA, usar NormalModel.total_probability().
        """
        mu_h = max(projection.expected_home, 0.1)
        mu_a = max(projection.expected_away, 0.1)
        mu_total = mu_h + mu_a

        # Para totales altos, usar la suma de Poisson directamente
        # via CDF de Poisson(mu_h + mu_a) — más eficiente que iterar pares
        from core.utils.math.poisson_math import cdf as poisson_cdf, sf as poisson_sf
        if side == 'over':
            return round(float(poisson_sf(line, mu_total)), 4)
        else:
            # Under: excluir push si linea entera
            import math as _math
            if line == _math.floor(line):
                return round(float(poisson_cdf(line - 1, mu_total)), 4)
            return round(float(poisson_cdf(_math.floor(line), mu_total)), 4)

    def model_version(self) -> str:
        return f'skellam-v1.0-maxdiff{self.max_diff}'

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
            draw_prob=win_probs['draw'],
            spread_home_prob=spread_home,
            spread_away_prob=spread_away,
            over_prob=over_prob,
            under_prob=under_prob,
            model_name=self.model_version(),
            projection=projection,
        )