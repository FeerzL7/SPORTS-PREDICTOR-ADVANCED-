"""
core/simulation/ensemble.py

EnsembleModel: ajusta una proyección Poisson combinándola con la
tendencia de regresión lineal sobre los scores recientes del equipo.

Generaliza analysis/ensemble.py del sistema MLB-PREDICTOR-ADVANCED a
cualquier deporte (ver SPORTS_PREDICTOR_ARCHITECTURE.md §8.3).

Por qué el ensemble existe
---------------------------
La distribución de Poisson asume equidispersión (Var = μ). Algunos
equipos tienen scoring "todo o nada" — alta varianza relativa a su
media — donde Poisson subestima sistemáticamente la probabilidad de
resultados extremos. La regresión lineal sobre los últimos N scores
captura una señal distinta: la TENDENCIA temporal del equipo, que
Poisson (al depender solo de la media) no ve en absoluto.

    Ejemplo: equipo con scores recientes [2, 3, 4, 5, 6]
    Poisson ve: media=4.0 (igual que [6,5,4,3,2], que tiene tendencia
    opuesta). La regresión lineal sí distingue ambos casos.

Por qué está activado por defecto (a diferencia del sistema MLB)
--------------------------------------------------------------------
En el sistema MLB original, ENABLE_ENSEMBLE=False Y además
runs_recientes_lista=[] en producción (bug F1/F2 documentado en
CRITICAL_FINDINGS_VALIDATION.md) — el ensemble nunca corrió realmente,
con o sin el flag. Aquí, TeamFeatures.recent_avg se recalcula SIEMPRE
desde recent_scores en __post_init__ (core/contracts/features.py),
haciendo estructuralmente imposible repetir ese bug. El ensemble
puede activarse con confianza.

La decisión de SI se invoca el ensemble pertenece a la capa de
configuración (config/{sport}.yaml vía ProjectionModel del sport
plugin), NO a este módulo. EnsembleModel no tiene un flag
enabled=False interno — adjust() está "siempre activo cuando se
invoca". Esto evita que existan dos lugares (YAML + parámetro de
constructor) donde la misma decisión pudiera divergir silenciosamente.

Fórmula (idéntica a la del sistema MLB original, generalizada)
-------------------------------------------------------------------
    CV = std(recent_scores) / mean(recent_scores)
    alpha = adaptive_alpha(CV)  — interpolación lineal entre umbrales
    regression_pred = linear_regression(recent_scores)
    adjusted = alpha × poisson_proj + (1 - alpha) × regression_pred

CV bajo (equipo consistente) → alpha alto → Poisson domina.
CV alto (equipo "todo o nada") → alpha bajo → regresión domina.

Uso típico:
    from core.simulation.ensemble import EnsembleModel

    ensemble = EnsembleModel()  # defaults documentados, o desde YAML
    adjusted_proj, detail = ensemble.adjust(
        poisson_proj=4.8,
        team_features=home_features,
    )
    # detail = {'cv': 0.42, 'alpha': 0.70, 'regression_pred': 5.1, ...}
"""

from __future__ import annotations

import math

from core.contracts import TeamFeatures

# Defaults idénticos a los documentados en el sistema MLB original
# (analysis/ensemble.py), preservados porque ya estaban bien calibrados
# y la arquitectura no exige cambiarlos — solo generalizar su aplicación.
_DEFAULT_ALPHA_BASE = 0.70
_DEFAULT_ALPHA_MIN = 0.50
_DEFAULT_CV_THRESHOLD_LOW = 0.60
_DEFAULT_CV_THRESHOLD_HIGH = 1.00
_DEFAULT_PROJ_MIN = 1.5

# Mínimo de scores para calcular una regresión lineal con algún
# sentido estadístico. Coincide deliberadamente con MIN_RECENT_SAMPLE
# de core/contracts/features.py — incluso si has_sufficient_sample
# diera True con un umbral de data_quality distinto, la regresión en
# sí necesita al menos este número de puntos para no ser ruido.
_MIN_SAMPLES_FOR_REGRESSION = 5


class EnsembleModel:
    """
    Ajusta una proyección Poisson combinándola con regresión lineal
    sobre la tendencia reciente del equipo, ponderado por un alpha
    adaptativo según el coeficiente de variación (CV) del equipo.

    Parámetros
    ----------
    alpha_base         -- Peso de Poisson cuando CV <= cv_threshold_low
                         (equipo consistente). Default 0.70.
    alpha_min           -- Peso de Poisson cuando CV >= cv_threshold_high
                         (equipo "todo o nada"). Default 0.50.
    cv_threshold_low     -- Umbral de CV bajo el cual alpha=alpha_base.
                         Default 0.60.
    cv_threshold_high     -- Umbral de CV sobre el cual alpha=alpha_min.
                         Default 1.00.
    proj_min              -- Clamp inferior de la proyección ajustada.
                         Default 1.5, razonable cruzado entre deportes
                         (ningún deporte real tiene proyección de
                         scoring por debajo de este valor).
    proj_max               -- Clamp superior de la proyección ajustada.
                         Default None (sin clamp superior). A diferencia
                         del sistema MLB original (PROJ_MAX=12.0 fijo
                         para carreras de béisbol), aquí no hay default
                         absoluto porque el rango de scoring varía
                         radicalmente entre deportes — un total NBA de
                         113 puntos rebasaría 12.0 por completo. El
                         sport plugin debe proveer proj_max desde
                         config/{sport}.yaml cuando el deporte lo
                         requiera (ej. NBA: proj_max=150).
    """

    def __init__(
        self,
        alpha_base: float = _DEFAULT_ALPHA_BASE,
        alpha_min: float = _DEFAULT_ALPHA_MIN,
        cv_threshold_low: float = _DEFAULT_CV_THRESHOLD_LOW,
        cv_threshold_high: float = _DEFAULT_CV_THRESHOLD_HIGH,
        proj_min: float = _DEFAULT_PROJ_MIN,
        proj_max: float | None = None,
    ) -> None:
        if cv_threshold_low >= cv_threshold_high:
            raise ValueError(
                f"cv_threshold_low={cv_threshold_low} debe ser < "
                f"cv_threshold_high={cv_threshold_high}."
            )
        if not 0.0 <= alpha_min <= alpha_base <= 1.0:
            raise ValueError(
                f"Se requiere 0 <= alpha_min({alpha_min}) <= "
                f"alpha_base({alpha_base}) <= 1."
            )

        self.alpha_base = alpha_base
        self.alpha_min = alpha_min
        self.cv_threshold_low = cv_threshold_low
        self.cv_threshold_high = cv_threshold_high
        self.proj_min = proj_min
        self.proj_max = proj_max

    # ── Componentes internos ───────────────────────────────────────────────────

    def _coefficient_of_variation(self, scores: list[float]) -> float:
        """
        CV = std(scores) / mean(scores).

        Retorna 0.0 si mean <= 0 o len(scores) < 2 — sin varianza
        calculable de forma significativa, se trata como "consistente"
        (CV bajo), lo que produce alpha=alpha_base (Poisson domina) en
        el caso degenerado, comportamiento seguro por defecto.
        """
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        if mean <= 0:
            return 0.0
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return math.sqrt(variance) / mean

    def _adaptive_alpha(self, cv: float) -> float:
        """
        Interpolación lineal de alpha entre cv_threshold_low y
        cv_threshold_high. Preserva exactamente la lógica del sistema
        MLB original (analysis/ensemble.py): evita un salto discreto
        de confianza en Poisson ante pequeñas variaciones de CV
        alrededor de los umbrales.

            CV <= cv_threshold_low  → alpha = alpha_base
            CV >= cv_threshold_high → alpha = alpha_min
            En medio                → interpolación lineal
        """
        if cv <= self.cv_threshold_low:
            return self.alpha_base
        if cv >= self.cv_threshold_high:
            return self.alpha_min

        t = (cv - self.cv_threshold_low) / (
            self.cv_threshold_high - self.cv_threshold_low
        )
        return self.alpha_base + t * (self.alpha_min - self.alpha_base)

    def _linear_regression_prediction(
        self,
        scores: list[float],
    ) -> float | None:
        """
        Ajusta una regresión lineal simple sobre scores (eje x = índice
        temporal 0..n-1) y predice el valor para el siguiente partido
        (posición n).

        Retorna None si:
        - hay menos de _MIN_SAMPLES_FOR_REGRESSION puntos
        - la varianza de x o y es 0 (línea degenerada, sin pendiente
          calculable)
        - R² < 0.05 (la tendencia es estadísticamente ruido, no vale
          la pena confiar en la predicción de regresión sobre Poisson)
        """
        if len(scores) < _MIN_SAMPLES_FOR_REGRESSION:
            return None

        n = len(scores)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(scores) / n

        ss_xx = sum((xi - x_mean) ** 2 for xi in x)
        ss_yy = sum((yi - y_mean) ** 2 for yi in scores)
        if ss_xx <= 0 or ss_yy <= 0:
            return None

        ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores))
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

        if r_squared < 0.05:
            return None

        prediction = intercept + slope * n  # predicción para el "siguiente" partido
        return float(prediction)

    # ── Punto de entrada principal ─────────────────────────────────────────────

    def adjust(
        self,
        poisson_proj: float,
        team_features: TeamFeatures,
    ) -> tuple[float, dict]:
        """
        Ajusta poisson_proj combinándolo con la tendencia de regresión
        lineal sobre team_features.recent_scores, ponderado por un
        alpha adaptativo según el coeficiente de variación del equipo.

        Si team_features.has_sufficient_sample es False, retorna
        (poisson_proj, detail) sin modificar — la proyección Poisson
        pura se usa tal cual cuando no hay datos suficientes para una
        regresión confiable. has_sufficient_sample ya encapsula la
        regla recent_n >= MIN_RECENT_SAMPLE and data_quality >= 0.5
        definida en core/contracts/features.py — no se reimplementa
        aquí un chequeo paralelo que pudiera divergir.

        Retorna
        -------
        (adjusted_projection, detail) donde detail es un dict con
        trazabilidad completa para logs y backtesting:
            cv, alpha, regression_pred, poisson_proj, adjusted,
            team_type ('consistent' | 'moderate' | 'all_or_nothing'
            | 'insufficient_sample' | 'flat_trend')
        """
        detail = {
            'poisson_proj': round(poisson_proj, 3),
            'cv': 0.0,
            'alpha': 1.0,
            'regression_pred': None,
            'adjusted': round(poisson_proj, 3),
            'team_type': 'insufficient_sample',
        }

        def _clamp(value: float) -> float:
            """Aplica proj_min y proj_max a cualquier proyección final."""
            result = max(self.proj_min, value)
            if self.proj_max is not None:
                result = min(result, self.proj_max)
            return round(result, 3)

        if not team_features.has_sufficient_sample:
            clamped = _clamp(poisson_proj)
            detail['adjusted'] = clamped
            return clamped, detail

        scores = team_features.recent_scores
        cv = self._coefficient_of_variation(scores)
        alpha = self._adaptive_alpha(cv)
        regression_pred = self._linear_regression_prediction(scores)

        detail['cv'] = round(cv, 4)
        detail['alpha'] = round(alpha, 4)
        detail['team_type'] = (
            'all_or_nothing' if cv >= self.cv_threshold_high else
            'consistent' if cv <= self.cv_threshold_low else
            'moderate'
        )

        if regression_pred is None:
            # Tendencia no significativa (R² bajo) — Poisson puro,
            # pero sí se aplica el clamp de proj_min/proj_max porque
            # la proyección Poisson de entrada puede ser válida
            # matemáticamente pero fuera del rango configurado para
            # este deporte (ej. proj_min=3.0 con poisson_proj=0.5).
            detail['team_type'] = 'flat_trend'
            clamped = _clamp(poisson_proj)
            detail['adjusted'] = clamped
            return clamped, detail

        adjusted = _clamp(alpha * poisson_proj + (1 - alpha) * regression_pred)

        detail['regression_pred'] = round(regression_pred, 3)
        detail['adjusted'] = adjusted

        return adjusted, detail
