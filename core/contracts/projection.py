"""
core/contracts/projection.py

Output del ProjectionModel específico de cada deporte.

Projection es el puente entre el conocimiento deportivo (TeamFeatures)
y el motor de probabilidades genérico (ProbabilityModel). El Core
consume Projection para simular probabilidades de mercado sin saber
qué fórmula deportiva la produjo — ERA/FIP/OPS para MLB, xG/xGA para
Soccer, ORtg/DRtg/pace para NBA.

Origen del diseño — continúa el principio anti-bug establecido en
TeamFeatures (ver CRITICAL_FINDINGS_VALIDATION.md F1/F2/F5):

    expected_total se deriva SIEMPRE de expected_home + expected_away
    en __post_init__, igual que recent_avg se deriva de recent_scores.
    Es estructuralmente imposible que estos tres campos diverjan.

    Las probabilidades de victoria (home/away/draw) se validan en
    construcción: deben sumar 1.0 con tolerancia de redondeo. Pequeñas
    desviaciones por masa de probabilidad residual (Poisson con
    max_score finito) se normalizan automáticamente; desviaciones
    mayores indican un bug real en el ProbabilityModel y abortan
    la construcción.

Uso típico:
    projection = Projection(
        event_id       = "a1b2c3d4-...",
        sport          = "mlb",
        expected_home  = 4.8,
        expected_away  = 3.9,
        expected_total = 0.0,   # se ignora — recalculado en __post_init__
        home_win_prob  = 0.54,
        away_win_prob  = 0.46,
        draw_prob      = 0.0,
        distribution   = "poisson",
        distribution_params = {"max_score": 20},
        confidence     = 0.85,
        model_version  = "mlb-v1.0",
        model_inputs   = {"era_rival_home": 3.5, "park_factor": 1.34},
        created_at     = "2026-06-08T17:00:00Z",
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# Distribuciones de probabilidad reconocidas por el DistributionFactory.
# Conjunto MUTABLE a propósito: registrar una distribución nueva es un
# paso deliberado de una línea (VALID_DISTRIBUTIONS.add("mi_distribucion"))
# antes de usarla en un ProjectionModel — no una barrera contra
# experimentación, sino una protección contra typos silenciosos que
# fallarían recién en el DistributionFactory, varios stages después.
VALID_DISTRIBUTIONS: set[str] = {
    "poisson",
    "bivariate_poisson",
    "neg_binomial",
    "normal",
    "skellam",
    "bradley_terry",
}

# Tolerancia para la suma de probabilidades de victoria.
# Desviaciones dentro de esta tolerancia se normalizan automáticamente
# (redondeo esperado de simulaciones con max_score finito). Desviaciones
# mayores indican un ProbabilityModel con un bug real y abortan la
# construcción del objeto.
PROBABILITY_SUM_TOLERANCE = 0.01


@dataclass
class Projection:
    """
    Salida del ProjectionModel de un sport plugin. Consumida por
    ProbabilityModel para simular probabilidades de mercado.

    Mutable a propósito, igual que TeamFeatures: permite que capas
    posteriores como EnsembleModel ajusten expected_home/expected_away
    tras la proyección base sin reconstruir el objeto completo.

    Campos
    ------
    Identidad:
        event_id        -- Igual a Event.event_id. Vincula la proyección
                          con su evento de origen.
        sport            -- Igual a Event.sport.

    Puntuación esperada:
        expected_home    -- Puntuación esperada del equipo local, en la
                          unidad del deporte (runs, goals, points).
        expected_away    -- Puntuación esperada del equipo visitante.
        expected_total   -- CALCULADO AUTOMÁTICAMENTE en __post_init__.
                          Siempre igual a expected_home + expected_away.
                          Cualquier valor pasado al constructor es
                          IGNORADO Y SOBRESCRITO.

    Probabilidades de victoria (pre-blending, antes de mezclar con mercado):
        home_win_prob    -- Probabilidad de victoria local, calculada
                          por el modelo deportivo puro.
        away_win_prob    -- Probabilidad de victoria visitante.
        draw_prob        -- Probabilidad de empate. 0.0 para deportes
                          sin empate (MLB, NBA, NFL, Tennis). Default
                          0.0 — nunca None, evita manejo de null disperso
                          en el ValueEngine.

                          VALIDADO en __post_init__: home + away + draw
                          debe sumar 1.0 dentro de PROBABILITY_SUM_TOLERANCE.
                          Desviaciones pequeñas (redondeo) se normalizan
                          automáticamente; desviaciones grandes lanzan
                          ValueError — señal de un ProbabilityModel roto.

    Distribución recomendada para simular:
        distribution        -- Identificador de la distribución de
                              probabilidad a usar. Debe estar en
                              VALID_DISTRIBUTIONS. Validado en
                              __post_init__.
        distribution_params -- Parámetros propios de esa distribución.
                              Poisson: {'max_score': 20}.
                              Normal: {'std_home': 11.5, 'std_away': 11.2}.
                              Opaco para el Core — cada ProbabilityModel
                              interpreta sus propios parámetros.

    Trazabilidad y calidad:
        confidence       -- Confianza del modelo en esta proyección,
                          0.0 a 1.0. A diferencia de TeamFeatures.data_quality,
                          NO se calcula automáticamente: es responsabilidad
                          declarativa de cada ProjectionModel, que conoce
                          su propia lógica de confianza (datos completos,
                          rival desconocido, muestra pequeña, etc.).
        model_version    -- Identificador de versión del modelo que generó
                          esta proyección. Usado por BacktestEngine para
                          comparar rendimiento entre versiones.
        model_inputs     -- Dict de trazabilidad libre, sin interpretación
                          por parte del Core. Registra qué datos
                          alimentaron el modelo para poder reconstruir
                          el razonamiento ante un resultado inesperado.
                          MLB: {'era_rival_home': 3.5, 'park_factor': 1.34}.
                          Soccer: {'xg_home': 1.8, 'xga_away': 1.1}.
        created_at       -- Timestamp ISO-8601 UTC de cuándo se generó
                          la proyección.

    Propiedades derivadas
    ----------------------
        is_draw_possible     -- True si draw_prob > 0.0.
        favored_side          -- 'home', 'away' o 'draw' — el resultado
                                con mayor probabilidad según el modelo.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    event_id: str
    sport:    str

    # ── Puntuación esperada ────────────────────────────────────────────────────
    # expected_total se recalcula SIEMPRE en __post_init__.
    expected_home:  float
    expected_away:  float
    expected_total: float = field(default=0.0)

    # ── Probabilidades de victoria ─────────────────────────────────────────────
    home_win_prob: float = 0.0
    away_win_prob: float = 0.0
    draw_prob:     float = 0.0

    # ── Distribución recomendada ───────────────────────────────────────────────
    distribution:        str  = "poisson"
    distribution_params: dict = field(default_factory=dict)

    # ── Trazabilidad y calidad ─────────────────────────────────────────────────
    confidence:    float = 1.0
    model_version: str   = "unknown"
    model_inputs:  dict  = field(default_factory=dict)
    created_at:    str   = ""

    # ── Validación e invariantes ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        """
        Aplica las invariantes estructurales del contrato.

        Se ejecuta automáticamente al construir el objeto. Si se mutan
        expected_home/expected_away o las probabilidades después de la
        construcción, llamar a recompute() para volver a aplicar estas
        invariantes.
        """
        self._validate_distribution()
        self._recompute_expected_total()
        self._validate_and_normalize_probabilities()

    def _validate_distribution(self) -> None:
        """
        distribution debe estar registrada en VALID_DISTRIBUTIONS.

        Falla en construcción, no tres stages después en el
        DistributionFactory con un KeyError críptico.
        """
        if self.distribution not in VALID_DISTRIBUTIONS:
            raise ValueError(
                f"distribution='{self.distribution}' no está registrada en "
                f"VALID_DISTRIBUTIONS. Valores válidos: "
                f"{sorted(VALID_DISTRIBUTIONS)}. Si es una distribución "
                f"nueva, regístrala explícitamente con "
                f"VALID_DISTRIBUTIONS.add('{self.distribution}') antes de "
                f"construir esta Projection."
            )

    def _recompute_expected_total(self) -> None:
        """
        expected_total se deriva EXCLUSIVAMENTE de expected_home + expected_away.

        Mismo principio que TeamFeatures._recompute_recent_form(): hace
        estructuralmente imposible que expected_total diverja de la suma
        real de sus componentes, sin depender de que cada ProjectionModel
        recuerde mantenerlos sincronizados.
        """
        self.expected_total = round(self.expected_home + self.expected_away, 3)

    def _validate_and_normalize_probabilities(self) -> None:
        """
        home_win_prob + away_win_prob + draw_prob debe sumar 1.0.

        Desviaciones pequeñas (masa de probabilidad residual por
        max_score finito en simulaciones discretas) se normalizan
        proporcionalmente. Desviaciones mayores a
        PROBABILITY_SUM_TOLERANCE indican un ProbabilityModel con un
        bug real — la construcción aborta con ValueError en lugar de
        propagar probabilidades inválidas al ValueEngine.
        """
        total = self.home_win_prob + self.away_win_prob + self.draw_prob

        if total <= 0:
            raise ValueError(
                f"Suma de probabilidades inválida ({total}) para "
                f"event_id='{self.event_id}'. home={self.home_win_prob}, "
                f"away={self.away_win_prob}, draw={self.draw_prob}."
            )

        # Se añade un epsilon de punto flotante al límite de tolerancia para
        # que el valor documentado de PROBABILITY_SUM_TOLERANCE sea inclusivo
        # en la práctica: sumas como 0.495+0.495 no producen exactamente 0.99
        # en binario (dan 0.9900000000000001), lo que sin este margen
        # rechazaría desviaciones que están exactamente en el límite aceptado.
        if not math.isclose(total, 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE + 1e-9):
            raise ValueError(
                f"Suma de probabilidades fuera de tolerancia para "
                f"event_id='{self.event_id}': home={self.home_win_prob} + "
                f"away={self.away_win_prob} + draw={self.draw_prob} = "
                f"{total}, esperado ≈1.0 (tolerancia "
                f"±{PROBABILITY_SUM_TOLERANCE}). Esto indica un bug en el "
                f"ProbabilityModel que generó esta proyección."
            )

        # Normalización proporcional dentro de tolerancia (corrige
        # redondeo de simulaciones discretas sin alterar las proporciones
        # relativas entre home/away/draw).
        if total != 1.0:
            self.home_win_prob = round(self.home_win_prob / total, 4)
            self.away_win_prob = round(self.away_win_prob / total, 4)
            self.draw_prob     = round(self.draw_prob / total, 4)

    def recompute(self) -> None:
        """
        Reaplica las invariantes del contrato.

        Llamar explícitamente si expected_home/expected_away o las
        probabilidades se mutan después de la construcción inicial
        (por ejemplo, EnsembleModel ajustando la proyección base).
        """
        self.__post_init__()

    # ── Propiedades derivadas ──────────────────────────────────────────────────

    @property
    def is_draw_possible(self) -> bool:
        """True si el deporte de esta proyección admite empate."""
        return self.draw_prob > 0.0

    @property
    def favored_side(self) -> str:
        """
        Resultado con mayor probabilidad según el modelo puro
        (pre-blending con el mercado).

        Retorna 'home', 'away' o 'draw'.
        """
        probs = {
            "home": self.home_win_prob,
            "away": self.away_win_prob,
            "draw": self.draw_prob,
        }
        return max(probs, key=probs.get) # type: ignore