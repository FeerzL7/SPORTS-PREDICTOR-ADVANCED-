"""
core/simulation/normal.py

Modelo de probabilidad basado en distribución Normal (Gaussiana).

Diseñado para deportes de alto scoring donde el total de puntos es la
suma de muchos eventos pequeños (touchdowns, canastas, drives) y
converge hacia una distribución Normal por el Teorema Central del Límite.

Deportes objetivo (ver SPORTS_PREDICTOR_ARCHITECTURE.md §8.1):
    NFL:  NormalModel(default_sigma=10.0) para todos los mercados
    NBA:  NormalModel(default_sigma=11.0) para totales
          SkellamModel para spread (tarea 2.5 del roadmap)
    Golf: NormalModel para strokes (Fase 4)

Por qué NOT Poisson para NFL/NBA
---------------------------------
Poisson asume Var = μ (equidispersión). Para NFL (μ≈24 puntos):
    Var_real ≈ 100  (σ≈10)  >>  μ=24
Para NBA (μ≈113 puntos):
    Var_real ≈ 121  (σ≈11)  >>  μ=113

Además, el scoring en estos deportes no son eventos raros e
independientes — los drives y posesiones están correlacionados entre
sí dentro del mismo partido (momentum, time-of-possession). La Normal
captura mejor la distribución empírica de resultados finales.

Implementación sin scipy
------------------------
Usa math.erfc() de stdlib para la CDF Normal estándar:

    Φ(x) = 0.5 * erfc(-x / sqrt(2))

Precisión: error < 1e-7 para |x| < 10, suficiente para cualquier
cálculo de probabilidad de mercado deportivo.

Parámetro σ: dos niveles
--------------------------
1. Projection.metrics['sigma_home'] / Projection.metrics['sigma_away']:
   sigma individual por equipo, provisto por el sport plugin cuando
   tiene datos suficientes (forma reciente, varianza histórica).
   Más preciso para equipos con scoring muy consistente o muy variable.

2. default_sigma del constructor: fallback cuando Projection.metrics
   no provee sigma. Configurable via config/base.yaml y sobreescrito
   por config/{sport}.yaml. El DistributionFactory (tarea 2.6) lo
   instancia leyendo el valor correcto para cada deporte.

   NormalModel(default_sigma=10.0)  ← NFL
   NormalModel(default_sigma=11.0)  ← NBA totales

Over + Under = 1.0 siempre
---------------------------
La distribución Normal es continua: P(X = línea_exacta) = 0. No existe
push matemáticamente, por lo que Over + Under = 1.0 en todos los casos,
incluyendo líneas enteras. Esto contrasta con PoissonModel (discreto)
donde sí existe masa de probabilidad en valores exactos.
"""

from __future__ import annotations

import math

from core.contracts import Projection
from core.simulation.protocols import SimulationResult


def _norm_cdf(x: float) -> float:
    """
    CDF de la distribución Normal estándar N(0,1) evaluada en x.

    Implementada via math.erfc() de stdlib Python:
        Φ(x) = 0.5 * erfc(-x / sqrt(2))

    Sin dependencia de scipy — precisión suficiente para probabilidades
    de mercado deportivo (error < 1e-7 para |x| < 10).
    """
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_sf(x: float) -> float:
    """
    Survival Function = 1 - CDF = P(X > x) para N(0,1).

    Equivalente a scipy.stats.norm.sf(x) pero sin dependencia externa.
    Numéricamente más estable que 1 - _norm_cdf(x) para x muy grandes
    porque evita cancelación catastrófica.
    """
    return 0.5 * math.erfc(x / math.sqrt(2))


class NormalModel:
    """
    Modelo Normal para deportes de scoring alto (NFL, NBA, Golf).

    Implementa ProbabilityModel via duck typing estructural — satisface
    el Protocol sin herencia explícita.

    La distribución de la diferencia de puntos y del total son también
    Normales (por la propiedad de cierre de la Normal bajo sumas lineales
    de variables independientes):

        Diff = X_home - X_away ~ N(μ_h - μ_a, √(σ_h² + σ_a²))
        Total = X_home + X_away ~ N(μ_h + μ_a, √(σ_h² + σ_a²))

    Nota: independencia asumida entre scoring home y away. Para deportes
    donde la correlación importa (NBA pace-of-play), el sport plugin
    puede pasar sigmas ajustados en Projection.metrics.

    Parámetros
    ----------
    default_sigma   -- Desviación estándar por defecto cuando
                      Projection.metrics no provee sigma individual
                      por equipo. Valores documentados en literatura:
                        NFL: ~10.0 (sd de puntos por equipo por partido)
                        NBA: ~11.0 (sd de puntos por equipo por partido)
                        Golf: ~3.5 (sd de strokes por ronda)
                      Configurable via config/{sport}.yaml bajo la key
                      simulation.normal.default_sigma.
    """

    def __init__(self, default_sigma: float = 10.0) -> None:
        if default_sigma <= 0:
            raise ValueError(
                f"default_sigma={default_sigma} debe ser > 0. "
                f"Valores típicos: NFL=10.0, NBA=11.0, Golf=3.5."
            )
        self.default_sigma = default_sigma

    # ── Resolución de sigma ────────────────────────────────────────────────────

    def _get_sigmas(
        self,
        projection: Projection,
    ) -> tuple[float, float]:
        """
        Resuelve σ_home y σ_away con dos niveles de prioridad:

        1. Projection.distribution_params['sigma_home'] / ['sigma_away']
           si existen y son positivos — provisto por el sport plugin
           con sigma calculado desde forma reciente del equipo.
        2. default_sigma del constructor como fallback — valor calibrado
           por deporte, configurable desde YAML.
        """
        params = projection.distribution_params or {}
        sigma_h = float(params.get('sigma_home', self.default_sigma))
        sigma_a = float(params.get('sigma_away', self.default_sigma))

        # Proteger contra valores inválidos provenientes de params externos
        if sigma_h <= 0:
            sigma_h = self.default_sigma
        if sigma_a <= 0:
            sigma_a = self.default_sigma

        return sigma_h, sigma_a

    # ── ProbabilityModel interface ─────────────────────────────────────────────

    def win_probabilities(
        self,
        projection: Projection,
    ) -> dict[str, float]:
        """
        P(home wins) y P(away wins) modelando la diferencia de scoring
        como variable Normal.

        Derivación:
            Diff = X_home - X_away ~ N(μ_diff, σ_diff)
            μ_diff = μ_h - μ_a
            σ_diff = √(σ_h² + σ_a²)

            P(home wins) = P(Diff > 0) = 1 - Φ(-μ_diff / σ_diff)
                                        = Φ(μ_diff / σ_diff)

        draw=0.0: NFL y NBA no tienen empate. Con distribución continua,
        P(empate exacto)=0 matemáticamente, consistente con la realidad
        deportiva (overtime existe precisamente para evitar empate).
        """
        sigma_h, sigma_a = self._get_sigmas(projection)
        mu_diff  = projection.expected_home - projection.expected_away
        sigma_diff = math.sqrt(sigma_h ** 2 + sigma_a ** 2)

        p_home = _norm_cdf(mu_diff / sigma_diff)
        p_away = 1.0 - p_home

        return {
            'home': round(p_home, 4),
            'away': round(p_away, 4),
            'draw': 0.0,
        }

    def spread_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(equipo cubre el spread) modelando la diferencia como Normal.

        Convención de line: es el handicap para el equipo seleccionado,
        igual que MarketOdds.line. Ejemplo NFL spread home -7.5:
            side='home', line=-7.5  → P(home - away > 7.5)
            side='away', line=-7.5  → P(away cubre +7.5) = P(away - home > -7.5)
                                     = P(Diff < 7.5) = CDF((7.5 - mu_diff) / σ)

        Propiedad garantizada: spread_home(line) + spread_away(line) = 1.0
        porque los eventos son complementarios (empate tiene prob=0 con
        distribución continua).

        Derivación para side='home', line=L (negativo para favorito):
            P(home - away > -L) = P(Diff > -L) = SF((-L - mu_diff) / σ)

        Derivación para side='away', line=L (mismo L negativo):
            away recibe +(-L) puntos de ventaja
            P(away - home > -(-L)) = P(-Diff > -L) = P(Diff < -L)
                                   = CDF((-L - mu_diff) / σ)
                                   = 1 - SF((-L - mu_diff) / σ)
        """
        sigma_h, sigma_a = self._get_sigmas(projection)
        mu_diff    = projection.expected_home - projection.expected_away
        sigma_diff = math.sqrt(sigma_h ** 2 + sigma_a ** 2)

        # threshold: margen que el equipo debe superar
        # Para home con line=-3.5: threshold = 3.5 (home debe ganar por más de 3.5)
        # Para away con line=-3.5: away recibe +3.5, threshold = -3.5 desde perspectiva Diff
        threshold = -line  # = 3.5 cuando line=-3.5

        z = (threshold - mu_diff) / sigma_diff

        if side == 'home':
            # P(Diff > threshold)
            prob = _norm_sf(z)
        else:
            # P(Diff < threshold) = complemento exacto
            prob = _norm_cdf(z)

        return round(float(prob), 4)

    def total_probability(
        self,
        projection: Projection,
        line: float,
        side: str,
    ) -> float:
        """
        P(total > line) o P(total < line) modelando la suma como Normal.

        Total = X_home + X_away ~ N(μ_total, σ_total)
        μ_total = μ_h + μ_a
        σ_total = √(σ_h² + σ_a²)  (independencia asumida)

        Over + Under = 1.0 siempre: con distribución continua no existe
        push (P(Total = line_exacto) = 0), a diferencia de PoissonModel
        donde líneas enteras producen masa de probabilidad en el push.
        """
        sigma_h, sigma_a = self._get_sigmas(projection)
        mu_total    = projection.expected_home + projection.expected_away
        sigma_total = math.sqrt(sigma_h ** 2 + sigma_a ** 2)

        z = (line - mu_total) / sigma_total

        if side == 'over':
            return round(float(_norm_sf(z)), 4)
        else:
            return round(float(_norm_cdf(z)), 4)

    def model_version(self) -> str:
        return f'normal-v1.0-sigma{self.default_sigma}'

    # ── Método de conveniencia ─────────────────────────────────────────────────

    def simulate(
        self,
        projection: Projection,
        spread_line: float | None = None,
        spread_side: str | None = None,
        total_line: float | None = None,
    ) -> SimulationResult:
        """
        Calcula todos los mercados en una sola llamada.
        σ se resuelve una vez y se reutiliza para los tres mercados.
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
            draw_prob=0.0,
            spread_home_prob=spread_home,
            spread_away_prob=spread_away,
            over_prob=over_prob,
            under_prob=under_prob,
            model_name=self.model_version(),
            projection=projection,
        )