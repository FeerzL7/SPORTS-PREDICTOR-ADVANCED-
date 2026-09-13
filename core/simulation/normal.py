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

    def _get_sigma_margin(self, projection: Projection) -> float:
        """
        σ del MARGEN (home - away), con tres niveles de prioridad.

            1. distribution_params['sigma_margin'] — valor empírico
               directo. Es la vía preferente.
            2. √(σ_h² + σ_a²) desde las sigmas por equipo.
            3. √2 · default_sigma.

        Por qué el valor directo tiene prioridad
        -----------------------------------------
        Los niveles 2 y 3 asumen Cov(home, away) = 0, es decir, que las
        anotaciones de ambos equipos son independientes. En NFL esa
        suposición es falsa y el error es cuantificable.

        Partiendo de los valores empíricos de config/nfl.yaml
        (σ_margen = 13.5, σ_total = 10.0) y de las identidades

            Var(H+A) = Var(H) + Var(A) + 2·Cov
            Var(H-A) = Var(H) + Var(A) - 2·Cov

        se obtiene σ por equipo ≈ 8.40 y Cov ≈ -20.6, esto es, una
        correlación de -0.29. La causa es el game script: el equipo que
        va ganando corre el balón y consume reloj, lo que suprime la
        anotación de AMBOS equipos.

        El coste de ignorarlo no es cosmético. La fórmula de
        independencia produce √(2·70.6) = 11.88 tanto para el margen
        como para el total — es estructuralmente incapaz de
        distinguirlos, cuando los valores reales son 13.5 y 10.0.
        Eso son 12% de error en el margen y 19% en el total, en
        direcciones opuestas.

        Sobre el mercado de totales el efecto es grave: inflar σ_total
        un 19% acerca artificialmente las probabilidades de over/under
        al 50% y anula casi todo el EV que el modelo podría detectar.

        Los deportes donde la independencia sí es razonable (o donde no
        hay calibración empírica del margen) siguen funcionando por los
        niveles 2 y 3 sin cambio alguno.
        """
        params = projection.distribution_params or {}

        direct = params.get('sigma_margin')
        if direct is not None:
            try:
                value = float(direct)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass

        sigma_h, sigma_a = self._get_sigmas(projection)
        return math.sqrt(sigma_h ** 2 + sigma_a ** 2)

    def _get_sigma_total(self, projection: Projection) -> float:
        """
        σ del TOTAL (home + away), con la misma jerarquía de prioridad.

            1. distribution_params['sigma_total'] — valor empírico.
            2. √(σ_h² + σ_a²) asumiendo independencia.
            3. √2 · default_sigma.

        Ver la nota de _get_sigma_margin sobre por qué el valor directo
        importa: en NFL la correlación negativa entre anotaciones hace
        que σ_total real (10.0) sea sensiblemente menor que el 11.88
        que produce la fórmula de independencia.
        """
        params = projection.distribution_params or {}

        direct = params.get('sigma_total')
        if direct is not None:
            try:
                value = float(direct)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass

        sigma_h, sigma_a = self._get_sigmas(projection)
        return math.sqrt(sigma_h ** 2 + sigma_a ** 2)

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
        mu_diff    = projection.expected_home - projection.expected_away
        sigma_diff = self._get_sigma_margin(projection)

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

        Convención de line: es el handicap PROPIO del equipo
        seleccionado, igual que MarketOdds.line. Ejemplo NFL con el
        local favorito por 7.5:
            side='home', line=-7.5  → P(Diff >  7.5)
            side='away', line=+7.5  → P(Diff <  7.5)

        Propiedad garantizada: para las líneas OPUESTAS del mismo
        mercado (las que realmente cotiza el book),
            spread_probability(L,  'home') +
            spread_probability(-L, 'away') = 1.0
        porque los eventos son complementarios (el empate tiene
        probabilidad 0 con distribución continua).

        Es la misma convención que SkellamModel. Pasar la MISMA línea a
        ambos lados no suma 1.0 y no corresponde a ningún mercado real:
        si el local es -3.5, el visitante es +3.5.

        Derivación, con Diff = home - away:
            side='home', line=L → home cubre si Diff > -L
                                → SF((-L - mu_diff) / σ)
            side='away', line=L → away cubre si Diff <  L
                                → CDF((L - mu_diff) / σ)
        """
        mu_diff    = projection.expected_home - projection.expected_away
        sigma_diff = self._get_sigma_margin(projection)

        # `line` es el handicap PROPIO de la selección, tal como llega
        # en MarketOdds.line: negativo para el favorito, positivo para
        # el underdog. El pipeline (core/pipeline/runner.py) lo pasa sin
        # transformar, así que una selección visitante con +3.5 llega
        # aquí como line=+3.5.
        #
        # CORRECCIÓN (auditoría NFL, tarea 10.11): la versión anterior
        # calculaba `threshold = -line` para AMBOS lados. Eso es correcto
        # para el local pero invierte el signo para el visitante:
        #
        #     visitante +3.5, local favorito por 7, σ=13.5
        #       correcto : P(margen < +3.5) = 0.3977
        #       anterior : P(margen < -3.5) = 0.2184   ← 18 pp de error
        #
        # El efecto era sistemático: subestimaba a todo underdog
        # visitante, descartando picks con valor real y sesgando el
        # libro hacia favoritos locales. A cuota 1.91 el EV calculado
        # se desviaba 34 puntos porcentuales del real.
        #
        # SkellamModel (NBA) ya usaba la convención correcta; eran dos
        # modelos del mismo Protocol con semánticas opuestas.
        #
        # Umbral expresado sobre Diff = home - away:
        #     home con line=L  → cubre si Diff > -L
        #     away con line=L  → cubre si Diff <  L
        if side == 'home':
            threshold = -line
            z = (threshold - mu_diff) / sigma_diff
            prob = _norm_sf(z)
        else:
            threshold = line
            z = (threshold - mu_diff) / sigma_diff
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
        σ_total = distribution_params['sigma_total'] si el plugin lo
                  provee; en su defecto √(σ_h² + σ_a²), que asume
                  independencia entre las anotaciones de ambos equipos.

        La independencia NO se sostiene en NFL: el game script
        correlaciona negativamente las anotaciones (r ≈ -0.29) y la
        fórmula sobreestima σ_total un 19%. Ver _get_sigma_total().

        Over + Under = 1.0 siempre: con distribución continua no existe
        push (P(Total = line_exacto) = 0), a diferencia de PoissonModel
        donde líneas enteras producen masa de probabilidad en el push.
        """
        mu_total    = projection.expected_home + projection.expected_away
        sigma_total = self._get_sigma_total(projection)

        z = (line - mu_total) / sigma_total

        if side == 'over':
            return round(float(_norm_sf(z)), 4)
        else:
            return round(float(_norm_cdf(z)), 4)

    def model_version(self) -> str:
        return f'normal-v1.2-sigma{self.default_sigma}'

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

        # `spread_line` es la línea del LOCAL. La del visitante es su
        # negación: si el local es -3.5, el visitante es +3.5. Pasar la
        # misma línea a ambos lados calcularía dos veces el mismo lado
        # del mercado y la suma no daría 1.0.
        #
        # (El simulate() de SkellamModel arrastra este mismo defecto;
        #  queda pendiente de corregir en su propia tarea, ya que NBA
        #  no está aún en producción.)
        spread_home = (
            self.spread_probability(projection, spread_line, 'home')
            if spread_line is not None else None
        )
        spread_away = (
            self.spread_probability(projection, -spread_line, 'away')
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