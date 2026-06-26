"""
core/simulation/protocols.py

Punto de entrada del subsistema de simulación estadística.

Reexporta ProbabilityModel desde core.pipeline.stage para que los
modelos concretos de simulación (poisson.py, bivariate_poisson.py,
normal.py, etc.) importen desde core.simulation.protocols en vez de
desde core.pipeline.stage — manteniendo la separación de capas: los
modelos de simulación son consumidores internos del Core, no deben
conocer la capa de pipeline directamente.

Jerarquía de implementaciones (ver SPORTS_PREDICTOR_ARCHITECTURE.md
sección 8.1):

    ProbabilityModel (Protocol)
        ├── PoissonModel          — MLB, NHL (core/simulation/poisson.py)
        ├── BivariatePoissonModel — Soccer  (core/simulation/bivariate_poisson.py)
        ├── NegBinomialModel      — NHL overdispersión (futuro)
        ├── NormalModel           — NFL     (core/simulation/normal.py)
        ├── SkellamModel          — NBA spreads (core/simulation/skellam.py)
        └── BradleyTerryModel     — Tennis  (futuro)

    DistributionFactory resuelve qué implementación usar por deporte
    y mercado (core/simulation/factory.py).

Uso desde un modelo concreto:
    from core.simulation.protocols import ProbabilityModel
    # o directamente:
    from core.simulation.protocols import ProbabilityModel, SimulationResult
"""

from __future__ import annotations

from dataclasses import dataclass

# Reexportar ProbabilityModel desde donde vive canónicamente.
# Los modelos concretos usan este import, nunca core.pipeline.stage
# directamente — si en el futuro ProbabilityModel se mueve, solo
# cambia esta línea.
from core.pipeline.stage import ProbabilityModel

from core.contracts import Projection


@dataclass(frozen=True)
class SimulationResult:
    """
    Resultado intermedio de una simulación, antes del blending con el
    mercado. Contiene las probabilidades brutas del modelo estadístico
    para los tres tipos de mercado: moneyline, spread y total.

    Producido por ProbabilityModel y consumido por ValueEngine
    (core/value/engine.py, Bloque 3) que aplica no-vig, blending
    y calcula EV/edge sobre estas probabilidades.

    Campos
    ------
    home_win_prob   -- P(home gana) en moneyline. Excluye empate —
                      home + away = 1.0 para deportes sin draw.
    away_win_prob    -- P(away gana) en moneyline.
    draw_prob         -- P(empate). 0.0 para MLB, NBA, NFL, NHL.
                        > 0 para Soccer/1X2.
    spread_home_prob  -- P(home cubre el handicap publicado).
                        None si el modelo no calculó este mercado.
    spread_away_prob   -- P(away cubre el handicap publicado).
                        None si el modelo no calculó este mercado.
    over_prob          -- P(total > línea publicada).
                        None si el modelo no calculó este mercado.
    under_prob          -- P(total < línea publicada). Excluye push
                          cuando la línea es entera.
                          None si el modelo no calculó este mercado.
    model_name           -- Nombre del modelo que produjo este resultado.
                          Para trazabilidad en logs y backtesting.
                          Ejemplos: 'poisson', 'bivariate_poisson',
                          'normal', 'skellam'.
    projection            -- La Projection de entrada que originó esta
                          simulación. Permite al ValueEngine reconstruir
                          el contexto completo del cálculo sin estado
                          adicional.
    """

    home_win_prob:  float
    away_win_prob:  float
    draw_prob:      float

    spread_home_prob: float | None
    spread_away_prob: float | None
    over_prob:         float | None
    under_prob:         float | None

    model_name:  str
    projection:  Projection


__all__ = [
    "ProbabilityModel",
    "SimulationResult",
]