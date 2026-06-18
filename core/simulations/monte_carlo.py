import random

from core.models.event import Event
from core.models.prediction import Prediction

from core.simulations.base import (
    SimulationModel,
)


class MonteCarloSimulationModel(
    SimulationModel
):
    """
    Modelo Monte Carlo genérico.

    Simula miles de partidos utilizando
    las proyecciones esperadas de ambos equipos.

    Funciona para cualquier deporte donde exista:

        home_projection
        away_projection
    """

    DEFAULT_ITERATIONS = 10000

    def __init__(
        self,
        iterations: int = DEFAULT_ITERATIONS,
    ):
        self.iterations = iterations

    @property
    def name(self) -> str:
        return "monte_carlo"

    def predict(
        self,
        event: Event,
    ) -> Prediction:

        home_projection = float(
            event.metadata.get(
                "home_projection",
                0.0,
            )
        )

        away_projection = float(
            event.metadata.get(
                "away_projection",
                0.0,
            )
        )

        allow_draw = (
            event.sport.lower()
            == "soccer"
        )

        home_wins = 0
        away_wins = 0
        draws = 0

        for _ in range(self.iterations):

            home_score = random.gauss(
                home_projection,
                max(home_projection * 0.20, 1.0),
            )

            away_score = random.gauss(
                away_projection,
                max(away_projection * 0.20, 1.0),
            )

            if home_score > away_score:

                home_wins += 1

            elif away_score > home_score:

                away_wins += 1

            else:

                draws += 1

        home_probability = (
            home_wins
            / self.iterations
        )

        away_probability = (
            away_wins
            / self.iterations
        )

        draw_probability = (
            draws
            / self.iterations
        )

        if not allow_draw:

            decided = (
                home_probability
                + away_probability
            )

            if decided > 0:

                home_probability /= decided
                away_probability /= decided

            draw_probability = 0.0

        return Prediction(
            home_projection=round(
                home_projection,
                3,
            ),
            away_projection=round(
                away_projection,
                3,
            ),
            total_projection=round(
                home_projection
                + away_projection,
                3,
            ),
            probabilities={
                "home_win": round(
                    home_probability,
                    4,
                ),
                "away_win": round(
                    away_probability,
                    4,
                ),
                "draw": round(
                    draw_probability,
                    4,
                ),
            },
            metrics={
                "model": self.name,
                "iterations": self.iterations,
            },
        )