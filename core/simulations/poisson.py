from core.models.event import Event
from core.models.prediction import Prediction

from core.probability.simulation_engine import (
    SimulationEngine,
)

from core.simulations.base import (
    SimulationModel,
)


class PoissonSimulationModel(
    SimulationModel
):
    """
    Modelo Poisson genérico.

    Recibe las proyecciones ofensivas ya calculadas
    por el adapter del deporte y transforma esas
    proyecciones en probabilidades de mercado.

    Funciona para:

        MLB
        Soccer
        NHL

    y cualquier deporte donde los eventos puedan
    modelarse mediante Poisson.
    """

    @property
    def name(self) -> str:
        return "poisson"

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

        result = (
            SimulationEngine
            .win_probabilities(
                home_projection=home_projection,
                away_projection=away_projection,
                allow_draw=(
                    event.sport.lower()
                    == "soccer"
                ),
            )
        )

        return Prediction(
            home_projection=result.expected_home_score,
            away_projection=result.expected_away_score,
            total_projection=result.total_projection,
            probabilities={
                "home_win":
                    result.home_win_probability,

                "away_win":
                    result.away_win_probability,

                "draw":
                    result.draw_probability,
            },
            metrics={
                "model": self.name,
            },
        )