from core.models.event import Event
from core.models.prediction import Prediction

from core.simulations.base import (
    SimulationModel,
)


class EloSimulationModel(
    SimulationModel
):
    """
    Modelo basado en ratings ELO.

    Utiliza:

        home_elo
        away_elo

    para estimar probabilidades de victoria.

    Funciona especialmente bien para:

        NFL
        NBA
        NHL
        Soccer

    cuando existe un sistema ELO actualizado.
    """

    HOME_ADVANTAGE = 65

    @property
    def name(self) -> str:
        return "elo"

    def predict(
        self,
        event: Event,
    ) -> Prediction:

        home_elo = float(
            event.metadata.get(
                "home_elo",
                1500,
            )
        )

        away_elo = float(
            event.metadata.get(
                "away_elo",
                1500,
            )
        )

        home_elo += self.HOME_ADVANTAGE

        home_probability = (
            1.0
            /
            (
                1.0
                +
                10
                ** (
                    (away_elo - home_elo)
                    / 400
                )
            )
        )

        away_probability = (
            1.0
            - home_probability
        )

        return Prediction(
            probabilities={
                "home_win": round(
                    home_probability,
                    4,
                ),
                "away_win": round(
                    away_probability,
                    4,
                ),
            },
            metrics={
                "model": self.name,
                "home_elo": home_elo,
                "away_elo": away_elo,
            },
        )