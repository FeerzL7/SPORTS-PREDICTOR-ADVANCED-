# core/simulations/base.py

from abc import ABC, abstractmethod

from core.models.event import Event
from core.models.prediction import Prediction


class SimulationModel(ABC):
    """
    Contrato base para cualquier modelo de simulación.

    Todos los modelos deben transformar:

        Event
            ↓

        Prediction

    Ejemplos:

        PoissonSimulationModel
        EloSimulationModel
        MonteCarloSimulationModel
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nombre único del modelo.

        Ejemplos:
            poisson
            elo
            monte_carlo
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        event: Event,
    ) -> Prediction:
        """
        Genera una predicción para un evento.
        """
        raise NotImplementedError