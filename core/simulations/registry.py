from core.simulations.base import (
    SimulationModel,
)


class SimulationRegistry:
    """
    Registro central de modelos predictivos.

    Permite obtener modelos por nombre:

        registry.get("poisson")

        registry.get("elo")

        registry.get("monte_carlo")
    """

    def __init__(self):
        self._models: dict[
            str,
            SimulationModel,
        ] = {}

    def register(
        self,
        model: SimulationModel,
    ) -> None:

        self._models[
            model.name.lower()
        ] = model

    def get(
        self,
        model_name: str,
    ) -> SimulationModel:

        key = model_name.lower()

        if key not in self._models:

            available = ", ".join(
                sorted(self._models.keys())
            )

            raise ValueError(
                f"Simulation model '{model_name}' "
                f"not registered. "
                f"Available: [{available}]"
            )

        return self._models[key]

    def exists(
        self,
        model_name: str,
    ) -> bool:

        return (
            model_name.lower()
            in self._models
        )

    def available_models(
        self,
    ) -> list[str]:

        return sorted(
            self._models.keys()
        )