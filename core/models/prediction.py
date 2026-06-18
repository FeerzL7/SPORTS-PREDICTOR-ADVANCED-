from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Prediction:
    """
    Resultado de los modelos predictivos.

    MLB:
        carreras proyectadas

    Soccer:
        goles proyectados

    NBA:
        puntos proyectados

    NFL:
        puntos proyectados
    """

    home_projection: float = 0.0

    away_projection: float = 0.0

    total_projection: float = 0.0

    probabilities: dict[str, float] = field(default_factory=dict)

    metrics: dict[str, Any] = field(default_factory=dict)

    def probability(self, market: str) -> float:
        return float(self.probabilities.get(market, 0.0))

    def to_dict(self) -> dict:
        return {
            "home_projection": self.home_projection,
            "away_projection": self.away_projection,
            "total_projection": self.total_projection,
            "probabilities": self.probabilities,
            "metrics": self.metrics,
        }