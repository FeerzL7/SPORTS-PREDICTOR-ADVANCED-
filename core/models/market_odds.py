# core/models/market_odds.py

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MarketOdds:
    """
    Representa una opción individual de un mercado de apuestas.

    Ejemplos:

    MLB Moneyline:
        Yankees @ 2.05
        Red Sox @ 1.80

    MLB Total:
        Over 8.5 @ 1.91
        Under 8.5 @ 1.91

    MLB Runline:
        Yankees -1.5 @ 2.10
        Red Sox +1.5 @ 1.75

    NBA Spread:
        Lakers -4.5 @ 1.91
        Celtics +4.5 @ 1.91

    Soccer:
        Arsenal @ 2.15
        Draw @ 3.40
        Liverpool @ 3.10
    """

    # Tipo de mercado:
    # ML, TOTAL, SPREAD, HANDICAP, BTTS, etc.
    market: str

    # Selección específica
    # Yankees, Over, Under, Arsenal, Draw...
    selection: str

    # Cuota decimal
    odds: float

    # Línea asociada al mercado si aplica
    # Ej:
    # TOTAL 8.5
    # SPREAD -1.5
    # HANDICAP +0.25
    line: float | None = None

    # Información extra opcional
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def implied_probability(self) -> float:
        """
        Probabilidad implícita de la cuota.

        Ejemplo:
            2.00 -> 0.50
            1.80 -> 0.5556
        """
        if self.odds <= 1:
            return 0.5

        return round(1 / self.odds, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "line": self.line,
            "metadata": self.metadata,
            "implied_probability": self.implied_probability,
        }

    def __str__(self) -> str:
        if self.line is None:
            return (
                f"{self.market} | "
                f"{self.selection} @ {self.odds}"
            )

        return (
            f"{self.market} | "
            f"{self.selection} "
            f"({self.line:+g}) @ {self.odds}"
        )