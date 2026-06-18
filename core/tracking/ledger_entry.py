from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class LedgerEntry:
    """
    Registro histórico de una apuesta.

    Una vez almacenado en el ledger,
    sirve para:

    - ROI
    - Yield
    - Bankroll tracking
    - Performance tracking
    - Reporting
    - Backtesting
    """

    event_id: str

    sport: str

    market: str

    selection: str

    odds: float

    probability: float

    market_probability: float

    edge: float

    expected_value: float

    stake_pct: float

    confidence: int

    result: str = "pending"

    profit_loss: float = 0.0

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    settled_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_pending(self) -> bool:
        return self.result == "pending"

    @property
    def is_win(self) -> bool:
        return self.result == "win"

    @property
    def is_loss(self) -> bool:
        return self.result == "loss"

    @property
    def is_push(self) -> bool:
        return self.result == "push"

    def settle(
        self,
        result: str,
        profit_loss: float,
    ) -> None:

        self.result = result

        self.profit_loss = round(
            profit_loss,
            2,
        )

        self.settled_at = datetime.utcnow()

    def to_dict(self) -> dict:

        return {
            "event_id": self.event_id,
            "sport": self.sport,
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "probability": self.probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
            "expected_value": self.expected_value,
            "stake_pct": self.stake_pct,
            "confidence": self.confidence,
            "result": self.result,
            "profit_loss": self.profit_loss,
            "created_at": self.created_at.isoformat(),
            "settled_at": (
                self.settled_at.isoformat()
                if self.settled_at
                else None
            ),
            "metadata": self.metadata,
        }