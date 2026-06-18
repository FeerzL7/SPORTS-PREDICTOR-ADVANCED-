from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class LedgerEntry:
    """
    Registro permanente del bankroll.
    """

    date: str

    game: str

    market: str

    selection: str

    odds: float

    probability: float

    expected_value: float

    stake_pct: int

    stake_amount: float

    bankroll_before: float

    bankroll_after: float = 0.0

    profit_amount: float = 0.0

    result: str = "pending"

    created_at: datetime | None = None

    settled_at: datetime | None = None

    def roi(self) -> float:
        if self.stake_amount <= 0:
            return 0.0

        return round(
            (self.profit_amount / self.stake_amount) * 100,
            2
        )

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "game": self.game,
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "probability": self.probability,
            "expected_value": self.expected_value,
            "stake_pct": self.stake_pct,
            "stake_amount": self.stake_amount,
            "bankroll_before": self.bankroll_before,
            "bankroll_after": self.bankroll_after,
            "profit_amount": self.profit_amount,
            "result": self.result,
            "created_at": self.created_at,
            "settled_at": self.settled_at,
        }