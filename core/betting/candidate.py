from dataclasses import dataclass


@dataclass(slots=True)
class CandidatePick:

    market: str

    selection: str

    probability: float

    market_probability: float

    edge: float

    odds: float

    expected_value: float

    stake_pct: float

    confidence: int = 0