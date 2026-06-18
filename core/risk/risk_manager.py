from dataclasses import dataclass

from core.betting.candidate import CandidatePick


@dataclass(slots=True)
class RiskDecision:

    approved: bool

    reasons: list[str]


class RiskManager:

    def __init__(
        self,
        min_ev: float = 3.0,
        min_edge: float = 0.02,
        min_confidence: int = 60,
        min_stake_pct: float = 0.10,
    ):
        self.min_ev = min_ev
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.min_stake_pct = min_stake_pct

    def evaluate(
        self,
        pick: CandidatePick,
    ) -> RiskDecision:

        reasons: list[str] = []

        if pick.expected_value < self.min_ev:
            reasons.append(
                f"EV<{self.min_ev}"
            )

        if pick.edge < self.min_edge:
            reasons.append(
                f"EDGE<{self.min_edge:.2%}"
            )

        if pick.confidence < self.min_confidence:
            reasons.append(
                f"CONF<{self.min_confidence}"
            )

        if pick.stake_pct < self.min_stake_pct:
            reasons.append(
                f"STAKE<{self.min_stake_pct}"
            )

        return RiskDecision(
            approved=len(reasons) == 0,
            reasons=reasons,
        )