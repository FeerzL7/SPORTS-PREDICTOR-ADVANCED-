from dataclasses import dataclass

from core.betting.candidate import CandidatePick
from core.risk.risk_profile import (
    RiskProfile,
    DEFAULT_PROFILE,
)


@dataclass(slots=True)
class RiskAssessment:
    """
    Resultado de evaluación de riesgo.
    """

    allowed: bool

    risk_level: str

    reason: str


class RiskEvaluator:
    """
    Evalúa si un pick es apostable.

    Revisa:

    - Edge
    - EV
    - Probabilidad
    - Stake

    y asigna:

    LOW
    MEDIUM
    HIGH
    REJECTED
    """

    def __init__(
        self,
        profile: RiskProfile = DEFAULT_PROFILE,
    ):
        self.profile = profile

    def evaluate(
        self,
        pick: CandidatePick,
    ) -> RiskAssessment:

        # -------------------------
        # Filtros duros
        # -------------------------

        if pick.odds <= 1:
            return RiskAssessment(
                allowed=False,
                risk_level="REJECTED",
                reason="Invalid odds",
            )

        if pick.probability <= 0:
            return RiskAssessment(
                allowed=False,
                risk_level="REJECTED",
                reason="Invalid probability",
            )

        if pick.expected_value < self.profile.min_ev:
            return RiskAssessment(
                allowed=False,
                risk_level="REJECTED",
                reason="EV below threshold",
            )

        if pick.edge < self.profile.min_edge:
            return RiskAssessment(
                allowed=False,
                risk_level="REJECTED",
                reason="Edge below threshold",
            )

        if pick.probability < self.profile.min_probability:
            return RiskAssessment(
                allowed=False,
                risk_level="REJECTED",
                reason="Probability below threshold",
            )

        # -------------------------
        # Clasificación riesgo
        # -------------------------

        risk_level = self._classify_risk(pick)

        return RiskAssessment(
            allowed=True,
            risk_level=risk_level,
            reason="Approved",
        )

    def _classify_risk(
        self,
        pick: CandidatePick,
    ) -> str:

        score = 0

        if pick.edge >= 0.05:
            score += 1

        if pick.expected_value >= 8:
            score += 1

        if pick.probability >= 0.60:
            score += 1

        if pick.stake_pct >= 1:
            score += 1

        if score >= 4:
            return "LOW"

        if score >= 2:
            return "MEDIUM"

        return "HIGH"