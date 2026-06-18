from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """
    Perfil de riesgo universal.

    Define los mínimos requeridos para
    considerar una apuesta válida.

    Se utilizará posteriormente por:

    - RiskEvaluator
    - PickBuilder
    - ConfidenceScorer
    - BankrollManager
    """

    name: str

    min_edge: float

    min_ev: float

    min_probability: float

    max_stake_pct: float


CONSERVATIVE = RiskProfile(
    name="CONSERVATIVE",
    min_edge=0.03,
    min_ev=4.0,
    min_probability=0.55,
    max_stake_pct=1.00,
)

BALANCED = RiskProfile(
    name="BALANCED",
    min_edge=0.02,
    min_ev=3.0,
    min_probability=0.53,
    max_stake_pct=1.25,
)

AGGRESSIVE = RiskProfile(
    name="AGGRESSIVE",
    min_edge=0.01,
    min_ev=2.0,
    min_probability=0.51,
    max_stake_pct=2.00,
)


DEFAULT_PROFILE = BALANCED