class KellyCriterion:

    def __init__(
        self,
        fraction: float = 0.18,
        max_stake_pct: float = 1.25,
    ):
        self.fraction = fraction
        self.max_stake_pct = max_stake_pct

    def calculate(
        self,
        probability: float,
        odds: float,
    ) -> float:

        if probability <= 0:
            return 0.0

        if odds <= 1:
            return 0.0

        b = odds - 1

        kelly_full = (
            probability * (b + 1)
            - 1
        ) / b

        if kelly_full <= 0:
            return 0.0

        return round(
            min(
                kelly_full
                * self.fraction
                * 100,
                self.max_stake_pct,
            ),
            2,
        )