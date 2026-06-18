class OddsConverter:

    @staticmethod
    def implied_probability(decimal_odds: float) -> float:

        if decimal_odds <= 1:
            return 0.5

        return 1 / decimal_odds

    @staticmethod
    def no_vig_probability(
        odds_a: float,
        odds_b: float,
    ) -> tuple[float, float]:

        pa = OddsConverter.implied_probability(
            odds_a
        )

        pb = OddsConverter.implied_probability(
            odds_b
        )

        total = pa + pb

        if total <= 0:
            return 0.5, 0.5

        return (
            pa / total,
            pb / total,
        )