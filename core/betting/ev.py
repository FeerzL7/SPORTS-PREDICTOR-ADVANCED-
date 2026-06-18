class ExpectedValue:

    @staticmethod
    def calculate(
        probability: float,
        odds: float,
    ) -> float:

        return round(
            (probability * odds - 1)
            * 100,
            2,
        )

    @staticmethod
    def edge(
        model_probability: float,
        market_probability: float,
    ) -> float:

        return round(
            model_probability
            - market_probability,
            4,
        )