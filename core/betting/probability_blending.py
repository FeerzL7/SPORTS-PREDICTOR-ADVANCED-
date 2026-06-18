class ProbabilityBlender:

    @staticmethod
    def blend(
        model_probability: float,
        market_probability: float,
        model_weight: float,
        probability_cap: float,
    ) -> float:

        probability = (
            model_probability
            * model_weight
            +
            market_probability
            * (1 - model_weight)
        )

        probability = max(
            0.02,
            probability,
        )

        probability = min(
            probability,
            probability_cap,
        )

        return round(
            probability,
            4,
        )