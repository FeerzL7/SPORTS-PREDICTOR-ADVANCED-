from core.betting.candidate import CandidatePick
from core.betting.ev import ExpectedValue
from core.betting.kelly import KellyCriterion
from core.betting.odds import OddsConverter
from core.betting.probability_blending import ProbabilityBlender
from core.models.market_odds import MarketOdds


class PickBuilder:

    def __init__(
        self,
        model_weight: float = 0.65,
        probability_cap: float = 0.85,
        kelly_fraction: float = 0.18,
        kelly_max_stake: float = 1.25,
    ):
        self.model_weight = model_weight
        self.probability_cap = probability_cap

        self.kelly = KellyCriterion(
            fraction=kelly_fraction,
            max_stake_pct=kelly_max_stake,
        )

    def build_two_way_market(
        self,
        option_a: MarketOdds,
        option_b: MarketOdds,
        probability_a: float,
        probability_b: float,
    ) -> list[CandidatePick]:

        self._validate_probability(probability_a)
        self._validate_probability(probability_b)

        self._validate_odds(option_a.odds)
        self._validate_odds(option_b.odds)

        market_a, market_b = (
            OddsConverter.no_vig_probability(
                option_a.odds,
                option_b.odds,
            )
        )

        return [
            self._build_candidate(
                option=option_a,
                model_probability=probability_a,
                market_probability=market_a,
            ),
            self._build_candidate(
                option=option_b,
                model_probability=probability_b,
                market_probability=market_b,
            ),
        ]

    def _build_candidate(
        self,
        option: MarketOdds,
        model_probability: float,
        market_probability: float,
    ) -> CandidatePick:

        blended_probability = (
            ProbabilityBlender.blend(
                model_probability=model_probability,
                market_probability=market_probability,
                model_weight=self.model_weight,
                probability_cap=self.probability_cap,
            )
        )

        edge = ExpectedValue.edge(
            model_probability=model_probability,
            market_probability=market_probability,
        )

        ev = ExpectedValue.calculate(
            probability=blended_probability,
            odds=option.odds,
        )

        stake = self.kelly.calculate(
            probability=blended_probability,
            odds=option.odds,
        )

        confidence = self._confidence_score(
            probability=blended_probability,
            edge=edge,
            ev=ev,
        )

        return CandidatePick(
            market=option.market,
            selection=option.selection,
            probability=blended_probability,
            market_probability=market_probability,
            edge=edge,
            odds=option.odds,
            expected_value=ev,
            stake_pct=stake,
            confidence=confidence,
        )

    @staticmethod
    def _validate_probability(
        probability: float,
    ) -> None:

        if not 0 <= probability <= 1:
            raise ValueError(
                f"Probability inválida: {probability}"
            )

    @staticmethod
    def _validate_odds(
        odds: float,
    ) -> None:

        if odds <= 1:
            raise ValueError(
                f"Odds inválidas: {odds}"
            )

    @staticmethod
    def _confidence_score(
        probability: float,
        edge: float,
        ev: float,
    ) -> int:
        """
        Score 0-100.

        Componentes:

        Probabilidad:
            0-40

        Edge:
            0-35

        EV:
            0-25
        """

        probability_component = (
            probability * 40
        )

        edge_component = min(
            max(edge * 350, 0),
            35,
        )

        ev_component = min(
            max(ev * 0.5, 0),
            25,
        )

        score = (
            probability_component
            + edge_component
            + ev_component
        )

        return int(
            round(
                min(score, 100)
            )
        )