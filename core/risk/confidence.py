from core.betting.candidate import CandidatePick


class ConfidenceScorer:
    """
    Genera una puntuación universal de confianza.

    Escala:

    50 - 59  = Muy baja
    60 - 69  = Baja
    70 - 79  = Media
    80 - 89  = Alta
    90 - 100 = Elite

    La puntuación se basa en:

    - Probabilidad del modelo
    - Edge
    - EV
    - Stake sugerido
    """

    MIN_CONFIDENCE = 50
    MAX_CONFIDENCE = 100

    def score(
        self,
        pick: CandidatePick,
    ) -> int:

        score = self.MIN_CONFIDENCE

        score += self._probability_score(
            pick.probability
        )

        score += self._edge_score(
            pick.edge
        )

        score += self._ev_score(
            pick.expected_value
        )

        score += self._stake_score(
            pick.stake_pct
        )

        score = max(
            self.MIN_CONFIDENCE,
            min(score, self.MAX_CONFIDENCE),
        )

        return int(round(score))

    @staticmethod
    def _probability_score(
        probability: float,
    ) -> int:

        if probability >= 0.70:
            return 18

        if probability >= 0.65:
            return 15

        if probability >= 0.60:
            return 12

        if probability >= 0.55:
            return 8

        return 0

    @staticmethod
    def _edge_score(
        edge: float,
    ) -> int:

        if edge >= 0.10:
            return 15

        if edge >= 0.07:
            return 12

        if edge >= 0.05:
            return 9

        if edge >= 0.03:
            return 6

        return 0

    @staticmethod
    def _ev_score(
        ev: float,
    ) -> int:

        if ev >= 20:
            return 12

        if ev >= 15:
            return 10

        if ev >= 10:
            return 8

        if ev >= 5:
            return 5

        return 0

    @staticmethod
    def _stake_score(
        stake_pct: float,
    ) -> int:

        if stake_pct >= 1.25:
            return 5

        if stake_pct >= 1.00:
            return 3

        return 0