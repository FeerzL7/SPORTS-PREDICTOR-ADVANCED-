from dataclasses import dataclass

from core.probability.poisson import PoissonDistribution


@dataclass(slots=True)
class SimulationResult:

    home_win_probability: float

    away_win_probability: float

    draw_probability: float

    expected_home_score: float

    expected_away_score: float

    total_projection: float


class SimulationEngine:

    @staticmethod
    def win_probabilities(
        home_projection: float,
        away_projection: float,
        max_score: int = 20,
        allow_draw: bool = False,
    ) -> SimulationResult:

        home_win = 0.0
        away_win = 0.0
        draw = 0.0

        for h in range(max_score):

            ph = PoissonDistribution.pmf(
                h,
                home_projection
            )

            for a in range(max_score):

                pa = PoissonDistribution.pmf(
                    a,
                    away_projection
                )

                p = ph * pa

                if h > a:
                    home_win += p

                elif a > h:
                    away_win += p

                else:
                    draw += p

        if not allow_draw:

            decided = home_win + away_win

            if decided > 0:
                home_win /= decided
                away_win /= decided

            draw = 0.0

        return SimulationResult(
            home_win_probability=round(home_win, 4),
            away_win_probability=round(away_win, 4),
            draw_probability=round(draw, 4),
            expected_home_score=round(home_projection, 3),
            expected_away_score=round(away_projection, 3),
            total_projection=round(
                home_projection + away_projection,
                3,
            ),
        )

    @staticmethod
    def handicap_probability(
        team_projection: float,
        opponent_projection: float,
        handicap: float,
        max_score: int = 20,
    ) -> float:

        probability = 0.0

        for team_score in range(max_score):

            pt = PoissonDistribution.pmf(
                team_score,
                team_projection,
            )

            for opponent_score in range(max_score):

                po = PoissonDistribution.pmf(
                    opponent_score,
                    opponent_projection,
                )

                if team_score + handicap > opponent_score:

                    probability += pt * po

        return round(probability, 4)

    @staticmethod
    def over_probability(
        total_line: float,
        home_projection: float,
        away_projection: float,
        max_score: int = 20,
    ) -> float:

        probability = 0.0

        for h in range(max_score):

            ph = PoissonDistribution.pmf(
                h,
                home_projection,
            )

            for a in range(max_score):

                pa = PoissonDistribution.pmf(
                    a,
                    away_projection,
                )

                if h + a > total_line:

                    probability += ph * pa

        return round(probability, 4)