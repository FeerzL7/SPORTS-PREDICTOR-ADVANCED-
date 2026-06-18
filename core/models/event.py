from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Event:
    """
    Evento deportivo universal.

    Puede representar:
    - MLB
    - NBA
    - NFL
    - Soccer
    - NHL
    - Golf
    - Tenis
    """

    sport: str

    home_team: str
    away_team: str

    start_time: datetime | None = None

    venue_name: str | None = None

    home_player: str | None = None
    away_player: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    def to_dict(self) -> dict:
        return {
            "sport": self.sport,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "start_time": self.start_time.isoformat()
            if self.start_time
            else None,
            "venue_name": self.venue_name,
            "home_player": self.home_player,
            "away_player": self.away_player,
            "metadata": self.metadata,
        }