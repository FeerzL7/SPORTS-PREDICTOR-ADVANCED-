from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TeamFeatures:
    """
    Estadísticas crudas de un equipo.

    MLB:
        offense
        defense
        bullpen
        pitching

    Soccer:
        attack
        defense
        xg

    NBA:
        offensive_rating
        defensive_rating
        pace

    NFL:
        offense
        defense
        qb
    """

    team_name: str

    offense: dict[str, Any] = field(default_factory=dict)

    defense: dict[str, Any] = field(default_factory=dict)

    special: dict[str, Any] = field(default_factory=dict)

    context: dict[str, Any] = field(default_factory=dict)

    def get(self, section: str, key: str, default=None):
        container = getattr(self, section, {})
        return container.get(key, default)

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "offense": self.offense,
            "defense": self.defense,
            "special": self.special,
            "context": self.context,
        }