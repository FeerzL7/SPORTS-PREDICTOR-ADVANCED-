from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Pick:
    """
    Apuesta oficial aprobada por el sistema.

    Flujo:

    Prediction
        ↓
    CandidatePick
        ↓
    RiskEvaluator
        ↓
    ConfidenceScorer
        ↓
    Pick

    Pick representa una apuesta que ya pasó todos
    los filtros del sistema y puede ser enviada a:

    - Reporting
    - Telegram
    - Tracking
    - Ledger
    - Bankroll
    """

    # ─────────────────────────────
    # Identificación
    # ─────────────────────────────

    sport: str

    event_id: str

    # ─────────────────────────────
    # Mercado
    # ─────────────────────────────

    market: str

    selection: str

    odds: float

    # ─────────────────────────────
    # Modelo
    # ─────────────────────────────

    probability: float

    market_probability: float

    edge: float

    expected_value: float

    # ─────────────────────────────
    # Gestión de riesgo
    # ─────────────────────────────

    stake_pct: float

    confidence: int

    risk_level: str = "UNKNOWN"

    # ─────────────────────────────
    # Trazabilidad
    # ─────────────────────────────

    source_model: str = ""

    rank: int = 0

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    # ─────────────────────────────
    # Resultado
    # ─────────────────────────────

    result: str = "pending"

    profit_loss: float = 0.0

    # ─────────────────────────────
    # Extras
    # ─────────────────────────────

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # =====================================================
    # Helpers
    # =====================================================

    def is_pending(self) -> bool:
        return self.result == "pending"

    def is_win(self) -> bool:
        return self.result == "win"

    def is_loss(self) -> bool:
        return self.result == "loss"

    def is_push(self) -> bool:
        return self.result == "push"

    # =====================================================
    # Settlement
    # =====================================================

    def settle(
        self,
        result: str,
        profit_loss: float,
    ) -> None:

        self.result = result

        self.profit_loss = round(
            profit_loss,
            2,
        )

    # =====================================================
    # Serialización
    # =====================================================

    def to_dict(self) -> dict:

        return {
            "sport": self.sport,
            "event_id": self.event_id,
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "probability": self.probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
            "expected_value": self.expected_value,
            "stake_pct": self.stake_pct,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "source_model": self.source_model,
            "rank": self.rank,
            "created_at": self.created_at.isoformat(),
            "result": self.result,
            "profit_loss": self.profit_loss,
            "metadata": self.metadata,
        }