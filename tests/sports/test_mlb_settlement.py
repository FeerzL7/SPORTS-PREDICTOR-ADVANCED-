"""
tests/sports/test_mlb_settlement.py

Liquidación de MLB desde el ledger (MLBSettlementProvider._settle_from_entry)
y el formato con el que register_pick() guarda la selección.

Dos fallos que estos tests fijan (2026-10):
    - register_pick() guardaba 'Over' sin la línea → todo total se
      liquidaba como void.
    - _selection_is_winner() ignoraba la selección → todo moneyline se
      liquidaba como ganado.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Any

import pytest

from core.tracking.roi_tracker import ledger_selection
from sports.mlb.settlement import MLBGameResult, MLBSettlementProvider

HOME, AWAY = "New York Yankees", "Baltimore Orioles"


def make_entry(market: str, selection: str, price: float = 1.91):
    from core.contracts.ledger import BetLedgerEntry
    fields: dict[str, Any] = dict(
        entry_id=f"823490_{market}_x", sport="mlb", league="MLB",
        date="2026-09-27", event=f"{AWAY} @ {HOME}", market=market,
        selection=selection, price=price, model_prob=0.55, ev=5.0,
        stake_pct=1, stake_amount=10.0, bankroll_before=1000.0,
        result="pending", model_version="mlb-v1.0",
        created_at="2026-09-27T14:00:00Z",
    )
    return BetLedgerEntry(**fields)


def game(home: int, away: int) -> MLBGameResult:
    return MLBGameResult(game_pk=823490, status="final", home_score=home,
                         away_score=away, innings=9, is_official=True,
                         home_team_id=147, away_team_id=110)


settle = MLBSettlementProvider._settle_from_entry


# ── Formato de la selección en el ledger ──────────────────────────────────────

class TestLedgerSelection:
    @pytest.mark.parametrize("market,selection,line,expected", [
        ("TOTAL",    "Over",          8.5, "Over 8.5"),
        ("TOTAL",    "Under",         9,   "Under 9.0"),
        ("SPREAD",   HOME,           -1.5, f"{HOME} -1.5"),
        ("SPREAD",   AWAY,            1.5, f"{AWAY} +1.5"),
        ("ML",       HOME,            None, HOME),
        ("TOTAL_F5", "Over",          4.5, "Over 4.5"),
    ])
    def test_line_is_embedded(self, market, selection, line, expected):
        assert ledger_selection(market, selection, line) == expected


# ── Moneyline ─────────────────────────────────────────────────────────────────

class TestMoneyline:
    def test_home_pick_wins_when_home_wins(self):
        assert settle(make_entry("ML", HOME), game(5, 3)) == "win"

    def test_away_pick_loses_when_home_wins(self):
        # Antes de la corrección esto devolvía "win": el ML siempre ganaba.
        assert settle(make_entry("ML", AWAY), game(5, 3)) == "lose"

    def test_away_pick_wins_when_away_wins(self):
        assert settle(make_entry("ML", AWAY), game(2, 6)) == "win"

    def test_selection_is_case_insensitive(self):
        assert settle(make_entry("ML", AWAY.upper()), game(2, 6)) == "win"

    def test_unknown_team_voids(self):
        assert settle(make_entry("ML", "Boston Red Sox"), game(5, 3)) == "void"

    def test_exactly_one_side_wins(self):
        results = {settle(make_entry("ML", t), game(4, 1)) for t in (HOME, AWAY)}
        assert results == {"win", "lose"}


# ── Runline ───────────────────────────────────────────────────────────────────

class TestRunline:
    @pytest.mark.parametrize("home,away,selection,expected", [
        (6, 3, f"{HOME} -1.5", "win"),    # favorito gana por 3
        (4, 3, f"{HOME} -1.5", "lose"),   # favorito gana por 1
        (4, 3, f"{AWAY} +1.5", "win"),    # underdog pierde por 1
        (6, 3, f"{AWAY} +1.5", "lose"),   # underdog pierde por 3
        (2, 5, f"{AWAY} -1.5", "win"),    # visitante favorito gana por 3
    ])
    def test_uses_embedded_line(self, home, away, selection, expected):
        assert settle(make_entry("SPREAD", selection), game(home, away)) == expected

    def test_legacy_entry_without_line_falls_back_to_price(self):
        # underdog (+1.5) si cotiza por encima de 2.0
        entry = make_entry("SPREAD", AWAY, price=2.30)
        assert settle(entry, game(4, 3)) == "win"


# ── Totales ───────────────────────────────────────────────────────────────────

class TestTotal:
    @pytest.mark.parametrize("selection,home,away,expected", [
        ("Over 8.5",  5, 4, "win"),
        ("Under 8.5", 5, 4, "lose"),
        ("Under 8.5", 3, 2, "win"),
        ("Over 9.0",  5, 4, "null"),
    ])
    def test_total_with_embedded_line(self, selection, home, away, expected):
        assert settle(make_entry("TOTAL", selection), game(home, away)) == expected

    def test_total_without_line_voids(self):
        # El formato viejo ('Over' sin línea) no puede liquidarse: void
        # es correcto, adivinar la línea no.
        assert settle(make_entry("TOTAL", "Over"), game(5, 4)) == "void"

    def test_registered_total_round_trips(self):
        sel = ledger_selection("TOTAL", "Over", 8.5)
        assert settle(make_entry("TOTAL", sel), game(6, 4)) == "win"
