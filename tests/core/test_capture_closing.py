"""
tests/core/test_capture_closing.py

Emparejamiento del cierre en scripts/capture_closing.py.

El fallo que fijan (2026-10): el índice de precios era global por
(mercado, selección, línea), así que en totales dos partidos con la
misma línea compartían clave y un pick podía quedar con el cierre de
otro partido.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import importlib.util
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'capture_closing.py')
_spec = importlib.util.spec_from_file_location("capture_closing", _SCRIPT)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

NOW = datetime(2026, 10, 18, 16, 0, tzinfo=timezone.utc)


def iso(minutes_from_now: float) -> str:
    return (NOW + timedelta(minutes=minutes_from_now)).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote(market, selection, line, price):
    return SimpleNamespace(market=market, selection=selection, line=line, price=price)


def raw_event(away, home, start_in, quotes):
    return SimpleNamespace(away_team=away, home_team=home,
                           commence_time=iso(start_in), quotes=quotes)


@pytest.fixture
def fake_normalizer(monkeypatch):
    """extract_best devuelve las cuotas que cada evento falso trae."""
    import core.odds.normalizer as normalizer
    monkeypatch.setattr(normalizer.OddsNormalizer, "extract_best",
                        lambda self, raw_event, markets, preferred_line=None: raw_event.quotes)


def make_entry(event, market, selection, price=1.91, entry_id=None):
    from core.contracts.ledger import BetLedgerEntry
    fields: dict[str, Any] = dict(
        entry_id=entry_id or f"{event}_{market}_{selection}".replace(" ", "_"),
        sport="nfl", league="NFL", date="2026-10-18", event=event,
        market=market, selection=selection, price=price, model_prob=0.55,
        ev=5.0, stake_pct=1, stake_amount=10.0, bankroll_before=1000.0,
        result="pending", model_version="nfl-v1.0",
        created_at="2026-10-18T12:00:00Z",
    )
    return BetLedgerEntry(**fields)


# ── Índice por partido ────────────────────────────────────────────────────────

class TestIndexByEvent:
    def test_same_total_line_in_two_games_does_not_mix(self, fake_normalizer):
        events = [
            raw_event("Las Vegas Raiders", "Kansas City Chiefs", 30,
                      [quote("TOTAL", "Over", 44.5, 1.80)]),
            raw_event("Denver Broncos", "Los Angeles Chargers", 30,
                      [quote("TOTAL", "Over", 44.5, 2.10)]),
        ]
        by_event = cc._index_prices_by_event(events, ["totals"], None)

        kc = make_entry("Las Vegas Raiders @ Kansas City Chiefs", "TOTAL", "Over 44.5")
        lac = make_entry("Denver Broncos @ Los Angeles Chargers", "TOTAL", "Over 44.5")

        assert cc._find_price(kc, by_event[cc._entry_event_key(kc)]["prices"]) == 1.80
        assert cc._find_price(lac, by_event[cc._entry_event_key(lac)]["prices"]) == 2.10

    def test_event_key_ignores_case_and_accents(self):
        e = make_entry("Atlético Madrid @ REAL MADRID", "1X2", "draw")
        assert cc._entry_event_key(e) == cc._event_key("atletico madrid", "Real Madrid")

    def test_entry_without_separator_has_no_key(self):
        assert cc._entry_event_key(make_entry("KC vs LV", "ML", "KC")) is None

    def test_events_without_prices_are_dropped(self, fake_normalizer):
        by_event = cc._index_prices_by_event([raw_event("A", "B", 30, [])], ["h2h"], None)
        assert by_event == {}


# ── Ventana de captura ────────────────────────────────────────────────────────

class TestWindow:
    @pytest.mark.parametrize("start_in,max_lead,expected", [
        (30,   60,   "capture"),
        (120,  60,   "deferred"),
        (-5,   60,   "started"),
        (600,  None, "capture"),
        (-5,   None, "started"),
    ])
    def test_status(self, start_in, max_lead, expected):
        assert cc._window_status(iso(start_in), NOW, max_lead) == expected

    def test_unreadable_start_is_captured(self):
        assert cc._window_status("no-es-fecha", NOW, 60) == "capture"


# ── Captura sobre un ledger real ──────────────────────────────────────────────

class TestCaptureLedger:
    def test_captures_only_games_in_window(self, tmp_path, fake_normalizer):
        from core.bankroll.tracker import BankrollTracker, CsvLedgerStore

        store = CsvLedgerStore(str(tmp_path / "nfl_roi_tracking.csv"))
        tracker = BankrollTracker(store=store)
        soon  = make_entry("Las Vegas Raiders @ Kansas City Chiefs", "TOTAL", "Under 44.5",
                           price=1.95, entry_id="soon")
        night = make_entry("Denver Broncos @ Los Angeles Chargers", "TOTAL", "Under 44.5",
                           price=1.95, entry_id="night")
        for e in (soon, night):
            tracker.register(e)

        by_event = cc._index_prices_by_event([
            raw_event("Las Vegas Raiders", "Kansas City Chiefs", 30,
                      [quote("TOTAL", "Under", 44.5, 1.85)]),
            raw_event("Denver Broncos", "Los Angeles Chargers", 400,
                      [quote("TOTAL", "Under", 44.5, 1.70)]),
        ], ["totals"], None)

        args = SimpleNamespace(dry_run=False, max_lead_minutes=75)
        cc._capture_ledger("operable", store, tracker, [soon, night], by_event, NOW, args)

        assert store.load_by_id("soon").clv is not None
        assert store.load_by_id("soon").clv > 0          # tomó 1.95, cerró 1.85
        assert store.load_by_id("night").clv is None     # queda para otra corrida

    def test_ledger_paths(self):
        args = SimpleNamespace(paper=False, with_paper=True)
        assert [label for label, _ in cc._ledger_paths("nfl", args)] == ["operable", "papel"]
        args = SimpleNamespace(paper=True, with_paper=False)
        assert cc._ledger_paths("nfl", args) == [
            ("papel", "output/ledger/nfl_paper_roi_tracking.csv")]
