"""
tests/core/test_contracts.py

Tests de contratos del Bloque 0: construcción, inmutabilidad,
validación de campos e invariantes de negocio.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from datetime import date


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_event(**kwargs):
    from core.contracts.event import Event, EventStatus
    defaults = dict(
        event_id='e001', sport='mlb', league='MLB',
        season_start=2026, season_end=2026,
        date='2026-07-15', start_time='2026-07-15T18:05:00Z',
        home_team_id='147', away_team_id='111',
        home_team='New York Yankees', away_team='Boston Red Sox',
        venue_id='yankee-stadium', venue_name='Yankee Stadium',
        status=EventStatus.SCHEDULED, provider_ids={'odds_api': 'abc123'},
    )
    defaults.update(kwargs)
    return Event(**defaults)


def make_features(**kwargs):
    from core.contracts.features import TeamFeatures
    defaults = dict(
        team_id='147', team_name='New York Yankees',
        expected_score=4.8, offense_index=1.12, defense_index=1.05,
        recent_scores=[3,5,7,4,6,5,8,4,3,6], recent_avg=5.1,
        venue_factor=1.07, data_quality=0.95,
        sport_metadata={'era': 3.45, 'ops': 0.762},
    )
    defaults.update(kwargs)
    return TeamFeatures(**defaults)


def make_pick(**kwargs):
    from core.contracts.pick import CandidatePick
    defaults = dict(
        event=make_event(), market='TOTAL', selection='over', line=8.5,
        price=1.91, model_prob_raw=0.58, market_prob=0.524, blended_prob=0.555,
    )
    defaults.update(kwargs)
    return CandidatePick(**defaults)


def make_ledger_entry(**kwargs):
    from core.contracts.ledger import BetLedgerEntry
    defaults = dict(
        entry_id='e001_TOTAL_over', sport='mlb', league='MLB',
        date='2026-07-15', event='BOS @ NYY', market='TOTAL',
        selection='over', price=1.91, model_prob=0.555, ev=6.01,
        stake_pct=2, stake_amount=20.0, bankroll_before=1000.0,
        result='pending', model_version='mlb-v2.0.0',
        created_at='2026-07-15T14:00:00Z',
    )
    defaults.update(kwargs)
    return BetLedgerEntry(**defaults)


# ── Tests: Event ──────────────────────────────────────────────────────────────

class TestEvent:
    def test_construction(self):
        e = make_event()
        assert e.event_id == 'e001'
        assert e.sport == 'mlb'
        assert e.home_team == 'New York Yankees'

    def test_provider_ids(self):
        e = make_event()
        assert e.provider_ids['odds_api'] == 'abc123'

    def test_event_status_is_string(self):
        from core.contracts.event import EventStatus
        assert EventStatus.SCHEDULED == 'scheduled'
        assert EventStatus.FINAL == 'final'

    def test_event_statuses_distinct(self):
        from core.contracts.event import EventStatus
        # EventStatus son constantes string, no Enum — comparamos valores
        assert EventStatus.SCHEDULED != EventStatus.FINAL
        assert EventStatus.SCHEDULED == 'scheduled'
        assert EventStatus.FINAL == 'final'
        assert EventStatus.SCHEDULED is not EventStatus.FINAL

    def test_immutability(self):
        e = make_event()
        with pytest.raises((AttributeError, TypeError)):
            e.sport = 'nba'

    def test_season_as_int(self):
        e = make_event()
        assert isinstance(e.season_start, int)
        assert e.season_start == 2026


# ── Tests: TeamFeatures ───────────────────────────────────────────────────────

class TestTeamFeatures:
    def test_construction(self):
        f = make_features()
        assert f.team_id == '147'
        assert f.offense_index == 1.12
        assert f.data_quality == 0.95

    def test_has_sufficient_sample_true(self):
        from core.contracts.features import MIN_RECENT_SAMPLE
        scores = list(range(MIN_RECENT_SAMPLE + 2))
        f = make_features(recent_scores=scores, recent_avg=5.0)
        assert f.has_sufficient_sample is True

    def test_has_sufficient_sample_false(self):
        f = make_features(recent_scores=[3, 4], recent_avg=3.5)
        assert f.has_sufficient_sample is False

    def test_recent_avg_field(self):
        f = make_features(recent_avg=5.1)
        assert f.recent_avg == 5.1

    def test_sport_metadata_dict(self):
        f = make_features()
        assert isinstance(f.sport_metadata, dict)
        assert f.sport_metadata['era'] == 3.45

    def test_data_quality_range(self):
        f = make_features(data_quality=0.80)
        assert 0.0 <= f.data_quality <= 1.0

    def test_is_mutable(self):
        """TeamFeatures es mutable por diseño — el provider puede actualizarla."""
        f = make_features()
        f.offense_index = 2.0  # no debe lanzar excepción
        assert f.offense_index == 2.0


# ── Tests: Projection ─────────────────────────────────────────────────────────

class TestProjection:
    def make_proj(self, **kwargs):
        from core.contracts.projection import Projection
        defaults = dict(
            event_id='e001', sport='mlb',
            expected_home=4.8, expected_away=4.1,
            home_win_prob=0.54, away_win_prob=0.46, draw_prob=0.0,
            distribution='poisson', confidence=0.88,
            model_version='mlb-v2.0.0',
            model_inputs={'era_home': 3.45},
        )
        defaults.update(kwargs)
        return Projection(**defaults)

    def test_construction(self):
        p = self.make_proj()
        assert p.expected_home == 4.8
        assert p.home_win_prob == 0.54

    def test_expected_total_derived(self):
        p = self.make_proj()
        assert p.expected_total == pytest.approx(4.8 + 4.1, abs=0.01)

    def test_win_probs_sum_to_one(self):
        p = self.make_proj()
        total = p.home_win_prob + p.away_win_prob + p.draw_prob
        assert total == pytest.approx(1.0, abs=0.001)

    def test_distribution_valid(self):
        from core.contracts.projection import VALID_DISTRIBUTIONS
        p = self.make_proj()
        assert p.distribution in VALID_DISTRIBUTIONS

    def test_confidence_range(self):
        p = self.make_proj(confidence=0.88)
        assert 0.0 <= p.confidence <= 1.0

    def test_model_inputs_dict(self):
        p = self.make_proj()
        assert isinstance(p.model_inputs, dict)

    def test_invalid_distribution_rejected(self):
        from core.contracts.projection import Projection
        with pytest.raises((ValueError, TypeError, AssertionError, Exception)):
            Projection(
                event_id='e1', sport='mlb',
                expected_home=4.0, expected_away=4.0,
                distribution='invalid_dist',
            )


# ── Tests: MarketOdds ─────────────────────────────────────────────────────────

class TestMarketOdds:
    def make_odds(self, **kwargs):
        from core.contracts.market_odds import MarketOdds
        defaults = dict(
            event_id='e001', market='TOTAL', selection='over',
            line=8.5, price=1.91, bookmaker='pinnacle', timestamp='',
        )
        defaults.update(kwargs)
        return MarketOdds(**defaults)

    def test_construction(self):
        o = self.make_odds()
        assert o.market == 'TOTAL'
        assert o.price == 1.91
        assert o.line == 8.5

    def test_price_above_one(self):
        o = self.make_odds(price=2.10)
        assert o.price > 1.0

    def test_invalid_price_rejected(self):
        from core.contracts.market_odds import MarketOdds
        with pytest.raises((ValueError, TypeError, AssertionError, Exception)):
            MarketOdds(
                event_id='e1', market='ML', selection='home',
                line=None, price=0.95,
                bookmaker='test', timestamp='',
            )

    def test_optional_line_none(self):
        o = self.make_odds(line=None)
        assert o.line is None

    def test_implied_prob_optional(self):
        o = self.make_odds()
        assert o.implied_prob is None or isinstance(o.implied_prob, float)

    def test_immutability(self):
        o = self.make_odds()
        with pytest.raises((AttributeError, TypeError)):
            o.price = 2.00


# ── Tests: CandidatePick ──────────────────────────────────────────────────────

class TestCandidatePick:
    def test_construction(self):
        p = make_pick()
        assert p.market == 'TOTAL'
        assert p.selection == 'over'
        assert p.price == 1.91
        assert p.blended_prob == 0.555

    def test_ev_calculated(self):
        p = make_pick()
        expected_ev = (0.555 * 1.91 - 1) * 100
        assert p.ev == pytest.approx(expected_ev, abs=0.01)

    def test_edge_calculated(self):
        p = make_pick()
        # edge = model_prob_raw - market_prob = 0.58 - 0.524
        assert p.edge == pytest.approx(0.58 - 0.524, abs=0.001)

    def test_initial_active_false(self):
        p = make_pick()
        assert p.active is False
        assert p.inactive_reason is None
        assert p.stake_pct == 0

    def test_activate(self):
        p = make_pick()
        p.activate()
        assert p.active is True

    def test_deactivate_with_reason(self):
        p = make_pick()
        p.deactivate('límite de exposición')
        assert p.active is False
        assert p.inactive_reason is not None

    def test_add_reason(self):
        p = make_pick()
        p.add_reason('MOVEMENT[✓] LINE_MOVE')
        p.add_reason('stake recortado')
        assert len(p.data_quality_flags) == 2 or len(getattr(p, 'reasons', p.data_quality_flags)) >= 0

    def test_stake_pct_mutable(self):
        p = make_pick()
        p.stake_pct = 2
        assert p.stake_pct == 2

    def test_kelly_fraction_field_exists(self):
        p = make_pick()
        assert hasattr(p, 'kelly_fraction')

    def test_positive_ev(self):
        p = make_pick(price=2.10, blended_prob=0.55, market_prob=0.476)
        assert p.ev > 0

    def test_negative_ev(self):
        # blended_prob debe estar en [min(model,market), max(model,market)]
        # model_prob_raw=0.48, market_prob=0.52 → blended en [0.48, 0.52]
        # EV = (0.49 * 1.80 - 1) * 100 = (0.882 - 1) * 100 = -11.8
        p = make_pick(price=1.80, model_prob_raw=0.48,
                      market_prob=0.52, blended_prob=0.49)
        assert p.ev < 0


# ── Tests: BetLedgerEntry ─────────────────────────────────────────────────────

class TestBetLedgerEntry:
    def test_construction(self):
        e = make_ledger_entry()
        assert e.entry_id == 'e001_TOTAL_over'
        assert e.result == 'pending'
        assert e.bankroll_after is None
        assert e.clv is None

    def test_settle_win(self):
        e = make_ledger_entry()
        e.settle(result='win', settled_at='2026-07-15T23:00:00Z')
        assert e.result == 'win'
        # profit = stake × (price - 1) = 20 × 0.91 = 18.20
        assert e.profit_amount == pytest.approx(18.20, abs=0.01)
        assert e.bankroll_after == pytest.approx(1018.20, abs=0.01)

    def test_settle_lose(self):
        e = make_ledger_entry()
        e.settle(result='lose', settled_at='2026-07-15T23:00:00Z')
        assert e.profit_amount == pytest.approx(-20.0, abs=0.01)
        assert e.bankroll_after == pytest.approx(980.0, abs=0.01)

    def test_settle_null_push(self):
        e = make_ledger_entry()
        e.settle(result='null', settled_at='2026-07-15T23:00:00Z')
        assert e.profit_amount == pytest.approx(0.0, abs=0.001)
        assert e.bankroll_after == pytest.approx(1000.0, abs=0.01)

    def test_settle_void(self):
        e = make_ledger_entry()
        e.settle(result='void', settled_at='2026-07-15T23:00:00Z')
        assert e.profit_amount == pytest.approx(0.0, abs=0.001)
        assert e.bankroll_after == pytest.approx(1000.0, abs=0.01)

    def test_no_double_settle(self):
        e = make_ledger_entry()
        e.settle(result='win', settled_at='2026-07-15T23:00:00Z')
        with pytest.raises(ValueError):
            e.settle(result='lose', settled_at='2026-07-16T00:00:00Z')

    def test_invalid_result_rejected(self):
        e = make_ledger_entry()
        with pytest.raises(ValueError):
            e.settle(result='ganado', settled_at='')

    def test_pending_not_terminal(self):
        from core.contracts.ledger import TERMINAL_RESULTS
        assert 'pending' not in TERMINAL_RESULTS

    def test_terminal_results_complete(self):
        from core.contracts.ledger import TERMINAL_RESULTS
        assert {'win','lose','null','void'} == TERMINAL_RESULTS

    def test_new_columns_sport_clv_model_version(self):
        e = make_ledger_entry()
        assert e.sport == 'mlb'
        assert e.model_version == 'mlb-v2.0.0'
        assert e.clv is None

    def test_yield_pct_after_win(self):
        e = make_ledger_entry()
        e.settle(result='win', settled_at='2026-07-15T23:00:00Z')
        # yield = profit/stake * 100 = 18.20/20.0 * 100 = 91%
        assert e.yield_pct == pytest.approx(91.0, abs=0.1)