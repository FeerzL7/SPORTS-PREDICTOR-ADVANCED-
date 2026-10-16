"""
tests/core/test_paper_only.py

Mercados en SOLO-PAPEL (Fase 0): se registran para medir, nunca se
operan ni se notifican.

Cubre las tres piezas del recorrido:
    RiskManager   — separa los picks de papel sin que compitan por cupo
    PipelineRunner — los registra en su propio ledger (Stage 9 → 11)
    config/*.yaml  — qué mercados están en papel según la evidencia
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Any

import pytest

from core.risk.manager import PAPER_ONLY_REASON, RiskManager, RiskProfile


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_event(event_id='e001'):
    from core.contracts.event import Event, EventStatus
    return Event(
        event_id=event_id, sport='nfl', league='NFL',
        season_start=2026, season_end=2026,
        date='2026-10-18', start_time='2026-10-18T17:00:00Z',
        home_team_id='KC', away_team_id='LV',
        home_team='Kansas City Chiefs', away_team='Las Vegas Raiders',
        venue_id='arrowhead', venue_name='Arrowhead',
        status=EventStatus.SCHEDULED, provider_ids={},
    )


def make_pick(market='TOTAL', selection='under', line=44.5, event_id='e001',
              stake_pct=1, ev_reason=True):
    from core.contracts.pick import CandidatePick
    pick = CandidatePick(
        event=make_event(event_id), market=market, selection=selection,
        line=line, price=1.91, model_prob_raw=0.58, market_prob=0.524,
        blended_prob=0.55, stake_pct=stake_pct,
    )
    if ev_reason:
        pick.add_reason("FILTERS PASSED")
    return pick


class FakeConfig:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


# ── RiskManager ───────────────────────────────────────────────────────────────

class TestRiskManagerPaper:
    def test_from_config_reads_paper_markets(self):
        rm = RiskManager.from_config(FakeConfig({
            "risk.paper_only_markets": ["spread", "ML"],
        }))
        assert rm._profile.paper_only_markets == frozenset({"SPREAD", "ML"})

    def test_from_config_accepts_single_string(self):
        rm = RiskManager.from_config(FakeConfig({"risk.paper_only_markets": "1x2"}))
        assert rm._profile.paper_only_markets == frozenset({"1X2"})

    def test_default_has_no_paper_markets(self):
        assert RiskManager.from_config(FakeConfig({}))._profile.paper_only_markets == frozenset()

    def test_paper_picks_are_separated_and_tagged(self):
        rm = RiskManager(RiskProfile(paper_only_markets=frozenset({"SPREAD"})))
        live, paper = make_pick('TOTAL'), make_pick('SPREAD', 'Kansas City Chiefs', -3.5)
        summary = rm.apply([live, paper])

        assert summary.picks_active == [live]
        assert summary.picks_paper == [paper]
        assert paper.active  # aprobado para SU ledger
        assert any(PAPER_ONLY_REASON in r for r in paper.reasons)
        assert not any(PAPER_ONLY_REASON in r for r in live.reasons)

    def test_paper_picks_do_not_use_daily_budget(self):
        rm = RiskManager(RiskProfile(max_picks_daily=1, max_exposure_pct=1,
                                     paper_only_markets=frozenset({"SPREAD"})))
        paper = [make_pick('SPREAD', f'T{i}', -3.5, event_id=f'e{i}') for i in range(4)]
        live = make_pick('TOTAL', event_id='e9')
        summary = rm.apply(paper + [live])

        assert summary.picks_active == [live]
        assert len(summary.picks_paper) == 4
        assert summary.total_exposure_pct == 1

    def test_paper_picks_still_vetoed_by_contradicting_movement(self):
        from core.odds.line_movement import MOVEMENT_CONTRADICTS_PREFIX
        rm = RiskManager(RiskProfile(paper_only_markets=frozenset({"SPREAD"})))
        pick = make_pick('SPREAD', 'Kansas City Chiefs', -3.5)
        pick.add_reason(f"{MOVEMENT_CONTRADICTS_PREFIX} línea en contra")
        summary = rm.apply([pick])

        assert summary.picks_paper == []
        assert summary.picks_vetoed == [pick]

    def test_paper_stake_trimmed_like_live(self):
        rm = RiskManager(RiskProfile(max_stake_per_pick_pct=2,
                                     paper_only_markets=frozenset({"SPREAD"})))
        pick = make_pick('SPREAD', 'Kansas City Chiefs', -3.5, stake_pct=3)
        rm.apply([pick])
        assert pick.stake_pct == 2

    def test_log_summary_reports_paper_count(self):
        rm = RiskManager(RiskProfile(paper_only_markets=frozenset({"SPREAD"})))
        summary = rm.apply([make_pick('SPREAD', 'Kansas City Chiefs', -3.5)])
        assert "solo_papel=1" in summary.log_summary()


# ── PipelineRunner: Stage 9 → Stage 11 ────────────────────────────────────────

class _Plugin:
    sport_id  = "nfl"
    league_id = "NFL"


class _CLV:
    def __init__(self):
        self.recorded = []

    def record_opening_line(self, entry_id, opening_price, market, sport):
        self.recorded.append(entry_id)


@pytest.fixture
def runner_and_paths(tmp_path):
    from core.bankroll.tracker import BankrollTracker, CsvLedgerStore
    from core.pipeline.runner import PipelineRunner, RunnerConfig
    from core.tracking.roi_tracker import ROITracker

    live_path  = str(tmp_path / "nfl_roi_tracking.csv")
    paper_path = str(tmp_path / "nfl_paper_roi_tracking.csv")
    clv = _CLV()
    runner = PipelineRunner(
        plugin        = _Plugin(),
        odds_client   = None,
        normalizer    = None,
        line_detector = None,
        value_engine  = None,
        staking       = None,
        risk_manager  = RiskManager(RiskProfile(paper_only_markets=frozenset({"SPREAD"}))),
        roi_tracker   = ROITracker(BankrollTracker(store=CsvLedgerStore(live_path)),
                                   model_version="nfl-v1.0"),
        clv_tracker   = clv,
        config        = RunnerConfig(paper_ledger_path=paper_path),
    )
    return runner, live_path, paper_path, clv


class TestRunnerPaperLedger:
    def _context(self, picks):
        from core.pipeline.stage import PipelineContext
        ctx = PipelineContext(sport="nfl", date="2026-10-18")
        ctx.candidates.extend(picks)
        return ctx

    def test_paper_picks_go_to_their_own_ledger(self, runner_and_paths):
        from core.bankroll.tracker import CsvLedgerStore
        runner, live_path, paper_path, clv = runner_and_paths

        live  = make_pick('TOTAL', event_id='e1')
        paper = make_pick('SPREAD', 'Kansas City Chiefs', -3.5, event_id='e2')
        ctx = self._context([live, paper])

        runner._stage_9_risk(ctx)
        assert ctx.active_picks == [live]
        assert ctx.paper_picks == [paper]

        runner._stage_11_register(ctx)
        live_entries  = CsvLedgerStore(live_path).load_all()
        paper_entries = CsvLedgerStore(paper_path).load_all()

        assert [e.market for e in live_entries]  == ["TOTAL"]
        assert [e.market for e in paper_entries] == ["SPREAD"]
        assert paper_entries[0].model_version == "nfl-v1.0-papel"
        assert ctx.metadata["paper_registered_count"] == 1
        assert len(clv.recorded) == 2  # CLV también para el de papel

    def test_no_paper_ledger_without_paper_picks(self, runner_and_paths):
        runner, _, paper_path, _ = runner_and_paths
        ctx = self._context([make_pick('TOTAL', event_id='e1')])
        runner._stage_9_risk(ctx)
        runner._stage_11_register(ctx)
        assert not os.path.exists(paper_path)

    def test_result_exposes_paper_picks(self):
        from core.pipeline.runner import PipelineResult
        from core.pipeline.stage import PipelineContext
        ctx = PipelineContext(sport="nfl", date="2026-10-18")
        res = PipelineResult(sport="nfl", date="2026-10-18", active_picks=[],
                             candidates=[], settled_count=0, errors=[],
                             context=ctx, duration_seconds=0.0, dry_run=True)
        assert res.paper_picks == []
        assert "solo_papel=0" in res.summary()


# ── Configuración: la evidencia de la Fase 0 ──────────────────────────────────

class TestConfiguredPaperMarkets:
    @pytest.mark.parametrize("sport,expected", [
        ("soccer", {"1X2", "TOTAL", "BTTS"}),
        ("nfl",    {"SPREAD", "ML", "TOTAL"}),
        ("mlb",    {"ML", "SPREAD", "TOTAL"}),
    ])
    def test_markets_without_evidence_are_paper_only(self, sport, expected):
        from core.utils.config_loader import load_config
        root = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
        rm = RiskManager.from_config(load_config(sport=sport, base_dir=root))
        assert rm._profile.paper_only_markets == frozenset(expected)

    def test_nba_total_sigma_matches_measured_closing_error(self):
        # audit_nba_sources.py: σ(total real − línea de cierre) = 18.04
        import math
        from core.simulation.factory import DistributionFactory
        from core.utils.config_loader import load_config
        root = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
        model = DistributionFactory(config=load_config(sport="mlb", base_dir=root)) \
            .get_model("nba", "TOTAL")
        assert math.sqrt(2) * model.default_sigma == pytest.approx(18.04, abs=0.1)
