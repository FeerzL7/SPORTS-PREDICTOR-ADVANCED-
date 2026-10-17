"""
tests/core/test_runner_guards.py

Salvaguardas del runner encontradas al revisar la primera semana del
ciclo diario (2026-10):

    (a) la liquidación (Stage 10) corre aunque no haya candidatos;
    (b) los avisos de día sin trabajo no cuentan como error;
    (d) los eventos ya empezados se descartan (sus cuotas serían en vivo).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import importlib.util
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.pipeline.runner import NO_WORK_NOTICES, PipelineResult, PipelineRunner, RunnerConfig
from core.pipeline.stage import PipelineContext

NOW = datetime(2026, 10, 18, 17, 30, tzinfo=timezone.utc)


def make_event(event_id, start_time):
    from core.contracts.event import Event, EventStatus
    return Event(
        event_id=event_id, sport='nfl', league='NFL',
        season_start=2026, season_end=2026,
        date='2026-10-18', start_time=start_time,
        home_team_id='KC', away_team_id='LV',
        home_team='Kansas City Chiefs', away_team='Las Vegas Raiders',
        venue_id='arrowhead', venue_name='Arrowhead',
        status=EventStatus.SCHEDULED, provider_ids={},
    )


class FakeTracker:
    def __init__(self):
        self.settle_calls = 0

    def settle_pending(self, provider, sport):
        self.settle_calls += 1
        return ["liquidado"]


class FakePlugin:
    sport_id, league_id = "nfl", "NFL"

    def __init__(self, events=()):
        self._events = list(events)

    def get_data_provider(self):
        return SimpleNamespace(get_events=lambda date: list(self._events))

    def get_settlement_provider(self):
        return object()


def make_runner(plugin, tracker=None, **config):
    return PipelineRunner(
        plugin=plugin, odds_client=None, normalizer=None, line_detector=None,
        value_engine=None, staking=None, risk_manager=None,
        roi_tracker=tracker or FakeTracker(), clv_tracker=None,
        config=RunnerConfig(**config),
    )


# ── (a) Liquidación sin candidatos ────────────────────────────────────────────

class TestSettlementWithoutCandidates:
    def test_settles_on_a_day_without_games(self, tmp_path):
        tracker = FakeTracker()
        runner = make_runner(FakePlugin(), tracker,
                             paper_ledger_path=str(tmp_path / "papel.csv"))
        result = runner.run("2026-10-17")

        assert tracker.settle_calls == 1
        assert result.settled_count == 1

    def test_dry_run_never_settles(self, tmp_path):
        tracker = FakeTracker()
        runner = make_runner(FakePlugin(), tracker, dry_run=True,
                             paper_ledger_path=str(tmp_path / "papel.csv"))
        runner.run("2026-10-17")
        assert tracker.settle_calls == 0


# ── (b) Avisos de día sin trabajo ─────────────────────────────────────────────

class TestNoWorkNotices:
    def make_result(self, errors):
        ctx = PipelineContext(sport="nfl", date="2026-10-17")
        return PipelineResult(sport="nfl", date="2026-10-17", active_picks=[],
                              candidates=[], settled_count=0, errors=errors,
                              context=ctx, duration_seconds=0.0, dry_run=False)

    def test_day_without_games_has_no_real_errors(self, tmp_path):
        result = make_runner(FakePlugin(),
                             paper_ledger_path=str(tmp_path / "papel.csv")).run("2026-10-17")
        assert result.errors            # los avisos siguen visibles
        assert result.real_errors == []

    def test_real_errors_survive(self):
        result = self.make_result([
            "[Stage1] Sin eventos para esta fecha.",
            "[Stage10] ConnectionError: sin red",
        ])
        assert result.real_errors == ["[Stage10] ConnectionError: sin red"]

    @pytest.mark.parametrize("notice", NO_WORK_NOTICES)
    def test_each_notice_is_filtered(self, notice):
        assert self.make_result([notice + " ..."]).real_errors == []


@pytest.fixture(scope="module")
def run_daily():
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'run_daily.py')
    spec = importlib.util.spec_from_file_location("run_daily", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunDailyExitCode:
    def result(self, errors=(), active=(), paper=()):
        ctx = PipelineContext(sport="nfl", date="2026-10-17")
        return PipelineResult(sport="nfl", date="2026-10-17", active_picks=list(active),
                              candidates=[], settled_count=0, errors=list(errors),
                              context=ctx, duration_seconds=0.0, dry_run=False,
                              paper_picks=list(paper))

    def test_quiet_day_is_two_not_one(self, run_daily):
        r = self.result(["[Stage1] Sin eventos para esta fecha.",
                         "[PhaseB] Sin candidatos — stages 7-9 y 11 omitidos."])
        assert run_daily._exit_code(r) == 2

    def test_paper_picks_are_work_done(self, run_daily):
        assert run_daily._exit_code(self.result(paper=["pick"])) == 0

    def test_real_error_without_picks_is_one(self, run_daily):
        assert run_daily._exit_code(self.result(["[Stage10] sin red"])) == 1


# ── (d) Eventos ya empezados ──────────────────────────────────────────────────

class TestStartedEvents:
    def run_phase_a(self, events, **config):
        runner = make_runner(FakePlugin(events), clock=lambda: NOW,
                             skip_stages=frozenset({5}), **config)
        ctx = PipelineContext(sport="nfl", date="2026-10-18")
        return runner._run_phase_a("2026-10-18", ctx)

    def test_started_games_are_dropped(self):
        ctx = self.run_phase_a([
            make_event("early", "2026-10-18T17:00:00Z"),   # empezó hace 30 min
            make_event("late",  "2026-10-18T20:25:00Z"),
        ])
        assert [e.event_id for e in ctx.events] == ["late"]
        assert ctx.metadata["skipped_started"] == ["early"]

    def test_all_started_is_a_quiet_day(self):
        ctx = self.run_phase_a([make_event("early", "2026-10-18T17:00:00Z")])
        assert ctx.events == []
        assert any(e.startswith("[Stage1] Sin eventos por jugar") for e in ctx.errors)

    def test_unreadable_start_time_is_kept(self):
        ctx = self.run_phase_a([make_event("x", "sin-hora")])
        assert [e.event_id for e in ctx.events] == ["x"]

    def test_can_be_disabled_for_fixed_date_dry_runs(self):
        ctx = self.run_phase_a([make_event("early", "2026-10-18T17:00:00Z")],
                               skip_started_events=False)
        assert [e.event_id for e in ctx.events] == ["early"]
