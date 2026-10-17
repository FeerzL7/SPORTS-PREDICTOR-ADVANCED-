"""
tests/core/test_daily_cycle.py

Decisiones del ciclo diario (scripts/daily_cycle.py), sin red ni
créditos: los scripts y la consulta de /events se sustituyen por
funciones falsas que registran lo que se habría ejecutado.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import importlib.util
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, 'scripts', f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dc = _load("daily_cycle")
cc = _load("capture_closing")

NOW = datetime(2026, 10, 18, 16, 0, tzinfo=timezone.utc)
DATE = "2026-10-18"


def iso(minutes):
    return (NOW + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_entry(event, entry_id, clv=None, date=DATE):
    from core.contracts.ledger import BetLedgerEntry
    fields: dict[str, Any] = dict(
        entry_id=entry_id, sport="nfl", league="NFL", date=date, event=event,
        market="TOTAL", selection="Under 44.5", price=1.95, model_prob=0.55,
        ev=5.0, stake_pct=1, stake_amount=10.0, bankroll_before=1000.0,
        result="pending", model_version="nfl-v1.0",
        created_at="2026-10-18T12:00:00Z", clv=clv,
    )
    return BetLedgerEntry(**fields)


class Recorder:
    """Sustituye a run_script: guarda los comandos y devuelve un código."""
    def __init__(self, code=0):
        self.calls, self.code = [], code

    def __call__(self, cmd, log):
        self.calls.append(cmd)
        return self.code


def events(*games, remaining=400):
    payload = [{"away_team": a, "home_team": h, "commence_time": iso(m)} for a, h, m in games]
    return lambda sport_key: (payload, remaining)


@pytest.fixture
def log(tmp_path):
    return dc.CycleLog(DATE, "test", base_dir=tmp_path)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Ledger operable en un directorio temporal, con dos picks pendientes."""
    from core.bankroll.tracker import BankrollTracker, CsvLedgerStore
    monkeypatch.chdir(tmp_path)
    os.makedirs("output/ledger")
    store = CsvLedgerStore("output/ledger/nfl_roi_tracking.csv")
    tracker = BankrollTracker(store=store)
    tracker.register(make_entry("Las Vegas Raiders @ Kansas City Chiefs", "soon"))
    tracker.register(make_entry("Denver Broncos @ Los Angeles Chargers", "night"))
    tracker.register(make_entry("Denver Broncos @ Los Angeles Chargers", "done", clv=2.0))
    return store


# ── Fase picks ────────────────────────────────────────────────────────────────

class TestPicks:
    def test_runs_every_sport_with_local_date(self, log):
        rec = Recorder(code=2)  # 2 = sin picks, no es error
        code = dc.phase_picks(["nfl", "mlb"], DATE, log, plan=False, min_credits=40,
                              runner=rec, events_fn=events())
        assert code == 0
        assert rec.calls == [
            ["scripts/run_daily.py", "--sport", "nfl", "--date", DATE],
            ["scripts/run_daily.py", "--sport", "mlb", "--date", DATE],
        ]

    def test_runs_even_without_games_to_settle_yesterday(self, log):
        rec = Recorder()
        dc.phase_picks(["mlb"], DATE, log, plan=False, min_credits=40,
                       runner=rec, events_fn=events())  # sin partidos
        assert len(rec.calls) == 1

    def test_low_credits_settles_without_odds(self, log):
        rec = Recorder()
        dc.phase_picks(["nfl"], DATE, log, plan=False, min_credits=40,
                       runner=rec, events_fn=events(remaining=10))
        assert rec.calls[0][-2:] == ["--skip-stages", "5"]

    def test_error_code_is_reported(self, log):
        assert dc.phase_picks(["nfl"], DATE, log, plan=False, min_credits=40,
                              runner=Recorder(code=1), events_fn=events()) == 1

    def test_plan_executes_nothing(self, log):
        rec = Recorder()
        dc.phase_picks(["nfl"], DATE, log, plan=True, min_credits=40,
                       runner=rec, events_fn=events())
        assert rec.calls == []


# ── Fase closing ──────────────────────────────────────────────────────────────

class TestClosing:
    def test_no_pending_spends_nothing(self, log, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rec = Recorder()
        called = []
        code = dc.phase_closing(["nfl"], DATE, log, plan=False, window_minutes=45,
                                min_credits=40, now=NOW, runner=rec,
                                events_fn=lambda k: called.append(k) or ([], 400), cc=cc)
        assert code == 0 and rec.calls == [] and called == []  # ni siquiera /events

    def test_captures_when_a_game_is_about_to_start(self, log, ledger):
        rec = Recorder()
        dc.phase_closing(["nfl"], DATE, log, plan=False, window_minutes=45, min_credits=40,
                         now=NOW, runner=rec, cc=cc, events_fn=events(
                             ("Las Vegas Raiders", "Kansas City Chiefs", 30),
                             ("Denver Broncos", "Los Angeles Chargers", 300)))
        assert rec.calls == [["scripts/capture_closing.py", "--sport", "nfl", "--date", DATE,
                              "--with-paper", "--max-lead-minutes", "45"]]

    def test_waits_when_no_game_is_in_window(self, log, ledger):
        rec = Recorder()
        dc.phase_closing(["nfl"], DATE, log, plan=False, window_minutes=45, min_credits=40,
                         now=NOW, runner=rec, cc=cc, events_fn=events(
                             ("Las Vegas Raiders", "Kansas City Chiefs", 120),
                             ("Denver Broncos", "Los Angeles Chargers", 300)))
        assert rec.calls == []

    def test_low_credits_does_not_spend(self, log, ledger):
        rec = Recorder()
        dc.phase_closing(["nfl"], DATE, log, plan=False, window_minutes=45, min_credits=40,
                         now=NOW, runner=rec, cc=cc, events_fn=events(
                             ("Las Vegas Raiders", "Kansas City Chiefs", 30), remaining=5))
        assert rec.calls == []

    def test_pending_ignores_entries_with_clv_and_other_dates(self, ledger):
        ids = {e.entry_id for e in dc.pending_entries("nfl", DATE, cc)}
        assert ids == {"soon", "night"}

    def test_events_failure_is_an_error_but_spends_nothing(self, log, ledger):
        def boom(k):
            raise RuntimeError("sin red")
        rec = Recorder()
        code = dc.phase_closing(["nfl"], DATE, log, plan=False, window_minutes=45,
                                min_credits=40, now=NOW, runner=rec, events_fn=boom, cc=cc)
        assert code == 1 and rec.calls == []


def test_log_file_is_written(tmp_path):
    log = dc.CycleLog(DATE, "picks", base_dir=tmp_path)
    log("hola — ñandú")
    assert "hola — ñandú" in (tmp_path / f"{DATE}.log").read_text(encoding="utf-8")


# ── Capturas NBA (deporte sin modelo) ─────────────────────────────────────────

AVAIL = ["scripts/capture_availability.py", "--sport", "nba"]
BOOKS = ["scripts/capture_book_odds.py", "--sport", "nba", "--markets", "spreads,totals",
         "--max-lead-minutes", "45", "--min-credits", "40"]
MORNING = ["scripts/capture_book_odds.py", "--sport", "nba", "--markets", "spreads,totals",
           "--max-lead-minutes", "1440", "--min-credits", "40"]


def nba_events(*minutes, remaining=400):
    payload = [{"away_team": f"A{i}", "home_team": f"H{i}", "commence_time": iso(m)}
               for i, m in enumerate(minutes)]
    return lambda sport_key: (payload, remaining)


def capture(phase, events_fn, plan=False):
    rec = Recorder()
    log = _NullLog()
    failures = dc.capture_sport("nba", phase, log, plan, 45.0, 40, NOW, rec, events_fn)
    return rec.calls, failures


class _NullLog:
    def __call__(self, message=""):
        pass


class TestNbaCapture:
    def test_morning_snapshot_and_prices_when_games_today(self):
        calls, _ = capture("picks", nba_events(600))
        assert calls == [AVAIL, MORNING]

    def test_morning_prices_skipped_with_low_credits(self):
        calls, _ = capture("picks", nba_events(600, remaining=5))
        assert calls == [AVAIL]

    def test_morning_ignores_games_beyond_24h(self):
        calls, _ = capture("picks", nba_events(30 * 60))
        assert calls == []

    def test_no_snapshot_without_games(self):
        calls, _ = capture("picks", nba_events())
        assert calls == []

    def test_closing_captures_injuries_and_prices_for_next_wave(self):
        calls, _ = capture("closing", nba_events(30, 300))
        assert calls == [AVAIL, BOOKS]

    def test_injuries_only_when_wave_is_an_hour_away(self):
        calls, _ = capture("closing", nba_events(60))
        assert calls == [AVAIL]

    def test_wave_already_captured_is_not_paid_twice(self):
        # a 10 min: ya cayó en (15, 45] en la corrida anterior
        calls, _ = capture("closing", nba_events(10))
        assert calls == [AVAIL]

    def test_low_credits_keeps_free_capture_only(self):
        calls, _ = capture("closing", nba_events(30, remaining=5))
        assert calls == [AVAIL]

    def test_started_games_are_ignored(self):
        calls, _ = capture("closing", nba_events(-20))
        assert calls == []

    def test_plan_runs_nothing(self):
        calls, _ = capture("closing", nba_events(30), plan=True)
        assert calls == []

    def test_events_failure_counts_as_failure(self):
        def boom(k):
            raise RuntimeError("sin red")
        calls, failures = capture("closing", boom)
        assert calls == [] and failures == 1

    def test_phase_closing_runs_capture_sports(self, log, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rec = Recorder()
        dc.phase_closing([], DATE, log, plan=False, window_minutes=45, min_credits=40,
                         now=NOW, runner=rec, events_fn=nba_events(30), cc=cc,
                         capture_sports=["nba"])
        assert rec.calls == [AVAIL, BOOKS]


# ── Credenciales fuera del log ────────────────────────────────────────────────

class TestRedaction:
    def test_api_key_in_url_is_hidden(self):
        text = "Max retries exceeded with url: /v4/sports/x/events?apiKey=abc123def456&regions=us"
        assert dc.redact(text) == \
            "Max retries exceeded with url: /v4/sports/x/events?apiKey=***&regions=us"

    def test_literal_secret_values_are_hidden(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:SECRETOTOKEN")
        assert "SECRETOTOKEN" not in dc.redact("bot123456:SECRETOTOKEN/sendMessage")

    def test_log_file_never_contains_the_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "clave-super-secreta-42")
        log = dc.CycleLog(DATE, "closing", base_dir=tmp_path)
        log("fallo con clave-super-secreta-42 y ?apiKey=clave-super-secreta-42")
        content = (tmp_path / f"{DATE}.log").read_text(encoding="utf-8")
        assert "clave-super-secreta-42" not in content

    def test_fetch_events_error_does_not_leak_key(self, monkeypatch):
        import requests
        monkeypatch.setenv("ODDS_API_KEY", "clave-super-secreta-42")

        def fail(url, params, timeout):
            raise requests.ConnectionError(f"Max retries exceeded with url: {url}?apiKey={params['apiKey']}")
        monkeypatch.setattr(requests, "get", fail)
        with pytest.raises(RuntimeError) as info:
            dc.fetch_events("basketball_nba")
        assert "clave-super-secreta-42" not in str(info.value)


# ── Respaldo ──────────────────────────────────────────────────────────────────

class TestBackup:
    def make_project(self, root):
        for rel, content in [("output/ledger/nfl_roi_tracking.csv", "a,b\n"),
                             ("output/book_archive/basketball_nba/2026-10.csv", "x\n"),
                             ("output/availability_archive/nba/raw/2026-10-20T1500Z.json", "{}"),
                             ("output/cache/no_se_respalda.parquet", "cache")]:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def test_copies_archives_and_ledgers_only(self, tmp_path):
        project, dest = tmp_path / "proyecto", tmp_path / "respaldo"
        self.make_project(project)
        assert dc.backup_outputs(dest, _NullLog(), plan=False, project_root=project) == 0
        copied = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
        assert copied == [
            "sports-predictor/output/availability_archive/nba/raw/2026-10-20T1500Z.json",
            "sports-predictor/output/book_archive/basketball_nba/2026-10.csv",
            "sports-predictor/output/ledger/nfl_roi_tracking.csv",
        ]

    def test_second_run_copies_nothing_and_never_deletes(self, tmp_path):
        project, dest = tmp_path / "proyecto", tmp_path / "respaldo"
        self.make_project(project)
        dc.backup_outputs(dest, _NullLog(), plan=False, project_root=project)
        (project / "output/ledger/nfl_roi_tracking.csv").unlink()

        messages = []
        dc.backup_outputs(dest, messages.append, plan=False, project_root=project)
        assert "0 archivos" in messages[-1]
        assert (dest / "sports-predictor/output/ledger/nfl_roi_tracking.csv").exists()

    def test_changed_file_is_copied_again(self, tmp_path):
        project, dest = tmp_path / "proyecto", tmp_path / "respaldo"
        self.make_project(project)
        dc.backup_outputs(dest, _NullLog(), plan=False, project_root=project)
        (project / "output/ledger/nfl_roi_tracking.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        dc.backup_outputs(dest, _NullLog(), plan=False, project_root=project)
        assert (dest / "sports-predictor/output/ledger/nfl_roi_tracking.csv").read_text(
            encoding="utf-8") == "a,b\n1,2\n"

    def test_plan_copies_nothing(self, tmp_path):
        project, dest = tmp_path / "proyecto", tmp_path / "respaldo"
        self.make_project(project)
        dc.backup_outputs(dest, _NullLog(), plan=True, project_root=project)
        assert not dest.exists()

    def test_disabled_without_env_var(self, monkeypatch):
        monkeypatch.delenv(dc.BACKUP_ENV_VAR, raising=False)
        messages = []
        assert dc._maybe_backup(messages.append, plan=False) == 0
        assert "desactivado" in messages[0]
