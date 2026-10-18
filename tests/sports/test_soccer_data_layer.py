"""
tests/sports/test_soccer_data_layer.py

Capa de datos de fútbol reparada en 2026-10:

    - Understat por el endpoint JSON getLeagueData (con respaldo HTML)
    - cierre de Betfair Exchange cuando falta Pinnacle, y fuente forzable
    - Pinnacle en el formato extendido (PSCH, no PCH)
    - hora real de inicio desde la columna Time (hora del Reino Unido)
    - partidos futuros desde fixtures.csv
    - Elo propio sin look-ahead
    - el reparto de goles del Elo conserva la ventaja de campo

Sin red: todo se inyecta o se lee de un caché temporal.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from types import SimpleNamespace

import pytest

import sports.soccer.data_source as ds
from sports.soccer.data_source import MatchRow, SoccerDataSource
from sports.soccer.elo_local import EloParams, LocalEloClient
from sports.soccer.understat import UnderstatClient


# ── Understat ─────────────────────────────────────────────────────────────────

TEAMS = {
    "83": {"id": "83", "title": "Arsenal", "history": [
        {"h_a": "h", "xG": "1.85", "xGA": "0.56", "npxG": "1.85", "npxGA": "0.56",
         "scored": "3", "missed": "0", "result": "w", "date": "2026-08-21 19:00:00"},
    ]},
}


class TestUnderstat:
    def test_reads_json_endpoint(self):
        payload = json.dumps({"teams": TEAMS, "players": [], "dates": []})
        urls = []
        client = UnderstatClient(fetch_fn=lambda url: urls.append(url) or payload)
        rows = client.fetch_league_matches("EPL", 2026)
        assert urls == ["https://understat.com/getLeagueData/EPL/2026"]
        assert [(r.team, r.date, r.is_home, r.goals, round(r.xg, 2)) for r in rows] == \
               [("Arsenal", "2026-08-21", True, 3, 1.85)]

    def test_falls_back_to_old_html_format(self):
        escaped = json.dumps(TEAMS).encode("unicode_escape").decode("latin-1")
        html = f"<script>var teamsData = JSON.parse('{escaped}');</script>"
        rows = UnderstatClient(fetch_fn=lambda url: html).fetch_league_matches("EPL", 2026)
        assert len(rows) == 1

    def test_garbage_degrades_to_empty(self):
        assert UnderstatClient(fetch_fn=lambda url: "<html>sin datos</html>") \
            .fetch_league_matches("EPL", 2026) == []

    def test_closed_season_is_cached(self, tmp_path):
        client = UnderstatClient(cache_dir=tmp_path, current_season=2026)
        client._write_cache("EPL", 2024, json.dumps({"teams": TEAMS}))
        assert client._read_cache("EPL", 2024) is not None
        assert len(client.fetch_league_matches("EPL", 2024)) == 1   # sin red


# ── Cuotas de cierre ──────────────────────────────────────────────────────────

BASE = {"HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Date": "17/10/2026",
        "FTHG": "2", "FTAG": "1"}
PIN = {"PSCH": "2.00", "PSCD": "3.50", "PSCA": "4.00"}
BFE = {"BFECH": "2.06", "BFECD": "3.60", "BFECA": "4.10",
       "BFEC>2.5": "1.90", "BFEC<2.5": "2.00"}
B365 = {"B365CH": "1.95", "B365CD": "3.40", "B365CA": "3.80"}


class TestClosingSources:
    def test_pinnacle_first_when_present(self):
        assert ds._extract_odds({**BASE, **PIN, **BFE, **B365})["odds_source"] == "pinnacle"

    def test_betfair_exchange_before_bet365_without_pinnacle(self):
        out = ds._extract_odds({**BASE, **BFE, **B365})
        assert out["odds_source"] == "betfair_exchange"
        assert (out["odds_home"], out["odds_over"]) == (2.06, 1.90)

    def test_forced_source_does_not_fall_back(self):
        assert ds._extract_odds({**BASE, **BFE}, "pinnacle")["odds_source"] == ""
        assert ds._extract_odds({**BASE, **PIN, **BFE}, "betfair_exchange")["odds_home"] == 2.06

    def test_extended_format_reads_psch(self):
        # Liga MX / Brasil: el cierre de Pinnacle se llama PSCH, no PCH.
        out = ds._extract_odds_extra({"Home": "Atlas", "Away": "Necaxa", **PIN,
                                      "AvgCH": "1.9", "AvgCD": "3.3", "AvgCA": "3.9"})
        assert out["odds_source"] == "pinnacle"

    def test_unknown_forced_source_is_rejected(self):
        with pytest.raises(ValueError):
            SoccerDataSource(cache_dir=None, odds_source="pinacle")


# ── Hora de inicio ────────────────────────────────────────────────────────────

class TestKickoff:
    @pytest.mark.parametrize("date,time,expected", [
        ("2026-10-17", "15:00", "2026-10-17T14:00:00Z"),   # BST (UTC+1)
        ("2026-12-26", "15:00", "2026-12-26T15:00:00Z"),   # GMT
        ("2026-10-17", "00:30", "2026-10-16T23:30:00Z"),   # cruza la medianoche
        ("2026-10-17", "", ""),
        ("2026-10-17", None, ""),
    ])
    def test_uk_time_to_utc(self, date, time, expected):
        assert ds._kickoff_utc(date, time) == expected


# ── Partidos futuros ──────────────────────────────────────────────────────────

FIXTURES_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,BFEH,BFED,BFEA\n"
    "E0,18/10/2026,14:00,Bournemouth,Sunderland,1.80,3.80,4.50,1.84,3.90,4.70\n"
    "E1,18/10/2026,12:30,Leeds,Hull,2.10,3.30,3.60,2.14,3.40,3.70\n"
    "SP1,18/10/2026,20:00,Betis,Sevilla,2.20,3.20,3.40,2.24,3.30,3.50\n"
)


class TestFixtures:
    def make_source(self, tmp_path):
        source = SoccerDataSource(cache_dir=str(tmp_path), current_season=2026)
        (tmp_path / "fixtures_main.csv").write_text(FIXTURES_CSV, encoding="utf-8")
        return source

    def test_filters_by_division_and_uses_opening_odds(self, tmp_path):
        rows = self.make_source(tmp_path).load_fixtures("epl")
        assert [(r.home_team, r.away_team, r.season) for r in rows] == \
               [("Bournemouth", "Sunderland", 2026)]
        row = rows[0]
        assert row.odds_home is None                         # no hay cierre aún
        assert (row.open_home, row.open_source) == (1.84, "betfair_exchange")
        assert row.kickoff_utc == "2026-10-18T13:00:00Z"
        assert not row.is_final

    def test_schedule_merges_fixtures_without_duplicates(self, tmp_path):
        from sports.soccer.competitions import get_competition
        from sports.soccer.schedule import SoccerScheduleFetcher

        played = MatchRow(comp_id="epl", season=2026, date="2026-10-18",
                          home_team="Brighton", away_team="Crystal Palace",
                          home_goals=1, away_goals=0, kickoff_utc="2026-10-18T11:00:00Z")
        upcoming = MatchRow(comp_id="epl", season=2026, date="2026-10-18",
                            home_team="Bournemouth", away_team="Sunderland",
                            kickoff_utc="2026-10-18T13:00:00Z")
        source = SimpleNamespace(
            load_matches=lambda comp, season: [played],
            load_match_xg=lambda comp, season: [],
            load_fixtures=lambda comp: [upcoming, played],   # 'played' repetido
        )
        events = SoccerScheduleFetcher(data_source=source,
                                       competitions=[get_competition("epl")]).get_events("2026-10-18")
        assert [(e.home_team, e.start_time) for e in events] == [
            ("Brighton", "2026-10-18T11:00:00Z"),
            ("Bournemouth", "2026-10-18T13:00:00Z"),
        ]


# ── Elo propio ────────────────────────────────────────────────────────────────

def match(date, home, away, hg, ag, season=2024):
    return MatchRow(comp_id="epl", season=season, date=date, home_team=home,
                    away_team=away, home_goals=hg, away_goals=ag)


class FakeSource:
    def __init__(self, by_season):
        self.by_season = by_season

    def load_matches(self, comp, season):
        return self.by_season.get(season, [])


class TestLocalElo:
    def client(self, by_season, **params):
        return LocalEloClient(FakeSource(by_season), EloParams(seasons_back=1, **params))

    def test_no_lookahead_same_day_match_is_not_seen(self):
        elo = self.client({2024: [match("2024-08-17", "Arsenal", "Wolves", 3, 0)]})
        assert elo.rating_before("Arsenal", "2024-08-17", "epl") is None
        assert elo.rating_before("Arsenal", "2024-08-18", "epl") > 1500

    def test_home_win_moves_ratings_symmetrically(self):
        elo = self.client({2024: [match("2024-08-17", "Arsenal", "Wolves", 1, 0)]})
        a = elo.rating_before("Arsenal", "2024-09-01", "epl")
        w = elo.rating_before("Wolves", "2024-09-01", "epl")
        assert a > 1500 > w and round(a + w, 6) == 3000

    def test_bigger_win_moves_more(self):
        narrow = self.client({2024: [match("2024-08-17", "Arsenal", "Wolves", 1, 0)]})
        rout = self.client({2024: [match("2024-08-17", "Arsenal", "Wolves", 5, 0)]})
        assert rout.rating_before("Arsenal", "2024-09-01", "epl") > \
               narrow.rating_before("Arsenal", "2024-09-01", "epl")

    def test_promoted_team_enters_below_league_mean(self):
        previous = [match("2023-08-12", "Arsenal", "Wolves", 1, 1, season=2023)]
        current = [match("2024-08-17", "Leeds", "Arsenal", 0, 0)]
        elo = self.client({2023: previous, 2024: current})
        mean_prev = 1500.0   # el empate 1-1 apenas mueve los ratings
        leeds = elo.rating_before("Leeds", "2024-09-01", "epl")
        assert leeds < mean_prev - 50

    def test_difference_none_without_history(self):
        elo = self.client({2024: [match("2024-08-17", "Arsenal", "Wolves", 1, 0)]})
        assert elo.difference("Arsenal", "Chelsea", "2024-09-01", "epl") is None
        assert elo.difference("Arsenal", "Wolves", "2024-09-01", "epl") > 0


# ── Reparto del Elo en goles ──────────────────────────────────────────────────

class TestEloLambdas:
    def test_equal_elo_keeps_home_advantage(self):
        from sports.soccer.projections import SoccerProjectionModel
        model = SoccerProjectionModel()
        home, away = model._lambdas_from_elo({"elo_difference": 0.0}, 1.5, 1.2)
        assert (home, away) == (1.5, 1.2)

    def test_elo_edge_shifts_goals_from_away_to_home(self):
        from sports.soccer.projections import SoccerProjectionModel
        model = SoccerProjectionModel()
        home, away = model._lambdas_from_elo({"elo_difference": 100.0}, 1.5, 1.2)
        assert home > 1.5 and away < 1.2
        assert round(home + away, 6) == 2.7
