"""
tests/sports/test_nba_availability.py

Parser del informe de lesiones de ESPN (scripts/capture_availability.py).

El payload de prueba replica la estructura verificada con --inspect
(2026-10): equipos sin abreviatura, jugadores sin `id`, y `type` como
tipo de estado en vez de parte del cuerpo.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import csv
import importlib.util
import json
from datetime import datetime, timezone

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'capture_availability.py')
_spec = importlib.util.spec_from_file_location("capture_availability", _SCRIPT)
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)

MOMENT = datetime(2026, 10, 20, 22, 30, tzinfo=timezone.utc)


def player(name, espn_id, abbr, status, body_part, detail, fantasy, ret="2026-10-22"):
    return {
        "id": "534140",  # id de la LESIÓN, no del jugador
        "status": status,
        "date": "2026-10-19T21:36Z",
        "shortComment": f"{name} won't play Monday.",
        "longComment": "...",
        "type": {"id": "6", "name": "INJURY_STATUS_DAYTODAY", "abbreviation": "DD"},
        "athlete": {
            "displayName": name,
            "links": [{"href": f"https://www.espn.com/nba/player/_/id/{espn_id}/x"}],
            "headshot": {"href": f"https://a.espncdn.com/i/headshots/nba/players/full/{espn_id}.png"},
            "position": {"displayName": "Guard", "abbreviation": "G"},
            "team": {"abbreviation": abbr, "displayName": "Boston Celtics"},
        },
        "details": {"type": body_part, "detail": detail, "side": "Left",
                    "returnDate": ret,
                    "fantasyStatus": {"description": fantasy, "abbreviation": fantasy}},
    }


PAYLOAD = {
    "timestamp": "2026-10-20T22:30:00Z",
    "status": "success",
    "injuries": [
        {"id": "2", "displayName": "Boston Celtics", "injuries": [
            player("Jayson Tatum", "4065648", "BOS", "Out", "Achilles", "Tear", "OUT"),
            player("Derrick White", "3078576", "BOS", "Day-To-Day", "Knee", "Soreness", "GTD"),
        ]},
        {"id": "8", "displayName": "Detroit Pistons", "injuries": []},
    ],
}


class TestParser:
    def rows(self, payload=PAYLOAD):
        return ca.parse_injuries(payload, "nba", MOMENT)

    def test_one_row_per_injured_player(self):
        assert [r["athlete_name"] for r in self.rows()] == ["Jayson Tatum", "Derrick White"]

    def test_team_abbreviation_comes_from_athlete(self):
        assert {r["team_abbreviation"] for r in self.rows()} == {"BOS"}

    def test_athlete_id_comes_from_links_not_injury_id(self):
        assert [r["athlete_id"] for r in self.rows()] == ["4065648", "3078576"]

    def test_athlete_id_falls_back_to_headshot(self):
        p = player("X", "999", "BOS", "Out", "Knee", "Tear", "OUT")
        p["athlete"]["links"] = []
        rows = ca.parse_injuries({"injuries": [{"id": "2", "injuries": [p]}]}, "nba", MOMENT)
        assert rows[0]["athlete_id"] == "999"

    def test_injury_type_is_body_part_not_status_type(self):
        assert [r["injury_type"] for r in self.rows()] == ["Achilles", "Knee"]

    def test_status_fantasy_and_return_date(self):
        r = self.rows()[1]
        assert (r["status"], r["fantasy_status"], r["return_date"]) == \
               ("Day-To-Day", "GTD", "2026-10-22")

    def test_short_comment_is_kept(self):
        assert self.rows()[0]["short_comment"].startswith("Jayson Tatum")

    def test_every_field_is_present(self):
        for row in self.rows():
            assert set(ca.FIELDNAMES) <= set(row)

    @pytest.mark.parametrize("payload", [{}, {"injuries": None}, [], None, {"injuries": [{}]}])
    def test_unexpected_shapes_never_raise(self, payload):
        assert ca.parse_injuries(payload, "nba", MOMENT) == []


class TestRun:
    def test_saves_raw_first_and_appends_csv(self, tmp_path):
        args = argparse.Namespace(sport="nba", base_dir=str(tmp_path),
                                  inspect=False, reparse=False, verbose=False)
        assert ca.run(args, fetcher=lambda path: PAYLOAD) == 0

        raws = list((tmp_path / "nba" / "raw").glob("*.json"))
        assert len(raws) == 1
        assert json.loads(raws[0].read_text(encoding="utf-8")) == PAYLOAD

        csvs = list((tmp_path / "nba").glob("*.csv"))
        with open(csvs[0], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["athlete_id"] for r in rows] == ["4065648", "3078576"]

    def test_inspect_saves_nothing(self, tmp_path):
        args = argparse.Namespace(sport="nba", base_dir=str(tmp_path),
                                  inspect=True, reparse=False, verbose=False)
        assert ca.run(args, fetcher=lambda path: PAYLOAD) == 0
        assert not any(tmp_path.iterdir())

    def test_network_error_exit_code(self, tmp_path):
        def boom(path):
            raise ConnectionError("sin red")
        args = argparse.Namespace(sport="nba", base_dir=str(tmp_path),
                                  inspect=False, reparse=False, verbose=False)
        assert ca.run(args, fetcher=boom) == 1
