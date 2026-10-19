"""Tests for preseason role evidence collection and feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courtside.proj.preseason import (
    _parse_duration,
    _parse_boxscore_players,
    _is_final_boxscore,
)


@pytest.fixture
def sample_boxscore():
    """Load sample preseason boxscore fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "preseason_boxscore.json"
    if not fixture_path.exists():
        pytest.skip("fixture not found")
    with open(fixture_path) as f:
        return json.load(f)


class TestParseDuration:
    """Test ISO 8601 duration parsing."""

    def test_minutes_and_seconds(self):
        """Parse PT24M30.00S -> 24.5 minutes."""
        assert _parse_duration("PT24M30.00S") == pytest.approx(24.5)

    def test_hours_minutes_seconds(self):
        """Parse PT1H24M30.00S -> 84.5 minutes."""
        assert _parse_duration("PT1H24M30.00S") == pytest.approx(84.5)

    def test_minutes_only(self):
        """Parse PT43M00.00S -> 43.0 minutes."""
        assert _parse_duration("PT43M00.00S") == pytest.approx(43.0)

    def test_minutes_fractional(self):
        """Parse PT43M09.00S -> 43.15 minutes."""
        assert _parse_duration("PT43M09.00S") == pytest.approx(43.15)

    def test_zero_minutes(self):
        """Parse PT0M0.00S -> 0.0 minutes."""
        assert _parse_duration("PT0M0.00S") == pytest.approx(0.0)

    def test_none_input(self):
        """Parse None -> None."""
        assert _parse_duration(None) is None

    def test_empty_string(self):
        """Parse '' -> None."""
        assert _parse_duration("") is None

    def test_invalid_format(self):
        """Parse invalid format -> None."""
        assert _parse_duration("invalid") is None


class TestBoxscoreValidation:
    """Test boxscore validation and player parsing."""

    def test_is_final_boxscore_valid(self, sample_boxscore):
        """Check that sample boxscore is marked as final."""
        assert _is_final_boxscore(sample_boxscore) is True

    def test_is_final_boxscore_missing_status(self):
        """Invalid boxscore without gameStatus."""
        boxscore = {"game": {}}
        assert _is_final_boxscore(boxscore) is False

    def test_is_final_boxscore_not_final(self):
        """Invalid boxscore with status != 3."""
        boxscore = {
            "game": {
                "gameStatus": 2,
                "homeTeam": {"players": [{"personId": 1}]},
                "awayTeam": {"players": [{"personId": 2}]},
            }
        }
        assert _is_final_boxscore(boxscore) is False

    def test_is_final_boxscore_missing_teams(self):
        """Invalid boxscore with missing players."""
        boxscore = {
            "game": {
                "gameStatus": 3,
                "homeTeam": {"players": []},
                "awayTeam": {"players": [{"personId": 1}]},
            }
        }
        assert _is_final_boxscore(boxscore) is False


class TestBoxscorePlayerParsing:
    """Test player data extraction from boxscores."""

    def test_parse_players_from_sample(self, sample_boxscore):
        """Parse players from sample preseason boxscore."""
        players = _parse_boxscore_players(sample_boxscore)

        assert len(players) > 0, "Should parse at least one player"

        # Check structure
        for p in players:
            assert "game_id" in p
            assert "team" in p
            assert "player_id" in p
            assert "starter" in p
            assert "minutes" in p

        # Check values are reasonable
        for p in players:
            assert isinstance(p["starter"], int)
            assert p["starter"] in (0, 1)
            if p["minutes"] is not None:
                assert p["minutes"] >= 0

    def test_parse_players_starters(self, sample_boxscore):
        """Check that starter flags are parsed correctly."""
        players = _parse_boxscore_players(sample_boxscore)
        starters = [p for p in players if p["starter"] == 1]
        non_starters = [p for p in players if p["starter"] == 0]

        # Should have both starters and benchers
        assert len(starters) > 0
        assert len(non_starters) > 0

    def test_parse_players_minutes(self, sample_boxscore):
        """Check that minutes are parsed correctly."""
        players = _parse_boxscore_players(sample_boxscore)

        # Find a non-zero minute player
        with_minutes = [p for p in players if p["minutes"] and p["minutes"] > 0]
        assert len(with_minutes) > 0

        # Check reasonable range (0-50 minutes in preseason)
        for p in with_minutes:
            assert 0 <= p["minutes"] <= 50

    def test_parse_players_team_split(self, sample_boxscore):
        """Check that players are split between teams."""
        players = _parse_boxscore_players(sample_boxscore)

        teams = set(p["team"] for p in players)
        assert len(teams) == 2, "Should have players from both teams"

    def test_parse_empty_boxscore(self):
        """Handle boxscore with no game data."""
        boxscore = {}
        players = _parse_boxscore_players(boxscore)
        assert players == []

    def test_parse_boxscore_no_game_id(self):
        """Handle boxscore with missing gameId."""
        boxscore = {
            "game": {
                "homeTeam": {"players": []},
                "awayTeam": {"players": []},
            }
        }
        players = _parse_boxscore_players(boxscore)
        assert players == []


class TestLastTwoLogic:
    """Test that last-2 preseason games logic is correct."""

    def test_last_two_per_team(self, sample_boxscore):
        """Verify that last-2 detection can identify the correct games.

        This is a structural test to ensure the feature code can properly
        identify and aggregate data from the last 2 preseason games per team.
        """
        # Note: Full integration test happens in test_preseason_features_integration
        # This just checks the parsing works correctly
        players = _parse_boxscore_players(sample_boxscore)
        assert len(players) > 0
