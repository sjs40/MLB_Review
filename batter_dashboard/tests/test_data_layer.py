"""Data-layer tests for scaffold prompt."""

import pandas as pd
import pytest

from config import CONTACT_DESCRIPTIONS, SWING_DESCRIPTIONS, TEST_CASE, WHIFF_DESCRIPTIONS
from data.loader import get_baseline_data, get_bat_tracking, get_game_data
from data.normalization import add_swing_take_classification, dedupe_pitches, filter_non_pitches
from data.run_value_matrix import get_run_value_matrix


def test_get_game_data_returns_expected_pitches() -> None:
    df = get_game_data(TEST_CASE["player_id"], TEST_CASE["game_date"])
    assert len(df) == TEST_CASE["expected_pitches"]


def test_dedupe_removes_duplicates() -> None:
    df = pd.DataFrame.from_records([
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1, "pitch_type": "FF", "description": "ball"},
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1, "pitch_type": "FF", "description": "ball"},
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 2, "pitch_type": "SL", "description": "foul"},
    ])
    assert len(dedupe_pitches(df)) == 2


def test_filter_non_pitches_removes_nulls() -> None:
    df = pd.DataFrame.from_records([
        {"pitch_type": "FF", "description": "ball"},
        {"pitch_type": None, "description": "ball"},
    ])
    assert len(filter_non_pitches(df)) == 1


def test_swing_take_classification_correct() -> None:
    records = [{"description": d} for d in sorted(SWING_DESCRIPTIONS | {"ball", "intent_ball"})]
    df = add_swing_take_classification(pd.DataFrame.from_records(records))
    for desc in SWING_DESCRIPTIONS:
        row = df[df["description"] == desc].iloc[0]
        assert bool(row["is_swing"])
        assert bool(row["is_contact"]) == (desc in CONTACT_DESCRIPTIONS)
        assert bool(row["is_whiff"]) == (desc in WHIFF_DESCRIPTIONS)
    intent = df[df["description"] == "intent_ball"].iloc[0]
    assert not bool(intent["is_take"])


def test_bat_tracking_source_logged() -> None:
    _, source = get_bat_tracking(TEST_CASE["player_id"], TEST_CASE["game_date"], TEST_CASE["game_date"])
    assert isinstance(source, str) and bool(source.strip())


def test_run_value_matrix_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="delta_run_exp"):
        get_run_value_matrix(2025)


def test_baseline_excludes_target_date() -> None:
    df = get_baseline_data(TEST_CASE["player_id"], TEST_CASE["game_date"], "last_30")
    assert TEST_CASE["game_date"] not in set(df.get("game_date", pd.Series(dtype=str)).astype(str).tolist())
