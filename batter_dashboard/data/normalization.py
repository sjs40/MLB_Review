"""Normalization pipeline for raw pitch-level Statcast data."""

from __future__ import annotations

import logging

import pandas as pd

from MLB_Review.batter_dashboard.config import (
    CONTACT_DESCRIPTIONS,
    DECISION_EXCLUDED_DESCRIPTIONS,
    LOG_FORMAT,
    LOG_LEVEL,
    SWING_DESCRIPTIONS,
    TEST_CASE,
    WHIFF_DESCRIPTIONS,
)

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def dedupe_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate pitches on (game_pk, at_bat_number, pitch_number).

    Expected columns: game_pk, at_bat_number, pitch_number.
    """
    before = len(df)
    out = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="first")
    logger.info("dedupe removed=%s", before - len(out))
    return out


def filter_non_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null pitch_type.

    Expected columns: pitch_type.
    """
    before = len(df)
    out = df[df["pitch_type"].notna()].copy()
    logger.info("filter_non_pitches removed=%s", before - len(out))
    return out


def add_swing_take_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Add swing/take classification booleans.

    Expected columns: description plus added columns is_swing, is_contact, is_whiff, is_take.
    """
    out = df.copy()
    out["is_swing"] = out["description"].isin(SWING_DESCRIPTIONS)
    out["is_contact"] = out["description"].isin(CONTACT_DESCRIPTIONS)
    out["is_whiff"] = out["description"].isin(WHIFF_DESCRIPTIONS)
    out["is_take"] = (~out["is_swing"]) & (~out["description"].isin(DECISION_EXCLUDED_DESCRIPTIONS))
    return out


def normalize_pitch_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full normalization pipeline: dedupe → filter → classify.

    Expected columns: raw Statcast columns including game_pk, at_bat_number,
    pitch_number, pitch_type, description; returns those + classification columns.
    """
    return add_swing_take_classification(filter_non_pitches(dedupe_pitches(df)))


if __name__ == "__main__":
    from MLB_Review.batter_dashboard.data.pybaseball_source import fetch_batter_game

    case = TEST_CASE
    raw = fetch_batter_game(case["player_id"], case["game_date"], use_cache=True)
    norm = normalize_pitch_data(raw)
    print(f"smoke: normalized rows={len(norm)}")
