"""Public data API that orchestrates raw sources and normalization."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd

from config import BAT_TRACKING_COLUMNS, LOG_FORMAT, LOG_LEVEL, TEST_CASE
from data.normalization import normalize_pitch_data
from data.pybaseball_source import fetch_batter_game, fetch_batter_range, fetch_pitcher_season
from data.savant_scraper import fetch_batter_pitches_savant

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def _season_start_for_date(iso_date: str) -> str:
    d = date.fromisoformat(iso_date)
    return f"{d.year}-03-01"


def get_game_data(player_id: int, game_date: str) -> pd.DataFrame:
    """Return normalized pitch-level dataframe for one batter, one game.

    Expected columns include normalized pitch columns and classification booleans.
    """
    return normalize_pitch_data(fetch_batter_game(player_id, game_date))


def get_baseline_data(player_id: int, end_date: str, window: Literal["season", "last_30", "career"]) -> pd.DataFrame:
    """Return normalized pitch-level dataframe for a baseline window ending before end_date.

    Expected columns include raw Statcast pitch columns plus classification booleans.
    """
    end_dt = date.fromisoformat(end_date)
    inclusive_end = (end_dt - timedelta(days=1)).isoformat()
    if window == "season":
        start_date = _season_start_for_date(end_date)
    elif window == "last_30":
        start_date = (end_dt - timedelta(days=30)).isoformat()
    elif window == "career":
        start_date = "2015-01-01"
    else:
        raise ValueError(f"Unsupported window: {window}")
    return normalize_pitch_data(fetch_batter_range(player_id, start_date, inclusive_end))


def _bat_tracking_population(df: pd.DataFrame) -> float:
    swings = df[df["description"].isin({"hit_into_play", "foul", "swinging_strike", "foul_tip", "swinging_strike_blocked", "foul_bunt"})]
    if swings.empty or "bat_speed" not in swings.columns:
        return 0.0
    return float(swings["bat_speed"].notna().mean())


def get_bat_tracking(player_id: int, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    """Return normalized dataframe and source_used string for bat tracking.

    Expected columns include raw Statcast pitch columns and bat tracking fields.
    """
    season = int(start_date.split("-")[0])
    py_df: pd.DataFrame | None = None
    try:
        py_df = fetch_batter_range(player_id, start_date, end_date)
        py_ratio = _bat_tracking_population(py_df)
        if py_ratio >= 0.5:
            logger.info("Bat tracking source used: pybaseball")
            return normalize_pitch_data(py_df), "pybaseball"
    except Exception:
        logger.exception("pybaseball bat tracking fetch failed; will attempt Savant fallback")

    try:
        sav_df = fetch_batter_pitches_savant(player_id, start_date, end_date, season)
        sav_ratio = _bat_tracking_population(sav_df)
        if sav_ratio >= 0.5:
            logger.info("Bat tracking source used: savant_scrape")
            return normalize_pitch_data(sav_df), "savant_scrape"
    except Exception:
        logger.exception("Savant fallback failed")

    fallback_df = py_df if py_df is not None else pd.DataFrame(columns=BAT_TRACKING_COLUMNS)
    logger.info("Bat tracking source used: unavailable")
    return normalize_pitch_data(fallback_df) if not fallback_df.empty else fallback_df, "unavailable"


def get_pitcher_attack_data(pitcher_id: int, current_season: int) -> dict[str, pd.DataFrame]:
    """Return {'current': df, 'prior': df} normalized pitcher dataframes.

    Expected columns include raw pitcher Statcast pitch columns plus classification booleans.
    """
    current = normalize_pitch_data(fetch_pitcher_season(pitcher_id, current_season))
    prior = normalize_pitch_data(fetch_pitcher_season(pitcher_id, current_season - 1))
    return {"current": current, "prior": prior}


if __name__ == "__main__":
    case = TEST_CASE
    game_df = get_game_data(case["player_id"], case["game_date"])
    print(f"Game normalized rows={len(game_df)} expected={case['expected_pitches']}")
    bt_df, source = get_bat_tracking(case["player_id"], case["game_date"], case["game_date"])
    print(f"Bat tracking source={source} rows={len(bt_df)}")
    baseline = get_baseline_data(case["player_id"], case["game_date"], "last_30")
    print(f"Baseline rows={len(baseline)} up_to<{case['game_date']}")
