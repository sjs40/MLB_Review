"""Raw pybaseball source wrappers with deterministic parquet caching."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import cache
from pathlib import Path

import pandas as pd
from pybaseball import statcast_batter, statcast_pitcher

from config import CACHE_DIR, LOG_FORMAT, LOG_LEVEL, TEST_CASE

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def _cache_path(name: str) -> Path:
    """Return parquet cache path for a cache key."""
    return CACHE_DIR / f"{name}.parquet"


def _read_or_fetch(cache_name: str, fetch_fn: Callable[[], pd.DataFrame], use_cache: bool = True) -> pd.DataFrame:
    """Read cached parquet or fetch and persist.

    Expected columns: raw Statcast schema from pybaseball source functions.
    """
    path = _cache_path(cache_name)
    if use_cache and path.exists():
        logger.info("Cache hit: %s", path.name)
        return pd.read_parquet(path)
    logger.info("Cache miss: %s — fetching", path.name)
    df = fetch_fn()
    df.to_parquet(path)
    return df


@cache
def _season_bounds(season: int) -> tuple[str, str]:
    """Return standard regular-season bounds used for pitcher season pulls."""
    return (f"{season}-03-01", f"{season}-10-31")


def fetch_batter_game(player_id: int, game_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Raw statcast_batter pull for a single game date.

    Expected columns: raw pybaseball statcast_batter schema.
    Cache key: cache/batter_{player_id}_{game_date}.parquet
    """
    cache_name = f"batter_{player_id}_{game_date}"
    return _read_or_fetch(
        cache_name,
        lambda: statcast_batter(game_date, game_date, player_id),
        use_cache=use_cache,
    )


def fetch_batter_range(player_id: int, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Raw statcast_batter pull for a date range.

    Expected columns: raw pybaseball statcast_batter schema.
    Cache key: cache/batter_{player_id}_{start_date}_{end_date}.parquet
    """
    cache_name = f"batter_{player_id}_{start_date}_{end_date}"
    return _read_or_fetch(
        cache_name,
        lambda: statcast_batter(start_date, end_date, player_id),
        use_cache=use_cache,
    )


def fetch_pitcher_season(pitcher_id: int, season: int, use_cache: bool = True) -> pd.DataFrame:
    """Raw statcast_pitcher pull for a full regular season.

    Expected columns: raw pybaseball statcast_pitcher schema.
    Cache key: cache/pitcher_{pitcher_id}_{season}.parquet
    Date range is hardcoded as standard MLB regular season (Mar-Oct).
    """
    start_date, end_date = _season_bounds(season)
    cache_name = f"pitcher_{pitcher_id}_{season}"
    return _read_or_fetch(
        cache_name,
        lambda: statcast_pitcher(start_date, end_date, pitcher_id),
        use_cache=use_cache,
    )


if __name__ == "__main__":
    case = TEST_CASE
    result = fetch_batter_game(case["player_id"], case["game_date"], use_cache=True)
    print(f"smoke: raw batter game rows={len(result)}")
