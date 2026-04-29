"""Direct Baseball Savant CSV scraper used as bat-tracking fallback."""

from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests

from MLB_Review.batter_dashboard.config import LOG_FORMAT, LOG_LEVEL, TEST_CASE

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def fetch_batter_pitches_savant(player_id: int, start_date: str, end_date: str, season: int) -> pd.DataFrame:
    """Direct CSV scrape from Baseball Savant for batter pitches in date range.

    Expected columns: same schema as pybaseball.statcast_batter raw output.
    Sends User-Agent: Mozilla/5.0 header. Logs response time.
    Raises requests.HTTPError on non-200 responses.
    """
    params = {
        "all": "true",
        "hfSea": f"{season}|",
        "player_type": "batter",
        "batters_lookup[]": str(player_id),
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "type": "details",
        "min_pitches": "0",
        "min_results": "0",
        "min_pas": "0",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    start = time.perf_counter()
    response = requests.get(SAVANT_CSV_URL, params=params, headers=headers, timeout=30)
    elapsed = time.perf_counter() - start
    logger.info("Savant response time %.3fs for player_id=%s", elapsed, player_id)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


if __name__ == "__main__":
    case = TEST_CASE
    season = int(case["game_date"].split("-")[0])
    df = fetch_batter_pitches_savant(case["player_id"], case["game_date"], case["game_date"], season)
    print(f"smoke: savant rows={len(df)}")
