"""Pitcher attack pattern metrics by count bucket."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from MLB_Review.batter_dashboard.config import (
    ATTACK_PATTERN_MIN_PITCHES_1_SEASON,
    ATTACK_PATTERN_MIN_PITCHES_2_SEASONS,
    COUNT_BUCKETS,
    TEST_CASE,
)
from MLB_Review.batter_dashboard.data.loader import get_game_data, get_pitcher_attack_data


@dataclass(frozen=True)
class CountBucketAttack:
    balls: int
    strikes: int
    n_pitches_in_game: int
    pitch_mix_in_game: dict[str, float]
    n_pitches_in_baseline: int
    baseline_window: str
    pitch_mix_baseline: dict[str, float]
    deviation_score: float


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _mix(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    counts = df["pitch_type"].value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def _l1_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def compute_attack_pattern(
    game_df: pd.DataFrame,
    pitcher_current_season_df: pd.DataFrame,
    pitcher_prior_season_df: pd.DataFrame,
) -> list[CountBucketAttack]:
    """Compute per-count game vs baseline pitch-mix attack pattern.

    Required input columns (all dfs): pitch_type, balls, strikes.
    """
    required = {"pitch_type", "balls", "strikes"}
    _require_columns(game_df, required)
    _require_columns(pitcher_current_season_df, required)
    _require_columns(pitcher_prior_season_df, required)

    out: list[CountBucketAttack] = []
    for balls, strikes in COUNT_BUCKETS:
        g = game_df[(game_df["balls"] == balls) & (game_df["strikes"] == strikes)]
        if g.empty:
            continue
        c = pitcher_current_season_df[(pitcher_current_season_df["balls"] == balls) & (pitcher_current_season_df["strikes"] == strikes)]
        p = pitcher_prior_season_df[(pitcher_prior_season_df["balls"] == balls) & (pitcher_prior_season_df["strikes"] == strikes)]
        c_n, p_n = len(c), len(p)
        if c_n >= ATTACK_PATTERN_MIN_PITCHES_1_SEASON:
            baseline = c
            baseline_window = "1_season"
        elif (c_n + p_n) >= ATTACK_PATTERN_MIN_PITCHES_2_SEASONS:
            baseline = pd.concat([c, p], ignore_index=True)
            baseline_window = "2_seasons"
        else:
            baseline = pd.DataFrame(columns=g.columns)
            baseline_window = "insufficient_data"

        gm = _mix(g)
        bm = _mix(baseline)
        dev = _l1_distance(gm, bm) if baseline_window != "insufficient_data" else float(np.nan)
        out.append(
            CountBucketAttack(
                balls=balls,
                strikes=strikes,
                n_pitches_in_game=int(len(g)),
                pitch_mix_in_game=gm,
                n_pitches_in_baseline=int(len(baseline)),
                baseline_window=baseline_window,
                pitch_mix_baseline=bm,
                deviation_score=dev,
            )
        )
    return out


if __name__ == "__main__":
    case = TEST_CASE
    game_df = get_game_data(case["player_id"], case["game_date"])
    pitcher_id = int(game_df["pitcher"].iloc[0])
    season = int(case["game_date"].split("-")[0])
    attack = get_pitcher_attack_data(pitcher_id, season)
    print(compute_attack_pattern(game_df, attack["current"], attack["prior"]))
