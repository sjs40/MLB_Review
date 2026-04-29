"""Bat-tracking game vs baseline delta metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from MLB_Review.batter_dashboard.config import BAT_TRACKING_COLUMNS, TEST_CASE
from MLB_Review.batter_dashboard.data.loader import get_baseline_data, get_bat_tracking


@dataclass(frozen=True)
class BatTrackingDeltas:
    n_swings_game: int
    n_swings_baseline: int
    bat_speed_game: float
    bat_speed_baseline: float
    bat_speed_delta: float
    swing_length_game: float
    swing_length_baseline: float
    swing_length_delta: float
    attack_angle_game: float
    attack_angle_baseline: float
    attack_angle_delta: float
    swing_path_tilt_game: float
    swing_path_tilt_baseline: float
    swing_path_tilt_delta: float
    bat_tracking_coverage_game: float
    bat_tracking_coverage_baseline: float


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _swing_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_swing"]].copy()


def _safe_mean(series: pd.Series) -> float:
    value = series.mean(skipna=True)
    return float(value) if pd.notna(value) else float(np.nan)


def _coverage(swings_df: pd.DataFrame) -> float:
    return float(swings_df["bat_speed"].notna().mean()) if not swings_df.empty else float(np.nan)


def compute_bat_tracking_deltas(game_df: pd.DataFrame, baseline_df: pd.DataFrame) -> BatTrackingDeltas:
    """Compute bat-tracking mean deltas on swings.

    Required input columns (both): is_swing plus config.BAT_TRACKING_COLUMNS.
    """
    required = {"is_swing", *BAT_TRACKING_COLUMNS}
    _require_columns(game_df, required)
    _require_columns(baseline_df, required)

    gs = _swing_subset(game_df)
    bs = _swing_subset(baseline_df)
    n_game, n_base = int(len(gs)), int(len(bs))

    def means(df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {"bat_speed": float(np.nan), "swing_length": float(np.nan), "attack_angle": float(np.nan), "swing_path_tilt": float(np.nan)}
        return {
            "bat_speed": _safe_mean(df["bat_speed"]),
            "swing_length": _safe_mean(df["swing_length"]),
            "attack_angle": _safe_mean(df["attack_angle"]),
            "swing_path_tilt": _safe_mean(df["swing_path_tilt"]),
        }

    gm, bm = means(gs), means(bs)
    return BatTrackingDeltas(
        n_swings_game=n_game,
        n_swings_baseline=n_base,
        bat_speed_game=gm["bat_speed"],
        bat_speed_baseline=bm["bat_speed"],
        bat_speed_delta=float(gm["bat_speed"] - bm["bat_speed"]) if pd.notna(gm["bat_speed"]) and pd.notna(bm["bat_speed"]) else float(np.nan),
        swing_length_game=gm["swing_length"],
        swing_length_baseline=bm["swing_length"],
        swing_length_delta=float(gm["swing_length"] - bm["swing_length"]) if pd.notna(gm["swing_length"]) and pd.notna(bm["swing_length"]) else float(np.nan),
        attack_angle_game=gm["attack_angle"],
        attack_angle_baseline=bm["attack_angle"],
        attack_angle_delta=float(gm["attack_angle"] - bm["attack_angle"]) if pd.notna(gm["attack_angle"]) and pd.notna(bm["attack_angle"]) else float(np.nan),
        swing_path_tilt_game=gm["swing_path_tilt"],
        swing_path_tilt_baseline=bm["swing_path_tilt"],
        swing_path_tilt_delta=float(gm["swing_path_tilt"] - bm["swing_path_tilt"]) if pd.notna(gm["swing_path_tilt"]) and pd.notna(bm["swing_path_tilt"]) else float(np.nan),
        bat_tracking_coverage_game=_coverage(gs),
        bat_tracking_coverage_baseline=_coverage(bs),
    )


if __name__ == "__main__":
    case = TEST_CASE
    game_df, _ = get_bat_tracking(case["player_id"], case["game_date"], case["game_date"])
    baseline_df = get_baseline_data(case["player_id"], case["game_date"], "season")
    print(compute_bat_tracking_deltas(game_df, baseline_df))
