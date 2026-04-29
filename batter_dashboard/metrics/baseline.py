"""Baseline metric computations for game and reference windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import IN_ZONE_VALUES, OUT_ZONE_VALUES, TEST_CASE
from data.loader import get_baseline_data, get_game_data


@dataclass(frozen=True)
class BaselineStats:
    """Aggregate hitter stats over a baseline window."""

    n_pitches: int
    n_swings: int
    n_takes: int
    n_contact: int
    n_whiffs: int
    n_bip: int
    chase_rate: float
    zone_swing_rate: float
    whiff_rate: float
    contact_rate: float
    avg_ev: float
    avg_la: float
    avg_xwoba_con: float
    avg_bat_speed: float
    avg_swing_length: float


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else float(np.nan)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float(np.nan)
    value = series.mean(skipna=True)
    return float(value) if pd.notna(value) else float(np.nan)


def compute_baseline_stats(pitches_df: pd.DataFrame) -> BaselineStats:
    """Compute aggregate stats from normalized pitch-level dataframe.

    Required input columns: is_swing, is_take, is_contact, is_whiff, zone,
    description, launch_speed, launch_angle, estimated_woba_using_speedangle,
    bat_speed, swing_length.
    """
    required = {
        "is_swing", "is_take", "is_contact", "is_whiff", "zone", "description", "launch_speed",
        "launch_angle", "estimated_woba_using_speedangle", "bat_speed", "swing_length",
    }
    _require_columns(pitches_df, required)

    n_pitches = int(len(pitches_df))
    n_swings = int(pitches_df["is_swing"].sum())
    n_takes = int(pitches_df["is_take"].sum())
    n_contact = int(pitches_df["is_contact"].sum())
    n_whiffs = int(pitches_df["is_whiff"].sum())
    bip = pitches_df[pitches_df["description"] == "hit_into_play"]
    n_bip = int(len(bip))

    in_zone = pitches_df["zone"].isin(IN_ZONE_VALUES)
    out_zone = pitches_df["zone"].isin(OUT_ZONE_VALUES)
    in_zone_n = int(in_zone.sum())
    out_zone_n = int(out_zone.sum())
    swings_in_zone = int((pitches_df["is_swing"] & in_zone).sum())
    swings_out_zone = int((pitches_df["is_swing"] & out_zone).sum())

    swing_mask = pitches_df["is_swing"].fillna(False).astype(bool)
    swings_df = pitches_df.loc[swing_mask]

    return BaselineStats(
        n_pitches=n_pitches,
        n_swings=n_swings,
        n_takes=n_takes,
        n_contact=n_contact,
        n_whiffs=n_whiffs,
        n_bip=n_bip,
        chase_rate=_safe_ratio(swings_out_zone, out_zone_n),
        zone_swing_rate=_safe_ratio(swings_in_zone, in_zone_n),
        whiff_rate=_safe_ratio(n_whiffs, n_swings),
        contact_rate=_safe_ratio(n_contact, n_swings),
        avg_ev=_safe_mean(bip["launch_speed"]),
        avg_la=_safe_mean(bip["launch_angle"]),
        avg_xwoba_con=_safe_mean(bip["estimated_woba_using_speedangle"]),
        avg_bat_speed=_safe_mean(swings_df["bat_speed"]),
        avg_swing_length=_safe_mean(swings_df["swing_length"]),
    )


def compute_game_vs_baseline_deltas(game_stats: BaselineStats, baseline_stats: BaselineStats) -> dict[str, float]:
    """Return game-minus-baseline deltas for selected rate/quality fields."""
    fields = [
        "chase_rate", "zone_swing_rate", "whiff_rate", "contact_rate", "avg_ev", "avg_la",
        "avg_xwoba_con", "avg_bat_speed", "avg_swing_length",
    ]
    out: dict[str, float] = {}
    game_dict = asdict(game_stats)
    base_dict = asdict(baseline_stats)
    for field in fields:
        bval = base_dict[field]
        out[f"{field}_delta"] = float(np.nan) if np.isnan(bval) else float(game_dict[field] - bval)
    return out


if __name__ == "__main__":
    game_df = get_game_data(TEST_CASE["player_id"], TEST_CASE["game_date"])
    baseline_df = get_baseline_data(TEST_CASE["player_id"], TEST_CASE["game_date"], "season")
    game_stats = compute_baseline_stats(game_df)
    baseline_stats = compute_baseline_stats(baseline_df)
    print(game_stats)
    print(compute_game_vs_baseline_deltas(game_stats, baseline_stats))
