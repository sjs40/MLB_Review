"""Contact quality extraction and summarization metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from MLB_Review.batter_dashboard.config import TEST_CASE
from MLB_Review.batter_dashboard.data.loader import get_game_data


@dataclass(frozen=True)
class BipResult:
    pitch_idx: int
    at_bat_number: int
    inning: int
    pitch_type: str
    launch_speed: float
    launch_angle: float
    xwoba: float
    xba: float
    actual_event: str
    actual_woba: float


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def extract_bip_results(game_df: pd.DataFrame) -> list[BipResult]:
    """Extract one BipResult per ball in play.

    Required input columns: description, events, at_bat_number, inning,
    pitch_type, launch_speed, launch_angle, estimated_woba_using_speedangle,
    estimated_ba_using_speedangle, woba_value.
    """
    required = {
        "description", "events", "at_bat_number", "inning", "pitch_type", "launch_speed",
        "launch_angle", "estimated_woba_using_speedangle", "estimated_ba_using_speedangle", "woba_value",
    }
    _require_columns(game_df, required)
    bip_df = game_df[game_df["description"] == "hit_into_play"]
    results: list[BipResult] = []
    for idx, row in bip_df.iterrows():
        results.append(
            BipResult(
                pitch_idx=int(idx),
                at_bat_number=int(row["at_bat_number"]),
                inning=int(row["inning"]),
                pitch_type=str(row["pitch_type"]),
                launch_speed=float(row["launch_speed"]),
                launch_angle=float(row["launch_angle"]),
                xwoba=float(row["estimated_woba_using_speedangle"]),
                xba=float(row["estimated_ba_using_speedangle"]),
                actual_event=str(row["events"]),
                actual_woba=float(row["woba_value"]) if pd.notna(row["woba_value"]) else 0.0,
            )
        )
    return results


@dataclass(frozen=True)
class ContactQualitySummary:
    n_bip: int
    sum_actual_woba: float
    sum_xwoba: float
    woba_above_expected: float
    avg_actual_woba: float
    avg_xwoba: float


def summarize_contact_quality(bip_results: list[BipResult]) -> ContactQualitySummary:
    """Aggregate BIP results into game-level summary."""
    if not bip_results:
        return ContactQualitySummary(0, 0.0, 0.0, 0.0, float(np.nan), float(np.nan))
    actual = np.array([r.actual_woba for r in bip_results], dtype=float)
    xwoba = np.array([r.xwoba for r in bip_results], dtype=float)
    sum_actual = float(actual.sum())
    sum_xwoba = float(xwoba.sum())
    return ContactQualitySummary(
        n_bip=len(bip_results),
        sum_actual_woba=sum_actual,
        sum_xwoba=sum_xwoba,
        woba_above_expected=float(sum_actual - sum_xwoba),
        avg_actual_woba=float(actual.mean()),
        avg_xwoba=float(xwoba.mean()),
    )


if __name__ == "__main__":
    game_df = get_game_data(TEST_CASE["player_id"], TEST_CASE["game_date"])
    bips = extract_bip_results(game_df)
    print(summarize_contact_quality(bips))
