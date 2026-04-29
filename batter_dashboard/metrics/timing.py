"""Timing diagnostics based on attack_direction by pitch category."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from MLB_Review.batter_dashboard.config import PITCH_CATEGORY_MAP, PULL_THRESHOLD_DEGREES, TEST_CASE
from MLB_Review.batter_dashboard.data.loader import get_baseline_data, get_game_data

PitchCategory = Literal["fastball", "breaking", "offspeed"]


@dataclass(frozen=True)
class TimingDiagnostics:
    category: PitchCategory
    n_contact_game: int
    n_contact_baseline: int
    mean_attack_direction_game: float
    mean_attack_direction_baseline: float
    attack_direction_delta: float
    pull_rate_game: float
    pull_rate_baseline: float


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def categorize_pitch_type(pitch_type: str) -> PitchCategory | None:
    """Map Statcast pitch_type abbreviations to pitch category."""
    cat = PITCH_CATEGORY_MAP.get(pitch_type)
    return cat if cat in {"fastball", "breaking", "offspeed"} else None


def _normalize_attack_direction(df: pd.DataFrame) -> pd.Series:
    adjusted = df["attack_direction"].astype(float).copy()
    left_mask = df["stand"] == "L"
    adjusted[left_mask] = -adjusted[left_mask]
    return adjusted


def _category_stats(df: pd.DataFrame, category: PitchCategory) -> tuple[int, float, float]:
    cdf = df[df["pitch_category"] == category]
    n = int(len(cdf))
    if n == 0:
        return 0, float(np.nan), float(np.nan)
    mean_dir = float(cdf["attack_direction_adj"].mean())
    pull_rate = float((cdf["attack_direction_adj"] < PULL_THRESHOLD_DEGREES).mean())
    return n, mean_dir, pull_rate


def compute_timing_diagnostics(game_df: pd.DataFrame, baseline_df: pd.DataFrame) -> list[TimingDiagnostics]:
    """Compute timing diagnostics by pitch category.

    Required input columns: pitch_type, is_contact, attack_direction, stand.
    """
    required = {"pitch_type", "is_contact", "attack_direction", "stand"}
    _require_columns(game_df, required)
    _require_columns(baseline_df, required)

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df[(df["is_contact"]) & (df["attack_direction"].notna())].copy()
        out["pitch_category"] = out["pitch_type"].map(categorize_pitch_type)
        out = out[out["pitch_category"].notna()].copy()
        out["attack_direction_adj"] = _normalize_attack_direction(out)
        return out

    game_p = prep(game_df)
    base_p = prep(baseline_df)
    results: list[TimingDiagnostics] = []
    for category in ["fastball", "breaking", "offspeed"]:
        ng, mg, pg = _category_stats(game_p, category)
        nb, mb, pb = _category_stats(base_p, category)
        delta = float(mg - mb) if pd.notna(mg) and pd.notna(mb) else float(np.nan)
        results.append(TimingDiagnostics(category, ng, nb, mg, mb, delta, pg, pb))
    return results


if __name__ == "__main__":
    case = TEST_CASE
    game_df = get_game_data(case["player_id"], case["game_date"])
    baseline_df = get_baseline_data(case["player_id"], case["game_date"], "season")
    print(compute_timing_diagnostics(game_df, baseline_df))
