"""Decision quality metrics for swing/take choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config import (
    DECISION_EXCLUDED_DESCRIPTIONS,
    DEFAULT_DECISION_QUALITY_METHOD,
    IN_ZONE_VALUES,
    OUT_ZONE_VALUES,
    TEST_CASE,
)
from data.loader import get_game_data


@dataclass(frozen=True)
class DecisionQualityResult:
    pitch_idx: int
    at_bat_number: int
    pitch_number: int
    balls: int
    strikes: int
    zone: int
    description: str
    is_swing: bool
    rv_chosen: float
    correct: bool | None


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def compute_decision_quality(
    game_df: pd.DataFrame,
    method: Literal["delta_run_exp", "rv_matrix"] | None = None,
    rv_matrix: pd.DataFrame | None = None,
) -> list[DecisionQualityResult]:
    """Compute per-pitch decision quality for every eligible pitch in game_df.

    Required input columns: at_bat_number, pitch_number, balls, strikes, zone,
    description, is_swing, delta_run_exp.
    """
    required = {"at_bat_number", "pitch_number", "balls", "strikes", "zone", "description", "is_swing", "delta_run_exp"}
    _require_columns(game_df, required)
    use_method = method or DEFAULT_DECISION_QUALITY_METHOD
    if use_method == "rv_matrix" and rv_matrix is None:
        raise ValueError("rv_matrix method requires rv_matrix dataframe")

    matrix_lookup: dict[tuple[int, int, int, str], float] = {}
    if rv_matrix is not None:
        needed = {"balls", "strikes", "zone", "swing_or_take", "mean_delta_run_exp"}
        _require_columns(rv_matrix, needed)
        for _, mrow in rv_matrix.iterrows():
            matrix_lookup[(int(mrow["balls"]), int(mrow["strikes"]), int(mrow["zone"]), str(mrow["swing_or_take"]))] = float(mrow["mean_delta_run_exp"])

    results: list[DecisionQualityResult] = []
    filtered = game_df[~game_df["description"].isin(DECISION_EXCLUDED_DESCRIPTIONS)]
    for idx, row in filtered.iterrows():
        zone_val = row["zone"]
        zone = int(zone_val) if pd.notna(zone_val) else -1
        is_swing = bool(row["is_swing"])
        if use_method == "delta_run_exp":
            rv = float(row["delta_run_exp"])
            correct: bool | None = None
        else:
            if pd.isna(zone_val):
                rv = float(np.nan)
                correct = None
            else:
                swing_rv = matrix_lookup.get((int(row["balls"]), int(row["strikes"]), int(zone_val), "swing"))
                take_rv = matrix_lookup.get((int(row["balls"]), int(row["strikes"]), int(zone_val), "take"))
                chosen = "swing" if is_swing else "take"
                chosen_rv = matrix_lookup.get((int(row["balls"]), int(row["strikes"]), int(zone_val), chosen))
                if swing_rv is None or take_rv is None or chosen_rv is None:
                    rv = float(np.nan)
                    correct = None
                else:
                    rv = float(chosen_rv)
                    correct = bool((is_swing and swing_rv >= take_rv) or ((not is_swing) and take_rv >= swing_rv))
        results.append(DecisionQualityResult(int(idx), int(row["at_bat_number"]), int(row["pitch_number"]), int(row["balls"]), int(row["strikes"]), zone, str(row["description"]), is_swing, rv, correct))
    return results


@dataclass(frozen=True)
class DecisionQualitySummary:
    n_decisions: int
    sum_rv: float
    avg_rv: float
    n_correct: int | None
    n_incorrect: int | None
    correct_rate: float | None
    chase_rate_in_game: float
    zone_swing_rate_in_game: float


def summarize_decision_quality(results: list[DecisionQualityResult], game_df: pd.DataFrame) -> DecisionQualitySummary:
    """Aggregate decision-quality per-pitch results to game-level summary.

    Required columns in game_df: zone, is_swing.
    """
    _require_columns(game_df, {"zone", "is_swing"})
    n = len(results)
    rv_values = np.array([r.rv_chosen for r in results], dtype=float) if results else np.array([], dtype=float)
    sum_rv = float(np.nansum(rv_values))
    avg_rv = float(np.nanmean(rv_values)) if rv_values.size else float(np.nan)

    any_correct = any(r.correct is not None for r in results)
    if any_correct:
        n_correct = sum(1 for r in results if r.correct is True)
        n_incorrect = sum(1 for r in results if r.correct is False)
        denom = n_correct + n_incorrect
        correct_rate = float(n_correct / denom) if denom else float(np.nan)
    else:
        n_correct = None
        n_incorrect = None
        correct_rate = None

    in_zone = game_df["zone"].isin(IN_ZONE_VALUES)
    out_zone = game_df["zone"].isin(OUT_ZONE_VALUES)
    chase = float((game_df["is_swing"] & out_zone).sum() / out_zone.sum()) if int(out_zone.sum()) else float(np.nan)
    zone_swing = float((game_df["is_swing"] & in_zone).sum() / in_zone.sum()) if int(in_zone.sum()) else float(np.nan)

    return DecisionQualitySummary(n, sum_rv, avg_rv, n_correct, n_incorrect, correct_rate, chase, zone_swing)


if __name__ == "__main__":
    game_df = get_game_data(TEST_CASE["player_id"], TEST_CASE["game_date"])
    res = compute_decision_quality(game_df)
    print(summarize_decision_quality(res, game_df))
