"""Metric-layer tests for Prompt 2 modules."""

import numpy as np
import pandas as pd
import pytest

from config import DECISION_EXCLUDED_DESCRIPTIONS
from metrics.attack_pattern import compute_attack_pattern
from metrics.baseline import (
    BaselineStats,
    compute_baseline_stats,
    compute_game_vs_baseline_deltas,
)
from metrics.bat_tracking import compute_bat_tracking_deltas
from metrics.contact_quality import (
    extract_bip_results,
    summarize_contact_quality,
)
from metrics.decision_quality import (
    compute_decision_quality,
    summarize_decision_quality,
)
from metrics.timing import categorize_pitch_type, compute_timing_diagnostics


# === baseline.py tests ===
def test_baseline_compute_basic() -> None:
    df = pd.DataFrame.from_records([
        {"is_swing": True, "is_take": False, "is_contact": True, "is_whiff": False, "zone": 1, "description": "hit_into_play", "launch_speed": 100.0, "launch_angle": 20.0, "estimated_woba_using_speedangle": 0.7, "bat_speed": 74.0, "swing_length": 7.0},
        {"is_swing": True, "is_take": False, "is_contact": False, "is_whiff": True, "zone": 12, "description": "swinging_strike", "launch_speed": np.nan, "launch_angle": np.nan, "estimated_woba_using_speedangle": np.nan, "bat_speed": 72.0, "swing_length": 7.2},
        {"is_swing": False, "is_take": True, "is_contact": False, "is_whiff": False, "zone": 12, "description": "ball", "launch_speed": np.nan, "launch_angle": np.nan, "estimated_woba_using_speedangle": np.nan, "bat_speed": np.nan, "swing_length": np.nan},
        {"is_swing": False, "is_take": True, "is_contact": False, "is_whiff": False, "zone": 2, "description": "called_strike", "launch_speed": np.nan, "launch_angle": np.nan, "estimated_woba_using_speedangle": np.nan, "bat_speed": np.nan, "swing_length": np.nan},
    ])
    stats = compute_baseline_stats(df)
    assert stats.n_pitches == 4
    assert stats.chase_rate == pytest.approx(0.5)
    assert stats.zone_swing_rate == pytest.approx(0.5)
    assert stats.whiff_rate == pytest.approx(0.5)


def test_baseline_empty() -> None:
    df = pd.DataFrame(columns=["is_swing", "is_take", "is_contact", "is_whiff", "zone", "description", "launch_speed", "launch_angle", "estimated_woba_using_speedangle", "bat_speed", "swing_length"])
    stats = compute_baseline_stats(df)
    assert stats.n_pitches == 0
    assert np.isnan(stats.whiff_rate)


def test_baseline_all_takes() -> None:
    df = pd.DataFrame.from_records([
        {"is_swing": False, "is_take": True, "is_contact": False, "is_whiff": False, "zone": 1, "description": "ball", "launch_speed": np.nan, "launch_angle": np.nan, "estimated_woba_using_speedangle": np.nan, "bat_speed": np.nan, "swing_length": np.nan}
    ])
    stats = compute_baseline_stats(df)
    assert np.isnan(stats.whiff_rate)
    assert np.isnan(stats.contact_rate)


def test_baseline_deltas() -> None:
    g = BaselineStats(1, 1, 0, 1, 0, 1, 0.4, 0.5, 0.2, 0.8, 90.0, 15.0, 0.4, 75.0, 7.0)
    b = BaselineStats(1, 1, 0, 1, 0, 1, 0.3, 0.4, 0.1, 0.7, 88.0, 10.0, 0.3, 73.0, 6.5)
    d = compute_game_vs_baseline_deltas(g, b)
    assert d["chase_rate_delta"] == pytest.approx(0.1)
    assert d["avg_ev_delta"] == pytest.approx(2.0)


# === contact_quality.py tests ===
def test_contact_quality_basic() -> None:
    df = pd.DataFrame.from_records([
        {"description": "hit_into_play", "events": "single", "at_bat_number": 1, "inning": 1, "pitch_type": "FF", "launch_speed": 100, "launch_angle": 15, "estimated_woba_using_speedangle": 0.6, "estimated_ba_using_speedangle": 0.7, "woba_value": 0.9},
        {"description": "hit_into_play", "events": "field_out", "at_bat_number": 2, "inning": 2, "pitch_type": "SL", "launch_speed": 90, "launch_angle": 5, "estimated_woba_using_speedangle": 0.4, "estimated_ba_using_speedangle": 0.3, "woba_value": 0.0},
        {"description": "hit_into_play", "events": "double", "at_bat_number": 3, "inning": 3, "pitch_type": "CH", "launch_speed": 105, "launch_angle": 25, "estimated_woba_using_speedangle": 0.8, "estimated_ba_using_speedangle": 0.8, "woba_value": 1.3},
        {"description": "foul", "events": None, "at_bat_number": 4, "inning": 4, "pitch_type": "FF", "launch_speed": np.nan, "launch_angle": np.nan, "estimated_woba_using_speedangle": np.nan, "estimated_ba_using_speedangle": np.nan, "woba_value": np.nan},
    ])
    bips = extract_bip_results(df)
    assert len(bips) == 3
    summary = summarize_contact_quality(bips)
    assert summary.sum_actual_woba == pytest.approx(2.2)
    assert summary.sum_xwoba == pytest.approx(1.8)


def test_contact_quality_no_bip() -> None:
    df = pd.DataFrame.from_records([{"description": "foul", "events": None, "at_bat_number": 1, "inning": 1, "pitch_type": "FF", "launch_speed": np.nan, "launch_angle": np.nan, "estimated_woba_using_speedangle": np.nan, "estimated_ba_using_speedangle": np.nan, "woba_value": np.nan}])
    summary = summarize_contact_quality(extract_bip_results(df))
    assert summary.n_bip == 0
    assert np.isnan(summary.avg_xwoba)


# === decision_quality.py tests ===
def test_decision_quality_delta_run_exp() -> None:
    df = pd.DataFrame.from_records([
        {"at_bat_number": 1, "pitch_number": 1, "balls": 0, "strikes": 0, "zone": 1, "description": "called_strike", "is_swing": False, "delta_run_exp": -0.02},
        {"at_bat_number": 1, "pitch_number": 2, "balls": 0, "strikes": 1, "zone": 12, "description": "swinging_strike", "is_swing": True, "delta_run_exp": -0.03},
        {"at_bat_number": 1, "pitch_number": 3, "balls": 0, "strikes": 2, "zone": 12, "description": "intent_ball", "is_swing": False, "delta_run_exp": 0.01},
        {"at_bat_number": 2, "pitch_number": 1, "balls": 1, "strikes": 0, "zone": 2, "description": "foul", "is_swing": True, "delta_run_exp": -0.01},
        {"at_bat_number": 2, "pitch_number": 2, "balls": 1, "strikes": 1, "zone": 11, "description": "ball", "is_swing": False, "delta_run_exp": 0.02},
    ])
    results = compute_decision_quality(df, method="delta_run_exp")
    assert all(r.description not in DECISION_EXCLUDED_DESCRIPTIONS for r in results)
    summary = summarize_decision_quality(results, df)
    assert summary.n_decisions == 4
    assert summary.sum_rv == pytest.approx(-0.04)


def test_decision_quality_rv_matrix() -> None:
    df = pd.DataFrame.from_records([
        {"at_bat_number": 1, "pitch_number": 1, "balls": 0, "strikes": 0, "zone": 1, "description": "foul", "is_swing": True, "delta_run_exp": -0.01},
        {"at_bat_number": 1, "pitch_number": 2, "balls": 0, "strikes": 1, "zone": 12, "description": "ball", "is_swing": False, "delta_run_exp": 0.01},
    ])
    matrix = pd.DataFrame.from_records([
        {"balls": 0, "strikes": 0, "zone": 1, "swing_or_take": "swing", "mean_delta_run_exp": 0.1},
        {"balls": 0, "strikes": 0, "zone": 1, "swing_or_take": "take", "mean_delta_run_exp": 0.0},
        {"balls": 0, "strikes": 1, "zone": 12, "swing_or_take": "swing", "mean_delta_run_exp": -0.2},
        {"balls": 0, "strikes": 1, "zone": 12, "swing_or_take": "take", "mean_delta_run_exp": 0.1},
    ])
    results = compute_decision_quality(df, method="rv_matrix", rv_matrix=matrix)
    assert results[0].correct is True
    assert results[1].correct is True


def test_decision_quality_missing_matrix_raises() -> None:
    df = pd.DataFrame.from_records([{"at_bat_number": 1, "pitch_number": 1, "balls": 0, "strikes": 0, "zone": 1, "description": "ball", "is_swing": False, "delta_run_exp": 0.0}])
    with pytest.raises(ValueError):
        compute_decision_quality(df, method="rv_matrix", rv_matrix=None)


def test_decision_quality_empty() -> None:
    df = pd.DataFrame(columns=["at_bat_number", "pitch_number", "balls", "strikes", "zone", "description", "is_swing", "delta_run_exp"])
    results = compute_decision_quality(df)
    summary = summarize_decision_quality(results, pd.DataFrame(columns=["zone", "is_swing"]))
    assert summary.n_decisions == 0


# === bat_tracking.py tests ===
def test_bat_tracking_deltas_basic() -> None:
    g = pd.DataFrame.from_records([
        {"is_swing": True, "bat_speed": 70.0, "swing_length": 7.0, "attack_angle": 10.0, "attack_direction": 5.0, "swing_path_tilt": 20.0, "intercept_ball_minus_batter_pos_x_inches": 1.0, "intercept_ball_minus_batter_pos_y_inches": 1.0},
        {"is_swing": True, "bat_speed": 74.0, "swing_length": 7.5, "attack_angle": 12.0, "attack_direction": 6.0, "swing_path_tilt": 22.0, "intercept_ball_minus_batter_pos_x_inches": 1.0, "intercept_ball_minus_batter_pos_y_inches": 1.0},
    ])
    b = pd.DataFrame.from_records([
        {"is_swing": True, "bat_speed": 72.0, "swing_length": 7.2, "attack_angle": 11.0, "attack_direction": 0.0, "swing_path_tilt": 21.0, "intercept_ball_minus_batter_pos_x_inches": 0.0, "intercept_ball_minus_batter_pos_y_inches": 0.0}
    ] * 100)
    d = compute_bat_tracking_deltas(g, b)
    assert d.bat_speed_game == pytest.approx(72.0)
    assert d.bat_speed_delta == pytest.approx(0.0)


def test_bat_tracking_empty_or_null() -> None:
    cols = ["is_swing", "bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt", "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches"]
    g = pd.DataFrame(columns=cols)
    b = pd.DataFrame(columns=cols)
    d = compute_bat_tracking_deltas(g, b)
    assert np.isnan(d.bat_speed_delta)


# === attack_pattern.py tests ===
def test_attack_pattern_basic_and_bucket_filtering() -> None:
    game = pd.DataFrame.from_records([
        {"pitch_type": "FF", "balls": 0, "strikes": 0},
        {"pitch_type": "FF", "balls": 0, "strikes": 0},
        {"pitch_type": "SL", "balls": 1, "strikes": 1},
    ])
    cur = pd.DataFrame.from_records([
        *[{"pitch_type": "FF", "balls": 0, "strikes": 0} for _ in range(50)],
        *[{"pitch_type": "SL", "balls": 1, "strikes": 1} for _ in range(20)],
    ])
    prior = pd.DataFrame.from_records([*[{"pitch_type": "CH", "balls": 1, "strikes": 1} for _ in range(20)]])
    out = compute_attack_pattern(game, cur, prior)
    assert len(out) == 2
    by_count = {(r.balls, r.strikes): r for r in out}
    assert by_count[(0, 0)].baseline_window == "1_season"
    assert by_count[(1, 1)].baseline_window == "2_seasons"


def test_attack_pattern_insufficient() -> None:
    game = pd.DataFrame.from_records([{"pitch_type": "FF", "balls": 2, "strikes": 2}])
    cur = pd.DataFrame.from_records([])
    prior = pd.DataFrame.from_records([])
    out = compute_attack_pattern(game, cur, prior)
    assert out[0].baseline_window == "insufficient_data"


# === timing.py tests ===
def test_timing_basic_and_unknown_exclusion() -> None:
    game = pd.DataFrame.from_records([
        {"pitch_type": "FF", "is_contact": True, "attack_direction": -20.0, "stand": "R"},
        {"pitch_type": "SL", "is_contact": True, "attack_direction": 10.0, "stand": "R"},
        {"pitch_type": "XX", "is_contact": True, "attack_direction": -30.0, "stand": "R"},
    ])
    baseline = pd.DataFrame.from_records([
        {"pitch_type": "FF", "is_contact": True, "attack_direction": -10.0, "stand": "R"},
        {"pitch_type": "SL", "is_contact": True, "attack_direction": 20.0, "stand": "R"},
    ])
    out = compute_timing_diagnostics(game, baseline)
    fastball = [r for r in out if r.category == "fastball"][0]
    assert fastball.mean_attack_direction_game == pytest.approx(-20.0)
    assert categorize_pitch_type("XX") is None


def test_timing_lhb_inversion() -> None:
    game = pd.DataFrame.from_records([{"pitch_type": "FF", "is_contact": True, "attack_direction": 10.0, "stand": "L"}])
    baseline = pd.DataFrame.from_records([{"pitch_type": "FF", "is_contact": True, "attack_direction": 0.0, "stand": "R"}])
    out = compute_timing_diagnostics(game, baseline)
    fastball = [r for r in out if r.category == "fastball"][0]
    assert fastball.mean_attack_direction_game == pytest.approx(-10.0)
