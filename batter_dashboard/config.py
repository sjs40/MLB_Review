"""Central configuration for the batter dashboard.

All thresholds, paths, classification sets, and tunable values live here.
No magic numbers in other modules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TEST_CASE = {
    "player_id": 663538,
    "game_date": "2026-04-15",
    "expected_pitches": 13,
}

SWING_DESCRIPTIONS = frozenset(
    {
        "hit_into_play",
        "foul",
        "swinging_strike",
        "foul_tip",
        "swinging_strike_blocked",
        "foul_bunt",
    }
)
CONTACT_DESCRIPTIONS = frozenset({"hit_into_play", "foul", "foul_tip", "foul_bunt"})
WHIFF_DESCRIPTIONS = frozenset({"swinging_strike", "swinging_strike_blocked"})
DECISION_EXCLUDED_DESCRIPTIONS = frozenset(
    {"intent_ball", "pitchout", "automatic_ball", "automatic_strike"}
)

IN_ZONE_VALUES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
OUT_ZONE_VALUES = frozenset({11, 12, 13, 14})

ATTACK_PATTERN_MIN_PITCHES_1_SEASON = 50
ATTACK_PATTERN_MIN_PITCHES_2_SEASONS = 30

COUNT_BUCKETS = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2),
]

DECISION_QUALITY_METHODS = ("delta_run_exp", "rv_matrix")
DEFAULT_DECISION_QUALITY_METHOD = "delta_run_exp"

BAT_TRACKING_PRIMARY_SOURCE = "pybaseball"
BAT_TRACKING_FALLBACK_SOURCE = "savant_scrape"
BAT_TRACKING_MIN_COVERAGE = 0.5

BAT_TRACKING_COLUMNS = [
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]

PITCH_CATEGORY_MAP = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball", "FA": "fastball",
    "SL": "breaking", "CU": "breaking", "KC": "breaking", "ST": "breaking",
    "SV": "breaking", "CS": "breaking", "EP": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed",
    "KN": "offspeed",
}
PULL_THRESHOLD_DEGREES = -15.0

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = "INFO"
