"""League-wide run value matrix for rigorous decision-quality analysis.

NOT IMPLEMENTED IN PROMPT 1. Will be built in a dedicated follow-up prompt.
The matrix requires pulling a full league season of Statcast data
(~700K pitches), which is its own engineering exercise.

For v1, decision quality uses the delta_run_exp shortcut path.
This module exists so the toggle in config.DECISION_QUALITY_METHODS works
end-to-end (selecting 'rv_matrix' will raise NotImplementedError with a
clear message until this is built).
"""

import pandas as pd


def get_run_value_matrix(season: int, use_cache: bool = True) -> pd.DataFrame:
    """Return league-wide run value matrix indexed by (balls, strikes, zone, swing_or_take).

    Returns DataFrame with columns:
        balls: int
        strikes: int
        zone: int (Statcast zone, 1-14)
        swing_or_take: str ("swing" or "take")
        mean_delta_run_exp: float
        n_pitches: int

    Currently raises NotImplementedError. To be implemented in a later prompt.
    """
    _ = (season, use_cache)
    raise NotImplementedError(
        "rv_matrix decision-quality method requires building the league-wide "
        "run value matrix. Use DECISION_QUALITY_METHOD='delta_run_exp' for v1, "
        "or implement this function in a follow-up prompt."
    )


if __name__ == "__main__":
    try:
        get_run_value_matrix(2025)
    except NotImplementedError as exc:
        print(f"not implemented in v1: {exc}")
