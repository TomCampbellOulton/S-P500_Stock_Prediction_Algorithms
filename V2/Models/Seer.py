# Seer — ground-truth OHLC for the test window.
# Returns the same DataFrame format as all models for direct comparison.

import numpy as np

from upgraded_utilities import get_data, feature_engineering, data_split, get_features_and_labels
from model_utils import build_targets, build_result_df, build_result_df_chained, TARGET_COLS


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.0, 0.2),
         N_STEPS=5):
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df = feature_engineering(data)        # ← use df, not data
    df = build_targets(df, N_STEPS)

    features, labels = get_features_and_labels(df, TARGET_COLS)   # (n, n_feat), (n, 4)

    _, _, _, _, _, label_test = data_split(features, labels, DATA_SPLIT_RATIOS)

    n = len(df)
    train_end = int(n * DATA_SPLIT_RATIOS[0])
    val_end = train_end + int(n * DATA_SPLIT_RATIOS[1])

    # actual_opens is the open price at each row's reference time —
    # the denominator used in build_targets so we can reconstruct true prices.
    actual_opens = df["open"].values[val_end: val_end + len(label_test)]
    dates = df["Date"].values[val_end: val_end + len(label_test)]
    seed_open = df["open"].values[val_end - 1]   # last known open before test window

    return build_result_df(label_test, actual_opens, idx=dates, seed_open=seed_open, n_steps=N_STEPS)
