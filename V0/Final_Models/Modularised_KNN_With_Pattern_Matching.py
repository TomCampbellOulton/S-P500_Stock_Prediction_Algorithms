# Predicts open, high, low, close RELATIVE TO CURRENT OPEN (pattern-matching KNN)
# N_STEPS=0  → nowcast  |  N_STEPS≥1 → N-step ahead

import numpy as np

from upgraded_utilities import get_data, feature_engineering, walk_forward_validation, normalise_patterns
from model_utils import build_result_df, compute_rmse, build_result_df_chained


def cosine_distance(x1, x2):
    x1, x2 = x1.flatten(), x2.flatten()
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8)


def knn_regression(feature_train, label_train, feature_test, k=3):
    """Inverse-distance weighted KNN returning (n_test, 4) predictions."""
    predictions = []
    for test in feature_test:
        distances = [(cosine_distance(feature_train[i], test), label_train[i])
                     for i in range(len(feature_train))]
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        weights   = np.array([1 / (d + 1e-8) for d, _ in k_nearest])
        values    = np.array([label for _, label in k_nearest])   # (k, 4)
        predictions.append(np.sum(weights[:, None] * values, axis=0) / np.sum(weights))
    return np.array(predictions)   # (n_test, 4)


def build_pattern_dataset_ohlc(returns, open_rel, high_rel, low_rel, close_rel,
                                window=20, n_steps=1):
    """
    features[i] = window of past returns ending at day i.
    labels[i]   = [open_rel, high_rel, low_rel, close_rel] at day:
                    n_steps > 0 → i + n_steps - 1
                    n_steps = 0 → i - 1  (nowcast)
    """
    features, labels = [], []
    for i in range(window, len(returns) - max(n_steps, 1) + 1):
        pattern   = returns[i - window: i]
        label_idx = (i + n_steps - 1) if n_steps > 0 else (i - 1)
        if label_idx < 0 or label_idx >= len(returns):
            continue
        label = np.array([open_rel[label_idx], high_rel[label_idx],
                          low_rel[label_idx],  close_rel[label_idx]])
        features.append(pattern)
        labels.append(label)
    return np.array(features), np.array(labels)   # (n, window), (n, 4)


def run_pattern_knn(df, window, k, training_ratio, validation_ratio, testing_ratio, n_steps):
    returns   = df["return"].values
    high_rel  = (df["high"]  / df["open"]).values
    low_rel   = (df["low"]   / df["open"]).values
    close_rel = (df["close"] / df["open"]).values
    open_rel  = (df["open"].shift(-n_steps) / df["open"]).values if n_steps > 0 \
                else (df["open"] / df["open"]).values   # always 1.0 for nowcast

    features, labels = build_pattern_dataset_ohlc(
        returns, open_rel, high_rel, low_rel, close_rel, window=window, n_steps=n_steps)
    features = normalise_patterns(features)

    predictions, actuals, rmse_val = walk_forward_validation(
        features, labels, knn_regression,
        data_split_ratios=(training_ratio, validation_ratio, testing_ratio), k=k,
    )
    return rmse_val, predictions, actuals


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT=(0.8, 0.1, 0.1),
         k=2,
         N_STEPS=5,
         rmse_mode="price"):
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df   = feature_engineering(data)

    training_ratio, validation_ratio, testing_ratio = DATA_SPLIT
    pattern_windows  = [5, 10, 20, 50]
    predictions_list = []
    actuals          = None

    for window in pattern_windows:
        _, predictions, actuals = run_pattern_knn(
            df, window=window, k=k,
            training_ratio=training_ratio,
            validation_ratio=validation_ratio,
            testing_ratio=testing_ratio,
            n_steps=N_STEPS,
        )
        predictions_list.append(predictions)

    min_len          = min(len(p) for p in predictions_list)
    predictions_list = [p[-min_len:] for p in predictions_list]
    predicted        = np.mean(predictions_list, axis=0)   # (min_len, 4)
    actuals          = actuals[-min_len:]

    # actual_opens: the open prices at the reference time for each prediction
    actual_opens = df["open"].values[-len(predicted):]
    dates        = df["Date"].values[-len(predicted):]
    seed_open    = df["open"].values[-len(predicted) - 1]   # last known training open

    rmse_scores = compute_rmse(predicted, actuals, actual_opens, mode=rmse_mode)
    print(f"Pattern KNN (k={k}, {N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    return build_result_df(predicted, actual_opens, idx=dates, seed_open=seed_open, n_steps=N_STEPS)
