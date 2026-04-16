import numpy as np

from upgraded_utilities import get_data, feature_engineering, walk_forward_validation
from model_utils import build_targets, build_result_df, compute_rmse, build_result_df_chained, TARGET_COLS


def cosine_distance(x1, x2):
    return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8)


def knn_regression(feature_train, label_train, feature_test, k=3):
    # Uses Cosine Distance o-o (Olga's slides suggested??)
    predictions = []
    for test in feature_test:
        distances = [(cosine_distance(feature_train[i], test), label_train[i]) for i in range(len(feature_train))]
        distances.sort(key=lambda x: x[0])
        k_values = [label for (_, label) in distances[:k]]
        predictions.append(np.mean(k_values, axis=0))   # mean over k neighbours, shape (4,)
    return np.array(predictions)   # (n_test, 4)


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),k=2,N_STEPS=5,rmse_mode="price"):
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    df = feature_engineering(data)
    df = build_targets(df, N_STEPS)
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna().reset_index(drop=True)

    feature_cols = ["return_t-1", "return_t-2", "return_t-3", "ma_5", "ma_10", "volatility_5", "momentum_5", "momentum_10"]

    features = df[feature_cols].values
    labels = df[TARGET_COLS].values   # (n, 4)

    predicted, actual, _ = walk_forward_validation(features, labels, knn_regression, data_split_ratios=DATA_SPLIT_RATIOS, k=k,)

    training_window = int(len(labels) * DATA_SPLIT_RATIOS[0])
    actual_opens = df["open"].values[training_window: training_window + len(predicted)]
    dates = df["Date"].values[training_window: training_window + len(predicted)]
    seed_open = df["open"].values[training_window - 1]   # last known training open

    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode=rmse_mode)
    print(f"KNN (k={k}, {N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    return build_result_df(predicted, actual_opens, idx=dates, seed_open=seed_open, n_steps=N_STEPS)
