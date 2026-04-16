# Predicts open, high, low, close RELATIVE TO CURRENT OPEN
# N_STEPS=0  → nowcast  |  N_STEPS≥1 → N-step ahead

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from upgraded_utilities import feature_engineering, walk_forward_validation, get_data, get_features_and_labels
from model_utils import build_targets, build_result_df, compute_rmse, build_result_df_chained, TARGET_COLS


class ANNModel(nn.Module):

    def __init__(self, input_size, width=256, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, width),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width,      width // 2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 2, width // 4),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 4, width // 8),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 8, max(width // 16, 8)), nn.ReLU(),
            nn.Linear(max(width // 16, 8), 4),
        )

    def forward(self, x):
        return self.model(x)


def make_ann_predict(epochs=50, lr=0.001, width=256, dropout=0.2, batch_size=32):
    """Factory so hyperparams can be injected from the tuning layer."""
    def ann_predict(feature_train_data, label_train_data, feature_test_data, k=None):
        # feature data is already standardised by walk_forward_validation
        feature_train = torch.tensor(feature_train_data, dtype=torch.float32)
        label_train   = torch.tensor(label_train_data,   dtype=torch.float32)
        feature_test  = torch.tensor(feature_test_data,  dtype=torch.float32)

        model         = ANNModel(feature_train_data.shape[1], width=width, dropout=dropout)
        optimiser     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_function = nn.MSELoss()
        loader        = DataLoader(TensorDataset(feature_train, label_train),
                                   batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for features_b, labels_b in loader:
                loss = loss_function(model(features_b), labels_b)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        model.eval()
        with torch.no_grad():
            return model(feature_test).numpy()   # (n, 4)

    return ann_predict


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         N_STEPS=5,
         rmse_mode="price",
         # ── Tunable hyperparameters ──────────────────────────────────────
         epochs=50,
         lr=0.001,
         width=256,
         dropout=0.2,
         batch_size=32,
         # ── Evaluation flag ─────────────────────────────────────────────
         return_metrics=False):
    """
    return_metrics=True → returns (DataFrame, rmse_dict) instead of just DataFrame.
    Used by the tuning layer to retrieve scores without re-running the pipeline.
    """
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    data = feature_engineering(data)
    data = build_targets(data, N_STEPS)

    if len(data) == 0:
        raise ValueError(f"No data remaining after feature engineering for {START_DATE}–{END_DATE}.")

    features, labels = get_features_and_labels(data, TARGET_COLS)

    predicted, actual, _ = walk_forward_validation(
        features, labels,
        make_ann_predict(epochs=epochs, lr=lr, width=width,
                         dropout=dropout, batch_size=batch_size),
        data_split_ratios=DATA_SPLIT_RATIOS,
    )

    if len(predicted) == 0:
        raise ValueError("walk_forward_validation produced no predictions.")

    training_window = max(1, int(len(features) * DATA_SPLIT_RATIOS[0]))
    actual_opens    = data["open"].values[training_window: training_window + len(predicted)]
    dates           = data["Date"].values[training_window: training_window + len(predicted)]
    seed_open       = data["open"].values[training_window - 1]

    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode=rmse_mode)
    print(f"ANN ({N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    result_df = build_result_df(predicted, actual_opens, idx=dates,
                                seed_open=seed_open, n_steps=N_STEPS)
    return (result_df, rmse_scores) if return_metrics else result_df
