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

    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),          nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.model(x)


def ann_predict(feature_train_data, label_train_data, feature_test_data, k=None):
    # feature data is already standardised by walk_forward_validation
    feature_train = torch.tensor(feature_train_data, dtype=torch.float32)
    label_train   = torch.tensor(label_train_data,   dtype=torch.float32)
    feature_test  = torch.tensor(feature_test_data,  dtype=torch.float32)

    model         = ANNModel(feature_train_data.shape[1])
    optimiser     = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_function = nn.MSELoss()
    loader        = DataLoader(TensorDataset(feature_train, label_train), batch_size=32, shuffle=True)

    model.train()
    for _ in range(50):
        for features_b, labels_b in loader:
            loss = loss_function(model(features_b), labels_b)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        return model(feature_test).numpy()   # (n, 4)


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         N_STEPS=5,
         rmse_mode="price"):
    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    data = feature_engineering(data)
    data = build_targets(data, N_STEPS)

    if len(data) == 0:
        raise ValueError(f"No data remaining after feature engineering for {START_DATE}–{END_DATE}.")

    features, labels = get_features_and_labels(data, TARGET_COLS)

    predicted, actual, _ = walk_forward_validation(
        features, labels, ann_predict, data_split_ratios=DATA_SPLIT_RATIOS,
    )

    if len(predicted) == 0:
        raise ValueError("walk_forward_validation produced no predictions.")

    training_window = max(1, int(len(features) * DATA_SPLIT_RATIOS[0]))
    actual_opens    = data["open"].values[training_window: training_window + len(predicted)]
    dates           = data["Date"].values[training_window: training_window + len(predicted)]
    seed_open       = data["open"].values[training_window - 1]   # last known training open

    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode=rmse_mode)
    print(f"ANN ({N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    return build_result_df(predicted, actual_opens, idx=dates, seed_open=seed_open, n_steps=N_STEPS)
