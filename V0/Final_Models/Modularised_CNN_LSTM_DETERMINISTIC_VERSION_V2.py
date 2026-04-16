# Predicts open, high, low, close RELATIVE TO CURRENT OPEN (deterministic v2)
# N_STEPS=0  → nowcast  |  N_STEPS≥1 → N-step ahead

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from upgraded_utilities import feature_engineering, walk_forward_validation, get_data, get_features_and_labels
from model_utils import build_targets, build_result_df, compute_rmse, build_result_df_chained, TARGET_COLS


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNLSTMModel(nn.Module):

    def __init__(self, input_features, seq_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),             nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc   = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def create_sequences(features, labels, seq_len, n_steps):
    X, y = [], []
    for i in range(len(features) - seq_len - max(n_steps - 1, 0)):
        X.append(features[i: i + seq_len])
        y.append(labels[i + seq_len + n_steps - 1])
    return np.array(X), np.array(y)


def make_cnn_lstm_predict(n_steps):
    def cnn_lstm_predict(feature_train_data, label_train_data, feature_test_data, k=None):
        set_seed(42)
        seq_len = 40

        # feature data is already standardised by walk_forward_validation
        X_train, y_train = create_sequences(feature_train_data, label_train_data, seq_len, n_steps)
        X_test,  _       = create_sequences(feature_test_data,
                                             np.zeros((len(feature_test_data), 4)), seq_len, n_steps)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)

        model         = CNNLSTMModel(X_train.shape[2], seq_len)
        optimiser     = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        loss_function = nn.MSELoss()
        loader        = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)

        model.train()
        for _ in range(50):
            for features_b, labels_b in loader:
                loss = loss_function(model(features_b), labels_b)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).detach().numpy()   # (n_test_seq, 4)

        # Per-column bias correction using training-set mean as reference
        fold_bias = preds.mean(axis=0) - label_train_data.mean(axis=0)
        preds     = preds - fold_bias

        pad              = seq_len + max(n_steps - 1, 0)
        output           = np.empty((len(feature_test_data), 4), dtype=np.float64)
        output[pad:]     = preds
        if len(preds) > 0:
            output[:pad] = preds[0]
        return output

    return cnn_lstm_predict


def main(START_DATE="2010-01-01", END_DATE="2025-12-31",
         DATA_SPLIT_RATIOS=(0.8, 0.1, 0.1),
         N_STEPS=5,
         rmse_mode="price"):
    set_seed(42)

    data = get_data(START_DATE, END_DATE)
    data["Date"] = data.index
    data = feature_engineering(data)
    data = build_targets(data, N_STEPS)

    features, labels = get_features_and_labels(data, TARGET_COLS)

    predicted, actual, _ = walk_forward_validation(
        features, labels, make_cnn_lstm_predict(N_STEPS),
        data_split_ratios=DATA_SPLIT_RATIOS,
    )

    training_window = int(len(features) * DATA_SPLIT_RATIOS[0])
    actual_opens    = data["open"].values[training_window: training_window + len(predicted)]
    dates           = data["Date"].values[training_window: training_window + len(predicted)]
    seed_open       = data["open"].values[training_window - 1]   # last known training open

    rmse_scores = compute_rmse(predicted, actual, actual_opens, mode=rmse_mode)
    print(f"CNN-LSTM-DET-V2 ({N_STEPS}-step ahead) RMSE [{rmse_mode}]: {rmse_scores}")

    return build_result_df(predicted, actual_opens, idx=dates, seed_open=seed_open, n_steps=N_STEPS)
