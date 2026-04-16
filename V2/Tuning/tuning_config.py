# Tuning/tuning_config.py
# Defines the hyperparameter search spaces used by every tuning runner.
#
# PARAM_GRIDS   — explicit discrete grids for grid search.
# SEARCH_SPACES — distributions for random / Bayesian search.
#   Each parameter entry is a dict with a "type" key:
#     {"type": "int_choice",   "values": [...]}
#     {"type": "float_choice", "values": [...]}
#     {"type": "float",        "low": lo, "high": hi, "log": bool}
#     {"type": "int",          "low": lo, "high": hi}
#     {"type": "categorical",  "values": [...]}   (any type, returned as-is)
#
# MODEL_META — per-model metadata consumed by tuning_utils.run_trial():
#   split_kwarg  : the keyword that accepts the (train, val, test) split tuple.
#   exclude_from : list of runner names that should skip this model
#                  ("grid" is sensible for large DL grids).


# ─────────────────────────────────────────────────────────────────────────────
# Grid search param grids  (kept small — DL grids cap at ~27 combos each)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRIDS = {

    "Modularised_ANN": {
        "epochs":     [30, 50, 100],
        "lr":         [1e-4, 1e-3, 5e-3],
        "width":      [128, 256],
        "dropout":    [0.1, 0.2, 0.3],
        "batch_size": [32, 64],
    },

    "Modularised_CNN-LSTM": {
        "seq_len":     [10, 20, 40],
        "epochs":      [30, 50],
        "lr":          [5e-4, 1e-3],
        "hidden_size": [32, 64],
        "batch_size":  [32],
    },

    "Modularised_CNN_LSTM_DETERMINISTIC_VERSION_V2": {
        "seq_len":     [20, 40],
        "epochs":      [30, 50],
        "lr":          [1e-4, 5e-4],
        "hidden_size": [32, 64, 128],
        "batch_size":  [32],
    },

    "Modularised_DTR": {
        "max_depth":             [1, 2, 3, 5],
        "min_samples_split":     [2, 5, 10],
        "min_samples_leaf":      [5, 10, 20],
        "min_impurity_decrease": [1e-7, 1e-6, 1e-5],
    },

    "Modularised_GRU": {
        "seq_len":     [30, 60],
        "hidden_size": [32, 64],
        "num_layers":  [2, 4],
        "epochs":      [20, 30],
        "lr":          [5e-4, 1e-3],
        "batch_size":  [64],
    },

    "Modularised_KNN": {
        "k": [2, 3, 5, 7, 10],
    },

    "Modularised_KNN_With_Pattern_Matching": {
        "k":              [2, 3, 5, 7],
        "pattern_windows": [
            (5, 10, 20),
            (5, 10, 20, 50),
            (10, 20, 50),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Search spaces for random / Bayesian search
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACES = {

    "Modularised_ANN": {
        "epochs":     {"type": "int_choice",   "values": [30, 50, 75, 100, 150]},
        "lr":         {"type": "float",        "low": 1e-4, "high": 1e-2, "log": True},
        "width":      {"type": "int_choice",   "values": [64, 128, 256, 512]},
        "dropout":    {"type": "float",        "low": 0.0, "high": 0.5,  "log": False},
        "batch_size": {"type": "int_choice",   "values": [16, 32, 64]},
    },

    "Modularised_CNN-LSTM": {
        "seq_len":     {"type": "int_choice",   "values": [10, 20, 30, 40, 60]},
        "epochs":      {"type": "int_choice",   "values": [30, 50, 75, 100]},
        "lr":          {"type": "float",        "low": 1e-4, "high": 5e-3, "log": True},
        "hidden_size": {"type": "int_choice",   "values": [16, 32, 64, 128]},
        "batch_size":  {"type": "int_choice",   "values": [16, 32, 64]},
    },

    "Modularised_CNN_LSTM_DETERMINISTIC_VERSION_V2": {
        "seq_len":     {"type": "int_choice",   "values": [20, 30, 40, 60, 80]},
        "epochs":      {"type": "int_choice",   "values": [30, 50, 75, 100]},
        "lr":          {"type": "float",        "low": 5e-5, "high": 2e-3, "log": True},
        "hidden_size": {"type": "int_choice",   "values": [32, 64, 128, 256]},
        "batch_size":  {"type": "int_choice",   "values": [16, 32, 64]},
    },

    "Modularised_DTR": {
        "max_depth":             {"type": "int",          "low": 1, "high": 8},
        "min_samples_split":     {"type": "int_choice",   "values": [2, 5, 10, 20]},
        "min_samples_leaf":      {"type": "int_choice",   "values": [1, 5, 10, 15, 20, 30]},
        "min_impurity_decrease": {"type": "float",        "low": 1e-8, "high": 1e-4, "log": True},
    },

    "Modularised_GRU": {
        "seq_len":     {"type": "int_choice",   "values": [30, 60, 90, 120]},
        "hidden_size": {"type": "int_choice",   "values": [32, 64, 128, 256]},
        "num_layers":  {"type": "int",          "low": 1, "high": 6},
        "epochs":      {"type": "int_choice",   "values": [20, 30, 50]},
        "lr":          {"type": "float",        "low": 1e-4, "high": 5e-3, "log": True},
        "batch_size":  {"type": "int_choice",   "values": [32, 64, 128]},
    },

    "Modularised_KNN": {
        "k": {"type": "int", "low": 1, "high": 15},
    },

    "Modularised_KNN_With_Pattern_Matching": {
        "k":              {"type": "int",        "low": 1, "high": 10},
        "pattern_windows": {"type": "categorical", "values": [
            (5, 10, 20),
            (5, 10, 20, 50),
            (10, 20, 50),
            (5, 20, 50),
            (5, 10, 50),
        ]},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────────────────────────────────────

MODEL_META = {
    "Modularised_ANN": {
        "split_kwarg":   "DATA_SPLIT_RATIOS",
        "exclude_from":  [],
    },
    "Modularised_CNN-LSTM": {
        "split_kwarg":   "DATA_SPLIT_RATIOS",
        # Grid for CNN-LSTM is 3*2*2*2 = 24 combos — acceptable
        "exclude_from":  [],
    },
    "Modularised_CNN_LSTM_DETERMINISTIC_VERSION_V2": {
        "split_kwarg":   "DATA_SPLIT_RATIOS",
        "exclude_from":  [],
    },
    "Modularised_DTR": {
        "split_kwarg":   "DATA_SPLIT_RATIOS",
        "exclude_from":  [],
    },
    "Modularised_GRU": {
        "split_kwarg":   "DATA_SPLIT_RATIOS",
        "exclude_from":  [],
    },
    "Modularised_KNN": {
        "split_kwarg":   "DATA_SPLIT_RATIOS",
        "exclude_from":  [],
    },
    "Modularised_KNN_With_Pattern_Matching": {
        # This model uses DATA_SPLIT, not DATA_SPLIT_RATIOS
        "split_kwarg":   "DATA_SPLIT",
        "exclude_from":  [],
    },
}


# Ordered list of model names — controls execution order in all runners.
TUNABLE_MODELS = list(MODEL_META.keys())
