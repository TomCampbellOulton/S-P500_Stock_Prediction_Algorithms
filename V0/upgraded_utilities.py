import pandas as pd
import numpy as np


def get_data_yf(start_date, end_date, ticker="^GSPC"):
    import yfinance as yf
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace=True)
    return df


def get_data(start_date, end_date, data_file="Data/S&P 500 Composite.csv"):
    df = pd.read_csv(data_file, usecols=["YYYYMMDD", "DlyPrcRet", "DlyPrcInd"])

    df["Date"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
    df.set_index("Date", inplace=True)

    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)

    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    filtered_df.sort_index(inplace=True)

    # Fetch OHLC ratios from yfinance
    y_finance = get_data_yf(start_date, end_date)
    y_finance = y_finance.reindex(filtered_df.index)

    relative_high  = y_finance["high"]  / y_finance["open"]
    relative_low   = y_finance["low"]   / y_finance["open"]
    relative_close = y_finance["close"] / y_finance["open"]

    base_price = filtered_df["DlyPrcInd"]

    filtered_df["open"]  = base_price
    filtered_df["high"]  = base_price * relative_high
    filtered_df["low"]   = base_price * relative_low
    filtered_df["close"] = base_price * relative_close

    # Drop non-numeric columns (YYYYMMDD, etc.)
    filtered_df = filtered_df.drop(columns=["YYYYMMDD"], errors="ignore")
    filtered_df = filtered_df.select_dtypes(include=["number"])

    return filtered_df


def get_features_and_labels(df: pd.DataFrame, target_cols: list):
    """
    Split a numeric DataFrame into feature and label numpy arrays.

    Parameters
    ----------
    df          : DataFrame — must already contain all feature-engineered columns
                  AND the target columns listed in target_cols.
    target_cols : list of str — columns to use as labels (excluded from features).

    Returns
    -------
    features : np.ndarray  (n, n_features)
    labels   : np.ndarray  (n, len(target_cols))
    """
    df_numeric   = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in df_numeric.columns if c not in target_cols]
    features     = df_numeric[feature_cols].values
    labels       = df_numeric[target_cols].values
    return features, labels


def get_feature_names(df: pd.DataFrame, target_cols: list) -> list:
    """Return the feature column names (everything numeric except target_cols)."""
    df_numeric = df.select_dtypes(include=[np.number])
    return [c for c in df_numeric.columns if c not in target_cols]


# ──────────────────────────────────────────────────────────────────
# Pattern-matching helpers
# ──────────────────────────────────────────────────────────────────

def build_pattern_dataset(returns, window=20):
    features, labels = [], []
    for i in range(window, len(returns) - 1):
        features.append(returns[i - window: i])
        labels.append(returns[i])
    return np.array(features), np.array(labels)


def normalise_patterns(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True) + 1e-8
    return (features - mean) / std


# ──────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────

def rmse(label, prediction):
    return np.sqrt(np.mean((label - prediction) ** 2))


# ──────────────────────────────────────────────────────────────────
# Train / validation / test split
# ──────────────────────────────────────────────────────────────────

def data_split(features, labels, split_ratio=(0.70, 0.15, 0.15)):
    """
    Split features and labels (numpy arrays or DataFrames) into
    train / validation / test portions.
    """
    n = len(features)
    train_end = int(n * split_ratio[0])
    val_end   = train_end + int(n * split_ratio[1])

    # Support both numpy arrays and DataFrames
    def _slice(arr, start, end):
        if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
            return arr.iloc[start:end]
        return arr[start:end]

    f_train = _slice(features, 0, train_end)
    l_train = _slice(labels,   0, train_end)
    f_val   = _slice(features, train_end, val_end)
    l_val   = _slice(labels,   train_end, val_end)
    f_test  = _slice(features, val_end, n)
    l_test  = _slice(labels,   val_end, n)

    return f_train, l_train, f_val, l_val, f_test, l_test


# ──────────────────────────────────────────────────────────────────
# Walk-forward validation
# ──────────────────────────────────────────────────────────────────

def walk_forward_validation(features, labels, prediction_algorithm,
                             data_split_ratios=(0.7, 0.15, 0.15), k=5):
    """
    Expanding-window walk-forward validation.

    Standardises features using only the current training window's statistics
    (no look-ahead leakage).

    Returns
    -------
    predictions : np.ndarray  (n_test_total, n_outputs)
    actuals     : np.ndarray  (n_test_total, n_outputs)
    rmse_per_col: np.ndarray  per-column RMSE (for quick sanity checks)
    """
    def _standardise(train, test):
        mean = train.mean(axis=0)
        std  = train.std(axis=0) + 1e-8
        return (train - mean) / std, (test - mean) / std

    # Ensure numpy
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    n               = len(features)
    training_window = int(n * data_split_ratios[0])
    testing_window  = max(int(n * data_split_ratios[2]), 1)

    predictions, actuals = [], []

    for start in range(0, n - training_window - testing_window + 1, testing_window):
        train_end = start + training_window
        test_end  = train_end + testing_window

        f_train = features[start:train_end]
        l_train = labels[start:train_end]
        f_test  = features[train_end:test_end]
        l_test  = labels[train_end:test_end]

        f_train_s, f_test_s = _standardise(f_train, f_test)

        preds = prediction_algorithm(f_train_s, l_train, f_test_s, k=k)

        predictions.extend(preds)
        actuals.extend(l_test)

    predictions = np.array(predictions)
    actuals     = np.array(actuals)
    rmse_vals   = np.sqrt(np.mean((predictions - actuals) ** 2, axis=0))

    return predictions, actuals, rmse_vals

def dtr_walk_forward_validation(features, labels, estimator_class,
                                 data_split_ratios=(0.8, 0.1, 0.1),
                                 k=None, **estimator_kwargs):
    def _standardise(train, test):
        mean = train.mean(axis=0)
        std  = train.std(axis=0) + 1e-8
        return (train - mean) / std, (test - mean) / std

    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    n               = len(features)
    training_window = int(n * data_split_ratios[0])
    testing_window  = max(int(n * data_split_ratios[2]), 1)

    predictions, actuals, test_indices = [], [], []  # ← added test_indices

    for start in range(0, n - training_window - testing_window + 1, testing_window):
        train_end = start + training_window
        test_end  = min(train_end + testing_window, n)

        f_train, l_train = features[start:train_end], labels[start:train_end]
        f_test,  l_test  = features[train_end:test_end], labels[train_end:test_end]

        f_train_s, f_test_s = _standardise(f_train, f_test)

        model = estimator_class(**estimator_kwargs)
        model.fit(f_train_s, l_train)
        preds = model.predict(f_test_s)

        predictions.extend(preds)
        actuals.extend(l_test)
        test_indices.extend(range(train_end, test_end))  # ← track exact rows used

    predictions  = np.array(predictions)
    actuals      = np.array(actuals)
    rmse_per_col = np.sqrt(np.mean((predictions - actuals) ** 2, axis=0))

    return predictions, actuals, rmse_per_col, test_indices  # ← returned

# ──────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────

def feature_engineering(data_frame: pd.DataFrame) -> pd.DataFrame:
    df = data_frame.copy()

    df["return"]     = df["DlyPrcRet"]
    df["log return"] = np.log(df["DlyPrcInd"] / df["DlyPrcInd"].shift(1))

    for w in [5, 10, 25, 50, 100, 200, 250]:
        df[f"ma_{w}"]         = df["DlyPrcInd"].rolling(w).mean()
        df[f"volatility_{w}"] = df["return"].rolling(w).std()
        df[f"momentum_{w}"]   = df["DlyPrcInd"] - df["DlyPrcInd"].shift(w)

    for lag in range(1, 26):
        df[f"return_t-{lag}"] = df["return"].shift(lag)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ──────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────

def reconstruct_prices(start_price, returns):
    prices, current = [], start_price
    for r in returns:
        current = current * (1 + r)
        prices.append(current)
    return np.array(prices)


def save_results(predictions, actuals, rmse=0, model_name="model"):
    import matplotlib.pyplot as plt
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize=(20, 10))
    plt.plot(actuals,     label="Actual closing")
    plt.plot(predictions, label="Predicted closing")
    plt.legend()
    plt.title(f"{model_name}  RMSE={rmse:.4f}")
    plt.savefig(f"{model_name}_rmse-{rmse}_{ts}.png")
    plt.close()
