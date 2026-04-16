import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close"]

# Column names for target construction — same order as output neurons
TARGET_COLS = ["target_open", "target_high", "target_low", "target_close"]


def build_targets(df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    """
    Add 4 relative target columns to df.
    n_steps=0 → predict current period (nowcast): ratio = value / open  (== 1.0 for open)
    n_steps≥1 → predict n_steps days ahead:       ratio = value[t+n] / open[t]

    All targets are relative to the CURRENT period's open (DlyPrcInd / open column).
    The original 'open' column is kept so callers can recover actual_opens for reconstruction.
    """
    df = df.copy()
    df["target_open"]  = df["open"].shift(-n_steps) / df["open"]
    df["target_high"]  = df["high"].shift(-n_steps) / df["open"]
    df["target_low"]   = df["low"].shift(-n_steps)  / df["open"]
    df["target_close"] = df["close"].shift(-n_steps) / df["open"]
    return df.dropna()#.reset_index(drop=True)

def build_result_df_chained(predicted_rel: np.ndarray,
                             seed_open: float,
                             n_steps: int = 1,
                             idx=None) -> pd.DataFrame:
    predicted_rel  = np.asarray(predicted_rel)
    n              = len(predicted_rel)

    # ── 1. Build chain on non-overlapping anchor points ───────────────────────
    chain_indices  = np.arange(0, n, max(n_steps, 1))        # [0, 5, 10, ...]
    chain_open_rel = predicted_rel[chain_indices, 0]          # (m,)
    m              = len(chain_open_rel)
    chain_opens    = np.empty(m)
    current_open   = seed_open
    for i in range(m):
        chain_opens[i] = chain_open_rel[i] * current_open
        current_open   = chain_opens[i]                       # feeds next link

    # ── 2. Interpolate chained open back to every step ────────────────────────
    full_open = np.interp(np.arange(n), chain_indices, chain_opens)   # (n,)

    # ── 3. Apply intraday ratios from raw model predictions ───────────────────
    safe_open_rel  = np.where(np.abs(predicted_rel[:, 0]) < 1e-8, 1.0, predicted_rel[:, 0])
    intraday_high  = predicted_rel[:, 1] / safe_open_rel
    intraday_low   = predicted_rel[:, 2] / safe_open_rel
    intraday_close = predicted_rel[:, 3] / safe_open_rel

    prices = np.column_stack([
        full_open,
        full_open * intraday_high,
        full_open * intraday_low,
        full_open * intraday_close,
    ])

    if idx is None:
        idx = pd.RangeIndex(n)

    return pd.DataFrame({
        "open_rel":  predicted_rel[:, 0],
        "high_rel":  predicted_rel[:, 1],
        "low_rel":   predicted_rel[:, 2],
        "close_rel": predicted_rel[:, 3],
        "open":      prices[:, 0],
        "high":      prices[:, 1],
        "low":       prices[:, 2],
        "close":     prices[:, 3],
    }, index=idx)

def compute_rmse(predicted_rel: np.ndarray,
                 actual_rel: np.ndarray,
                 actual_opens: np.ndarray,
                 mode: str = "price") -> dict:
    """
    Parameters
    ----------
    predicted_rel : (n, 4)  model predictions  [open_rel, high_rel, low_rel, close_rel]
    actual_rel    : (n, 4)  ground truth relatives
    actual_opens  : (n,)    the open price at each row's reference time t.
                            Used to reconstruct absolute prices: price = rel * open[t].
                            Pass df["open"].values[test_start : test_start + n] from your main().
    mode          : "price" | "relative"

    Returns
    -------
    dict  { "open": float, "high": float, "low": float, "close": float, "mean": float }
    """
    actual_opens = np.asarray(actual_opens).reshape(-1, 1)   # (n, 1) for broadcasting

    if mode == "price":
        pred  = predicted_rel * actual_opens
        truth = actual_rel    * actual_opens
    elif mode == "relative":
        pred  = predicted_rel
        truth = actual_rel
    else:
        raise ValueError(f"rmse_mode must be 'price' or 'relative', got '{mode}'")

    per_col = {col: float(np.sqrt(np.mean((pred[:, i] - truth[:, i]) ** 2)))
               for i, col in enumerate(COLS)}
    per_col["mean"] = float(np.mean(list(per_col.values())))
    return per_col


def build_result_df(predicted_rel: np.ndarray,
                    actual_opens: np.ndarray,
                    idx=None,
                    seed_open: float = None,
                    n_steps: int = 1) -> pd.DataFrame:
    """
    Parameters
    ----------
    predicted_rel : (n, 4)  [open_rel, high_rel, low_rel, close_rel]
    actual_opens  : (n,)    open price at each row's reference time t.
                            Reconstructed price[t] = predicted_rel[t] * actual_opens[t].
    idx           : array-like of timestamps, or None (falls back to RangeIndex)
    seed_open     : float   The last known open price from the training window
                            (i.e. df["open"].values[training_window - 1]).
                            Used as the starting point for daisy-chained predictions
                            so no test-set data is touched. If None, falls back to
                            actual_opens[0] (old behaviour — uses test data).
    n_steps       : int     Prediction horizon — passed to build_result_df_chained
                            so the chain subsamples correctly (see that function).

    Returns
    -------
    pd.DataFrame with columns:
        open_rel, high_rel, low_rel, close_rel, open, high, low, close
    """
    predicted_rel = np.asarray(predicted_rel)
    actual_opens = np.asarray(actual_opens).ravel()
    if seed_open is None:
        seed_open = float(actual_opens[0])   # fallback — uses test data
    actual_opens  = np.asarray(actual_opens).reshape(-1, 1)  # (n, 1) for broadcasting

    if idx is None:
        idx = pd.RangeIndex(len(predicted_rel))

    prices = predicted_rel * actual_opens   # (n, 4) — each row scaled by its own open

    daisy_chained_df = build_result_df_chained(predicted_rel, seed_open=seed_open, n_steps=n_steps)

    return pd.DataFrame({
        "open_rel":  predicted_rel[:, 0],
        "high_rel":  predicted_rel[:, 1],
        "low_rel":   predicted_rel[:, 2],
        "close_rel": predicted_rel[:, 3],
        "open":      prices[:, 0],
        "high":      prices[:, 1],
        "low":       prices[:, 2],
        "close":     prices[:, 3],
        "daisy_chained_open_rel":  daisy_chained_df["open_rel"].values,
        "daisy_chained_high_rel":  daisy_chained_df["high_rel"].values,
        "daisy_chained_low_rel":   daisy_chained_df["low_rel"].values,
        "daisy_chained_close_rel": daisy_chained_df["close_rel"].values,
        "daisy_chained_open":      daisy_chained_df["open"].values,
        "daisy_chained_high":      daisy_chained_df["high"].values,
        "daisy_chained_low":       daisy_chained_df["low"].values,
        "daisy_chained_close":     daisy_chained_df["close"].values,
    }, index=idx)
