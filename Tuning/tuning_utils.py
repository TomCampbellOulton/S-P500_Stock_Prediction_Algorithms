# Tuning/tuning_utils.py
# Shared utilities for all tuning runners.
#
# Public API
# ----------
# run_trial(model_name, params, ...)  → result dict  (params + rmse_* keys)
# save_results(results, model_name, method)  → path str
# print_best(results, model_name, method)
#
# BayesianOptimizer  — numpy-only GP + Expected-Improvement BO class.
# ParamSampler       — samples a single config from a SEARCH_SPACES entry.

import os
import sys
import math
import importlib
import importlib.util
import traceback

import numpy as np
import pandas as pd
from datetime import datetime

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


# ─────────────────────────────────────────────────────────────────────────────
# Module import helper
# ─────────────────────────────────────────────────────────────────────────────

def _import_model(model_name: str):
    """
    Import a model from Models/<model_name>.py.
    Handles filenames that contain hyphens (invalid Python identifiers).
    """
    models_dir = os.path.join(ROOT, "Models")
    file_path  = os.path.join(models_dir, f"{model_name}.py")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Model file not found: {file_path}\n"
            f"Expected models in {models_dir}/"
        )

    safe_name = model_name.replace("-", "_")
    spec      = importlib.util.spec_from_file_location(safe_name, file_path)
    module    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Trial runner
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(model_name: str,
              params: dict,
              split_kwarg:       str   = "DATA_SPLIT_RATIOS",
              start_date:        str   = "2010-01-01",
              end_date:          str   = "2025-12-31",
              data_split_ratios: tuple = (0.8, 0.1, 0.1),
              n_steps:           int   = 5,
              rmse_mode:         str   = "price") -> dict | None:
    """
    Run one trial of *model_name* with the given *params*.

    Returns a flat dict  {param: value, ..., rmse_open: float, ..., rmse_mean: float}
    or None if the trial raises an exception.
    """
    try:
        module = _import_model(model_name)
        if not hasattr(module, "main"):
            raise AttributeError(f"{model_name} has no main() function.")

        fixed = {
            "START_DATE":    start_date,
            "END_DATE":      end_date,
            split_kwarg:     data_split_ratios,
            "N_STEPS":       n_steps,
            "rmse_mode":     rmse_mode,
            "return_metrics": True,
        }

        _, rmse_scores = module.main(**fixed, **params)

        return {
            **params,
            "rmse_open":  rmse_scores["open"],
            "rmse_high":  rmse_scores["high"],
            "rmse_low":   rmse_scores["low"],
            "rmse_close": rmse_scores["close"],
            "rmse_mean":  rmse_scores["mean"],
        }

    except Exception:
        print(f"  [!] Trial failed for {model_name} with params={params}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list, model_name: str, method: str,
                 results_dir: str = None) -> str:
    """
    Persist *results* (list of dicts) to a timestamped CSV.
    Rows are sorted by rmse_mean ascending.
    Returns the path written.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(results_dir, f"{model_name}__{method}__{ts}.csv")

    df = pd.DataFrame(results).sort_values("rmse_mean").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  [Saved] {path}")
    return path


def load_results(path: str) -> pd.DataFrame:
    """Load a previously saved results CSV."""
    return pd.read_csv(path)


def print_best(results: list, model_name: str, method: str):
    """Pretty-print the best trial from *results*."""
    if not results:
        print(f"  No results to report for {model_name}.")
        return

    df   = pd.DataFrame(results).sort_values("rmse_mean").reset_index(drop=True)
    best = df.iloc[0]

    param_cols = [c for c in df.columns if not c.startswith("rmse_")]
    param_str  = "  ".join(f"{c}={best[c]}" for c in param_cols)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {method.upper()}  ·  {model_name}")
    print(f"  Trials run : {len(df)}")
    print(f"  Best RMSE  : mean={best['rmse_mean']:.5f}  "
          f"open={best['rmse_open']:.5f}  "
          f"close={best['rmse_close']:.5f}")
    print(f"  Best params: {param_str}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sampler (for random search and BO candidate generation)
# ─────────────────────────────────────────────────────────────────────────────

class ParamSampler:
    """Draw a random configuration from a SEARCH_SPACES entry."""

    def __init__(self, search_space: dict, rng: np.random.Generator = None):
        self.space = search_space
        self.rng   = rng or np.random.default_rng()

    def sample(self) -> dict:
        config = {}
        for name, spec in self.space.items():
            config[name] = self._draw(spec)
        return config

    def _draw(self, spec: dict):
        t = spec["type"]
        if t == "int_choice":
            return self.rng.choice(spec["values"]).item()
        if t == "float_choice":
            return float(self.rng.choice(spec["values"]))
        if t == "categorical":
            idx = self.rng.integers(len(spec["values"]))
            return spec["values"][idx]
        if t == "float":
            lo, hi = spec["low"], spec["high"]
            if spec.get("log", False):
                return float(np.exp(self.rng.uniform(math.log(lo), math.log(hi))))
            return float(self.rng.uniform(lo, hi))
        if t == "int":
            return int(self.rng.integers(spec["low"], spec["high"] + 1))
        raise ValueError(f"Unknown param type: {t!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian-Process surrogate + Expected-Improvement  (numpy-only)
# ─────────────────────────────────────────────────────────────────────────────

def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF via math.erf — no scipy required."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


class _GaussianProcess:
    """
    Minimal RBF-kernel GP surrogate (numpy-only).

    Inputs are expected to lie in [0, 1]^d (normalised by BayesianOptimizer).
    The target (RMSE) is internally z-scored for numerical stability.
    """

    def __init__(self, length_scale: float = 0.5, noise: float = 1e-4):
        self.ls    = length_scale
        self.noise = noise
        self._y_mean = 0.0
        self._y_std  = 1.0
        self.X = self.y_norm = self.alpha = self.K_inv = None

    # ------------------------------------------------------------------
    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """RBF: K[i,j] = exp( -‖A[i]-B[j]‖² / (2 ls²) )."""
        diff = A[:, None, :] - B[None, :, :]        # (n, m, d)
        sq   = np.einsum("nmd,nmd->nm", diff, diff)  # (n, m)
        return np.exp(-sq / (2.0 * self.ls ** 2))

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._y_mean = float(y.mean())
        self._y_std  = float(y.std()) + 1e-8
        y_norm       = (y - self._y_mean) / self._y_std

        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        try:
            self.K_inv = np.linalg.solve(K, np.eye(len(K)))
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)

        self.X      = X.copy()
        self.y_norm = y_norm
        self.alpha  = self.K_inv @ y_norm

    # ------------------------------------------------------------------
    def predict(self, X_star: np.ndarray):
        """Returns (mu, sigma²) in the original (un-normalised) scale."""
        K_s   = self._kernel(X_star, self.X)                     # (m, n)
        mu    = K_s @ self.alpha                                  # normalised
        var   = 1.0 - np.einsum("mi,ij,mj->m", K_s, self.K_inv, K_s)
        var   = np.maximum(var, 0.0)
        # Undo z-scoring
        mu_out  = mu  * self._y_std + self._y_mean
        var_out = var * self._y_std ** 2
        return mu_out, var_out


class BayesianOptimizer:
    """
    GP-based Bayesian Optimisation for minimising RMSE.

    Works with mixed (int / float / categorical) search spaces by encoding
    each parameter as a float in [0, 1] for the GP, then decoding back to
    native types when calling the model.

    Parameters
    ----------
    search_space   : dict as in SEARCH_SPACES
    n_initial      : number of random warm-up trials before GP kicks in
    xi             : exploration–exploitation trade-off for EI (default 0.01)
    n_candidates   : random candidates evaluated per acquisition step
    length_scale   : GP RBF kernel length scale (on the [0,1]^d space)
    rng_seed       : reproducibility seed
    """

    def __init__(self, search_space: dict,
                 n_initial:    int   = 5,
                 xi:           float = 0.01,
                 n_candidates: int   = 5_000,
                 length_scale: float = 0.5,
                 rng_seed:     int   = 0):
        self.space        = search_space
        self.n_initial    = n_initial
        self.xi           = xi
        self.n_candidates = n_candidates
        self.rng          = np.random.default_rng(rng_seed)
        self.sampler      = ParamSampler(search_space, rng=self.rng)
        self.gp           = _GaussianProcess(length_scale=length_scale)

        self._X: list = []   # encoded (float) observations
        self._y: list = []   # observed RMSE values
        self._configs: list = []   # original config dicts

        # Build per-param encoding metadata once
        self._enc_meta = self._build_encoding_meta()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _build_encoding_meta(self) -> list:
        """Return a list of (param_name, encode_fn, decode_fn) triples."""
        meta = []
        for name, spec in self.space.items():
            t = spec["type"]

            if t == "float" and spec.get("log", False):
                lo, hi = math.log(spec["low"]), math.log(spec["high"])
                enc = lambda v, lo=lo, hi=hi: (math.log(v) - lo) / (hi - lo)
                dec = lambda u, lo=lo, hi=hi: float(math.exp(u * (hi - lo) + lo))

            elif t == "float":
                lo, hi = spec["low"], spec["high"]
                enc = lambda v, lo=lo, hi=hi: (v - lo) / (hi - lo)
                dec = lambda u, lo=lo, hi=hi: float(u * (hi - lo) + lo)

            elif t == "int":
                lo, hi = spec["low"], spec["high"]
                enc = lambda v, lo=lo, hi=hi: (v - lo) / max(hi - lo, 1)
                dec = lambda u, lo=lo, hi=hi: int(round(u * (hi - lo) + lo))

            elif t in ("int_choice", "float_choice", "categorical"):
                vals = spec["values"]
                n    = len(vals)
                enc  = lambda v, vals=vals, n=n: vals.index(v) / max(n - 1, 1)
                dec  = lambda u, vals=vals, n=n: vals[int(round(u * (n - 1)))]

            else:
                raise ValueError(f"Unknown type: {t!r}")

            meta.append((name, enc, dec))
        return meta

    def _encode(self, config: dict) -> np.ndarray:
        return np.array([enc(config[name]) for name, enc, _ in self._enc_meta],
                        dtype=np.float64)

    def _decode(self, x: np.ndarray) -> dict:
        return {name: dec(float(x[i]))
                for i, (name, _, dec) in enumerate(self._enc_meta)}

    # ------------------------------------------------------------------
    # Core BO loop interface
    # ------------------------------------------------------------------

    def suggest(self) -> dict:
        """Return the next configuration to evaluate."""
        if len(self._X) < self.n_initial:
            return self.sampler.sample()

        X = np.array(self._X)
        y = np.array(self._y)
        self.gp.fit(X, y)

        # Sample a large random set of candidates in [0,1]^d
        d          = len(self._enc_meta)
        candidates = self.rng.uniform(0.0, 1.0, size=(self.n_candidates, d))

        mu, var = self.gp.predict(candidates)
        best    = float(np.min(y))
        sigma   = np.sqrt(np.maximum(var, 0.0)) + 1e-8
        Z       = (best - mu - self.xi) / sigma
        ei      = np.maximum((best - mu - self.xi) * _normal_cdf(Z)
                             + sigma * _normal_pdf(Z), 0.0)

        best_idx = int(np.argmax(ei))
        return self._decode(candidates[best_idx])

    def update(self, config: dict, rmse_mean: float):
        """Register an observed (config, score) pair."""
        self._X.append(self._encode(config))
        self._y.append(rmse_mean)
        self._configs.append(config)

    def best(self) -> tuple:
        """Return (best_config, best_rmse_mean)."""
        if not self._y:
            return None, float("inf")
        idx = int(np.argmin(self._y))
        return self._configs[idx], self._y[idx]
