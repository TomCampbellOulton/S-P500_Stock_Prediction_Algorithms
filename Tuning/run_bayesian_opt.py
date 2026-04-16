# Tuning/run_bayesian_opt.py
# Gaussian-Process Bayesian Optimisation for hyperparameter tuning.
# Uses a numpy-only RBF-kernel GP surrogate with Expected Improvement (EI)
# acquisition — no scipy, no scikit-learn, no optuna.
#
# Algorithm per model
# -------------------
#   1. n_initial  random warm-up trials (exploration).
#   2. For each subsequent trial:
#       a. Fit GP to all observations so far.
#       b. Sample n_candidates random points in [0,1]^d.
#       c. Evaluate EI at every candidate; propose the argmax.
#       d. Run the model; register (config, RMSE) with the GP.
#   3. Save and report the best found configuration.
#
# Run directly:
#   python -m Tuning.run_bayesian_opt
#   python -m Tuning.run_bayesian_opt --n_trials 25 --n_initial 5
#
# Results are saved to  Tuning/Results/<model>__bayesian_opt__<timestamp>.csv

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tuning.tuning_config import SEARCH_SPACES, MODEL_META, TUNABLE_MODELS
from Tuning.tuning_utils   import run_trial, save_results, print_best, BayesianOptimizer

METHOD = "bayesian_opt"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_N_TRIALS          = 25      # total trials per model (includes warm-up)
DEFAULT_N_INITIAL         = 5       # random warm-up trials before GP takes over
DEFAULT_XI                = 0.01    # EI exploration-exploitation trade-off
DEFAULT_N_CANDIDATES      = 5_000   # random candidates per acquisition step
DEFAULT_LENGTH_SCALE      = 0.5     # GP RBF kernel length scale (on [0,1]^d)
DEFAULT_START_DATE        = "2010-01-01"
DEFAULT_END_DATE          = "2025-12-31"
DEFAULT_DATA_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_N_STEPS           = 5
DEFAULT_RMSE_MODE         = "price"
DEFAULT_SEED              = 0


# ─────────────────────────────────────────────────────────────────────────────
# Per-model Bayesian optimisation
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_name: str,
              n_trials:          int   = DEFAULT_N_TRIALS,
              n_initial:         int   = DEFAULT_N_INITIAL,
              xi:                float = DEFAULT_XI,
              n_candidates:      int   = DEFAULT_N_CANDIDATES,
              length_scale:      float = DEFAULT_LENGTH_SCALE,
              start_date:        str   = DEFAULT_START_DATE,
              end_date:          str   = DEFAULT_END_DATE,
              data_split_ratios: tuple = DEFAULT_DATA_SPLIT_RATIOS,
              n_steps:           int   = DEFAULT_N_STEPS,
              rmse_mode:         str   = DEFAULT_RMSE_MODE,
              seed:              int   = DEFAULT_SEED) -> list:
    """
    Run Bayesian optimisation for *model_name*.
    Returns list of result dicts (all observed trials, sorted by rmse_mean).
    """
    if model_name not in SEARCH_SPACES:
        print(f"[bayesian_opt] No SEARCH_SPACE defined for {model_name} — skipping.")
        return []

    meta = MODEL_META[model_name]

    bo = BayesianOptimizer(
        search_space = SEARCH_SPACES[model_name],
        n_initial    = n_initial,
        xi           = xi,
        n_candidates = n_candidates,
        length_scale = length_scale,
        rng_seed     = seed,
    )

    n_random = min(n_initial, n_trials)
    n_guided = max(0, n_trials - n_random)

    print(f"\n[bayesian_opt] {model_name}  —  "
          f"{n_random} random + {n_guided} GP-guided  ({n_trials} total)")

    results = []

    for trial in range(1, n_trials + 1):
        phase  = "random" if trial <= n_initial else "GP-EI "
        params = bo.suggest()

        print(f"  [{trial}/{n_trials}] ({phase}) {params}", end=" ... ", flush=True)

        result = run_trial(
            model_name        = model_name,
            params            = params,
            split_kwarg       = meta["split_kwarg"],
            start_date        = start_date,
            end_date          = end_date,
            data_split_ratios = data_split_ratios,
            n_steps           = n_steps,
            rmse_mode         = rmse_mode,
        )

        if result is not None:
            rmse_mean = result["rmse_mean"]
            bo.update(params, rmse_mean)
            results.append(result)
            print(f"rmse_mean={rmse_mean:.5f}")
        else:
            # Register a high penalty so the GP steers away from this region
            bo.update(params, float("inf"))
            print("FAILED")

    # Filter out any +inf penalty rows before saving
    valid = [r for r in results if r is not None]
    if valid:
        save_results(valid, model_name, METHOD)
        print_best(valid, model_name, METHOD)

    best_cfg, best_rmse = bo.best()
    if best_cfg is not None:
        print(f"  → Optimal config: {best_cfg}  (rmse_mean={best_rmse:.5f})")

    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Run all models
# ─────────────────────────────────────────────────────────────────────────────

def run_all(models:             list  = None,
            n_trials:          int   = DEFAULT_N_TRIALS,
            n_initial:         int   = DEFAULT_N_INITIAL,
            xi:                float = DEFAULT_XI,
            n_candidates:      int   = DEFAULT_N_CANDIDATES,
            length_scale:      float = DEFAULT_LENGTH_SCALE,
            start_date:        str   = DEFAULT_START_DATE,
            end_date:          str   = DEFAULT_END_DATE,
            data_split_ratios: tuple = DEFAULT_DATA_SPLIT_RATIOS,
            n_steps:           int   = DEFAULT_N_STEPS,
            rmse_mode:         str   = DEFAULT_RMSE_MODE,
            seed:              int   = DEFAULT_SEED) -> dict:
    """
    Run Bayesian optimisation for every model in *models* (defaults to TUNABLE_MODELS).
    Returns  { model_name: [result_dicts] }.
    """
    if models is None:
        models = TUNABLE_MODELS

    all_results = {}
    for name in models:
        all_results[name] = run_model(
            name,
            n_trials          = n_trials,
            n_initial         = n_initial,
            xi                = xi,
            n_candidates      = n_candidates,
            length_scale      = length_scale,
            start_date        = start_date,
            end_date          = end_date,
            data_split_ratios = data_split_ratios,
            n_steps           = n_steps,
            rmse_mode         = rmse_mode,
            seed              = seed,
        )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bayesian (GP + EI) hyperparameter optimisation")
    parser.add_argument("--models",       nargs="*",  default=None)
    parser.add_argument("--n_trials",     type=int,   default=DEFAULT_N_TRIALS)
    parser.add_argument("--n_initial",    type=int,   default=DEFAULT_N_INITIAL)
    parser.add_argument("--xi",           type=float, default=DEFAULT_XI)
    parser.add_argument("--n_candidates", type=int,   default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--length_scale", type=float, default=DEFAULT_LENGTH_SCALE)
    parser.add_argument("--start_date",   default=DEFAULT_START_DATE)
    parser.add_argument("--end_date",     default=DEFAULT_END_DATE)
    parser.add_argument("--n_steps",      type=int,   default=DEFAULT_N_STEPS)
    parser.add_argument("--rmse_mode",    default=DEFAULT_RMSE_MODE,
                        choices=["price", "relative"])
    parser.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    args = parser.parse_args()

    run_all(
        models        = args.models,
        n_trials      = args.n_trials,
        n_initial     = args.n_initial,
        xi            = args.xi,
        n_candidates  = args.n_candidates,
        length_scale  = args.length_scale,
        start_date    = args.start_date,
        end_date      = args.end_date,
        n_steps       = args.n_steps,
        rmse_mode     = args.rmse_mode,
        seed          = args.seed,
    )
