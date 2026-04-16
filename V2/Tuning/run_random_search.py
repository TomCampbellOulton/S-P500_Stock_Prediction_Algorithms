# Tuning/run_random_search.py
# Random hyperparameter search sampled from the continuous/discrete distributions
# defined in SEARCH_SPACES (tuning_config.py).
#
# Run directly:
#   python -m Tuning.run_random_search
#   python -m Tuning.run_random_search --n_trials 30 --models Modularised_ANN
#
# Results are saved to  Tuning/Results/<model>__random_search__<timestamp>.csv

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from Tuning.tuning_config import SEARCH_SPACES, MODEL_META, TUNABLE_MODELS
from Tuning.tuning_utils   import run_trial, save_results, print_best, ParamSampler

METHOD = "random_search"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_N_TRIALS          = 20      # trials per model
DEFAULT_START_DATE        = "2010-01-01"
DEFAULT_END_DATE          = "2025-12-31"
DEFAULT_DATA_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_N_STEPS           = 5
DEFAULT_RMSE_MODE         = "price"
DEFAULT_SEED              = 42


# ─────────────────────────────────────────────────────────────────────────────
# Per-model random search
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_name: str,
              n_trials:          int   = DEFAULT_N_TRIALS,
              start_date:        str   = DEFAULT_START_DATE,
              end_date:          str   = DEFAULT_END_DATE,
              data_split_ratios: tuple = DEFAULT_DATA_SPLIT_RATIOS,
              n_steps:           int   = DEFAULT_N_STEPS,
              rmse_mode:         str   = DEFAULT_RMSE_MODE,
              seed:              int   = DEFAULT_SEED) -> list:
    """
    Run *n_trials* random trials for *model_name*.
    Returns list of result dicts.
    """
    if model_name not in SEARCH_SPACES:
        print(f"[random_search] No SEARCH_SPACE defined for {model_name} — skipping.")
        return []

    meta    = MODEL_META[model_name]
    rng     = np.random.default_rng(seed)
    sampler = ParamSampler(SEARCH_SPACES[model_name], rng=rng)

    print(f"\n[random_search] {model_name}  —  {n_trials} trials")

    results = []
    for trial in range(1, n_trials + 1):
        params = sampler.sample()
        print(f"  [{trial}/{n_trials}] {params}", end=" ... ", flush=True)

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
            results.append(result)
            print(f"rmse_mean={result['rmse_mean']:.5f}")
        else:
            print("FAILED")

    if results:
        save_results(results, model_name, METHOD)
        print_best(results, model_name, METHOD)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Run all models
# ─────────────────────────────────────────────────────────────────────────────

def run_all(models:             list  = None,
            n_trials:          int   = DEFAULT_N_TRIALS,
            start_date:        str   = DEFAULT_START_DATE,
            end_date:          str   = DEFAULT_END_DATE,
            data_split_ratios: tuple = DEFAULT_DATA_SPLIT_RATIOS,
            n_steps:           int   = DEFAULT_N_STEPS,
            rmse_mode:         str   = DEFAULT_RMSE_MODE,
            seed:              int   = DEFAULT_SEED) -> dict:
    """
    Run random search for every model in *models* (defaults to TUNABLE_MODELS).
    Returns  { model_name: [result_dicts] }.
    """
    if models is None:
        models = TUNABLE_MODELS

    all_results = {}
    for name in models:
        all_results[name] = run_model(
            name,
            n_trials          = n_trials,
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

    parser = argparse.ArgumentParser(description="Random hyperparameter search")
    parser.add_argument("--models",    nargs="*",  default=None,
                        help="Model names to tune (default: all)")
    parser.add_argument("--n_trials",  type=int,   default=DEFAULT_N_TRIALS)
    parser.add_argument("--start_date", default=DEFAULT_START_DATE)
    parser.add_argument("--end_date",   default=DEFAULT_END_DATE)
    parser.add_argument("--n_steps",   type=int,   default=DEFAULT_N_STEPS)
    parser.add_argument("--rmse_mode", default=DEFAULT_RMSE_MODE,
                        choices=["price", "relative"])
    parser.add_argument("--seed",      type=int,   default=DEFAULT_SEED)
    args = parser.parse_args()

    run_all(
        models     = args.models,
        n_trials   = args.n_trials,
        start_date = args.start_date,
        end_date   = args.end_date,
        n_steps    = args.n_steps,
        rmse_mode  = args.rmse_mode,
        seed       = args.seed,
    )
