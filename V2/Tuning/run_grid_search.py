# Tuning/run_grid_search.py
# Exhaustive grid search over the discrete parameter grids defined in tuning_config.py.
#
# Run directly:
#   python -m Tuning.run_grid_search
#
# Or from main.py:
#   from Tuning.run_grid_search import run_all
#   run_all()
#
# Results are saved to  Tuning/Results/<model>__grid_search__<timestamp>.csv
# with rows sorted by rmse_mean ascending.

import os
import sys
import itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Tuning.tuning_config import PARAM_GRIDS, MODEL_META, TUNABLE_MODELS
from Tuning.tuning_utils   import run_trial, save_results, print_best

METHOD = "grid_search"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — override via run_all() kwargs or edit here
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_START_DATE        = "2010-01-01"
DEFAULT_END_DATE          = "2025-12-31"
DEFAULT_DATA_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_N_STEPS           = 5
DEFAULT_RMSE_MODE         = "price"


# ─────────────────────────────────────────────────────────────────────────────
# Per-model grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_name: str,
              start_date:        str   = DEFAULT_START_DATE,
              end_date:          str   = DEFAULT_END_DATE,
              data_split_ratios: tuple = DEFAULT_DATA_SPLIT_RATIOS,
              n_steps:           int   = DEFAULT_N_STEPS,
              rmse_mode:         str   = DEFAULT_RMSE_MODE) -> list:
    """
    Run an exhaustive grid search for *model_name*.
    Returns the list of result dicts (sorted by rmse_mean).
    """
    if model_name not in PARAM_GRIDS:
        print(f"[grid_search] No PARAM_GRID defined for {model_name} — skipping.")
        return []

    meta = MODEL_META[model_name]
    if "grid" in meta.get("exclude_from", []):
        print(f"[grid_search] {model_name} is excluded from grid search — skipping.")
        return []

    grid  = PARAM_GRIDS[model_name]
    keys  = list(grid.keys())
    vals  = [grid[k] for k in keys]
    combos = list(itertools.product(*vals))
    total  = len(combos)

    print(f"\n[grid_search] {model_name}  —  {total} combinations")

    results = []
    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"  [{idx}/{total}] {params}", end=" ... ", flush=True)

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
            start_date:        str   = DEFAULT_START_DATE,
            end_date:          str   = DEFAULT_END_DATE,
            data_split_ratios: tuple = DEFAULT_DATA_SPLIT_RATIOS,
            n_steps:           int   = DEFAULT_N_STEPS,
            rmse_mode:         str   = DEFAULT_RMSE_MODE) -> dict:
    """
    Run grid search for every model in *models* (defaults to TUNABLE_MODELS).

    Returns a dict  { model_name: [result_dicts] }.
    """
    if models is None:
        models = TUNABLE_MODELS

    all_results = {}
    for name in models:
        all_results[name] = run_model(
            name,
            start_date        = start_date,
            end_date          = end_date,
            data_split_ratios = data_split_ratios,
            n_steps           = n_steps,
            rmse_mode         = rmse_mode,
        )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid-search hyperparameter tuning")
    parser.add_argument("--models",      nargs="*",  default=None,
                        help="Model names to tune (default: all)")
    parser.add_argument("--start_date",  default=DEFAULT_START_DATE)
    parser.add_argument("--end_date",    default=DEFAULT_END_DATE)
    parser.add_argument("--n_steps",     type=int,   default=DEFAULT_N_STEPS)
    parser.add_argument("--rmse_mode",   default=DEFAULT_RMSE_MODE,
                        choices=["price", "relative"])
    args = parser.parse_args()

    run_all(
        models    = args.models,
        start_date = args.start_date,
        end_date   = args.end_date,
        n_steps    = args.n_steps,
        rmse_mode  = args.rmse_mode,
    )
