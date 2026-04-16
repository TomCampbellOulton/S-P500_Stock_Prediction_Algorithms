import importlib
import os

from upgraded_utilities import save_models_data, plot_candlestick_graphs, plot_mp_graphs


# Hyperparemeter Tuning toggle
RUN_TUNING = False

# Which tuning method(s) to run - Can be any subset of: "grid", "random", "bayesian"
TUNING_METHOD = "random"

# Which models to tune — None tunes all models in TUNABLE_MODELS.
# If only some need tuning, provide a list such as ["Modularised_ANN", "Modularised_DTR"]
TUNING_MODELS = None

# Shared tuning parameters — used by all three tuning methods
TUNING_KWARGS = dict(
    start_date        = "2010-01-01",
    end_date          = "2025-12-31",
    data_split_ratios = (0.8, 0.1, 0.1),
    n_steps           = 5,
    rmse_mode         = "price",
)

# Method-specific overrides (merged with TUNING_KWARGS when tuning is turned on).

# No additional kwargs needed
GRID_KWARGS    = {}
# Keep a fixed seed for consistency and determinism
RANDOM_KWARGS  = dict(n_trials=20, seed=42)
BAYESIAN_KWARGS = dict(n_trials=25, n_initial=5)


# Hyper parameter tuning (runs before model evaluation IF run tuning is set to true)
if RUN_TUNING:
    if TUNING_METHOD == "grid":
        from Tuning.run_grid_search import run_all as _tune
        _tune(models=TUNING_MODELS, **TUNING_KWARGS, **GRID_KWARGS)

    elif TUNING_METHOD == "random":
        from Tuning.run_random_search import run_all as _tune
        _tune(models=TUNING_MODELS, **TUNING_KWARGS, **RANDOM_KWARGS)

    elif TUNING_METHOD == "bayesian":
        from Tuning.run_bayesian_opt import run_all as _tune
        _tune(models=TUNING_MODELS, **TUNING_KWARGS, **BAYESIAN_KWARGS)

    else:
        raise ValueError(f"Unknown TUNING_METHOD={TUNING_METHOD!r}. Please choose from 'grid', 'random', or 'bayesian'.")


# Now evaluate the models

# All the model files are in a subdirectory called Models, all end in .py and any files to not be tested should start with _
files = [f[:-3] for f in os.listdir("./Models") if f.endswith(".py") and not f.startswith("_")]

# Stores the main function for each model as the value, with the key being the name of the file
models        = {}
# Stores a pandas dataframe outputted by the main function from each model as the value, with the key being the name of the model's file
model_results = {}

# Try each model, if it has a 'main' function, execute it and record the results
for name in files:
    module = importlib.import_module(f"Models.{name}")
    if hasattr(module, "main"):
        models[name] = module.main
    else:
        print(f"{name} has no main() — skipping")

# Iterates through every model, retrieving the name and main function
for name, main_function in models.items():
    try:
        # Each model returns a DataFrame with columns:
        # open_rel, high_rel, low_rel, close_rel, open, high, low, close
        # And the daisy chained versions for each, where the relatives (returns) should be identical to before,
        # but are displayed as a sanity check - if unequal there's an error
        model_results[name] = main_function()
    except Exception as e:
        print(f"{name} failed: {e}")



# --- Save all the models data into CSV files ---
for name in models.keys():
    save_models_data(name, model_results[name])


# Display the results in a candlestick plot and save the matplotlib plots
for name, df in model_results.items():
    # Call the plot matplot lib graphs function to render and save those plots
    plot_mp_graphs(df, name, save=True, show=False)
    # And call the pyplot candlesticks function to render, save and display those results
    plot_candlestick_graphs(df, name, save=True, show=True)
    