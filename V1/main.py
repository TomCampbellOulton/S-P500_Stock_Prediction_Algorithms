import importlib
import os

from upgraded_utilities import save_models_data, plot_candlestick_graphs, plot_mp_graphs


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
    