import importlib
import os
import matplotlib.pyplot as plt

files = [f[:-3] for f in os.listdir("./Models") if f.endswith(".py") and not f.startswith("_")]

models        = {}
model_results = {}

for name in files:
    module = importlib.import_module(f"Models.{name}")
    if hasattr(module, "main"):
        models[name] = module.main
    else:
        print(f"{name} has no main() — skipping")

for name, main_function in models.items():
    try:
        # Each model returns a DataFrame with columns:
        # open_rel, high_rel, low_rel, close_rel, open, high, low, close
        model_results[name] = main_function()
    except Exception as e:
        print(f"{name} failed: {e}")

# ---- Plot relative values (left) and absolute prices (right) and then the daisy chained / domino ones o-o ----
price_cols = ["open", "high", "low", "close"]
rel_cols   = ["open_rel", "high_rel", "low_rel", "close_rel"]
d_c_price_cols = ["daisy_chained_open", "daisy_chained_high", "daisy_chained_low", "daisy_chained_close"] 
d_c_rel_cols = ["daisy_chained_open_rel", "daisy_chained_high_rel", "daisy_chained_low_rel", "daisy_chained_close_rel"]

fig, axes = plt.subplots(4, 4, figsize=(16, 14))

for row, (rel_col, abs_col, d_c_rel_col, d_c_abs_col) in enumerate(zip(rel_cols, price_cols, d_c_rel_cols, d_c_price_cols)):
    ax_rel, ax_abs, ax_dc_rel, ax_dc_abs = axes[row][0], axes[row][1], axes[row][2], axes[row][3]
    for name, df in model_results.items():
        required_cols = [rel_col, abs_col, d_c_rel_col, d_c_abs_col]
        if all(col in df.columns for col in required_cols):
            ax_rel.plot(df.index, df[rel_col].values, label=name)
            ax_abs.plot(df.index, df[abs_col].values, label=name)
            ax_dc_rel.plot(df.index, df[d_c_rel_col].values, label=name)
            ax_dc_abs.plot(df.index, df[d_c_abs_col].values, label=name)
            
    ax_rel.set_title(f"Predicted {rel_col.capitalize()} (relative to open)")
    ax_rel.set_xlabel("Date")
    ax_rel.legend(fontsize=6)
    ax_abs.set_title(f"Predicted {abs_col.capitalize()} (price)")
    ax_abs.set_xlabel("Date")
    ax_abs.legend(fontsize=6)

    ax_dc_rel.set_title(f"Predicted {d_c_rel_col.capitalize()} (relative to open)")
    ax_dc_rel.set_xlabel("Date")
    ax_dc_rel.legend(fontsize=6)
    ax_dc_abs.set_title(f"Predicted {d_c_abs_col.capitalize()} (price)")
    ax_dc_abs.set_xlabel("Date")
    ax_dc_abs.legend(fontsize=6)

plt.tight_layout()
plt.show()
