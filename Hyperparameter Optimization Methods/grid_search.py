import itertools
import numpy as np

class GridSearch:
    def __init__(self, search_space, run_model, budget=1.0):
        """
        search_space: dict
        e.g. {"k": [2,3,5], "N_STEPS": [1,3,5]}
        """
        self.search_space = search_space
        self.run_model = run_model
        self.budget = budget

    def run(self):
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())

        best_config = None
        best_score = -np.inf

        for i, combo in enumerate(itertools.product(*values)):
            config = dict(zip(keys, combo))
            score = self.run_model(config, self.budget)

            if score > best_score:
                best_score = score
                best_config = config

            print(f"[GridSearch] Iter {i+1} → {score:.4f}")

        return best_config, best_score