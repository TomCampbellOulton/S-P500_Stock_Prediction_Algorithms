import numpy as np

class RandomSearch:
    def __init__(self, get_config, run_model, n_trials=50, budget=1.0):
        self.get_config = get_config
        self.run_model = run_model
        self.n_trials = n_trials
        self.budget = budget

    def run(self):
        best_config = None
        best_score = -np.inf

        for i in range(self.n_trials):
            config = self.get_config()
            score = self.run_model(config, self.budget)

            if score > best_score:
                best_score = score
                best_config = config

            print(f"[RandomSearch] Trial {i+1}/{self.n_trials} → {score:.4f}")

        return best_config, best_score