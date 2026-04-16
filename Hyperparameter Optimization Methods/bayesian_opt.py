import numpy as np
import random

class SimpleBayesianOpt:
    def __init__(self, search_space, run_model, n_init=10, n_iter=40, budget=1.0):
        self.search_space = search_space
        self.run_model = run_model
        self.n_init = n_init
        self.n_iter = n_iter
        self.budget = budget
        self.history = []

    def sample_random(self):
        return {
            k: random.choice(v)
            for k, v in self.search_space.items()
        }

    def run(self):
        best_config = None
        best_score = -np.inf

        # initial random phase
        for _ in range(self.n_init):
            config = self.sample_random()
            score = self.run_model(config, self.budget)
            self.history.append((config, score))

        # exploitation phase
        for i in range(self.n_iter):
            # pick best so far and mutate
            best_past = max(self.history, key=lambda x: x[1])[0]

            new_config = best_past.copy()
            key = random.choice(list(self.search_space.keys()))
            new_config[key] = random.choice(self.search_space[key])

            score = self.run_model(new_config, self.budget)
            self.history.append((new_config, score))

            if score > best_score:
                best_score = score
                best_config = new_config

            print(f"[BayesOpt] Iter {i+1} → {score:.4f}")

        return best_config, best_score