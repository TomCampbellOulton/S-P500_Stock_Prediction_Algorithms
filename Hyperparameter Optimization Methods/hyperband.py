import numpy as np
import math

class Hyperband:
    def __init__(self, get_config, run_model, max_resources=1.0, eta=3):
        self.get_config = get_config
        self.run_model = run_model
        self.R = max_resources
        self.eta = eta
        self.s_max = int(math.log(self.R, self.eta)) if self.R > 1 else 0
        self.B = (self.s_max + 1) * self.R

    def run(self):
        best_config = None
        best_score = -np.inf

        for s in reversed(range(self.s_max + 1)):
            n = int((self.B / self.R) * (self.eta ** s) / (s + 1))
            r = self.R * (self.eta ** (-s))

            configs = [self.get_config() for _ in range(n)]

            for i in range(s + 1):
                n_i = max(1, int(n * (self.eta ** (-i))))
                r_i = r * (self.eta ** i)

                scores = []
                for config in configs:
                    score = self.run_model(config, r_i)
                    scores.append((score, config))

                    if score > best_score:
                        best_score = score
                        best_config = config

                scores.sort(reverse=True, key=lambda x: x[0])
                configs = [c for _, c in scores[:max(1, int(n_i / self.eta))]]

        return best_config, best_score