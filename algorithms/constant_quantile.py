import numpy as np
from algorithms.algorithm import Algorithm

class constant_quantile_algorithm(Algorithm) :
    def __init__(self, nb_products, demands, horizon, alpha) :
        self.nb_products = nb_products
        self.decision = np.quantile(demands[1:horizon+1], alpha, axis=0)

    def next_decision(self, t, state, subgradient, sales, demands) :
        return self.decision

    def __str__(self) :
        return "ConstantQuantile"