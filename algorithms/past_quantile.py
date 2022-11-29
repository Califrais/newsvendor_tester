import numpy as np
import pandas as pd
from algorithms.algorithm import Algorithm

class past_quantile_algorithm(Algorithm) :
    def __init__(self, nb_products, demands, alpha) :
        self.nb_products = nb_products
        self.quantiles = np.zeros(demands.shape)
        for k in range(nb_products) :
            self.quantiles[1:,k] = np.array(pd.Series(demands[1:,k]).expanding().quantile(alpha))

    def next_decision(self, t, state, subgradient, sales, demands) :
        return self.quantiles[t]
    
    def __str__(self) :
        return "PastQuantile"