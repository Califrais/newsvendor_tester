from typing import List
from algorithms.algorithm import Algorithm
import numpy as np

class Parallelizer(Algorithm) :
    
    def __init__(self, algorithm_list : List[Algorithm], name : str ="Parallelizer") :
        self.nb_products = len(algorithm_list)
        self.algorithm_list = algorithm_list
        self.name = name

    def next_decision(self,t,state, subgradient, sales,demands) :
        decision = np.zeros(self.nb_products)
        for product_id, algorithm in enumerate(self.algorithm_list) :
            decision[product_id] = algorithm.next_decision(
                t,
                state[product_id]*np.ones(1),
                subgradient[product_id]*np.ones(1),
                sales[product_id]*np.ones(1),
                demands[product_id]*np.ones(1)
            )[0]
        return decision
    
    def __str__(self) :
        return self.name

    def reset(self) :
        for algorithm in self.algorithm_list :
            algorithm.reset()