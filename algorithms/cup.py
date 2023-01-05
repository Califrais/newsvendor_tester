from algorithms.rcosd_generic import RCOSD_generic_algorithm
import numpy as np

class CUP_algorithm(RCOSD_generic_algorithm) :
    def __init__(self, initial_decision, y_min, y_max, gamma, G) :
        self.gamma = gamma
        self.diameter = np.sqrt((y_max-y_min)*(y_max-y_min))
        self.G = G

        def learning_rate(t) :
            return self.gamma*self.diameter/(self.G*np.sqrt(self.cycle_counter))

        projection = lambda y,state : np.clip(y,y_min,y_max)
        trigger_event = lambda t, state, subgradient, sales, demands : (state==0).all()
        relaxation_parameter = lambda t : 0

        super().__init__(initial_decision, learning_rate, projection, trigger_event, relaxation_parameter)

    def __str__(self) :
        return r"CUP $\gamma={}$".format(self.gamma)