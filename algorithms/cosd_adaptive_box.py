from algorithms.rcosd_generic import RCOSD_generic_algorithm
import numpy as np

class COSD_Adaptive_Box_algorithm(RCOSD_generic_algorithm) :
    def __init__(self, initial_decision, y_min, y_max, gamma) :
        self.gamma = gamma
        self.diameter = np.sqrt((y_max-y_min)*(y_max-y_min))

        def learning_rate(t) :
            if(self.accumulated_cycle_gradients_norm_squared == 0) :
                return 0
            return self.gamma*self.diameter/np.sqrt(self.accumulated_cycle_gradients_norm_squared)

        projection = lambda y, state : np.clip(y,y_min,y_max)
        trigger_event = lambda t, state, subgradient, sales, demands : (demands>0).all()
        relaxation_parameter = lambda t : 0

        super().__init__(initial_decision, learning_rate, projection, trigger_event, relaxation_parameter)

    def __str__(self) :
        return r"COSD $\gamma={}$".format(self.gamma)