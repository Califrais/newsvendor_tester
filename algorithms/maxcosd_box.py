from typing import Callable
from algorithms.cosd_generic import COSD_generic_algorithm
import utils
import numpy as np

class MaxCOSD_Box_algorithm(COSD_generic_algorithm) :
    
    def __init__(self, initial_decision, y_min, y_max, gamma) :
        self.gamma = gamma
        self.diameter = np.sqrt((y_max-y_min)*(y_max-y_min))
        
        def learning_rate(t) :
            if(self.accumulated_cycle_gradients_norm_squared == 0) :
                return 0
            return self.gamma*self.diameter/np.sqrt(self.accumulated_cycle_gradients_norm_squared)

        projection = lambda y, state : np.clip(y,y_min,y_max)

        def trigger_event(t, state, subgradient, sales, demands) :
            learning_rate_value = 0
            if(self.accumulated_cycle_gradients_norm_squared+np.sum(self.cycle_gradient*self.cycle_gradient) > 0 ) :
                learning_rate_value = self.gamma*self.diameter/np.sqrt(self.accumulated_cycle_gradients_norm_squared+np.sum(self.cycle_gradient*self.cycle_gradient))

            return ( state <= projection(self.decision-learning_rate_value*self.cycle_gradient, state) ).all()
        
        super().__init__(initial_decision, learning_rate, projection, trigger_event)

    
    def __str__(self) :
        return r"MaxCOSD $\gamma={:.3e}$".format(self.gamma)