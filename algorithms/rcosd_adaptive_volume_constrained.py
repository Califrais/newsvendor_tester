from typing import Callable
from algorithms.rcosd_generic import RCOSD_generic_algorithm
import utils
import numpy as np

class RCOSD_Adaptive_Volume_Constrained_algorithm(RCOSD_generic_algorithm) :
    
    def __init__(self, initial_decision:np.array, volumes:np.array, total_volume:float, gamma) :
        self.gamma = gamma
        self.diameter = np.minimum(total_volume*np.sqrt(2)/np.min(volumes), total_volume*np.sqrt(np.sum(1/volumes**2)))
        
        def learning_rate(t,cycle_counter, accumulated_cycle_gradients_norm_squared) :
            if(accumulated_cycle_gradients_norm_squared == 0) :
                return 0
            return self.gamma*self.diameter/np.sqrt(accumulated_cycle_gradients_norm_squared)

        projection = lambda y : utils.projection(y, volumes, total_volume)
        trigger_event = lambda t, state, subgradient, sales, demands : (sales>0).all()
        relaxation_parameter = lambda t,cycle_counter,last_update_period : 1/(2**(np.sqrt(cycle_counter))*(t-last_update_period))

        super().__init__(initial_decision, learning_rate, projection, trigger_event, relaxation_parameter)

    
    def __str__(self) :
        return r"RCOSD $\gamma={}$".format(self.gamma)