from typing import Callable
from algorithms.rcosd_generic import RCOSD_generic_algorithm
import utils
import numpy as np

class COSD_Variant2_Volume_Constrained_algorithm(RCOSD_generic_algorithm) :
    
    def __init__(self, initial_decision:np.array, volumes:np.array, total_volume:float, gamma) :
        self.gamma = gamma
        self.diameter = np.minimum(total_volume*np.sqrt(2)/np.min(volumes), total_volume*np.sqrt(np.sum(1/volumes**2)))
        
        def learning_rate(t) :
            if(self.accumulated_cycle_gradients_norm_squared == 0) :
                return 0
            return self.gamma*self.diameter/np.sqrt(self.accumulated_cycle_gradients_norm_squared)

        def projection(y,state) : 
            return utils.projection(y, volumes, total_volume)

        def trigger_event(t, state, subgradient, sales, demands) :
            learning_rate_value = 0
            if(self.accumulated_cycle_gradients_norm_squared+np.sum(self.cycle_gradient*self.cycle_gradient) > 0 ) :
                learning_rate_value = self.gamma*self.diameter/np.sqrt(self.accumulated_cycle_gradients_norm_squared+np.sum(self.cycle_gradient*self.cycle_gradient))

            return ( state <= utils.projection(self.decision-learning_rate_value*self.cycle_gradient, volumes, total_volume) ).all()

        relaxation_parameter = lambda t : 0
        
        super().__init__(initial_decision, learning_rate, projection, trigger_event, relaxation_parameter)

    
    def __str__(self) :
        return r"COSD Variant2 $\gamma={}$".format(self.gamma)