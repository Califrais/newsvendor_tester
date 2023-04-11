from typing import Callable
from algorithms.cosd_generic import COSD_generic_algorithm
import utils
import numpy as np

class OSD_Dynamic_Projection_Volume_Constrained_algorithm(COSD_generic_algorithm) :
    
    def __init__(self, initial_decision:np.array, volumes:np.array, total_volume:float, gamma) :
        self.gamma = gamma
        self.diameter = np.minimum(total_volume*np.sqrt(2)/np.min(volumes), total_volume*np.sqrt(np.sum(1/volumes**2)))
        
        def learning_rate(t) :
            if(self.accumulated_cycle_gradients_norm_squared == 0) :
                return 0
            return self.gamma*self.diameter/np.sqrt(self.accumulated_cycle_gradients_norm_squared)

        def projection(y,state) : 
            output = utils.projection(y, volumes, total_volume, min_bounds=state)
            return output

        def trigger_event(t, state, subgradient, sales, demands) :
            return True
            
        super().__init__(initial_decision, learning_rate, projection, trigger_event)

    
    def __str__(self) :
        return r"OSD DP Variant $\gamma={:.3e}$".format(self.gamma)