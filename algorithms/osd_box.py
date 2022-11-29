from algorithms.rcosd_generic import RCOSD_generic_algorithm
import numpy as np

class OSD_Box_algorithm(RCOSD_generic_algorithm) :
   
    def __init__(self, initial_decision:np.array, y_min:np.array, y_max:np.array, gamma, G) :
        self.gamma = gamma
        self.diameter = np.sqrt((y_max-y_min)*(y_max-y_min))
        self.G = G

        def learning_rate(t,cycle_counter, accumulated_cycle_gradients_norm_squared) :
            return self.gamma*self.diameter/(self.G*np.sqrt((t-1)))

        projection = lambda y : np.clip(y,y_min,y_max)
        trigger_event = lambda t, state, subgradient, sales, demands : True
        relaxation_parameter = lambda t,cycle_counter,last_update_period : 0

        super().__init__(initial_decision, learning_rate, projection, trigger_event, relaxation_parameter)

    def __str__(self) :
        return r"OSD_Box $\gamma={}$".format(self.gamma)