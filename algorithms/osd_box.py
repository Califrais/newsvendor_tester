from algorithms.cosd_generic import COSD_generic_algorithm
import numpy as np

class OSD_Box_algorithm(COSD_generic_algorithm) :
   
    def __init__(self, initial_decision:np.array, y_min:np.array, y_max:np.array, gamma, G) :
        self.gamma = gamma
        self.diameter = np.sqrt((y_max-y_min)*(y_max-y_min))
        self.G = G

        def learning_rate(t) :
            return self.gamma*self.diameter/(self.G*np.sqrt((t-1)))

        projection = lambda y,state : np.clip(y,y_min,y_max)
        trigger_event = lambda t, state, subgradient, sales, demands : True

        super().__init__(initial_decision, learning_rate, projection, trigger_event)

    def __str__(self) :
        return r"OSD_Box $\gamma={:.3e}$".format(self.gamma)