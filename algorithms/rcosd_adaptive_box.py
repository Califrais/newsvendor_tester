from typing import Callable
from algorithms.rcosd_generic import RCOSD_generic_algorithm
import numpy as np

class RCOSD_Adaptive_Box_algorithm(RCOSD_generic_algorithm) :
    
    def __init__(self, initial_decision:np.array, y_min:np.array, y_max:np.array, gamma) :
        self.gamma = gamma
        self.diameter = np.sqrt((y_max-y_min)*(y_max-y_min))
        
        def learning_rate(t,cycle_counter, accumulated_cycle_gradients_norm_squared) :
            if(accumulated_cycle_gradients_norm_squared == 0) :
                return 0
            return self.gamma*self.diameter/np.sqrt(accumulated_cycle_gradients_norm_squared)

        projection = lambda y : np.clip(y,y_min,y_max)
        trigger_event = lambda t, state, subgradient, sales, demands : (sales>0).all()
        relaxation_parameter = lambda t,cycle_counter,last_update_period : 1/(2**(np.sqrt(cycle_counter))*(t-last_update_period))

        super().__init__(initial_decision, learning_rate, projection, trigger_event, relaxation_parameter)

    
    def __str__(self) :
        return r"RCOSD_Adaptive_Box $\gamma={}$".format(self.gamma)

    #     self.nb_products = nb_products
    #     self.gamma = gamma
    #     self.y_min = y_min
    #     self.y_max = y_max
    #     self.G = G

    #     self.D = np.max(y_max-y_min)
    #     self.decision = np.array(y_min+y_max)/2

    #     self.accumulated_gradients = np.zeros(nb_products)
    #     self.learning_rate_denominator_squared = 0.0
    #     self.cycle_counter = 1
    #     self.last_update_period = 1

    # def next_decision(self, t, state, subgradient, sales) :
    #     if(t==1) :
    #         self.decision = np.array(self.y_min+self.y_max)/2
    #         return self.decision
    #     else:             
    #         self.accumulated_gradients += subgradient

    #         if((sales>0).all()) :
    #             theta = 1/(2**(np.sqrt(self.cycle_counter))*(t-self.last_update_period))
    #             self.learning_rate_denominator_squared += np.sum(self.accumulated_gradients*self.accumulated_gradients)
    #             if(self.learning_rate_denominator_squared == 0) :
    #                 learning_rate = 0
    #             else :
    #                 learning_rate = self.gamma*self.D/np.sqrt(self.learning_rate_denominator_squared)
    #             self.decision = (1-theta)*np.clip(self.decision-learning_rate*self.accumulated_gradients,self.y_min,self.y_max) + theta*self.decision

    #             self.cycle_counter += 1
    #             self.last_update_period = t
    #             self.accumulated_gradients = np.zeros(self.nb_products)
    #             return np.array(self.decision)
    #         else :
    #             return np.array(self.decision)