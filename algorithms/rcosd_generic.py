from typing import Callable
from algorithms.algorithm import Algorithm
import numpy as np

class RCOSD_generic_algorithm(Algorithm) :
    
    def __init__(self, initial_decision: np.array, learning_rate: Callable, projection: Callable, trigger_event: Callable, relaxation_parameter:Callable) :
        self.nb_products, = initial_decision.shape
        self.initial_decision = initial_decision
        self.learning_rate = learning_rate
        self.projection = projection
        self.trigger_event = trigger_event
        self.relaxation_parameter = relaxation_parameter

        self.reset()

    def next_decision(self, t, state, subgradient, sales, demands):
        if(t==1) :
            self.decision = np.array(self.initial_decision,dtype=np.float64)
        else :
            self.cycle_gradient += subgradient
            if(self.trigger_event(t,state,subgradient,sales,demands)) :
                self.accumulated_cycle_gradients_norm_squared += np.sum(self.cycle_gradient*self.cycle_gradient)

                relaxation_paramater_value = self.relaxation_parameter(t,self.cycle_counter,self.last_update_period)
                learning_rate_value = self.learning_rate(t,self.cycle_counter,self.accumulated_cycle_gradients_norm_squared)
                self.decision = (1-relaxation_paramater_value)*self.projection(self.decision-learning_rate_value*self.cycle_gradient) + relaxation_paramater_value*self.decision

                self.cycle_counter += 1
                self.last_update_period = t
                self.cycle_gradient = np.zeros(self.nb_products)
        return self.decision
    
    def reset(self) :
        self.cycle_gradient = np.zeros(self.nb_products)
        self.accumulated_cycle_gradients_norm_squared = 0.0
        self.cycle_counter = 1
        self.last_update_period = 1

        self.decision = np.array(self.initial_decision,dtype=np.float64)

