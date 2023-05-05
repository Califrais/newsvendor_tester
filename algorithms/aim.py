from algorithms.algorithm import Algorithm
import numpy as np

class AIM_algorithm(Algorithm) :
    def __init__(self, initial_decision, y_min, y_max, gamma, G, holding_costs, penalty_costs) :
        self.nb_products, = initial_decision.shape
        self.initial_decision = initial_decision
        self.gamma = gamma
        self.y_min = y_min
        self.y_max = y_max
        self.G = G
        self.holding_costs = holding_costs
        self.penalty_costs = penalty_costs

        self.D = np.sqrt(np.sum((y_max-y_min)*(y_max-y_min)))
        self.reset()

    def next_decision(self, t,state, subgradient, sales,demands) :
        if(t==1) :
            self.target_decision = np.array(self.initial_decision,dtype=np.float64)
            self.implemented_decision = np.array(self.initial_decision,dtype=np.float64)
        else: 
            target_subgradient = np.where(self.target_decision>sales,self.holding_costs,-self.penalty_costs)
            learning_rate = self.gamma*self.D/(self.G*np.sqrt(t-1))
            self.target_decision = np.clip(self.target_decision-learning_rate*target_subgradient,self.y_min,self.y_max)
            self.implemented_decision = np.maximum(state,self.target_decision)
            assert (state <= self.implemented_decision).all(), "Weird stuff happenning"
        return np.array(self.implemented_decision)

    def __str__(self) :
        return r"AIM $\gamma={:.3e}$".format(self.gamma)
    
    def reset(self) :
        self.target_decision = np.array(self.initial_decision,dtype=np.float64)
        self.implemented_decision = np.array(self.initial_decision,dtype=np.float64)