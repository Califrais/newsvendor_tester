import numpy as np
from algorithms.algorithm import Algorithm
import utils

class DDM_algorithm(Algorithm) :
    def __init__(self, initial_decision, volumes, total_volume, gamma, G, holding_costs, penalty_costs):
        self.initial_decision = initial_decision
        self.G = G
        self.volumes = volumes
        self.total_volume = total_volume
        self.gamma = gamma
        self.diameter = np.minimum(total_volume*np.sqrt(2)/np.min(volumes), total_volume*np.sqrt(np.sum(1/volumes**2)))
        self.holding_costs = holding_costs
        self.penalty_costs = penalty_costs
        
        self.reset()
    
    def reset(self) :
        self.implemented_decision = np.array(self.initial_decision,dtype=np.float64)
        self.target_decision = np.array(self.initial_decision,dtype=np.float64)

    def next_decision(self,t, state, subgradient, sales, demands) :
        if((self.implemented_decision >= self.target_decision).all()) :
            subgradient_at_target = np.where(self.target_decision>demands,self.holding_costs,-self.penalty_costs) 
            learning_rate = self.gamma * self.diameter / ( self.G*np.sqrt(t) )
            self.target_decision = utils.projection(self.target_decision-learning_rate*subgradient_at_target, self.volumes, self.total_volume)
        
        j_indexes = np.where(state>self.target_decision)[0]
        j_complementary_indexes = np.where(state<=self.target_decision)[0]
        self.implemented_decision[j_indexes] = state[j_indexes]

        if(len(j_complementary_indexes) > 0) :
            self.implemented_decision[j_complementary_indexes] = utils.projection(
                self.target_decision[j_complementary_indexes],
                self.volumes[j_complementary_indexes],
                self.total_volume-np.sum((self.volumes*state)[j_indexes]),
                state[j_complementary_indexes]
            )

        return np.array(self.implemented_decision)


    def __str__(self) :
        return r"DDM $\gamma={:.3e}$".format(self.gamma)

   