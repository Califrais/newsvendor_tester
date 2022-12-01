import numpy as np
import scipy.stats 

class Environment :

    def __init__(self) :
        pass

    def reset(self) :
        pass

    def get_state(self,t, previous_level) -> np.array :
        pass

    def get_loss(self,t, level) :
        pass

    def get_subgradient(self,t, level) -> np.array :
        pass

    def get_demand(self,t) -> np.array :
        pass

    def get_sales(self,t, level) -> np.array :
        pass

class Environment_NonPerishable_Newsvendor(Environment) :
    """
    Multi-product lost sales inventory system with non-perishable products with lognormal demands and l1-norm loss.
    """
    def __init__(self, demands, holding_costs, penalty_costs) :
        self.horizon, self.nb_products = demands.shape
        self.horizon -= 1
        self.demands = demands
        self.holding_costs = holding_costs
        self.penalty_costs = penalty_costs

    def reset(self) :
        pass

    def get_state(self, t, last_decision):
        if(t<=1) :
            return np.zeros(self.nb_products)
        return np.maximum(0,last_decision-self.demands[t-1])
    
    def get_loss(self, t, decision) :
        return np.sum(self.holding_costs*np.maximum(0,decision-self.demands[t])+self.penalty_costs*np.maximum(0,self.demands[t]-decision))

    def get_sales(self,t,decision) :
        return np.minimum(decision, self.demands[t])

    def get_subgradient(self,t, decision) :
        return np.where(decision>self.demands[t],self.holding_costs,-self.penalty_costs)
    
    def get_demand(self,t) :
        return self.demands[t]

class Environment_Perishable_Newsvendor(Environment) :
    """
    Multi-product lost sales perishable inventory system with lognormal demands and l1-norm loss.
    """
    def __init__(self, lifetime:int, demands, holding_costs, penalty_costs) :
        self.horizon, self.nb_products = demands.shape
        self.horizon -= 1
        self.demands = demands
        self.holding_costs = holding_costs
        self.penalty_costs = penalty_costs

        self.lifetime = lifetime
        self.reset()

    def reset(self) :
        self.state = np.zeros((self.nb_products,self.lifetime-1))

    def get_state(self,t,last_decision) :
        if(t<=1) :
            self.state = np.zeros((self.nb_products,self.lifetime-1))
        else :
            new_state = np.zeros(self.state.shape)
            for k in range(self.nb_products) :
                for i in range(self.lifetime-2) :
                    new_state[k,i] = np.maximum(0, self.state[k,i+1] - self.demands[t-1]-np.maximum(0,self.state[k,0]-self.demands[t-1]))
                new_state[k,self.lifetime-2] = np.maximum(0, last_decision - self.demands[t-1]-np.maximum(0,self.state[k,0]-self.demands[t-1]))
            self.state = new_state
        return np.array(self.state[:,self.lifetime-2])

    def get_loss(self, t, decision) :
        return np.sum(self.holding_costs*np.maximum(0,decision-self.demands[t])+self.penalty_costs*np.maximum(0,self.demands[t]-decision))

    def get_sales(self,t,decision) :
        return np.minimum(decision, self.demands[t])

    def get_subgradient(self,t, decision) :
        return np.where(decision>self.demands[t],self.holding_costs,-self.penalty_costs)

    def get_demand(self,t) :
        return self.demands[t]
