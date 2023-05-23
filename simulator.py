from typing import Callable, List
import numpy as np
from tqdm import tqdm
import scipy.stats

from environment import Environment 

class Simulator :

    def __init__(self, envs: List[Environment], nb_products:int, nb_samples: int, horizons:np.array, algs: list, condition_on_optimum:Callable, holding_costs, penalty_costs, first_seed = 1) :
        self.nb_algs = len(algs)
        self.algs = algs
        self.nb_products= nb_products
        self.nb_samples = nb_samples
        self.horizons = horizons
        self.envs = envs
        self.condition_on_optimum = condition_on_optimum
        self.holding_costs = holding_costs
        self.penalty_costs = penalty_costs
        self.first_seed = first_seed


    def run(self) :
        cum_losses = np.zeros((self.nb_samples, self.nb_algs+1,len(self.horizons)))

        for sample_id in range(self.nb_samples) : 
            env = self.envs[sample_id]
            for horizon_index in range(len(self.horizons)) :
                optimal_decision = np.zeros(self.nb_products)
                cum_losses[sample_id,0,horizon_index] = 0
                for i in range(self.nb_products) :
                    optimal_decision[i] = np.quantile(env.demands[1:self.horizons[horizon_index]+1,i], self.penalty_costs[i]/(self.holding_costs[i]+self.penalty_costs[i]))
                    cum_losses[sample_id,0,horizon_index] += np.sum(self.holding_costs[i]*np.maximum(0,optimal_decision[i]-env.demands[1:self.horizons[horizon_index]+1,i])
                    + self.penalty_costs[i]*np.maximum(0,env.demands[1:self.horizons[horizon_index]+1,i]-optimal_decision[i]))
                assert self.condition_on_optimum(optimal_decision), "The optimal order-up-to level does not satisfy the constraints, please consider less restrictive constraints."

            for alg_index in tqdm(range(0,self.nb_algs)) :
                yt, gt, st, dt = np.zeros(self.nb_products), np.zeros(self.nb_products), np.zeros(self.nb_products), np.zeros(self.nb_products)
                cum_lt = 0
                #flag_raised = False
                horizon_index = 0
                self.algs[alg_index].reset()

                for t in range(1,self.horizons[-1]+1) :
                    xt = env.get_state(t,yt)
                    yt = self.algs[alg_index].next_decision(t,xt,gt,st,dt)

                    if((yt-xt<-10**-5).any()) :
                        print("Undershooting error for alg {} at period {}:".format(alg_index,t))
                        print("x_t = {}, y_t = {}".format(xt,yt))
                        #flag_raised = True
                    
                    cum_lt += env.get_loss(t,yt)
                    if(t in self.horizons) :
                        cum_losses[sample_id,alg_index+1, horizon_index] = cum_lt
                        horizon_index += 1
                    
                    gt = env.get_subgradient(t,yt)
                    st = env.get_sales(t,yt)
                    dt = env.get_demand(t)
        return cum_losses
