from typing import Callable
import numpy as np
from tqdm import tqdm
import scipy.stats 

class Simulator :

    def __init__(self, env_generator: Callable, nb_products:int, nb_samples: int, horizons:np.array, algs: list, condition_on_optimum:Callable, holding_costs, penalty_costs, first_seed = 1, ) :
        self.nb_algs = len(algs)
        self.algs = algs
        self.nb_products= nb_products
        self.nb_samples = nb_samples
        self.horizons = horizons
        self.env_generator = env_generator
        self.condition_on_optimum = condition_on_optimum
        self.holding_costs = holding_costs
        self.penalty_costs = penalty_costs
        self.first_seed = first_seed


    def run(self) :
        cum_losses = np.zeros((self.nb_samples, self.nb_algs+1,len(self.horizons)))

        np.random.seed(self.first_seed)
        seeds = np.random.randint(0,self.nb_samples,self.nb_samples)

        for seed_id in tqdm(range(self.nb_samples)) : 

            np.random.seed(seeds[seed_id])
            env = self.env_generator()
            for horizon_index in range(len(self.horizons)) :
                optimal_decision = np.zeros(self.nb_products)
                for i in range(self.nb_products) :
                    optimal_decision[i] = np.quantile(env.demands[1:self.horizons[horizon_index]+1,i], self.penalty_costs[i]/(self.holding_costs[i]+self.penalty_costs[i]))
                cum_losses[seed_id,0,horizon_index] = np.sum(self.holding_costs*np.maximum(0,optimal_decision-env.demands[1:self.horizons[horizon_index]+1])
                    + self.penalty_costs*np.maximum(0,env.demands[1:self.horizons[horizon_index]+1]-optimal_decision))
                assert self.condition_on_optimum(optimal_decision), "Condition on optimum failed"

            for alg_index in range(0,self.nb_algs) :
                yt, gt, st, dt = np.zeros(self.nb_products), np.zeros(self.nb_products), np.zeros(self.nb_products), np.zeros(self.nb_products)
                cum_lt = 0
                #flag_raised = False
                horizon_index = 0
                self.algs[alg_index].reset()

                for t in range(1,self.horizons[-1]+1) :
                    xt = env.get_state(t,yt)
                    yt = self.algs[alg_index].next_decision(t,xt,gt,st,dt)

                    if((yt<xt).any()) :
                        print("Undershooting error for alg {} at period {}:".format(alg_index,t))
                        print("x_t = {}, y_t = {}".format(xt,yt))
                        #flag_raised = True
                    
                    cum_lt += env.get_loss(t,yt)
                    if(t in self.horizons) :
                        cum_losses[seed_id,alg_index+1, horizon_index] = cum_lt
                        horizon_index += 1
                    
                    gt = env.get_subgradient(t,yt)
                    st = env.get_sales(t,yt)
                    dt = env.get_demand(t)
        return cum_losses
