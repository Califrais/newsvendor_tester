from typing import Callable
from cost_structures import CostStructure
from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class MyAlgorithm(AbstractInventoryPolicy) :
    """
    Zero fixed cost, zero purchase cost.
    """
    def __init__(self,
            initial_base_stock:np.array,
            base_stock_lower_bound: np.array,
            base_stock_upper_bound: np.array,
            gamma:float,
            cost_structure:CostStructure,
            name:str="My Algorithm") :

        super().__init__(name)
        self.nb_products = len(initial_base_stock)
        self.gamma = gamma
        self.cost_structure = cost_structure
        self.base_stock_lower_bound = np.array(base_stock_lower_bound,dtype=float)
        self.base_stock_upper_bound = np.array(base_stock_upper_bound,dtype=float)

        self.target_levels = np.array(initial_base_stock,dtype=float)
        self.cycle_counter = 1
        self.last_positive_sales_period = 1
        self.min_positive_sales = np.array(base_stock_upper_bound,dtype=float)
        self.max_tau = np.ones(self.nb_products,dtype=float)

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        quantities = np.array(self.target_levels)
        if(t>1) :
            for k in range(self.nb_products) :
                # Updating the levels if positive sales in period t-1 :
                if(inventory_state.movements.loc[(t-1,k),"sales"] > 0) :

                    self.min_positive_sales[k] = np.minimum(self.min_positive_sales[k], inventory_state.movements.loc[(t-1,k),"sales"])
                    self.max_tau[k] = np.maximum(self.max_tau[k],t-self.last_positive_sales_period)

                    if(inventory_state.movements.loc[(t-1,k),"sales"] == inventory_state.movements.loc[(t-1,k),"interim_inventory_level"]) :
                        gradient = -self.cost_structure.stockout_costs[k]
                    else :
                        gradient = self.cost_structure.holding_costs[k]
                    gradient += self.cost_structure.holding_costs[k]*(t-self.last_positive_sales_period-1)
                    
                    learning_rate = self.min_positive_sales[k]*self.gamma*(self.base_stock_upper_bound-self.base_stock_lower_bound)/(
                        self.max_tau[k]*np.sqrt(self.cycle_counter)*np.maximum(self.cost_structure.holding_costs[k],self.cost_structure.stockout_costs[k])
                    )

                    self.target_levels[k] = np.clip(
                        self.target_levels[k] - learning_rate * gradient,
                        self.base_stock_lower_bound[k],
                        self.base_stock_upper_bound[k]
                    )
                    
                    self.last_positive_sales_period = t
                    self.cycle_counter += 1

                # Computing order quantities
                quantities[k] = np.maximum(0,self.target_levels[k] - inventory_state.movements.loc[(t,k),"starting_inventory_level"])
        return quantities

    def __str__(self) :
        return self.name