from typing import Callable
from cost_structures import CostStructure
from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class TargetLevelLinearizedFTRL(AbstractInventoryPolicy) :
    """
    Zero fixed cost, zero purchase cost.
    """
    def __init__(self, 
            initial_base_stock:np.array,
            base_stock_upper_bound: np.array,
            learning_rate:Callable,
            cost_structure:CostStructure,
            project_on_the_highest_dynamic_constraint:bool,
            name:str="TargetLevelLinearizedFTRL") :

        super().__init__(name)
        self.nb_products = len(initial_base_stock)
        self.learning_rate = learning_rate
        self.cost_structure = cost_structure
        self.base_stock_upper_bound = np.array(base_stock_upper_bound,dtype=float)
        self.project_on_the_highest_dynamic_constraint = project_on_the_highest_dynamic_constraint

        self.initial_base_stock = np.array(initial_base_stock,dtype=float)
        self.accumulated_gradients = np.zeros(self.nb_products,dtype=float)
        self.lower_bounds = np.zeros(self.nb_products,dtype=float)

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        quantities = self.initial_base_stock
        if(t>1) :
            for k in range(self.nb_products) :
                #
                starting_inventory_level = inventory_state.get_inventory_level(k)

                # Updating gradient estimates
                if(starting_inventory_level>0) :
                    last_gradient = self.cost_structure.holding_costs[k]
                else :
                    last_gradient = -self.cost_structure.stockout_costs[k]
                self.accumulated_gradients[k] += last_gradient

                # Computing the lower bound
                if(self.project_on_the_highest_dynamic_constraint) :
                    self.lower_bounds[k] = np.maximum(self.lower_bounds[k], starting_inventory_level)
                else :
                    self.lower_bounds[k] = starting_inventory_level
                
                # Computing target level
                target = np.clip(
                    self.initial_base_stock[k]-self.learning_rate(t)*self.accumulated_gradients[k],
                    self.lower_bounds[k],
                    self.base_stock_upper_bound[k]
                )
                assert ((target>=self.lower_bounds[k]) and (target<=self.base_stock_upper_bound[k])), "Projection error"
                assert ((target - starting_inventory_level) >= 0), "The target is below the current inventory level"

                quantities[k] = target - starting_inventory_level
        return quantities