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
        self.base_stock_upper_bound = np.array(base_stock_upper_bound)
        self.project_on_the_highest_dynamic_constraint = project_on_the_highest_dynamic_constraint

        self.initial_base_stock = np.array(initial_base_stock)
        self.implemented_target_levels = np.array(initial_base_stock)
        self.accumulated_gradients = np.zeros(self.nb_products)

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        quantities = self.implemented_target_levels
        if(t>1) :
            for k in range(self.nb_products) :
                self.accumulated_gradients[k] += (
                    self.cost_structure.costs_history.loc[(t-1,k),"holding_cost_gradient"]
                    + self.cost_structure.costs_history.loc[(t-1,k),"stockout_cost_gradient"]
                )
                
                self.implemented_target_levels[k] = np.clip(
                    self.initial_base_stock[k]-self.learning_rate(t)*self.accumulated_gradients[k],
                    np.max(inventory_state.movements.loc[(slice(0,t),k),"starting_inventory_level"])
                    if self.project_on_the_highest_dynamic_constraint 
                    else inventory_state.movements.loc[(t,k),"starting_inventory_level"],
                    self.base_stock_upper_bound[k]
                )

                quantities[k] = np.maximum(0,self.implemented_target_levels[k] - inventory_state.movements.loc[(t,k),"starting_inventory_level"])
        return quantities