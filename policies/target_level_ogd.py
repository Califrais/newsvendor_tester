from typing import Callable
from cost_structures import CostStructure
from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class TargetLevelOGD(AbstractInventoryPolicy) :
    """
    Zero fixed cost, zero purchase cost.
    """
    def __init__(self,
            initial_base_stock:np.array,
            base_stock_upper_bound: np.array,
            learning_rate:Callable,
            cost_structure:CostStructure,
            iterate_on_implemented_levels:bool,
            name:str="TargetLevelOGD") :

        super().__init__(name)
        self.nb_products = len(initial_base_stock)
        self.learning_rate = learning_rate
        self.cost_structure = cost_structure
        self.base_stock_upper_bound = np.array(base_stock_upper_bound,dtype=float)
        self.iterate_on_implemented_levels = iterate_on_implemented_levels

        self.unconstrained_target_levels = np.array(initial_base_stock,dtype=float)
        self.implemented_target_levels = np.array(initial_base_stock,dtype=float)

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        quantities = self.implemented_target_levels
        if(t>1) :
            for k in range(self.nb_products) :
                gradient = (
                    self.cost_structure.costs_history.loc[(t-1,k),"holding_cost_gradient"]
                    + self.cost_structure.costs_history.loc[(t-1,k),"stockout_cost_gradient"]
                )
                
                if(self.iterate_on_implemented_levels) :
                    self.unconstrained_target_levels[k] = np.clip(
                        self.unconstrained_target_levels[k]-self.learning_rate(t)*gradient,
                        0,
                        self.base_stock_upper_bound[k]
                    )
                    self.implemented_target_levels[k] = np.maximum(self.unconstrained_target_levels[k],inventory_state.movements.loc[(t,k),"starting_inventory_level"])
                else :
                    self.implemented_target_levels[k] = np.clip(
                        self.implemented_target_levels[k]-self.learning_rate(t)*gradient,
                        inventory_state.movements.loc[(t,k),"starting_inventory_level"],
                        self.base_stock_upper_bound[k]
                    )

                quantities[k] = np.maximum(0,self.implemented_target_levels[k] - inventory_state.movements.loc[(t,k),"starting_inventory_level"])
        return quantities

    def __str__(self) :
        return self.name