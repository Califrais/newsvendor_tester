from typing import Callable
from cost_structures import CostStructure
from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class OrderQuantityOGD(AbstractInventoryPolicy) :
    """
    Zero fixed cost, zero purchase cost.
    """
    def __init__(self,
            initial_order_quantities:np.array,
            order_quantities_upper_bound: np.array,
            learning_rate:Callable,
            cost_structure:CostStructure,
            iterate_on_implemented_levels:bool,
            name:str="OrderQuantityOGD") :

        super().__init__(name)
        self.nb_products = len(initial_order_quantities)
        self.learning_rate = learning_rate
        self.cost_structure = cost_structure
        self.order_quantity_upper_bound = np.array(order_quantities_upper_bound)
        self.iterate_on_implemented_levels = iterate_on_implemented_levels

        self.current_order_quantities = np.array(initial_order_quantities)

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        if(t>1) :
            for k in range(self.nb_products) :
                gradient = (
                    self.cost_structure.costs_history.loc[(t-1,k),"holding_cost_gradient"]
                    + self.cost_structure.costs_history.loc[(t-1,k),"stockout_cost_gradient"]
                )
                
                self.current_order_quantities[k] = np.clip(
                    self.current_order_quantities[k] - self.learning_rate(t)*gradient,
                    0,
                    self.order_quantity_upper_bound[k]
                )
        return self.current_order_quantities

    def __str__(self) :
        return self.name