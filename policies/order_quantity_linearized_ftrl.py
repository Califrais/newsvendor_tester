from typing import Callable
from cost_structures import CostStructure
from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class OrderQuantityLinearizedFTRL(AbstractInventoryPolicy) :
    """
    Zero fixed cost, zero purchase cost.
    """
    def __init__(self, 
            initial_order_quantities:np.array,
            order_quantity_upper_bound: np.array,
            learning_rate:Callable,
            cost_structure:CostStructure,
            name:str="OrderQuantityLinearizedFTRL") :

        super().__init__(name)
        self.nb_products = len(initial_order_quantities)
        self.learning_rate = learning_rate
        self.cost_structure = cost_structure
        self.order_quantity_upper_bound = np.array(order_quantity_upper_bound)

        self.initial_order_quantities = np.array(initial_order_quantities)
        self.accumulated_gradients = np.zeros(self.nb_products)

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        order_quantities = np.array(self.initial_order_quantities)
        if(t>1) :
            for k in range(self.nb_products) :
                self.accumulated_gradients[k] += (
                    self.cost_structure.costs_history.loc[(t-1,k),"holding_cost_gradient"]
                    + self.cost_structure.costs_history.loc[(t-1,k),"stockout_cost_gradient"]
                )
                
                order_quantities[k] = np.clip(
                    self.initial_order_quantities[k]-self.learning_rate(t)*self.accumulated_gradients[k],
                    0,
                    self.order_quantity_upper_bound[k]
                )
        return order_quantities