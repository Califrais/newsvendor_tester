from typing import Callable
from cost_structures import CostStructure
from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class Huh09OGD(AbstractInventoryPolicy) :
    """
    Zero fixed cost, zero purchase cost.
    """
    def __init__(self,
            initial_base_stock:np.array,
            base_stock_upper_bound: np.array,
            learning_rate:Callable,
            cost_structure:CostStructure,
            name:str="Huh09 OGD") :
        """
        See Huh, W. T., & Rusmevichientong, P. (2009).
        A nonparametric asymptotic analysis of inventory planning with censored demand.
        Mathematics of Operations Research, 34(1), 103-123.
        """

        super().__init__(name)
        self.nb_products = len(initial_base_stock)
        self.learning_rate = learning_rate
        self.cost_structure = cost_structure
        self.base_stock_upper_bound = np.array(base_stock_upper_bound,dtype=float)

        self.unconstrained_target_levels = np.array(initial_base_stock,dtype=float)
        self.implemented_target_levels = np.array(initial_base_stock,dtype=float)

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        quantities = np.array(self.implemented_target_levels)
        if(t>1) :
            for k in range(self.nb_products) :
                if(inventory_state.movements.loc[(t-1,k),"sales"] == inventory_state.movements.loc[(t-1,k),"interim_inventory_level"]) :
                    # there has been lost sales on the implemented system
                    gradient = -self.cost_structure.stockout_costs[k]
                else :
                    # no lost sales on the implemented system, thus, demand = sales
                    if(self.unconstrained_target_levels[k] > inventory_state.movements.loc[(t-1,k),"sales"]) :
                        # no lost sales on the unconstrainted system too
                        gradient = self.cost_structure.holding_costs[k]
                    else :
                        gradient = -self.cost_structure.stockout_costs[k]
                
                # Updating the unconstrained system
                self.unconstrained_target_levels[k] = np.clip(
                    self.unconstrained_target_levels[k]-self.learning_rate(t-1)*gradient,
                    0,
                    self.base_stock_upper_bound[k]
                )

                # Updating the implemented system
                self.implemented_target_levels[k] = np.maximum(self.unconstrained_target_levels[k],inventory_state.movements.loc[(t,k),"starting_inventory_level"])

                # Computing order quantities
                assert (self.implemented_target_levels[k] - inventory_state.movements.loc[(t,k),"starting_inventory_level"]) >= 0
                quantities[k] = self.implemented_target_levels[k] - inventory_state.movements.loc[(t,k),"starting_inventory_level"]
        return quantities

    def __str__(self) :
        return self.name