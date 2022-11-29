from typing import Callable
from cost_structures import CostStructure
from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class Zhang18OGD(AbstractInventoryPolicy) :
    """
    Zero fixed cost, zero purchase cost.
    """
    def __init__(self,
            initial_base_stock:np.array,
            base_stock_upper_bound: np.array,
            gamma:float,
            cost_structure:CostStructure,
            name:str="Zhang18 OGD") :

        super().__init__(name)
        self.nb_products = len(initial_base_stock)
        self.gamma = gamma
        self.cost_structure = cost_structure
        self.base_stock_upper_bound = np.array(base_stock_upper_bound,dtype=float)

        self.implemented_target_levels = np.array(initial_base_stock,dtype=float)
        self.cycle_counter = 1
        self.last_lost_sales_period = 1

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        quantities = np.array(self.implemented_target_levels)
        if(t>1) :
            for k in range(self.nb_products) :
                # Updating the levels if lost sales occured in period t-1 :
                if(inventory_state.movements.loc[(t,k),"starting_inventory_level"] == 0) :
                    gradient = self.cost_structure.holding_costs[k]*(t-self.last_lost_sales_period-1) - self.cost_structure.stockout_costs[k]
                    self.implemented_target_levels[k] = np.clip(
                        self.implemented_target_levels[k] - self.gamma/np.sqrt(self.cycle_counter) * gradient,
                        0,
                        self.base_stock_upper_bound[k]
                    )
                    
                    self.last_lost_sales_period = t
                    self.cycle_counter += 1

                # Computing order quantities
                quantities[k] = np.maximum(0,self.implemented_target_levels[k] - inventory_state.movements.loc[(t,k),"starting_inventory_level"])
        return quantities

    def __str__(self) :
        return self.name

    def OPTIMAL_GAMMA(mu:float, base_stock_upper_bound: float,cost_structure:CostStructure) :
        return mu*base_stock_upper_bound/(np.maximum(cost_structure.holding_costs[0],cost_structure.stockout_costs[0])*np.sqrt(4-2*mu))