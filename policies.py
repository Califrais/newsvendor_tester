from typing import Callable, List
from inventory_states import NonPerishableInventoryState
import numpy as np

class AbstractInventoryPolicy() :
    def __init__(self) :
        pass
    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        return []

class ConstantOrderQuantity(AbstractInventoryPolicy) :
    def __init__(self,constant_order_quantity : np.array) :
        self.constant_order_quantity = constant_order_quantity
        
    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        return self.constant_order_quantity

class OptimalLostSalesPolicy(AbstractInventoryPolicy) :
    """
    Optimal clairvoyant policy for the infinite-horizon lost sales problem with :
        * No volume constraints (the problem is separable per product)
        * Zero fixed cost
        * Zero lead time
        * Infinite life time
        * i.i.d demand
        * Discount factor (gamma) in [0,1]

    If q is the quantile function of the single period demand of a given product this policy is base-stock policy with target level q((p-c)/(h+p-gamma*c))
    
    See Paragraph 4.6.1 of (Fundamentals of Supply Chain Theory)
    """
    def __init__(self, nb_products, purchase_costs, holding_costs, stockout_costs, discount_factor : float, demand_quantile_functions : List[Callable]) :
        self.nb_products = nb_products
        self.base_stock_levels = np.zeros(nb_products)
        for k in range(nb_products) : 
            if(holding_costs[k]+stockout_costs[k]-discount_factor*purchase_costs[k] == 0) :
                print("Costs for product {} leads to a degenerate solution: infinite target level.".format(k))
            if((stockout_costs[k]-purchase_costs[k])*(holding_costs[k]+stockout_costs[k]-discount_factor*purchase_costs[k]) <= 0) :
                print("The optimal unconstrainted strategy for product {} is never order.".format(k))
            self.base_stock_levels[k] = demand_quantile_functions[k]((stockout_costs[k]-purchase_costs[k])/(holding_costs[k]+stockout_costs[k]-discount_factor*purchase_costs[k]))
        print("Optimal unconstrainted base-stock level: {}".format(self.base_stock_levels))

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        # Test whether a LOST_SALES transition is used in inventory_state
        assert inventory_state.transition_type == "LOST_SALES", (
            """OptimalLostSalesPolicy handles lost sales transitions only.
            But, the inventory_state parameter has transition: {}.""".format(inventory_state.transition_type)
        )

        # Compute and returns order quantities
        quantities = np.zeros(self.nb_products)
        for i in range(self.nb_products) :
            quantities[i] = np.maximum(0,self.base_stock_levels[i] - inventory_state.movements.loc[(t,i),"starting_inventory_level"])
        return quantities


"""
class OptimalConstrainedPolicy(AbstractInventoryPolicy) :
    
    Optimal clairvoyant policy for the multi-item volume-constrained lost sales problem with :
        * Zero fixed cost
        * Zero lead time

    See Theorem 1 of (Shi, C., Chen, W., & Duenyas, I. (2016). Nonparametric data-driven algorithms for multiproduct inventory systems with censored demand)
    or (Ignall, E., A. F. Veinott. 1969. Optimality of myopic inventory policies for several substitute products)
    pass
"""