import os
import pandas as pd
import numpy as np

from inventory_states import NonPerishableInventoryState

class CostStructure :

    COSTS = ["fixed_cost", "purchase_cost", "holding_cost", "stockout_cost"]

    def __init__(self, nb_products, fixed_costs, purchase_costs, holding_costs, stockout_costs) :
        self.nb_products = nb_products
        self.fixed_costs = fixed_costs
        self.purchase_costs = purchase_costs
        self.holding_costs = holding_costs
        self.stockout_costs = stockout_costs
        self.reset()

    def reset(self) -> None :
        self.costs_history = pd.DataFrame(
            np.nan,
            index = pd.MultiIndex.from_product([[1],np.arange(self.nb_products)],names=["period","product_id"]),
            columns = self.COSTS,
            dtype=np.float
        )

    def incur_cost(self, t:int, inventory_state: NonPerishableInventoryState, order_quantities: np.array) -> None :
        """
        It should be called AFTER `step`
        """
        for k in range(self.nb_products) :
            self.costs_history.loc[(t,k),"fixed_cost"] =  self.fixed_costs[k] if order_quantities[k]>0 else 0
            self.costs_history.loc[(t,k),"purchase_cost"] = self.purchase_costs[k]*order_quantities[k]
            self.costs_history.loc[(t,k),"holding_cost"] = self.holding_costs[k]*np.maximum(0,inventory_state.movements.loc[(t+1,k), "starting_inventory_level"])
            self.costs_history.loc[(t,k),"stockout_cost"] = self.stockout_costs[k]*inventory_state.movements.loc[(t,k), "unmet_demand"]
    
    def save_history_as_csv(self, file_name:str) -> None :
        self.costs_history.to_csv(os.path.join("cost_histories",file_name))