from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class ConstantBaseStock(AbstractInventoryPolicy) :
    def __init__(self,base_stock_levels : np.array, name:str="ConstantBaseStock") :
        super().__init__(name)
        self.base_stock_levels = base_stock_levels
        
    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        nb_products = len(self.base_stock_levels)
        quantities = np.zeros(nb_products)
        for k in range(nb_products) :
            quantities[k] = np.maximum(0, self.base_stock_levels[k] - inventory_state.movements.loc[(t,k),"starting_inventory_level"])
        return quantities