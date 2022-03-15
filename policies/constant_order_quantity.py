from inventory_states import NonPerishableInventoryState
import numpy as np

from policies.abstract_inventory_policy import AbstractInventoryPolicy

class ConstantOrderQuantity(AbstractInventoryPolicy) :
    def __init__(self,constant_order_quantity : np.array, name:str="ConstantOrderQuantity") :
        super().__init__(name)
        self.constant_order_quantity = constant_order_quantity
        
    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        return self.constant_order_quantity