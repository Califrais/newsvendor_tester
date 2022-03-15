from non_perishable_inventory_state import NonPerishableInventoryState
import numpy as np

class AbstractInventoryPolicy() :
    def __init__(self, name:str) :
        self.name = name

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        return np.array([])
    
    def __str__(self) :
        return self.name