from inventory_states import NonPerishableInventoryState
import numpy as np

class AbstractInventoryPolicy() :
    def __init__(self) :
        pass

    def get_order_quantity(self, t:int, inventory_state:NonPerishableInventoryState) -> np.array :
        return []