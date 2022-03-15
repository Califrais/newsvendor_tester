import os
import numpy as np
import pandas as pd

class NonPerishableInventoryState :

    INVENTORY_METRICS = [
        "starting_inventory_level",
        "starting_inventory_position",
        "order_quantities",
        "interim_inventory_level",
        "interim_inventory_position",
        "demands",
        "sales",
        "unmet_demand"
    ]
    TRANSITION_TYPES = ["LOST_SALES", "NO_CARRYOVER", "BACKLOGGING"]

    def __init__(self,nb_products: int, lead_times: np.array, transition_type:str) :
        self.nb_products = nb_products
        self.lead_times = lead_times
        self.transition_type = transition_type
        self.reset()

    def reset(self) :
        self.__state = [np.zeros(np.maximum(1,self.lead_times[k]),dtype=np.float) for k in range(self.nb_products)]
        self.movements = pd.DataFrame(
            np.nan,
            index = pd.MultiIndex.from_product([[1],np.arange(self.nb_products)],names=["period","product_id"]),
            columns = self.INVENTORY_METRICS,
            dtype=np.float
        )
        self.movements.loc[(1,slice(None)),["starting_inventory_level", "starting_inventory_position"]] = .0

    def __log(self,t:int, product_id:int, key:str, value:float) -> None :
        self.movements.loc[(t,product_id), key] = value

    def save_movements_as_csv(self, file_name:str) -> None :
        self.movements.to_csv(os.path.join("inventory_movements",file_name))


    def step(self, t:int, order_quantities: np.array, demands: np.array) -> None :
        for k in range(self.nb_products) :
            # Append new order_quantities
            if(self.lead_times[k] == 0) :
                self.__state[k][0] += order_quantities
            else :
                self.__state[k] = np.append(self.__state[k],order_quantities[k])

            # Computing interim quantities
            interim_inventory_level = self.__state[k][0]
            on_hand = np.maximum(.0, interim_inventory_level)
            sales = np.minimum(demands[k], on_hand)
            unmet_demand = demands[k]-sales
            carryover = .0
            if(self.transition_type == "LOST_SALES") :
                carryover = np.maximum(.0, on_hand - sales)
            elif (self.transition_type == "BACKLOGGING") :
                carryover = on_hand - sales

            # Interim metrics
            self.__log(t,k,"order_quantities", order_quantities[k])
            self.__log(t,k,"interim_inventory_level", interim_inventory_level)
            self.__log(t,k,"interim_inventory_position", self.__state[k].sum())
            self.__log(t,k,"demands", demands[k])
            self.__log(t,k,"sales", sales)
            self.__log(t,k,"unmet_demand", unmet_demand)

            # Transition
            if(self.lead_times[k] == 0) :
                self.__state[k][0] = carryover
            else :
                self.__state[k][1] += carryover
                self.__state[k] = self.__state[k][1:]

            # Saving metrics for next step
            self.__log(t+1,k,"starting_inventory_level", self.__state[k][0])
            self.__log(t+1,k,"starting_inventory_position", self.__state[k].sum())