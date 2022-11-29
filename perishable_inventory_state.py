import os
import numpy as np
import pandas as pd

class PerishableInventoryState :

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
    ISSUING_POLICIES = ["FIFO"]

    def __init__(self,nb_products: int, lead_times: np.array, life_times: np.array, transition_type:str, issuing_policy:str) :
        self.nb_products = nb_products
        self.lead_times = lead_times
        self.life_times = life_times
        self.transition_type = transition_type
        if(issuing_policy=="FIFO") :
            self.issuing_policy_method = self.__fifo_issuing_policy
        self.reset()

    def reset(self) :
        self.__state = [np.zeros(self.lead_times[k]+self.life_time[k]-1,dtype=np.float) for k in range(self.nb_products)]
        self.movements = pd.DataFrame(
            np.nan,
            index = pd.MultiIndex.from_product([[1],np.arange(self.nb_products),[]],names=["period","product_id","nb_remaining_periods"]),
            columns = self.INVENTORY_METRICS,
            dtype=np.float
        )
        self.movements.loc[(1,slice(None)),["starting_inventory_level", "starting_inventory_position"]] = .0

    def __log(self,t:int, product_id:int, nb_remaining_periods:int, key:str, value:float) -> None :
        self.movements.loc[(t,product_id, nb_remaining_periods), key] = value

    def save_movements_as_csv(self, file_name:str) -> None :
        self.movements.to_csv(os.path.join("inventory_movements",file_name))

    def __fifo_issuing_policy(self, t:int, product_id:int, demands: np.array) :
        sales = np.zeros(self.life_times[product_id])
        for j in range(self.life_times[product_id]-1) :
            sales[j] = np.minimum(self.__state[product_id], np.maximum(0,demands[product_id]-np.sum(sales)))
        #### sales[-1] = np.minimum(self.__state[product_id], np.maximum(0,demands[product_id]-np.sum(sales)))
        return sales

    def step(self, t:int, order_quantities: np.array, demands: np.array) -> None :
        for k in range(self.nb_products) :
            # Append new order_quantities
            self.__state[k] = np.append(self.__state[k],order_quantities[k])

            # Computing interim quantities
            interim_inventory_level = np.sum(self.__state[k][:self.life_times[k] +1])
            on_hand = np.maximum(.0, interim_inventory_level)
            sales_vector = self.self.issuing_policy_method(t,k,demands)
            unmet_demand = demands[k]-np.sum(sales_vector)
            carryover = .0
            if(self.transition_type == "LOST_SALES") :
                carryover = np.maximum(.0, on_hand - unmet_demand)
            elif (self.transition_type == "BACKLOGGING") :
                carryover = on_hand - unmet_demand

            # Interim metrics
            self.__log(t,k,"order_quantities", order_quantities[k])
            self.__log(t,k,"interim_inventory_level", interim_inventory_level)
            self.__log(t,k,"interim_inventory_position", self.__state[k].sum())
            self.__log(t,k,"demands", demands[k])
            self.__log(t,k,"sales", np.sum(sales_vector))
            self.__log(t,k,"unmet_demand", unmet_demand)

            # Transition
            self.__state[k] = self.__state[k][1:]

            # Saving metrics for next step
            self.__log(t+1,k,"starting_inventory_level", self.__state[k][0])
            self.__log(t+1,k,"starting_inventory_position", self.__state[k].sum())