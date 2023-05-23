from typing import List
import numpy as np
import pandas as pd
from datasetsforecast.m5 import M5
import itertools
import yaml

CONFIG_FILE = "m5_experiments/m5_data_import_conf.yml" 


def save_demands(Y_df:pd.DataFrame, S_df:pd.DataFrame, config) :
    """
    Subsampling, preprocessing and saving the demand series
    """
    product_list = S_df.item_id.unique() 

    start_date = Y_df.ds.min()
    end_date = Y_df.ds.max()
    nb_days = (end_date-start_date).days +1

    demands_df = Y_df.merge(S_df[["unique_id", "item_id", "store_id"]], on="unique_id")
    demands_df = demands_df.loc[demands_df.store_id == config["store_id"]] ###############
    demands_df["ds"] = (demands_df["ds"]-start_date).dt.days +1
    demands_df = (
        demands_df.drop(["unique_id", "store_id"],axis="columns")
        .set_index(["ds", "item_id"])
        .reindex(pd.MultiIndex.from_product([range(nb_days+1), product_list], names=["ds", "item_id"]), fill_value=0.0)
        .reset_index()
        .rename({"ds":"period", "item_id":"product_id", "y":"quantity"},axis="columns")
        .to_csv("m5_experiments/m5_demands.csv", index=False)
    )

def save_costs(X_df:pd.DataFrame, S_df:pd.DataFrame, config) :
    """
    Subsampling, preprocessing and saving the costs data
    """
    product_list = S_df.item_id.unique()
    selling_cost_to_holding_cost_factor = config["selling_cost_to_holding_cost_factor"]
    selling_cost_to_penalty_cost_factor = config["selling_cost_to_penalty_cost_factor"]
    costs = (
        X_df.merge(S_df[["unique_id", "item_id"]], on="unique_id")
        .groupby("item_id")
        [["sell_price"]]
        .mean()
        .loc[product_list]
    )
    costs["holding_cost"] = selling_cost_to_holding_cost_factor*costs.sell_price
    costs["penalty_cost"] = selling_cost_to_penalty_cost_factor*costs.sell_price
    costs = costs.drop("sell_price",axis="columns")
    costs.index = costs.index.rename("product_id")
    costs.to_csv("m5_experiments/m5_costs.csv")


if(__name__ == "__main__") :
    config = yaml.load(open(CONFIG_FILE, "r"), yaml.Loader)

    # Loading all M5 data
    m5_directory_path = config["m5_directory_path"]
    Y_df, X_df, S_df = M5.load(m5_directory_path, cache=False)

    save_demands(Y_df, S_df, config)
    save_costs(X_df, S_df, config)