#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import csv
import pandas as pd
import numpy as np

def func(x):
    return x.std() / np.sqrt(len(x))

def time_convert(x):
    times = x.split(':')
    return float(60*(60*float(times[0])+float(times[1]))+float(times[2]))

result_list=[]

for index, experiment in enumerate(os.listdir(sys.argv[1])):
    experiment_folder = os.path.join(sys.argv[1], experiment)

    result_string = ""
    result_list_single=[]

    with(open(os.path.join(experiment_folder, "config_" + experiment + ".json"))) as f:
        data = json.load(f)

        # dataset
        dataset = data["dataset"]

        # selection method
        selection_method = data["selection_method"]

        # epochs
        epochs = str(sum(num for num in data["proxy_epochs"]))

    result_string += dataset+","+selection_method+","+epochs
    result_list_single.append(dataset)
    result_list_single.append(selection_method)
    result_list_single.append(epochs)

    # Read proxy.csv, selection.csv, add everything up
    df_proxy_times = pd.read_csv(os.path.join(experiment_folder, "proxy.csv"))
    df_proxy_total_times = df_proxy_times["test_time"].apply(time_convert) + df_proxy_times["train_time"].apply(time_convert)
    df_selection_times = pd.read_csv(os.path.join(experiment_folder, "selection.csv"))
    df_selection_total_times = df_selection_times["total_time"].apply(time_convert)

    # print(df_proxy_times, df_selection_times)
    df_total_times = df_proxy_total_times + df_selection_total_times

    for idx, target in enumerate(["5000", "10000", "15000", "20000", "25000"]):
        target_folder = os.path.join(experiment_folder, "target", target)

        df = pd.read_csv(os.path.join(target_folder, "results.csv"))
        df_filtered = df[df["mode"] == "test"]

        top1_error = str(100-round(df_filtered["top1_accuracy"].max(), 1))
        result_string += ","+top1_error
        result_list_single.append(top1_error)

    # Add selection times
    for idx, target in enumerate(["5000", "10000", "15000", "20000", "25000"]):
        #print(df_total_times.iloc[0:idx+1])
        result_list_single.append(df_total_times.iloc[0:idx+1].sum())

    result_list.append(result_list_single)

df = pd.DataFrame(result_list, columns=["dataset", "selection_method", "epochs", "10%", "20%", "30%", "40%", "50%", "t_10%", "t_20%", "t_30%", "t_40%", "t_50%"])

float_cols = "10% 20% 30% 40% 50%".split()
for float_col in float_cols:
    df[float_col] = pd.to_numeric(df[float_col])

df1 = df.groupby("epochs", as_index=False).agg(['mean', func])
df2 = (df1.xs('mean', axis=1, level=1).round(1).astype(str) + ' ± ' + df1.xs('func', axis=1, level=1).round(2).astype(str))
df2 = df2.reset_index()

# Add other columns
df2.insert(loc=0, column="selection_method", value=selection_method)
df2.insert(loc=0, column="dataset", value=dataset)

print(df2)
