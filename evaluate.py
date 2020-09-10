#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def link_data(inferred, truth):
    inferred = inferred.set_index(['user_id', 'item_id'])
    truth = truth.set_index(['user_id', 'item_id'])
    data = inferred.join(truth)

    return data

def f_stat(data):
    user_attrs = data['user_attr'].dropna().unique()
    model_attrs = data['model_attr'].dropna().unique()
    M = len(user_attrs)
    N = len(model_attrs)

    average_error = data['error'].mean()
    V = 0
    U = 0
    for m in user_attrs:
        for n in model_attrs:
            market_segment = data[(data['user_attr'] == m) & (data['model_attr'] == n)]
            market_segment_average_error = market_segment['error'].mean()
            print(m, n, market_segment.shape)

            V += market_segment.shape[0] * (market_segment_average_error - average_error) ** 2

            u_market_segment = 0
            for index, row in market_segment.iterrows():
                error = row['error']
                u_market_segment += (error - market_segment_average_error) ** 2

            U += u_market_segment
    
    f = (V / (M * N - 1)) / (U / (data.shape[0] - (M * N)))
    return f


def evaluate_rating(data):
    f1 = f_stat(data)
    f2, p2 = sm.stats.anova_lm(ols('error ~ model_attr*user_attr - model_attr - user_attr', data=data).fit()).values[0, -2:]
    print(f"{f1=}, {f2=}, {p2=}")

    # return {"MSE": mse, "MAE": mae,"f-stat": f,"p": p}

def evaluate_ranking(inferred, truth):
    pass


if __name__ == '__main__':
    # data = pd.read_csv("data.csv")
    # data2 = pd.read_csv("data.csv", converters={"user_attr": str, 'model_attr': str})
    data = pd.read_csv("test.csv")
    data2 = pd.read_csv("test.csv", converters={"user_attr": str, 'model_attr': str})
    evaluate_rating(data)
    evaluate_rating(data2)
