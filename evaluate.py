import argparse
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def link_data(inferred, truth):
    inferred = inferred.set_index(['user_id', 'item_id'])
    truth = truth.set_index(['user_id', 'item_id'])
    data = inferred.join(truth)

    return data

def f_stat(data):
    user_attrs = data['user_attr'].dropn.unique()
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

            V += market_segment.shape[0] * (market_segment_average_error - average_error) ** 2

            u_market_segment = 0
            for index, row in market_segment.iterrows():
                error = row['error']
                u_market_segment += (error - market_segment_average_error) ** 2

            U += u_market_segment
    
    f = (V / (M * N - 1)) / (U / (data.shape[0] - (M * N)))
    return f


def evaluate_rating(inferred, truth):
    data = link_data(inferred, truth)

    true_rating = np.array(data['rating'])
    inferred_rating = (4 * np.array(data['inferred_rating'])) + 1
    errors = inferred_rating - true_rating
    data['error'] = errors

    print(f_stat(data))

    f, p = sm.stats.anova_lm(ols('error ~ model_attr*user_attr - model_attr - user_attr', data=data).fit()).values[0, -2:]

    mse = np.mean(errors * errors)
    mae = np.mean(np.absolute(errors))

    return {"MSE": mse, "MAE": mae,"f-stat": f,"p": p}

def evaluate_ranking(inferred, truth):
    pass
