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

def main():
    '''
    parser = argparse.ArgumentParser(description='Evaluate inferred.')

    parser.add_argument('--inferred', metavar=['INFERRED FILE'], nargs=1, required=True, help='Path to inferred from this file.')
    parser.add_argument('--truth', metavar=['TRUTH FILE'], nargs=1, required=True, help='Path to truth from this file.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ratings', action='store_true', help='Evaluate inferred as ratings.')
    group.add_argument('--rankings', action='store_true', help='Evaluate inferred as rankings.')

    args = parser.parse_args()
    '''

    #inferred_path = os.path.abspath(os.path.join(THIS_DIR, args.inferred[0]))
    inferred_path = os.path.abspath(os.path.join(THIS_DIR, '../data/predicates/model/mf_ratings_test.txt')) 
    #truth_path = os.path.abspath(os.path.join(THIS_DIR, args.truth[0]))
    truth_path = os.path.abspath(os.path.join(THIS_DIR, '../data/raw/test.txt'))

    inferred = pd.read_csv(inferred_path, sep='\t', lineterminator='\n', names=['user_id', 'item_id', 'inferred_rating'])
    truth = pd.read_csv(truth_path, usecols=['user_id', 'item_id', 'rating', 'user_attr', 'model_attr'], sep='\t', lineterminator='\n', converters={"user_attr": str})

    #if (args.ratings):
    if (True):
        evaluation = evaluate_rating(inferred, truth)
        print(', '.join([ "%s: %s" % (str(x), str(evaluation[x])) for x in evaluation.keys()]))
    '''
    if (args.rankings):
        pass
    '''

if (__name__ == '__main__'):
    main()   
