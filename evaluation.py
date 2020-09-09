import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(truth, inferred):
    comparison = truth.merge(inferred, on=["user_id", "item_id"], suffixes=["_truth", "_inferred"])
    # TODO: Normalize to 1 (once we can run MF code, )
    true_ratings = 4*comparison["rating_truth"] + 1
    predicted_ratings = 4*comparison["rating_inferred"] + 1
    MSE = mean_squared_error(true_ratings, predicted_ratings, squared=False)
    MAE = mean_absolute_error(true_ratings, predicted_ratings)
    print(MAE, MSE)