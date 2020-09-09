import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(predicate_dir, results):
    truth_file = predicate_dir / "eval" / "truth" / "Rating.txt"
    
    truth = pd.read_csv(truth_file, sep='\t', names=["user_id", "item_id", "rating"])
    inferred = results["Rating"].rename(columns={
        0: "user_id",
        1: "item_id",
        "truth": "rating"
    })
    comparison = truth.merge(inferred, on=["user_id", "item_id"], suffixes=["_truth", "_inferred"])
    # TODO: Normalize to 1 (once we can run MF code, )
    true_ratings = 4*comparison["rating_truth"] + 1
    predicted_ratings = 4*comparison["rating_inferred"] + 1
    MSE = mean_squared_error(true_ratings, predicted_ratings, squared=False)
    MAE = mean_absolute_error(true_ratings, predicted_ratings)
    print(MAE, MSE)