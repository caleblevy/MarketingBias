import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(predicate_dir, output_dir):
    truth_file = predicate_dir / "eval" / "truth" / "Rating.txt"
    inferred_file = output_dir / "inferred_predicates" / "RATING.txt"
    truth = pd.read_csv(truth_file, sep='\t', names=["user_id", "item_id", "rating"])
    inferred = pd.read_csv(inferred_file, sep='\t', names=["user_id", "item_id", "rating"])
    data = truth.merge(inferred, on=["user_id", "item_id"], suffixes=["_truth", "_inferred"])
    print(data)
    # TODO: Normalize to 1 (once we can run MF code, )
    data["rating_truth"] = 4*data["rating_truth"] + 1
    data["rating_inferred"] = 4*data["rating_inferred"] + 1
    print(data)
    MSE = mean_squared_error(data["rating_truth"], data["predicted_ratings"], squared=False)
    print(output_dir)
    print(f"{MSE=}")
    MAE = mean_absolute_error(data["rating_truth"], data["predicted_ratings"])
    print(f"{MAE=}")
    F = fstat(predicate_dir, data)
    print(F)


def f_stat(predicate_dir, data):
    item_group = pd.read_csv(predicate_dir / "eval" / "observations" / "ItemGroup.txt", sep='\t', names=["item_id", "item_attr"])
    user_group = pd.read_csv(predicate_dir / "eval" / "observations" / "UserGroup.txt", sep='\t', names=["user_id", "item_attr"])
    data = data.join(item_group, on="item_id").join(user_group, on="user_id")
    print(data)
    print(data)
    M = len(user_attrs)
    N = len(model_attrs)

    average_error = error.mean()
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