import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, auc, roc_curve
import statsmodels as sm
import statsmodels.api as sm
from statsmodels.formula.api import ols


def evaluate(model, eval_tokens):
    truth = model.load_eval_truth("Rating", ["user_id", "item_id"], "rating")
    inferred = model.load_inferred("Rating", ["user_id", "item_id"], "rating")
    data = truth.merge(inferred, on=["user_id", "item_id"], suffixes=["_truth", "_inferred"])
    # TODO: Normalize to 1 (once we can run MF code, )
    data["rating_truth"] = 4*data["rating_truth"] + 1
    data["rating_inferred"] = 4*data["rating_inferred"] + 1
    MSE = mean_squared_error(data["rating_truth"], data["rating_inferred"], squared=False)
    # print(output_dir)
    MAE = mean_absolute_error(data["rating_truth"], data["rating_inferred"])
    F = f_stat(data, model)
    AUC = _auc(data['rating_truth'], data['rating_inferred'], 4)
    eval_tokens["MAE"].append(MAE)
    eval_tokens["MSE"].append(MSE)
    eval_tokens["F-stat"].append(F)
    eval_tokens["AUC"].append(AUC)


def _auc(truth, inferred, threshold):
    truth = [int(x > threshold) for x in truth]
    inferred = [int(x > threshold) for x in inferred]

    fpr, tpr, thresholds = roc_curve(truth, inferred, pos_label=1)
    return auc(fpr, tpr)


def f_stat(data, model):
    item_group = model.load_eval_observations("ItemGroup", ["item_id", "model_attr"]).iloc[:, :-1]
    user_group = model.load_eval_observations("UserGroup", ["user_id", "user_attr"]).iloc[:, :-1]
    data = data.merge(item_group, on="item_id").merge(user_group, on="user_id")
    true_ratings = np.array(data["rating_truth"])
    inferred_ratings = np.array(data["rating_inferred"])
    errors = true_ratings - inferred_ratings
    data['error'] = errors
    data.fillna('unspecified')
    # data.to_csv("throwaway/test.csv", index=False)
    # data = pd.read_csv("throwaway/test.csv", converters={"model_attr": str, "user_attr": str})
    user_attrs = data["user_attr"].dropna().unique()
    model_attrs = data["model_attr"].dropna().unique()
    M = len(user_attrs)
    N = len(model_attrs)
    average_error = data["error"].mean()
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
    f1 = (V / (M * N - 1)) / (U / (data.shape[0] - (M * N)))
    f2, p2 = sm.stats.anova_lm(ols('error ~ model_attr*user_attr - model_attr - user_attr', data=data).fit()).values[0, -2:]
    # print(f1, f2, p2)
    return f1
