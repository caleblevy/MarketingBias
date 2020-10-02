import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, auc, roc_curve, average_precision_score
import statsmodels as sm


RATING_THRESHOLD = 3


def evaluate(model, eval_tokens):
    # Load truth and errors
    truth = model.load_eval_truth("Rating", ["user_id", "item_id"], "rating")
    inferred = model.load_inferred("Rating", ["user_id", "item_id"], "rating")
    data = truth.merge(inferred, on=["user_id", "item_id"], suffixes=["_truth", "_inferred"])
    data["rating_truth"] = 4*data["rating_truth"] + 1
    data["rating_inferred"] = 4*data["rating_inferred"] + 1
    true_ratings = np.array(data["rating_truth"])
    inferred_ratings = np.array(data["rating_inferred"])
    errors = true_ratings - inferred_ratings
    data['error'] = errors
    # Load item groups
    item_group = model.load_eval_observations("ItemGroup", ["item_id", "model_attr"]).iloc[:, :-1]
    user_group = model.load_eval_observations("UserGroup", ["user_id", "user_attr"]).iloc[:, :-1]
    data = data.merge(item_group, on="item_id", how="left").merge(user_group, on="user_id", how="left")
    # Get missing, string, and missing + string
    data_ = data.dropna()
    # TODO: Normalize to 1 (once we can run MF code, )
    MSE = mean_squared_error(data["rating_truth"], data["rating_inferred"])
    MAE = mean_absolute_error(data["rating_truth"], data["rating_inferred"])
    AUC_ROC = auc_roc(data['rating_truth'], data['rating_inferred'], RATING_THRESHOLD)
    AUPRC_POS, AUPRC_NEG = au_prc(data['rating_truth'], data['rating_inferred'], RATING_THRESHOLD)
    F_STAT = f_stat_def(data_)
    # Now for testing purposes
    eval_tokens["MAE"].append(MAE)
    eval_tokens["MSE"].append(MSE)
    eval_tokens["F-stat"].append(F_STAT)
    eval_tokens["AUC-ROC"].append(AUC_ROC)
    eval_tokens["Pos Class AUPRC"].append(AUPRC_POS)
    eval_tokens["Neg Class AUPRC"].append(AUPRC_NEG)



def auc_roc(truth, inferred, threshold):
    truth = [int(x >= threshold) for x in truth]
    inferred = [int(x >= threshold) for x in inferred]

    fpr, tpr, thresholds = roc_curve(truth, inferred, pos_label=1)
    return auc(fpr, tpr)


def au_prc(truth, inferred, threshold):
    truth = np.array([int(x >= threshold) for x in truth])
    inferred = np.array([int(x >= threshold) for x in inferred])
    pos_auprc = average_precision_score(truth, inferred, pos_label=1)
    neg_auprc = average_precision_score(1 - truth, 1 - inferred, pos_label=1)
    return pos_auprc, neg_auprc


def f_stat_def(data):
    user_attrs = data["user_attr"].dropna().unique()
    model_attrs = data["model_attr"].dropna().unique()
    M = len(user_attrs)
    N = len(model_attrs)
    average_error = data["error"].mean()
    V = 0
    U = 0
    # print("---")
    # print(M, N)
    for m in user_attrs:
        for n in model_attrs:
            # print(m, n)
            market_segment = data[(data['user_attr'] == m) & (data['model_attr'] == n)]
            market_segment_average_error = market_segment['error'].mean()
            V += market_segment.shape[0] * (market_segment_average_error - average_error) ** 2

            u_market_segment = 0
            for index, row in market_segment.iterrows():
                error = row['error']
                u_market_segment += (error - market_segment_average_error) ** 2

            U += u_market_segment
    # print('---')
    f_kyle = (V / (M * N - 1)) / (U / (data.shape[0] - (M * N)))
    return f_kyle


def f_stat_julian(data):
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    f, p = sm.stats.anova_lm(ols('error ~ model_attr*user_attr - model_attr - user_attr', data=data).fit()).values[0, -2:]
    return f
