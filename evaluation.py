import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, auc, roc_curve
import statsmodels as sm
import statsmodels.api as sm
from statsmodels.formula.api import ols


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
    data_["user_attr"] = data_["user_attr"].astype(int)
    data_nansegment = data.copy()
    data_nansegment['user_attr'].loc[data_nansegment['user_attr'].isna()] = 3
    data_nansegment['user_attr'] = data_nansegment['user_attr'].astype(int)
    data_stringcols = data_.copy()
    data_stringcols['user_attr'] = data_stringcols['user_attr'].astype(str)
    data_stringcols['model_attr'] = data_stringcols['model_attr'].astype(str)
    data_nansegment_stringcols = data_nansegment.copy()
    data_nansegment_stringcols['user_attr'] = data_nansegment_stringcols['user_attr'].astype(str)
    data_nansegment_stringcols['model_attr'] = data_nansegment_stringcols['model_attr'].astype(str)
    # TODO: Normalize to 1 (once we can run MF code, )
    MSE = mean_squared_error(data["rating_truth"], data["rating_inferred"])
    MAE = mean_absolute_error(data["rating_truth"], data["rating_inferred"])
    AUC = _auc(data['rating_truth'], data['rating_inferred'], 4)
    F_DEFINITION = f_stat_def(data_)
    F_DEFINITION_STRINGCOLS = f_stat_def(data_stringcols)
    F_DEFINITION_NANSEGMENT = f_stat_def(data_nansegment)
    F_DEFINITION_NANSEGMENT_STRINGCOLS = f_stat_def(data_nansegment_stringcols)
    F_JULIAN = f_stat_julian(data_)
    F_JULIAN_STRINGCOLS = f_stat_julian(data_stringcols)
    F_JULIAN_NANSEGMENT = f_stat_julian(data_nansegment)
    F_JULIAN_NANSEGMENT_STRINGCOLS = f_stat_julian(data_nansegment_stringcols)
    # Now for testing purposes
    eval_tokens["MAE"].append(MAE)
    eval_tokens["MSE"].append(MSE)
    eval_tokens["AUC"].append(AUC)
    eval_tokens["F-stat (Definition)"].append(F_DEFINITION)
    eval_tokens["F-stat (Definition + StringCols)"].append(F_DEFINITION_STRINGCOLS)
    eval_tokens["F-stat (Definition + NaNSegment)"].append(F_DEFINITION_NANSEGMENT)
    eval_tokens["F-stat (Definition + NaNSegment + StringCols)"].append(F_DEFINITION_NANSEGMENT_STRINGCOLS)
    eval_tokens["F-stat (Julian)"].append(F_JULIAN)
    eval_tokens["F-stat (Julian + StringCols)"].append(F_JULIAN_STRINGCOLS)
    eval_tokens["F-stat (Julian + NaNSegment)"].append(F_JULIAN_NANSEGMENT)
    eval_tokens["F-stat (Julian + NaNSegment + StringCols)"].append(F_JULIAN_NANSEGMENT_STRINGCOLS)
    # print(F_DEFINITION, F_DEFINITION_STRINGCOLS, F_DEFINITION_NANSEGMENT, F_DEFINITION_NANSEGMENT_STRINGCOLS)
    # print(F_JULIAN, F_JULIAN_STRINGCOLS, F_JULIAN_NANSEGMENT, F_JULIAN_NANSEGMENT_STRINGCOLS)
    # eval_tokens["F-stat (Kyle)"].append(F_KYLE)
    # eval_tokens["F-stat (statsmodels - str)"].append(F_A_STR)
    # eval_tokens["p (statsmodels - str)"].append(P_A_STR)
    # print(F_KYLE, F_A_STR, P_A_STR, F_A_NOSTR, P_A_NOSTR)


def _auc(truth, inferred, threshold):
    truth = [int(x > threshold) for x in truth]
    inferred = [int(x > threshold) for x in inferred]

    fpr, tpr, thresholds = roc_curve(truth, inferred, pos_label=1)
    return auc(fpr, tpr)


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
    f, p = sm.stats.anova_lm(ols('error ~ model_attr*user_attr - model_attr - user_attr', data=data).fit()).values[0, -2:]
    return f
