import numpy as np
import pandas as pd

modcloth_mf_pred = pd.read_csv("throwaway/mf_1_1_1_test_ratings.txt", sep='\t')
electronics_mf_pred = pd.read_csv("throwaway/electronics_MF_0.5_512_10_10_0.001_1_1_1_rating_test.csv")



def turn_mf_factor_into_predicate(df, dataset_name):
    error = np.array(df["error"])
    actual_rating = np.array(df["rating"])
    mf_inferred_rating = actual_rating + error
    mf_inferred_rating = (mf_inferred_rating-1)/4
    df["mf_rating"] = mf_inferred_rating
    df.to_csv(
        f"datasets/{dataset_name}/predicates/baseline_split/eval/observations/MFRating.txt",
        sep='\t', header=False, index=False, columns=["user_id", "item_id", "mf_rating"]
    )


if __name__ == '__main__':
    turn_mf_factor_into_predicate(modcloth_mf_pred, "modcloth")
    turn_mf_factor_into_predicate(electronics_mf_pred, "electronics")
