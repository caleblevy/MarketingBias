from pathlib import Path

import numpy as np
import pandas as pd


dfs = {
    'modcloth': pd.read_csv("throwaway/mf_1_1_1_test_ratings.txt", sep='\t'),
    'electronics': pd.read_csv("throwaway/electronics_MF_0.5_512_10_10_0.001_1_1_1_rating_test.csv")
}



def make_mf_into_predicate(dataset_name):
    df = dfs[dataset_name]
    error = np.array(df["error"])
    actual_rating = np.array(df["rating"])
    mf_inferred_rating = actual_rating + error
    mf_inferred_rating = (mf_inferred_rating-1)/4
    df["mf_rating"] = mf_inferred_rating
    fname = Path(f"datasets/{dataset_name}/predicates/baseline_split/eval/observations/MFRating.txt").absolute()
    print(f"Writing: {fname}")
    df.to_csv(
        fname,
        sep='\t', header=False, index=False, columns=["user_id", "item_id", "mf_rating"]
    )





if __name__ == '__main__':
    make_mf_into_predicate("modcloth")
    make_mf_into_predicate("electronics")
