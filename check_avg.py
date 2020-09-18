import pandas as pd

modcloth = pd.read_csv("throwaway/modcloth.csv")
item_avg = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMAVGBYUG.txt", sep='\t', names=["item_id", "user_attr", "rating"])
segment_item = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTITEMAVG.txt", sep='\t', names=["user_attr", "model_attr", "rating"])
item_sum = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMSUM.txt", sep='\t', names=["item_id", "user_attr", "model_attr", "sum_inferred"])
segment_avg = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTAVG.txt", sep='\t', names=["user_attr", "item_attr", "rating"])
inferred = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/RATING.txt", sep='\t', names=["user_id", "item_id", "rating"])


true = modcloth.query("split == 2")[["user_id", "item_id", "user_attr", "model_attr", "rating"]].dropna()
data = inferred.merge(true, on=["user_id", "item_id"], suffixes=(None, "_truth"))[["user_id", "item_id", "user_attr", "model_attr", "rating"]]
print(data.groupby(["user_attr", "model_attr"])["rating"].count())
print(data.groupby(["user_attr", "model_attr"])["rating"].mean())
print(segment_avg)
exit()


sums = (data.groupby(["item_id", "user_attr", "model_attr"])
            .sum("rating")
            .reset_index()
            .rename(columns={"rating": "sum"})
            [["item_id", "user_attr", "model_attr", "sum"]]
        )