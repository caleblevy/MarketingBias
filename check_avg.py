import pandas as pd

modcloth = pd.read_csv("throwaway/modcloth.csv")
item_avg = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMAVGBYUG.txt", sep='\t', names=["item_id", "user_attr", "rating"])
segment_item = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTITEMAVG.txt", sep='\t', names=["user_attr", "model_attr", "item_average"])
rating_segment = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETRATINGSEGMENT.txt", sep='\t', names=["user_id", "item_id", "user_attr", "model_attr", "rating"])
segment_avg = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTAVG.txt", sep='\t', names=["user_attr", "model_attr", "rating_average"])
inferred = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/RATING.txt", sep='\t', names=["user_id", "item_id", "rating"])


true = modcloth.query("split == 2")[["user_id", "item_id", "user_attr", "model_attr", "rating"]].dropna()
data = inferred.merge(true, on=["user_id", "item_id"], suffixes=(None, "_truth"))[["user_id", "item_id", "user_attr", "model_attr", "rating"]]
data["user_attr"] = data["user_attr"].astype(int)
# data = data.merge(rating_segment, on=["user_id", "item_id", "user_attr", "model_attr"], suffixes=(None, "_intermediate"))
# print(data)


avg = data.groupby(["user_attr", "model_attr"])["rating"].mean().reset_index().rename(columns={"rating": "rating_average"})
item = data.groupby(["user_attr", "model_attr", "item_id"])["rating"].mean().reset_index().groupby(["user_attr", "model_attr"])["rating"].mean().reset_index().rename(columns={"rating": "item_average"})
print(avg.merge(segment_avg, on=["user_attr", "model_attr"], suffixes=("_pandas", "_psl"))
         .merge(item, on=["user_attr", "model_attr"])
         .merge(segment_item, on=["user_attr", "model_attr"], suffixes=("_pandas", "_psl"))
    )