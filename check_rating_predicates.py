import pandas as pd
modcloth = pd.read_csv("throwaway/modcloth.csv")
inferred = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/RATING.txt", sep='\t', names=["user_id", "item_id", "rating"])
item_sums = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMSUM.txt", sep='\t', names=["item_id", "user_attr", "model_attr", "rating"])
target_segments = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTAVG.txt", sep='\t', names=["user_attr", "model_attr", "rating"])

t = item_sums[["item_id", "user_attr", "model_attr"]]
print(t)
print(len(t))
print(len(t.drop_duplicates()))


true = modcloth.query("split == 2")[["user_id", "item_id", "user_attr", "model_attr", "rating"]].dropna()
data = inferred.merge(true, on=["user_id", "item_id"], suffixes=(None, "_other"))
data = data.iloc[:,~data.columns.str.contains("_other")]
print(data.groupby(["user_attr", "model_attr"]).mean("rating").reset_index())

print(item_sums.drop("model_attr").groupby(["item_id"]).count())