import pandas as pd

modcloth = pd.read_csv("throwaway/modcloth.csv")
item_avg = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMAVGBYUG.txt", sep='\t', names=["item_id", "user_attr", "rating"])
segment_item = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTITEMAVG.txt", sep='\t', names=["user_attr", "model_attr", "rating"])
item_sum = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMSUM.txt", sep='\t', names=["item_id", "user_attr", "rating"])
segment_avg = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTAVG.txt", sep='\t', names=["user_attr", "item_attr", "rating"])
inferred = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/RATING.txt", sep='\t', names=["user_id", "item_id", "rating"])


true = modcloth.query("split == 2")[["user_id", "item_id", "user_attr", "model_attr", "rating"]].dropna()
data = inferred.merge(true, on=["user_id", "item_id"], suffixes=(None, "_truth"))

counts = data.groupby(["user_attr", "model_attr"]).count().reset_index()["rating"]
sums = data.groupby(["user_attr", "model_attr"]).sum().reset_index()["rating"]
print(counts, sums)
print(sums/counts)
print(segment_avg)

exit()

data["user_attr"] = data["user_attr"].astype(int)
means = data.groupby(["item_id", "user_attr"]).mean("rating").reset_index()
merged = means.merge(item_avg, on=["item_id", "user_attr"], suffixes=("_pandas", "_psl"))
averages = merged.groupby(["user_attr", "model_attr"]).mean().reset_index()
print(averages)
print(segment_item)
# import code
# code.interact(local=locals())
exit()


print(f"{segment_item}")
print(data.groupby(["user_attr", "model_attr", "item_id"])
          .mean("rating")
          .reset_index()
          .groupby(["user_attr", "model_attr"])
          .mean("rating")
          .reset_index()
          )