import pandas as pd
modcloth = pd.read_csv("throwaway/modcloth.csv")
inferred = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/RATING.txt", sep='\t', names=["user_id", "item_id", "rating"])
item_sums = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMSUM.txt", sep='\t', names=["item_id", "user_attr", "model_attr", "rating"])
target_segments = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTAVG.txt", sep='\t', names=["user_attr", "model_attr", "rating"])
item_avg = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/ITEMAVGBYUG.txt", sep='\t', names=["item_id", "user_attr", "rating"])
segment_item = pd.read_csv("results/modcloth/user_parity_fairness/baseline_split/inferred_predicates/TARGETSEGMENTITEMAVG.txt", sep='\t', names=["user_attr", "model_attr", "rating"])

t = item_sums[["item_id", "user_attr", "model_attr"]]
print(t)
print(len(t))
print(len(t.drop_duplicates()))


true = modcloth.query("split == 2")[["user_id", "item_id", "user_attr", "model_attr", "rating"]].dropna()
data = inferred.merge(true, on=["user_id", "item_id"], suffixes=(None, "_other"))
# data = data.iloc[:,~data.columns.str.contains("_other")]
print(data)
print(data.groupby(["user_attr", "model_attr"]).mean("rating").reset_index())
print(item_sums.groupby(["user_attr", "model_attr"]).sum("rating"))


with open("throwaway/ground.txt") as f:
    grounds = f.readlines()


grounds = [line for line in grounds if ('ITEMSUM' in line or 'TARGETSEGMENTRATING' in line)]
print(len(grounds))
counts = [line.count("RATING") for line in grounds]
print(counts)
print(sum(counts))


print(data.groupby(["item_id", "user_attr", "model_attr"]).sum("rating"))



print(segment_item)

print(data.groupby(["user_attr", "model_attr", "item_id"]).mean("rating").reset_index().groupby(["user_attr", "model_attr"]).mean("rating").reset_index())