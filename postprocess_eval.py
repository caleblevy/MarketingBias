from datetime import datetime

import pandas as pd
from toolz import interleave


def compute_stats(df):
    means = df.groupby(["model"]).mean().add_prefix("Average ").reset_index()
    std = df.groupby(["model"]).std(ddof=0).add_suffix(" (STD)").reset_index()
    model = means["model"]
    means = means.drop(columns=["model"])
    std = std.drop(columns=["model"])
    stats = pd.concat([means, std], axis=1)[list(interleave([means, std]))]
    stats.insert(0, 'model', model)
    return stats


def postprocess():
    ev = pd.read_csv("throwaway/evaluation.csv")
    ev = ev[["dataset", "model", "split", "MSE", "MAE", "AUC", "F-stat (Definition)"]]
    ev = ev.rename(columns={"F-stat (Definition)": "F-stat"}, errors='raise')
    modcloth_baseline = ev.query("(dataset == 'modcloth') & (split == 'baseline_split')").drop(columns=["dataset"])
    print(modcloth_baseline)
    modcloth_random = ev.query("(dataset == 'modcloth') & (split != 'baseline_split')").drop(columns=["dataset"])
    electronics_baseline = ev.query("(dataset == 'electronics') & (split == 'baseline_split')").drop(columns=["dataset"])
    electronics_random = ev.query("(dataset == 'electronics') & (split != 'baseline_split')").drop(columns=["dataset"])
    modcloth_random = compute_stats(modcloth_random)
    electronics_random = compute_stats(electronics_random)

    width_baseline = len(modcloth_baseline.columns)
    height_baseline = len(modcloth_baseline) + 1
    width_random = len(modcloth_random.columns)
    height_random = len(modcloth_random) + 1

    title_row_mb = 1
    start_row_mb = title_row_mb + 1
    title_row_mr = start_row_mb + height_baseline + 1
    start_row_mr = title_row_mr + 1

    title_row_eb = start_row_mr + height_random + 1
    start_row_eb = title_row_eb + 1
    title_row_er = start_row_eb + height_baseline + 1
    start_row_er = title_row_er + 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    with pd.ExcelWriter(f"throwaway/evaluation_.xlsx") as writer:
        ev.to_excel(writer, sheet_name='Evaluation')


if __name__ == '__main__':
    postprocess()
