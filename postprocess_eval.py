from datetime import datetime

import pandas as pd
from toolz import interleave


def compute_stats(df):
    means = df.groupby(["model"], sort=False).mean().add_prefix("Average ").reset_index()
    std = df.groupby(["model"], sort=False).std(ddof=0).add_suffix(" (STD)").reset_index()
    model = means["model"]
    means = means.drop(columns=["model"])
    std = std.drop(columns=["model"])
    stats = pd.concat([means, std], axis=1)[list(interleave([means, std]))]
    stats.insert(0, 'model', model)
    return stats


def postprocess():
    ev = pd.read_csv("throwaway/evaluation.csv")
    modcloth_baseline = ev.query("(dataset == 'modcloth') & (split == 'baseline_split')").drop(columns=["dataset", "split"])
    modcloth_random = ev.query("(dataset == 'modcloth') & (split != 'baseline_split')").drop(columns=["dataset"])
    electronics_baseline = ev.query("(dataset == 'electronics') & (split == 'baseline_split')").drop(columns=["dataset", "split"])
    electronics_random = ev.query("(dataset == 'electronics') & (split != 'baseline_split')").drop(columns=["dataset"])
    num_modcloth_splits = len(modcloth_random["split"].unique())
    num_electronics_splits = len(electronics_random["split"].unique())
    modcloth_random = compute_stats(modcloth_random)
    electronics_random = compute_stats(electronics_random)

    width_mb = len(modcloth_baseline.columns)
    height_mb = len(modcloth_baseline) + 1
    width_mr = len(modcloth_random.columns)
    height_mr = len(modcloth_random) + 1

    width_eb = len(electronics_baseline.columns)
    height_eb = len(electronics_baseline) + 1
    width_er = len(electronics_random.columns)
    height_er = len(electronics_random) + 1

    title_spacing = 1
    dataset_spacing = 2
    title_row_mb = 0
    start_row_mb = title_row_mb + title_spacing
    title_row_mr = start_row_mb + height_mb + dataset_spacing
    start_row_mr = title_row_mr + title_spacing

    title_row_eb = start_row_mr + height_mr + dataset_spacing
    start_row_eb = title_row_eb + title_spacing
    title_row_er = start_row_eb + height_eb + dataset_spacing
    start_row_er = title_row_er + title_spacing

    timestamp = datetime.now().strftime("%-m-%-d, %-I.%M %p")
    with pd.ExcelWriter("throwaway/evaluation.xlsx") as writer:
        sheet_name = f"Evaluation ({timestamp})"
        modcloth_baseline.to_excel(writer, sheet_name=sheet_name, startrow=start_row_mb, index=False, float_format="%.4f")  # Need to create the worksheet to access it
        modcloth_random.to_excel(writer, sheet_name=sheet_name, startrow=start_row_mr, index=False, float_format="%.4f")
        electronics_baseline.to_excel(writer, sheet_name=sheet_name, startrow=start_row_eb, index=False, float_format="%.4f")
        electronics_random.to_excel(writer, sheet_name=sheet_name, startrow=start_row_er, index=False, float_format="%.4f")
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        # Increase wdith for model name
        worksheet.set_column(0, 0, 25)
        for col in range(1, max(width_mb, width_mr, width_eb, width_er)):
            worksheet.set_column(col, col, 15)
        big_font = workbook.add_format({'font_size': 15})
        for row in [title_row_mb, title_row_mr, title_row_eb, title_row_er]:
            worksheet.set_row(row, None, big_font)
        worksheet.merge_range(title_row_mb, 0, title_row_mb, width_mb-1, 'Modcloth (Baseline Split)')
        worksheet.merge_range(title_row_mr, 0, title_row_mr, width_mr-1, f"Modcloth (Average of {num_modcloth_splits} Random Splits)")
        worksheet.merge_range(title_row_eb, 0, title_row_eb, width_eb-1, f"Electronics (Baseline Split)")
        worksheet.merge_range(title_row_er, 0, title_row_er, width_er-1, f"Electronics (Average of {num_electronics_splits} Random Splits)")


if __name__ == '__main__':
    postprocess()
