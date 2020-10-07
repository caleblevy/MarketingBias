from datetime import datetime

import pandas as pd
from toolz import interleave


def compute_stats(df):
    means = df.groupby(["model"], sort=False).mean().add_prefix("Average ").reset_index()
    std = df.groupby(["model"], sort=False).std().add_suffix(" (STD)").reset_index()
    model = means["model"]
    means = means.drop(columns=["model"])
    std = std.drop(columns=["model"])
    stats = pd.concat([means, std], axis=1)[list(interleave([means, std]))]
    stats.insert(0, 'model', model)
    return stats


def postprocess():
    ev = pd.read_csv("throwaway/evaluation.csv")
    modcloth_baseline = ev.query("(dataset == 'modcloth') & split.str.contains('baseline')").drop(columns=["dataset", "split"])
    modcloth_active = ev.query("(dataset == 'modcloth') & split.str.contains('active')").drop(columns=["dataset"])
    modcloth_random = ev.query("(dataset == 'modcloth') & split.str.contains('random')").drop(columns=["dataset"])

    electronics_baseline = ev.query("(dataset == 'electronics') & split.str.contains('baseline')").drop(columns=["dataset", "split"])
    electronics_active = ev.query("(dataset == 'electronics') & split.str.contains('active')").drop(columns=["dataset"])
    electronics_random = ev.query("(dataset == 'electronics') & split.str.contains('random')").drop(columns=["dataset"])

    num_splits_ma = len(modcloth_active["split"].unique())
    num_splits_mr = len(modcloth_random["split"].unique())

    num_splits_ea = len(electronics_active["split"].unique())
    num_splits_er = len(electronics_random["split"].unique())

    modcloth_active = compute_stats(modcloth_active)
    modcloth_random = compute_stats(modcloth_random)

    electronics_active = compute_stats(electronics_active)
    electronics_random = compute_stats(electronics_random)

    width_mb = len(modcloth_baseline.columns)
    width_ma = len(modcloth_active.columns)
    width_mr = len(modcloth_random.columns)

    height_mb = len(modcloth_baseline) + 1
    height_ma = len(modcloth_active) + 1
    height_mr = len(modcloth_random) + 1

    width_eb = len(electronics_baseline.columns)
    width_ea = len(electronics_active.columns)
    width_er = len(electronics_random.columns)

    height_eb = len(electronics_baseline) + 1
    height_ea = len(electronics_active) + 1
    height_er = len(electronics_random) + 1

    title_spacing = 1
    dataset_spacing = 2

    title_row_mb = 0
    start_row_mb = title_row_mb + title_spacing

    title_row_ma = start_row_mb + height_mb + dataset_spacing
    start_row_ma = title_row_ma + title_spacing

    title_row_mr = start_row_ma + height_ma + dataset_spacing
    start_row_mr = title_row_mr + title_spacing

    title_row_eb = start_row_mr + height_mr + dataset_spacing
    start_row_eb = title_row_eb + title_spacing

    title_row_ea = start_row_eb + height_eb + dataset_spacing
    start_row_ea = title_row_ea + dataset_spacing

    title_row_er = start_row_ea + height_ea + dataset_spacing
    start_row_er = title_row_er + title_spacing

    timestamp = datetime.now().strftime("%d-%b-%y, %-I.%M %p")

    with pd.ExcelWriter("throwaway/evaluation.xlsx") as writer:
        sheet_name = f"Evaluation ({timestamp})"

        modcloth_baseline.to_excel(writer, sheet_name=sheet_name, startrow=start_row_mb, index=False, float_format="%.4f")  # Need to create the worksheet to access it
        modcloth_active.to_excel(writer, sheet_name=sheet_name, startrow=start_row_ma, index=False, float_format="%.4f")
        modcloth_random.to_excel(writer, sheet_name=sheet_name, startrow=start_row_mr, index=False, float_format="%.4f")

        electronics_baseline.to_excel(writer, sheet_name=sheet_name, startrow=start_row_eb, index=False, float_format="%.4f")
        electronics_active.to_excel(writer, sheet_name=sheet_name, startrow=start_row_ea, index=False, float_format="%.4f")
        electronics_random.to_excel(writer, sheet_name=sheet_name, startrow=start_row_er, index=False, float_format="%.4f")

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Increase wdith for model name
        worksheet.set_column(0, 0, 25)
        num_cols = max(width_mb, width_ma, width_mr, width_eb, width_ea, width_er)
        for col in range(1, num_cols):
            ending_col_pad = 5*((num_cols - col) <= 6) + 5*((num_cols - col) <= 4)  # TODO: This is a total hack that works because of lucky alignment of where stretched columns are. Fix this if/when we ahve time.
            worksheet.set_column(col, col, 15 + ending_col_pad)
        big_font = workbook.add_format({'font_size': 15})
        for row in [title_row_mb, title_row_ma, title_row_mr, title_row_eb, title_row_ea, title_row_er]:
            worksheet.set_row(row, None, big_font)
        worksheet.merge_range(title_row_mb, 0, title_row_mb, width_mb-1, 'Modcloth (Baseline Split)')
        worksheet.merge_range(title_row_ma, 0, title_row_ma, width_mr-1, f"Modcloth (Average of {num_splits_mr} Active User Splits)")
        worksheet.merge_range(title_row_mr, 0, title_row_mr, width_mr-1, f"Modcloth (Average of {num_splits_mr} Random Splits)")
        worksheet.merge_range(title_row_eb, 0, title_row_eb, width_eb-1, f"Electronics (Baseline Split)")
        worksheet.merge_range(title_row_ea, 0, title_row_ea, width_mr-1, f"Electronics (Average of {num_splits_mr} Active User Splits)")
        worksheet.merge_range(title_row_er, 0, title_row_er, width_er-1, f"Electronics (Average of {num_splits_er} Random Splits)")


if __name__ == '__main__':
    postprocess()
