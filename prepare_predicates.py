from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["modcloth", "electronics"]
NUM_SPLITS = 5
TRAIN_TEST_RATIO = 80 / 20

PROTECTED_ATTR_MAPS = {
    # Deliberate choice to make Large == Small&Large)
    "modcloth": [
        ("Small", 0),
        ("Large", 1),
        ("Small&Large", 1)
    ],
    "electronics": [
        ("Male", 0),
        ("Female", 1),
        ("Female&Male", 2)
    ]
}

# Linear scaling
RATING_SCALE = [
    (1, 0),
    (2, 1/4),
    (3, 1/2),
    (4, 3/4),
    (5, 1)
]


REPOSITORY_DIR = Path(__file__).parent.absolute()


def make_attr_map_df(attr_map, file_path):
    df = pd.DataFrame.from_records(attr_map, columns=["attr", "label"])
    df.to_csv(file_path, index=False, header=False, sep='\t')
    return df


def make_index_column(map_dir, data, column_name):
    indices, values = pd.factorize(data[column_name])
    pd.DataFrame(values).to_csv(
        map_dir / (column_name + '.txt'),
        header=False,
        sep='\t')
    indices = pd.Series(indices).replace(-1, np.nan)
    return pd.DataFrame(indices, columns=[column_name])


def substitute_column(data, column_name, attr_map):
    return data[[column_name]].replace(attr_map["attr"].to_list(), attr_map["label"].to_list())


for dataset in DATASETS:
    dataset_dir = REPOSITORY_DIR / "datasets" / dataset
    raw_data = pd.read_csv(dataset_dir / "raw" / f"df_{dataset}.csv",
                           converters={"user_id": str})  # For modcloth NaN user
    map_dir = dataset_dir / "maps"
    map_dir.mkdir(exist_ok=True)
    # Factorize
    item_index = make_index_column(map_dir, raw_data, "item_id")
    user_index = make_index_column(map_dir, raw_data, "user_id")
    brand_index = make_index_column(map_dir, raw_data, "brand")
    category_index = make_index_column(map_dir, raw_data, "category")
    # TODO: Create scales for fit, size, and possibly dates
    # Make substitute dataframes
    protected_attr_map_df = make_attr_map_df(PROTECTED_ATTR_MAPS[dataset], map_dir / "protected_attribute.txt")
    rating_scale_df = make_attr_map_df(RATING_SCALE, map_dir / "rating_scale.txt")
    # Create substitution indices
    user_attr_index = substitute_column(raw_data, "user_attr", protected_attr_map_df)
    model_attr_index = substitute_column(raw_data, "model_attr", protected_attr_map_df)
    rating_normalized = substitute_column(raw_data, "rating", rating_scale_df)
    # TODO: Incorporate the other attributes