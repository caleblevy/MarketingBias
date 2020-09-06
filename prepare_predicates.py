from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASETS_DIR = Path(__file__).parent.absolute() / "datasets"
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


# NOTE: Code conventions throughout this file:
#        - major functions called by main()
#        - minor functions prepended by _, grouped by caller
#        - globals are called referenced only from main()


def main():
    for dataset in DATASETS:
        dataset_dir = DATASETS_DIR / dataset
        # TODO: Revisit use of string converter for modcloth NaN
        raw_data = pd.read_csv(dataset_dir / "raw" / f"df_{dataset}.csv", converters={"user_id": str})
        data = preprocess(raw_data, dataset_dir, protected_attr_map=PROTECTED_ATTR_MAPS[dataset], rating_scale=RATING_SCALE)
        for split in range(NUM_SPLITS):
            # TODO: Create validation, look into cross-validation
            # TODO: Select proper random seeds
            observations, test_set = train_test_split(data, test_size=1/(1+TRAIN_TEST_RATIO), random_state=split)


def preprocess(raw_data, dataset_dir, protected_attr_map, rating_scale):
    map_dir = dataset_dir / "maps"
    map_dir.mkdir(exist_ok=True)
    # Factorize
    item_index = _make_index_column(map_dir, raw_data, "item_id")
    user_index = _make_index_column(map_dir, raw_data, "user_id")
    brand_index = _make_index_column(map_dir, raw_data, "brand")
    category_index = _make_index_column(map_dir, raw_data, "category")
    # TODO: Discretize dates
    # TODO: Answer question: do we treat Modcloth[["size", "fit"]] as categorical or numerical?
    # Turn substitutions int dataframes and save them
    protected_attr_map_df = _make_attr_map_df(protected_attr_map, map_dir / "protected_attribute.txt")
    rating_scale_df = _make_attr_map_df(rating_scale, map_dir / "rating_scale.txt")
    # Create substitution indices
    user_attr_index = _substitute_column(raw_data, "user_attr", protected_attr_map_df)
    model_attr_index = _substitute_column(raw_data, "model_attr", protected_attr_map_df)
    rating_normalized = _substitute_column(raw_data, "rating", rating_scale_df)
    # TODO: Incorporate the other attributes
    return pd.concat(
        [
            user_index,
            item_index,
            rating_normalized,
            user_attr_index,
            model_attr_index,
            brand_index,
            category_index
        ],
        axis=1)


def _make_index_column(map_dir, data, column_name):
    indices, values = pd.factorize(data[column_name])
    pd.DataFrame(values).to_csv(
        map_dir / (column_name + '.txt'),
        header=False,
        sep='\t')
    indices = pd.Series(indices).replace(-1, np.nan).astype('Int64')
    return pd.DataFrame(indices, columns=[column_name])


def _make_attr_map_df(attr_map, file_path):
    df = pd.DataFrame.from_records(attr_map, columns=["attr", "label"])
    df.to_csv(file_path, index=False, header=False, sep='\t')
    return df


def _substitute_column(data, column_name, attr_map):
    sub = data[[column_name]].replace(attr_map["attr"].to_list(), attr_map["label"].to_list())
    if column_name != "rating":
        sub = sub.astype('Int64')
    return sub


if __name__ == '__main__':
    main()