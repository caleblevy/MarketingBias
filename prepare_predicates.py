from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.absolute() / "datasets"
DATASETS = ["modcloth", "electronics"]
NUM_SPLITS = 5
TRAIN_TEST_RATIO = 80 / 20


PROTECTED_ATTR_IDS = {
    "Small": 0,
    "Large": 1,
    "Small&Large": 1,  # This deliberately differs from Female&Male due to model semantics
    "Male": 0,
    "Female": 1,
    "Female&Male": 2
}




def write_string(directory, fname, string):
    with open(directory / fname, 'w') as f:
        f.write(string)


def _write_protected_attribute_maps(dataset, map_dir):
    if dataset == 'modcloth':
        map_ = "0\tSmall\n1\tLarge\n1\tSmall&Large\n"
    elif dataset == 'electronics':
        map_ = "0\tMale\n1\tFemale\n2\tFemale&Male\n"
    with open(map_dir / "protected_attributes.txt", 'w') as f:
        f.write(map_)


def make_index_column(map_dir, raw_data, column_name):
    indices, values = pd.factorize(raw_data[column_name])
    pd.DataFrame(values).to_csv(
        map_dir / (column_name + '.txt'),
        header=False,
        sep='\t')
    indices = pd.Series(indices).replace(-1, np.nan)
    indices = pd.DataFrame(indices, columns=[column_name])
    return indices


for dataset in DATASETS:
    raw_data = pd.read_csv(DATA_DIR / dataset / "raw" / ("df_" + dataset + '.csv'),
                           converters={"user_id": str})  # For modcloth NaN user
    map_dir = DATA_DIR / dataset / "maps"
    map_dir.mkdir(exist_ok=True)
    # Deal with fairness
    _write_protected_attribute_maps(dataset, map_dir)
    # Factorize
    item_index = make_index_column(map_dir, raw_data, "item_id")
    user_index = make_index_column(map_dir, raw_data, "user_id")
    brand_index = make_index_column(map_dir, raw_data, "brand")
    category_index = make_index_column(map_dir, raw_data, "category")
    rating_normalized = (raw_data[["rating"]] - 1) / 4
    user_attr = raw_data[["user_attr"]].replace(PROTECTED_ATTR_IDS)
    model_attr = raw_data[["model_attr"]].replace(PROTECTED_ATTR_IDS)
    # TODO: normalize timestamp