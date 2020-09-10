import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances


DATASETS_DIR = Path(__file__).parent.absolute() / "datasets"
DATASETS = [
    "modcloth",
    "electronics"
]
BASELINE_SPLIT = True
NUM_SPLITS = 0
EVAL_TRAIN_TEST_RATIO = 80 / 20
LEARN_TRAIN_TEST_RATIO = 90 / 10

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

# TODO: Fix argument propagation structure for similarity settings
SIMILARITY_SETTINGS = {
    "modcloth": {
        "user_required_rating_count": 2,
        "user_threshold": 0.3,
        "item_threshold": 0.1
    },
    "electronics": {
        "user_required_rating_count": 3,
        "user_threshold": 0.3,
        "item_threshold": 0.1
    },
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
        predicate_dir = dataset_dir / "predicates"
        predicate_dir.mkdir(exist_ok=True)
        similarity_settings = SIMILARITY_SETTINGS[dataset]
        with open(predicate_dir / "similarity_settings.json", 'w', encoding='utf-8') as f:
            json.dump(similarity_settings, f, ensure_ascii=False, indent=4)
        if BASELINE_SPLIT:
            create_baseline_split(data, predicate_dir, similarity_settings=similarity_settings)
        for split in range(NUM_SPLITS):
            # TODO: Create validation, look into cross-validation
            # TODO: Select proper random seeds
            create_random_split(data, predicate_dir, split,
                         eval_test_size=1/(1+EVAL_TRAIN_TEST_RATIO),
                         learn_test_size=1/(1+LEARN_TRAIN_TEST_RATIO),
                         similarity_settings=similarity_settings)


def create_baseline_split(data, predicate_dir, similarity_settings):
    split_dir = predicate_dir / "baseline_split"
    split_dir.mkdir(exist_ok=True)
    observations = data.query("(split == 0) | (split == 1)")
    test = data.query("split == 2")
    observations_learn = observations.query("split == 0")
    test_learn = observations.query("split == 1")
    _create_predicates(data, observations, test, split_dir / "eval", similarity_settings)
    _create_predicates(observations, observations_learn, test_learn, split_dir / "learn", similarity_settings)


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
            category_index,
            raw_data[["split"]]
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


def create_random_split(data, predicate_dir, split, eval_test_size, learn_test_size, similarity_settings):
    split_dir = predicate_dir / str(split)
    split_dir.mkdir(exist_ok=True)
    observations, test = train_test_split(data, test_size=eval_test_size, random_state=split)
    observations_learn, test_learn = train_test_split(observations, test_size=learn_test_size, random_state=split)
    _create_predicates(data, observations, test, split_dir / "eval", similarity_settings)
    _create_predicates(observations, observations_learn, test_learn, split_dir / "learn", similarity_settings)


def _create_predicates(full_data, train, test, output_dir, similarity_settings):
    output_dir.mkdir(exist_ok=True)
    observations_dir = output_dir / "observations"
    targets_dir = output_dir / "targets"
    truth_dir = output_dir / "truth"
    observations_dir.mkdir(exist_ok=True)
    targets_dir.mkdir(exist_ok=True)
    truth_dir.mkdir(exist_ok=True)
    # TODO: Come up with better name to distinguish predicate comprising valid group names from predicate describing the group of a given user
    # Blocking Predicates (TODO: Add category)
    _make_predicate('User', full_data, 'user_id', observations_dir)
    _make_predicate('Item', full_data, 'item_id', observations_dir)
    _make_predicate('ValidUserGroup', full_data, 'user_attr', observations_dir)
    _make_predicate('ValidItemGroup', full_data, 'model_attr', observations_dir)
    _make_predicate('Brand', full_data, 'brand', observations_dir)
    _make_predicate('ItemBrand', full_data, ['item_id', 'brand'], observations_dir)
    _make_predicate('UserGroup', full_data, ['user_id', 'user_attr'], observations_dir)
    _make_predicate('ItemGroup', full_data, ['item_id', 'model_attr'], observations_dir)
    _make_predicate('Rated', full_data, ['user_id', 'item_id'], observations_dir)
    # Average rating prior
    _make_average_rating_predicate('AverageItemRating', train, 'item_id', observations_dir)
    _make_average_rating_predicate('AverageUserRating', train, 'user_id', observations_dir)
    _make_average_rating_predicate('AverageBrandRating', train, 'brand', observations_dir)
    #TODO: Create matrix factorization priors
    _make_user_and_item_similarities(train, observations_dir, **similarity_settings)
    # Ratings in the train/test split
    _make_predicate('Rating', train, ['user_id', 'item_id', 'rating'], observations_dir)
    _make_predicate('Rating', test, ['user_id', 'item_id'], targets_dir)
    _make_predicate('Rating', test, ['user_id', 'item_id', 'rating'], truth_dir)


def _make_predicate(predicate_name, data, columns, output_dir):
    if isinstance(columns, str):
        columns = [columns]
    data = data[columns].dropna().drop_duplicates()
    _write_predicate(predicate_name, data, output_dir)


def _make_average_rating_predicate(predicate_name, data, groupby, output_dir):
    mean = data[[groupby, 'rating']].groupby(groupby).mean().add_prefix('average_').reset_index()
    _write_predicate(predicate_name, mean, output_dir)


def _make_user_and_item_similarities(data, output_dir,
                                     user_required_rating_count,
                                     user_threshold,
                                     item_threshold):
    data = data[["user_id", "item_id", "rating"]]
    user_counts = data[["user_id"]].value_counts().reset_index(name='count')
    frequent_raters = user_counts[user_counts["count"] >= user_required_rating_count][["user_id"]]
    data = data.merge(frequent_raters)
    _make_cosine_similarity_predicate("SimilarItem", data, output_dir,
                                      similarity_index="item_id",
                                      attribute_index="user_id",
                                      threshold=item_threshold)
    _make_cosine_similarity_predicate("SimilarUser", data, output_dir,
                                      similarity_index="user_id",
                                      attribute_index="item_id",
                                      threshold=user_threshold)

def _make_cosine_similarity_predicate(predicate_name, data, output_dir, similarity_index, attribute_index, threshold):
    expanded_data = data.set_index([similarity_index, attribute_index]).unstack()
    similarities = 1 - pairwise_distances(
        expanded_data.fillna(0),
        metric='cosine',
        force_all_finite='allow-nan'
    )
    similarities = pd.DataFrame(
        similarities,
        index=expanded_data.index,
        columns=expanded_data.index
    ).stack()
    similarities.index = similarities.index.set_names([f"{similarity_index}1", f"{similarity_index}2"])
    similarities = similarities.reset_index(name="similarity")
    similarities = similarities[
        (similarities['similarity'] >= threshold) &
        (similarities[f"{similarity_index}1"] < similarities[f"{similarity_index}2"])
    ]
    _write_predicate(predicate_name, similarities, output_dir)


def _write_predicate(predicate_name, data, output_dir):
    # TODO: Add f"{predicate_name}.txt" to mapping of predicate names to file names, loaded via json
    filepath = output_dir / f"{predicate_name}.txt"
    print(f"Writing: {filepath}")
    data.to_csv(
        filepath,
        header=False,
        index=False,
        sep='\t'
    )
    


if __name__ == '__main__':
    main()