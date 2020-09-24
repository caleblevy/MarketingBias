import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold


sys.path.insert(0, 'matrix_factorization')
from main import mf_rating
#from make_mf_into_predicate import make_mf_into_predicate

DATASETS_DIR = Path(__file__).parent.absolute() / "datasets"
DATASETS = [
    "modcloth",
    "electronics"
]
BASELINE_SPLIT = False
COMPUTE_MF_RATINGS = False
NUM_SPLITS = 1
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
        data.to_csv('throwaway/' + dataset+'.csv')
        predicate_dir = dataset_dir / "predicates"
        predicate_dir.mkdir(exist_ok=True)
        similarity_settings = SIMILARITY_SETTINGS[dataset]
        with open(predicate_dir / "similarity_settings.json", 'w', encoding='utf-8') as f:
            json.dump(similarity_settings, f, ensure_ascii=False, indent=4)
        if BASELINE_SPLIT:
            create_baseline_split(data, predicate_dir, similarity_settings=similarity_settings)
            #make_mf_into_predicate(dataset)

        # K-Fold
        kf = KFold(n_splits=NUM_SPLITS)
        for (train_index, test_index), split in zip(kf.split(data), range(kf.get_n_splits())):
            split_dir = predicate_dir / str(split)

            eval_data = create_k_fold_split(data, train_index, test_index, 1/(1+LEARN_TRAIN_TEST_RATIO))
            observations_eval = eval_data.query('split == 0 | split == 1')
            test_eval = eval_data.query('split == 2')

            learn_data = create_weight_learning_split(observations_eval, 1/(1+LEARN_TRAIN_TEST_RATIO))
            observations_learn = learn_data.query('split == 0 | split == 1')
            test_learn = learn_data.query('split == 2')

            create_predicates(data, observations_eval, test_eval, split_dir / "eval", similarity_settings, mf=COMPUTE_MF_RATINGS)
            create_predicates(learn_data, observations_learn, test_learn, split_dir / "learn", similarity_settings)

def create_k_fold_split(data, train_index, test_index, learn_test_size):
    train = data.iloc[train_index]
    train, validation = train_test_split(train, test_size=learn_test_size, random_state=0)

    train = train.assign(split=0)
    validation = validation.assign(split=1)
    test = data.iloc[test_index].assign(split=2)

    observations = train.append(validation)
    data = observations.append(test)

    return data

def create_weight_learning_split(data, validation_size):
    train = data.query('split == 0')
    test = data.query('split == 1').assign(split=2)

    train, validation = train_test_split(train, test_size=validation_size, random_state=0)
    train = train.assign(split=0)
    validation = validation.assign(split=1)

    observations = train.append(validation)
    data = observations.append(test)

    return data


def create_baseline_split(data, predicate_dir, similarity_settings):
    split_dir = predicate_dir / "baseline_split"
    split_dir.mkdir(exist_ok=True)
    observations = data.query("(split == 0) | (split == 1)")
    test = data.query("split == 2")
    observations_learn = observations.query("split == 0")
    test_learn = observations.query("split == 1")
    create_predicates(data, observations, test, split_dir / "eval", similarity_settings)
    create_predicates(observations, observations_learn, test_learn, split_dir / "learn", similarity_settings)


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


def _modify_data_splits(data, split, test_size, validation_size):
    observations, test = train_test_split(data, test_size=test_size, random_state=split)
    train, validation = train_test_split(observations, test_size=validation_size, random_state=split)

    train = train.assign(split=0)
    validation = validation.assign(split=1)
    test = test.assign(split=2)

    observations = train.append(validation)
    data = observations.append(test)

    return data


def create_random_split(data, predicate_dir, split, eval_test_size, learn_test_size, similarity_settings, compute_mf_ratings):
    split_dir = predicate_dir / str(split)
    split_dir.mkdir(exist_ok=True)
    data = _modify_data_splits(data, split, eval_test_size, learn_test_size)
    observations_eval = data.query('split == 0 | split == 1')
    test_eval = data.query('split == 2')
    observations_learn = data.query('split == 0')
    test_learn = data.query('split == 1')
    create_predicates(data, observations_eval, test_eval, split_dir / "eval", similarity_settings, mf=compute_mf_ratings)
    create_predicates(observations_eval, observations_learn, test_learn, split_dir / "learn", similarity_settings)


def create_predicates(full_data, train, test, output_dir, similarity_settings, mf=False):
    output_dir.mkdir(exist_ok=True)
    observations_dir = output_dir / "observations"
    targets_dir = output_dir / "targets"
    truth_dir = output_dir / "truth"
    observations_dir.mkdir(exist_ok=True)
    targets_dir.mkdir(exist_ok=True)
    truth_dir.mkdir(exist_ok=True)
    # TODO: Come up with better name to distinguish predicate comprising valid group names from predicate describing the group of a given user
    # ---- BLOCKING PREDICATES ---- (TODO: Add category)
    make_blocking_predicate('User', full_data, 'user_id', observations_dir)
    make_blocking_predicate('Item', full_data, 'item_id', observations_dir)
    make_blocking_predicate('Brand', full_data, 'brand', observations_dir)
    make_blocking_predicate('ItemBrand', full_data, ['item_id', 'brand'], observations_dir)
    make_blocking_predicate('Rated', full_data, ['user_id', 'item_id'], observations_dir)
    make_blocking_predicate('Target', test, ['user_id', 'item_id'], observations_dir)
    # ---- AVERAGE RATING PRIORS ----
    _make_average_rating_predicate('AverageItemRating', train, 'item_id', observations_dir)
    _make_average_rating_predicate('AverageUserRating', train, 'user_id', observations_dir)
    _make_average_rating_predicate('AverageBrandRating', train, 'brand', observations_dir)
    # ---- SIMILARITIES ----
    make_cosine_similarities(train, observations_dir, **similarity_settings)
    # ---- FAIRNESS PREDICATES ----
    # group blockings
    make_blocking_predicate('ValidUserGroup', full_data, 'user_attr', observations_dir)
    make_blocking_predicate('ValidItemGroup', full_data, 'model_attr', observations_dir)
    make_blocking_predicate('UserGroup', full_data, ['user_id', 'user_attr'], observations_dir)
    make_blocking_predicate('ItemGroup', full_data, ['item_id', 'model_attr'], observations_dir)
    # segment by rating
    _make_average_rating_predicate("ObsSegmentRatingAvg", train, ["user_attr", "model_attr"], observations_dir)
    _make_predicate("TargetSegmentRatingAvg", test, ["user_attr", "model_attr"], targets_dir)
    _make_predicate("TargetRatingSegment", test, ["user_id", "item_id", "user_attr", "model_attr"], targets_dir)
    # segment by item
    make_segment_average_predicate_by("ObsSegmentItemAvg", train, "item_id", observations_dir)
    _make_predicate("TargetSegmentItemAvg", test, ["user_attr", "model_attr"], targets_dir)
    _make_predicate("ItemAvgByUG", test, ["item_id", "user_attr"], targets_dir)
    # segment by user
    make_segment_average_predicate_by("ObsSegmentUserAvg", train, "user_id", observations_dir)
    _make_predicate("TargetSegmentUserAvg", test, ["user_attr", "model_attr"], targets_dir)
    _make_predicate("UserAvgByIG", test, ["user_id", "model_attr"], targets_dir)
    # ---- RATING PREDICATES ----
    # Ratings in the train/test split
    _make_predicate('Rating', train, ['user_id', 'item_id', 'rating'], observations_dir)
    _make_predicate('Rating', test, ['user_id', 'item_id'], targets_dir)
    _make_predicate('Rating', test, ['user_id', 'item_id', 'rating'], truth_dir)
    # ---- MATRIX FACTORIZATION ----
    if mf:
        _make_mf_predicate(full_data, observations_dir)


def make_segment_average_predicate_by(predicate_name, data, by, output_dir):
    data = (data.groupby(["user_attr", "model_attr", by])["rating"]
                .mean()
                .reset_index()
                .groupby(["user_attr", "model_attr"])
                .mean()
                .reset_index()
           )
    _write_predicate(predicate_name, data, output_dir)


def make_blocking_predicate(predicate_name, data, columns, output_dir):
    if isinstance(columns, str):
        columns = [columns]
    data = data[columns].dropna().drop_duplicates()
    data['truthiness'] = 1
    _write_predicate(predicate_name, data, output_dir)


def _make_predicate(predicate_name, data, columns, output_dir):
    if isinstance(columns, str):
        columns = [columns]
    data = data[columns].dropna().drop_duplicates()
    _write_predicate(predicate_name, data, output_dir)


def _make_average_rating_predicate(predicate_name, data, groupby, output_dir):
    if isinstance(groupby, str):
        groupby = [groupby]
    mean = data[groupby + ['rating']].groupby(groupby).mean().add_prefix('average_').reset_index()
    _write_predicate(predicate_name, mean, output_dir)


def make_cosine_similarities(data, output_dir,
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
    similarities['similarity'] = 1
    _write_predicate(predicate_name, similarities, output_dir)


def _make_mf_predicate(data, output_dir):
    ratings = mf_rating(data)
    _write_predicate('MFRatings', ratings, output_dir)


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
