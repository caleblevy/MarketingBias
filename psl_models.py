#!/usr/bin/env python3

from pathlib import Path
import os

import pandas as pd

from _pslmodel import Model
from evaluation import evaluate

# TODO: Model.PSL_JAR_PATH to custom in repository
# TODO: Figure out how capture logging out

DATA_DIR = Path(__file__).parent.absolute() / "datasets"
RESULT_DIR = Path(__file__).parent.absolute() / "results"
DATASETS = [
    "modcloth",
    "electronics"
]
SPLITS = ["baseline_split", 0, 1, 2, 3, 4]

PRINT_JAVA_OUTPUT = True
RUN_MODEL = False

# TODO Switch these to argparse commands --overwrite and --dry-run
OVERWRITE_OLD_DATA = False
# DRY_RUN = True

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'DEBUG'
}

ADDITIONAL_CLI_OPTIONS = [
    '--postgres'
    # '--satisfaction'
]

# TODO: If there are too many shared models, we can make "shared models" global variable
MODELS = {
    "baseline": set(),
    "prior": {
        "rating_priors"
    },
    "similarity": {
        "rating_priors",
        "similarities"
    },
    "mf_prior": {
        "rating_priors",
        "matrix_factorization_prior"
    },
    "mf_prior_similarity": {
        "rating_priors",
        "matrix_factorization_prior",
        "similarities"
    }
    # "parity_fairness": {
    #     "rating_priors",
    #     "similarities",
    #     "value_fairness"
    # }
}


# TODO: deal with weight learning (separate data adding)
def main():
    eval_tokens = {
        "dataset": [],
        "model": [],
        "split": [],
        "MAE": [],
        "MSE": [],
        "F-stat": []
    }
    for dataset in DATASETS:
        eval_tokens["dataset"]
        for split in SPLITS:
            predicate_dir = DATA_DIR / dataset / "predicates" / str(split)
            for model_name, ruleset in MODELS.items():
                print(dataset, model_name, split)
                if split != 'baseline_split' and 'mf' in model_name:
                    continue
                output_dir = RESULT_DIR / dataset / model_name / str(split)
                model = make_model(model_name, predicate_dir, output_dir, ruleset)
                if RUN_MODEL:
                    results = model.infer(additional_cli_options=ADDITIONAL_CLI_OPTIONS,
                                          psl_config=ADDITIONAL_PSL_OPTIONS,
                                          print_java_output=PRINT_JAVA_OUTPUT)
                eval_tokens["dataset"].append(dataset)
                eval_tokens["model"].append(model_name)
                eval_tokens["split"].append(str(split))
                evaluate(model, eval_tokens)
    eval_df = pd.DataFrame(eval_tokens)
    print(eval_df)
    eval_df.to_csv("evaluation.csv", index=False)


def make_model(model_name, predicate_dir, output_dir, ruleset):
    model = Model(model_name, predicate_dir, output_dir)
    add_baselines(model)
    if "rating_priors" in ruleset:
        add_rating_priors(model)
    if "matrix_factorization_prior" in ruleset:
        add_mf_prior(model)
    if "similarities" in ruleset:
        add_similarities(model)
    if "value_fairness" in ruleset:
        add_value_fairness(model)
    return model


def add_baselines(model):
    model.add_predicate("Rating", size=2, closed=False)
    model.add_predicate("ItemGroup", size=2, closed=True)
    model.add_predicate("UserGroup", size=2, closed=True)
    model.add_predicate("ValidUserGroup", size=1, closed=True)
    model.add_predicate("ValidItemGroup", size=1, closed=True)
    model.add_predicate("Rated", size=2, closed=True)
    model.add_predicate("Target", size=2, closed=True)
    # Negative prior
    model.add_rule("1: !Rating(U,I) ^2")


def add_rating_priors(model):
    model.add_predicate("AverageUserRating", size=1, closed=True)
    model.add_predicate("AverageItemRating", size=1, closed=True)
    model.add_predicate("AverageBrandRating", size=1, closed=True)
    model.add_predicate("ItemBrand", size=2, closed=True)
    # Average user rating prior
    model.add_rule("10: AverageUserRating(U) & Rated(U, I) -> Rating(U, I) ^2")
    # Average item rating prior
    model.add_rule("10: AverageItemRating(I) & Rated(U, I) -> Rating(U, I) ^2")
    # Average 
    model.add_rule("10: AverageBrandRating(B) & Rated(U, I) & ItemBrand(I, B) -> Rating(U, I) ^2")
    # Average user rating prior
    model.add_rule("10: Rating(U, I) & Rated(U, I) ->  AverageUserRating(U) ^2")
    # Average item rating prior
    model.add_rule("10: Rating(U, I) & Rated(U, I) -> AverageItemRating(I) ^2")
    # Average 
    model.add_rule("10: Rating(U, I) & Rated(U, I) & ItemBrand(I, B) -> AverageBrandRating(B)^2")


def add_similarities(model):
    model.add_predicate("SimilarUser", size=2, closed=True)
    model.add_predicate("SimilarItem", size=2, closed=True)
    model.add_rule("100: Rated(U1, I) & Rated(U2, I) & SimilarUser(U1, U2) & Rating(U1, I) -> Rating(U2, I) ^2")
    model.add_rule("100: Rated(U, I1) & Rated(U, I2) & SimilarItem(I1, I2) & Rating(U, I1) -> Rating(U, I2) ^2")
    model.add_rule("100: Rated(U1, I) & Rated(U2, I) & SimilarUser(U1, U2) & Rating(U2, I) -> Rating(U1, I) ^2")
    model.add_rule("100: Rated(U, I1) & Rated(U, I2) & SimilarItem(I1, I2) & Rating(U, I2) -> Rating(U, I1) ^2")


def add_mf_prior(model):
    model.add_predicate("MFRating", size=2, closed=True)
    model.add_rule("10: MFRating(U, I) -> Rating(U, I) ^2")
    model.add_rule("10: Rating(U, I) -> MFRating(U, I) ^2")


# rating(+U, +I) / DENOMINATOR_1 = group1_avg_rating(c) . {U: group_1(U)} {I: group_1_item_block(I) & target(U,I)}
# rating(+U, +I) / DENOMINATOR_2 = group2_avg_rating(c) . {U: group_2(U)} {I: group_2_item_block(I) & target(U,I)}


# def _combine(stringlist, k):
#     """Combine in a way that parens do not induce stack overflow"""
#     if len(stringlist) == 1:
#         return stringlist[0]
#     elif len(stringlist) == k:
#         return " + ".join(stringlist)
#     elif len(stringlist) > k:
#         strlen = len(stringlist) // 2
#         s1 = _combine(stringlist[:strlen])
#         s2 = _combine(stringlist[strlen:])
#         return f"({s1}) + ({s2})"
#
#
# def add_value_fairness(model):
#     model.add_predicate("AveragePredictedSegmentRating", closed=False, size=2)
#     Target = model.load_eval_observations("Target", ["user_id", "item_id"]).iloc[:, :-1]
#     UserGroup = model.load_eval_observations("UserGroup", ["user_id", "user_attr"]).iloc[:, :-1]
#     ItemGroup = model.load_eval_observations("ItemGroup", ["item_id", "model_attr"]).iloc[:, :-1]
#     user_attrs = model.load_eval_observations("ValidUserGroup")["col1"].unique()
#     item_attrs = model.load_eval_observations("ValidItemGroup")["col1"].unique()
#     # Fist Way: Gives stack overflow in parser
#     for ug in user_attrs:
#         for ig in item_attrs:
#             segment = (Target.merge(UserGroup.query("user_attr == @ug"), on="user_id")
#                              .merge(ItemGroup.query("model_attr == @ig"), on="item_id")
#                       )[["user_id", "item_id"]]
#             segment_size = max(1, len(segment))
#             segment_ratings = [f"Rating('{u}', '{i}')/{segment_size}" for (u, i) in zip(segment["user_id"].to_list(), segment["item_id"].to_list())]
#             body = _combine(segment_ratings)
#             model.add_rule(body + f" = AveragePredictedSegmentRating('{ug}', '{ig}') .")
#     # exit()
#
#     # Second Way: Times out in cross product
#     # user_groups = {}
#     # item_groups = {}
#     # for u in user_attrs:
#     #     model.add_predicate(f"Group{u}User", closed=True, size=1)
#     #     user_groups[u] = model.load_eval_observations(f"Group{u}User", ["user_id"]).iloc[:, :-1]
#     # for i in item_attrs:
#     #     model.add_predicate(f"Group{i}Item", closed=True, size=1)
#     #     item_groups[i] = model.load_eval_observations(f"Group{i}Item", ["item_id"]).iloc[:, :-1]
#     # for u in user_attrs:
#     #     for i in item_attrs:
#     #         normalization = max(1, len(target.merge(user_groups[u]).merge(item_groups[i])))
#     #         print(f"Rating(+U, +I) / {normalization} = AveragePredictedSegmentRating('{u}', '{i}') . {{U: Target(U, I) & Group{u}User(U)}} {{I: Target(U, I) & Group{i}Item(I)}}")
#     #         model.add_rule(f"Rating(+U, +I) / {normalization} = AveragePredictedSegmentRating('{u}', '{i}') . {{U: Target(U, I) & Group{u}User(U)}} {{I: Target(U, I) & Group{i}Item(I)}}", weighted=False)
#     # model.add_rule("10: AveragePredictedSegmentRating(UG1, IG1) = AveragePredictedSegmentRating(UG2, IG2)")


if (__name__ == '__main__'):
    main()
