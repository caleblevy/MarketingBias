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
SPLITS = ["baseline_split", 0, 1]

PRINT_JAVA_OUTPUT = True
RUN_MODEL = True

# TODO Switch these to argparse commands --overwrite and --dry-run
OVERWRITE_OLD_DATA = False
# DRY_RUN = True

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'DEBUG'
}

ADDITIONAL_CLI_OPTIONS = [
    '--postgres',
    '--int-ids',
    # '--groundrules', 'throwaway/ground.txt'
    # '--satisfaction'
]

ADDITIONAL_JAVA_OPTIONS = [
    "-Xmx8G",
]

# TODO: If there are too many shared models, we can make "shared models" global variable

# Abbrevs = {
#   Avg: "Average Rating Prior",
#   MF: "Matrix Factorization",
#   Sim: "Similarity",
#   Parity: "Rating Parity Fairness",
#   Value: "Rating Value Fairness"
# }
BASE_MODELS = [
    ["Avg"],
    ["Avg", "Sim"],
    ["MF"],
    ["Avg", "MF", "Sim"],
]


FAIRNESS_RULES = [
    ["Parity"],
    ["Value"],
    ["Parity", "Value"]
]


MODELS = {
    # "baseline": set(),
    # "prior": {
    #     "rating_priors"
    # },
    # "similarity": {
    #     "rating_priors",
    #     "similarities"
    # },
    # "mf_prior": {
    #     "rating_priors",
    #     "matrix_factorization_prior"
    # },
    # "mf_prior_similarity": {
    #     "rating_priors",
    #     "matrix_factorization_prior",
    #     "similarities"
    # },
    "user_parity_fairness": {
        "rating_priors",
        "similarities",
        "user_parity_fairness"
    }
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
                                          jvm_options=ADDITIONAL_JAVA_OPTIONS,
                                          print_java_output=PRINT_JAVA_OUTPUT)
                eval_tokens["dataset"].append(dataset)
                eval_tokens["model"].append(model_name)
                eval_tokens["split"].append(str(split))
                evaluate(model, eval_tokens)
                return  # TODO: Remove this
    eval_df = pd.DataFrame(eval_tokens)
    print(eval_df)
    eval_df.to_csv("evaluation.csv", index=False)


def make_model(model_name, predicate_dir, output_dir, ruleset):
    model = Model(model_name, predicate_dir, output_dir)
    add_baselines(model)
    if "rating_priors" in ruleset:
        add_rating_priors(model)
    if "matrix_factorization" in ruleset:
        add_mf_priors(model)
    if "similarities" in ruleset:
        add_similarities(model)
    if "user_parity_fairness" in ruleset:
        _prepare_rating_fairness(model)
        # _prepare_user_fairness(model)
        # _prepare_item_fairness(model)
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


# ---- RATING FAIRNESS ----


def _prepare_rating_fairness(model):
    model.add_predicate("TargetSegmentRatingAvg", size=2, closed=False)
    model.add_predicate("TargetRatingSegment", size=4, closed=False)
    UserGroup = model.load_eval_observations("UserGroup", ["user_id", "user_attr"]).iloc[:, :-1]
    ItemGroup = model.load_eval_observations("ItemGroup", ["item_id", "model_attr"]).iloc[:, :-1]
    Target = model.load_eval_observations("Target", ["user_id", "item_id"])
    user_groups = model.load_eval_observations("ValidUserGroup")["col1"].unique()
    item_groups = model.load_eval_observations("ValidItemGroup")["col1"].unique()
    model.add_rule(
        "Rating(U, I) = TargetRatingSegment(U, I, UG, IG) .", weighted=False
    )
    for ug in user_groups:
        for ig in item_groups:
            segment = (Target.merge(UserGroup.query("user_attr == @ug"), on="user_id")
                             .merge(ItemGroup.query("model_attr == @ig"), on="item_id")
                      ).dropna()
            segment_size = len(segment)
            model.add_rule(
                f"TargetRatingSegment(+U, +I, '{ug}', '{ig}') / {segment_size} = TargetSegmentRatingAvg('{ug}', '{ig}') .", weighted=False
            )


def add_segment_rating_parity(model):
    # TODO: Write out unique segments in a for-loop, since arithmetic rules don't support != predicate
    model.add_rule(
        "10: TargetSegmentAvg(UG1, IG1) = TargetSegmentAvg(UG2, IG2)"
    )


def add_segment_rating_value_fairness(model):
    model.add_predicate("ObsSegmentRatingAvg", closed=False, size=2)
    model.add_rule(
        "10: TargetSegmentRatingAvg(UG1, IG1) - ObsSegmentRatingAvg(UG1, IG1) = TargetSegmentRatingAvg(UG2, IG2) - ObsSegmentRatingAvg(UG2, IG2)"
    )


# ---- ITEM FAIRNESS ----


def _prepare_item_fairness(model):
    model.add_predicate("TargetSegmentItemAvg", closed=False, size=2)
    model.add_predicate("ItemAvgByUG", closed=False, size=2)
    model.add_rule(
        "Rating(+U, I) / |U| = ItemAvgByUG(I, UG) . {U: UserGroup(U, UG) & Target(U, I)}", weighted=0
    )
    model.add_rule(
        "ItemAvgByUG(+I, UG) / |I| = TargetSegmentItemAvg(UG, IG) . {I: ItemGroup(I, IG)}", weighted=0
    )


def add_segment_item_parity(model):
    model.add_rule(
        "10: TargetSegmentItemAvg(UG1, IG1) = TargetSegmentItemAvg(UG2, IG2)"
    )


def add_segment_item_value_fairness(model):
    model.add_predicate("ObsSegmentItemAvg", closed=False, size=2)
    model.add_rule(
        "10: TargetSegmentItemAvg(UG1, IG1) - ObsSegmentItemAvg(UG1, IG1) = TargetSegmentItemAvg(UG2, IG2) - ObsSegmentItemAvg(UG2, IG2)"
    )


# ---- USER FAIRNESS ----


def _prepare_user_fairness(model):
    model.add_predicate("TargetSegmentUserAvg", closed=False, size=2)
    model.add_predicate("UserAvgByIG", closed=False, size=2)
    model.add_rule(
        "Rating(U, +I) / |I| = UserAvgByIG(U, IG) . {I: ItemGroup(I, IG) & Target(U, I)}", weighted=0
    )
    model.add_rule(
        "UserAvgByIG(+U, IG) / |U| = TargetSegmentUserAvg(UG, IG) . {U: UserGroup(U, UG)}", weighted=0
    )


def add_segment_user_parity(model):
    model.add_rule(
        "10: TargetSegmentUserAvg(UG1, IG1) = TargetSegmentUserAvg(UG2, IG2)"
    )


def add_segment_user_value_fairness(model):
    model.load_predicate("ObsSegmentUserAvg", closed=False, size=2)
    model.add_rule(
        "10: TargetSegmentUserAvg(UG1, IG1) - ObsSegmentUserAvg(UG1, IG1) = TargetSegmentUserAvg(UG2, IG2) - ObsSegmentUserAvg(UG2, IG2)"
    )



if (__name__ == '__main__'):
    main()
