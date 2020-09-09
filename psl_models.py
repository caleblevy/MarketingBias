#!/usr/bin/env python3

from pathlib import Path
import os

from _pslmodel import Model, Predicate
from pslpython.partition import Partition
from pslpython.rule import Rule

# TODO: Model.PSL_JAR_PATH to custom in repository
# TODO: Figure out how capture logging out

DATA_DIR = Path(__file__).parent.absolute() / "datasets"
RESULT_DIR = Path(__file__).parent.absolute() / "results"
SPLITS = [0, 1, 2, 3, 4]

# TODO Switch these to argparse commands --overwrite and --dry-run
OVERWRITE_OLD_DATA = True
DRY_RUN = True

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
    '--satisfaction'
]

# TODO: If there are too many shared models, we can make "shared models" global variable
MODELS = {
    "modcloth": {
        "baseline": {},
        "priors": {
            "rating_priors": True
        },
        "similarities": {
            "rating_priors": True,
            "similarities": True
        }
    }
}
#         "similarities": {
#             "rating_priors": True,
#             "similarities": True
#         }
#     },
#
#     "electronics":{
#         "baseline": {},
#         "priors": {
#             "rating_priors": True
#         },
#         "similarities": {
#             "rating_priors": True,
#             "similarities": True
#         }
#     }
# }


# TODO: deal with weight learning (separate data adding)
def main():
    for dataset, models in MODELS.items():
        for split in SPLITS:
            predicate_dir = DATA_DIR / dataset / "predicates" / str(split)
            for model_name, ruleset in models.items():
                output_dir = RESULT_DIR / dataset / model_name / str(split)
                model = make_model(model_name, predicate_dir, output_dir, **ruleset)
                results = model.infer(additional_cli_options=ADDITIONAL_CLI_OPTIONS,
                                      psl_config=ADDITIONAL_PSL_OPTIONS)
                print(results)


def make_model(model_name, predicate_dir, output_dir,
                rating_priors=False,
                mf_prior=False,
                similarities=False):
    model = Model(model_name, predicate_dir, output_dir)
    add_baselines(model, predicate_dir)
    if rating_priors:
        add_rating_priors(model, predicate_dir)
    if mf_prior:
        add_mf_prior(model, predicate_dir)
    if similarities:
        add_similarities(model, predicate_dir)
    return model


def add_baselines(model, predicate_dir, sqare=True):
    model.add_predicate("Rating", size=2, closed=False)
    model.add_predicate("Rated", size=2, closed=True)
    # Negative prior
    model.add_rule(Rule("1: !Rating(U,I) ^2"))


def add_rating_priors(model, predicate_dir, square=True):
    model.add_predicate("AverageUserRating", size=1, closed=True)
    model.add_predicate("AverageItemRating", size=1, closed=True)
    model.add_predicate("AverageBrandRating", size=1, closed=True)
    model.add_predicate("ItemBrand", size=2, closed=True)
    # Average user rating prior
    model.add_rule(Rule("10: AverageUserRating(U) & Rated(U, I) -> Rating(U, I) ^2"))
    # Average item rating prior
    model.add_rule(Rule("10: AverageItemRating(I) & Rated(U, I) -> Rating(U, I) ^2"))
    # Average 
    model.add_rule(Rule("10: AverageBrandRating(B) & Rated(U, I) & ItemBrand(I, B) -> Rating(U, I) ^2"))


# def add_mf_prior(model, predicate_dir, square=True):
#     MF_Rating = Predicate("MF_Rating", size=2, closed=True)
#     model.add_predicate(MF_Rating)
#
#     MF_Rating.add_data_file(Partition.OBSERVATIONS, predicate_dir / "mf_ratings_test.txt")
#
#     # Matrix factorization prior
#     model.add(Rule("10: Rated(U, I) & MF_Rating(U, I) -> Rating(U, I) ^2"))


def add_similarities(model, predicate_dir, threshold=0.8, min_rating_count=3):
    model.add_predicate("SimilarUser", size=2, closed=True)
    model.add_predicate("SimilarItem", size=2, closed=True)
    model.add_rule(Rule("100: Rated(U1, I) & Rated(U2, I) & SimilarUser(U1, U2) & Rating(U1, I) -> Rating(U2, I) ^2"))
    model.add_rule(Rule("100: Rated(U, I1) & Rated(U, I2) & SimilarItem(I1, I2) & Rating(U, I1) -> Rating(U, I2) ^2"))


if (__name__ == '__main__'):
    main()
