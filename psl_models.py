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
SPLITS = ['baseline_split']

PRINT_JAVA_OUTPUT = True
RUN_MODEL = True

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
    "modcloth": {
        # "baseline": {},
        "priors": {
            "rating_priors": True
        },
        "similarities": {
            "rating_priors": True,
            "similarities": True
        },
        "mf_prior_similarities": {
            "rating_priors": True,
            "similarities": True,
            "mf_prior": True,
        },
        "mf_prior": {
            "rating_priors": True,
            "similarities": False,
            "mf_prior": True,
        }
    },
    "electronics":{
        # "baseline": {},
        "priors": {
            "rating_priors": True
        },
        "similarities": {
            "rating_priors": True,
            "similarities": True
        },
        "mf_prior_similarities": {
            "rating_priors": True,
            "similarities": True,
            "mf_prior": True,
        },
        "mf_prior": {
            "rating_priors": True,
            "similarities": False,
            "mf_prior": True,
        }
    }
}

# Need Targets


# TODO: deal with weight learning (separate data adding)
def main():
    for dataset, models in MODELS.items():
        for split in SPLITS:
            predicate_dir = DATA_DIR / dataset / "predicates" / str(split)
            for model_name, ruleset in models.items():
                output_dir = RESULT_DIR / dataset / model_name / str(split)
                if RUN_MODEL:
                    model = make_model(model_name, predicate_dir, output_dir, **ruleset)
                    results = model.infer(additional_cli_options=ADDITIONAL_CLI_OPTIONS,
                                          psl_config=ADDITIONAL_PSL_OPTIONS,
                                          print_java_output=PRINT_JAVA_OUTPUT)
                print(dataset, model_name)
                evaluate(predicate_dir, output_dir)


def make_model(model_name, predicate_dir, output_dir,
                rating_priors=False,
                mf_prior=False,
                similarities=False,
                market_parity=False):
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
    model.add_rule("1: !Rating(U,I) ^2")


def add_rating_priors(model, predicate_dir, square=True):
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


def add_similarities(model, predicate_dir):
    model.add_predicate("SimilarUser", size=2, closed=True)
    model.add_predicate("SimilarItem", size=2, closed=True)
    model.add_rule("100: Rated(U1, I) & Rated(U2, I) & SimilarUser(U1, U2) & Rating(U1, I) -> Rating(U2, I) ^2")
    model.add_rule("100: Rated(U, I1) & Rated(U, I2) & SimilarItem(I1, I2) & Rating(U, I1) -> Rating(U, I2) ^2")
    model.add_rule("100: Rated(U1, I) & Rated(U2, I) & SimilarUser(U1, U2) & Rating(U2, I) -> Rating(U1, I) ^2")
    model.add_rule("100: Rated(U, I1) & Rated(U, I2) & SimilarItem(I1, I2) & Rating(U, I2) -> Rating(U, I1) ^2")


def add_mf_prior(model, predicate_dir):
    model.add_predicate("MFRating", size=2, closed=True)
    model.add_rule("10: MFRating(U, I) -> Rating(U, I) ^2")
    model.add_rule("10: Rating(U, I) -> MFRating(U, I) ^2")


if (__name__ == '__main__'):
    main()
