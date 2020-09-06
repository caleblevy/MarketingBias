#!/usr/bin/env python3

from pathlib import Path
import os

from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

# TODO: Model.PSL_JAR_PATH to custom in repository
# TODO: Figure out how capture logging out

DATA_DIR = Path(__file__).parent.absolute() / "datasets-test"
RESULT_DIR = Path(__file__).parent.absolute() / "results"
SPLITS = [0]

# TODO Switch these to argparse commands --overwrite and --dry-run
OVERWRITE_OLD_DATA = True
DRY_RUN = True

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
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
    },

    "electronics":{
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


# TODO: deal with weight learning (separate data adding)
def main():
    for dataset, models in MODELS.items():
        for split in SPLITS:
            predicate_dir = DATA_DIR / dataset / "predicates" / str(split) / 'eval'
            for model_name, ruleset in models.items():
                print(ruleset)
                model = make_model(model_name, predicate_dir, **ruleset)
                output_dir = RESULT_DIR / dataset / model_name / str(split)
                infer(model, output_dir)


# TODO: Add run-specific cli-options
def infer(model, output_dir):
    results = model.infer(additional_cli_options=ADDITIONAL_CLI_OPTIONS,
                          psl_config=ADDITIONAL_PSL_OPTIONS)
    output_dir.mkdir(parents=True, exist_ok=True)
    inferred_predicate_dir = output_dir / "inferred_predicates"
    inferred_predicate_dir.mkdir(exist_ok=True)
    for predicate in model.get_predicates().values():
        if (predicate.closed()):
            continue
        out_path = inferred_predicate_dir / predicate.name()
        results[predicate].to_csv(out_path, sep = "\t", header = False, index = False)


def make_model(model_name, predicate_dir,
                rating_priors=False,
                mf_prior=False,
                similarities=False):
    model = Model(model_name)
    add_baselines(model, predicate_dir)
    if rating_priors:
        add_rating_priors(model, predicate_dir)
    if mf_prior:
        add_mf_prior(model, predicate_dir)
    if similarities:
        add_similarities(model, predicate_dir)
    return model


def add_baselines(model, predicate_dir, sqare=True):
    Rating = Predicate("Rating", size=2, closed=False)
    Rated = Predicate("Rated", size=2, closed=True)
    model.add_predicate(Rating)
    model.add_predicate(Rated)

    Rating.add_data_file(Partition.OBSERVATIONS, predicate_dir / "ratings_train.txt")
    Rating.add_data_file(Partition.TARGETS, predicate_dir / "ratings_test_target.txt")
    # Rating.add_data_file(Partition.TRUTH, predicate_dir / "ratings_test_truth.txt")
    Rated.add_data_file(Partition.OBSERVATIONS, predicate_dir / "rated_test.txt")

    # Negative prior
    model.add_rule(Rule("10: !Rating(U,I) ^2"))



def add_rating_priors(model, predicate_dir, square=True):
    UserAvg = Predicate("UserAvg", size=1, closed=True)
    ItemAvg = Predicate("ItemAvg", size=1, closed=True)
    BrandAvg = Predicate("BrandAvg", size=1, closed=True)
    Brand = Predicate("Brand", size=2, closed=True)  # Brand of the item
    model.add_predicate(UserAvg)
    model.add_predicate(ItemAvg)
    model.add_predicate(BrandAvg)
    model.add_predicate(Brand)

    UserAvg.add_data_file(Partition.OBSERVATIONS, predicate_dir / "user_avg.txt")
    ItemAvg.add_data_file(Partition.OBSERVATIONS, predicate_dir / "item_avg.txt")
    BrandAvg.add_data_file(Partition.OBSERVATIONS, predicate_dir / "brand_avg.txt")
    Brand.add_data_file(Partition.OBSERVATIONS, predicate_dir / "brands.txt")
    
    # Average user rating prior
    model.add_rule(Rule("10: UserAvg(U) & Rated(U, I) -> Rating(U, I) ^2"))
    # Average item rating prior
    model.add_rule(Rule("10: ItemAvg(I) & Rated(U, I) -> Rating(U, I) ^2"))
    # Average 
    model.add_rule(Rule("10: BrandAvg(B) & Rated(U, I) & Brand(I, B) -> Rating(U, I) ^2"))


def add_mf_prior(model, predicate_dir, square=True):
    MF_Rating = Predicate("MF_Rating", size=2, closed=True)
    model.add_predicate(MF_Rating)

    MF_Rating.add_data_file(Partition.OBSERVATIONS, predicate_dir / "mf_ratings_test.txt")

    # Matrix factorization prior
    model.add(Rule("10: Rated(U, I) & MF_Rating(U, I) -> Rating(U, I) ^2"))


def add_similarities(model, predicate_dir, threshold=0.8, min_rating_count=3):
    UserRatedK = Predicate("UserRatedK", size=1, closed=True)
    ItemRatedK = Predicate("ItemRatedK", size=1, closed=True)
    SimilarUsers = Predicate("SimilarUsers", size=2, closed=True)
    SimilarItems = Predicate("SimilarItems", size=2, closed=True)
    model.add_predicate(UserRatedK)
    model.add_predicate(ItemRatedK)
    model.add_predicate(SimilarUsers)
    model.add_predicate(SimilarItems)

    # TODO change K and threshold values programatically
    UserRatedK.add_data_file(Partition.OBSERVATIONS, predicate_dir / 'similar_users_3.txt')
    ItemRatedK.add_data_file(Partition.OBSERVATIONS, predicate_dir / 'similar_items_3.txt')
    SimilarUsers.add_data_file(Partition.OBSERVATIONS, predicate_dir / '3_0.8_user_user_cosine_similarity.txt')
    SimilarItems.add_data_file(Partition.OBSERVATIONS, predicate_dir / '3_0.8_item_item_cosine_similarity.txt')

    model.add_rule(Rule("100: Rated(U2, I) & UserRatedK(U1) & UserRatedK(U2) & SimilarUsers(U1, U2) & Rating(U1, I) -> Rating(U2, I) ^2"))
    model.add_rule(Rule("100: Rated(U, I2) & ItemRatedK(I1) & ItemRatedK(I2) & SimilarItems(I1, I2) & Rating(U, I1) -> Rating(U, I2) ^2"))


def add_value_fairness(model):
    pass


if (__name__ == '__main__'):
    main()
