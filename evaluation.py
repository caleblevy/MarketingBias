import pandas as pd


def evaluate(truth, inferred):
    print(truth.merge(inferred, on=["user_id", "item_id"], suffixes=["_truth", "_inferred"]))