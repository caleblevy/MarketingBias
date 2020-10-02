from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path(__file__).parent.absolute() / "throwaway" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def ratings_by_user_rating_counts(dataset_name, data):
    description = "Full Data"
    user_rating_counts = data[["user_id"]].value_counts().reset_index(name="User Rating Count")
    rating_count_frequency = user_rating_counts[["User Rating Count"]].value_counts().reset_index(name="Number of Users")
    fig, ax = plt.subplots()
    rating_count_frequency.hist("User Rating Count", weights=rating_count_frequency["Number of Users"], bins=range(0, user_rating_counts["User Rating Count"].max()+1), ax=ax)
    ax.set_yscale('log')
    plt.title(f"{dataset_name} {description}")
    plt.xlabel("User Rating Count")
    plt.ylabel("Number of Ratings")
    fig.savefig(FIG_DIR / f"{dataset_name}_{''.join(description.split())}Ratings.pdf")


def hist_of_train_rating_counts_for_users_in_test(dataset_name, data):
    train, test = [x for _, x in data.groupby(data["split"] == 2)]
    test_users = test[["user_id"]].drop_duplicates()
    train_users = train.merge(test_users, on="user_id")
    _hist_of_user_rating_counts(dataset_name, train_users, "Baseline Split", "Ratings in Train Set per User in Test Set")


def hist_of_rating_counts_for_full_data(dataset_name, data):
    _hist_of_user_rating_counts(dataset_name, data, "Full Data", "Ratings per User")


def _hist_of_user_rating_counts(dataset_name, data, description, xlabel):
    user_rating_counts = data[["user_id"]].value_counts().reset_index(name="User Rating Count")
    fig, ax = plt.subplots()
    user_rating_counts.hist("User Rating Count", bins=range(0, user_rating_counts["User Rating Count"].max()+1), ax=ax, density=True)
    plt.title(f"{dataset_name} {description}")
    plt.xlabel(xlabel)
    plt.ylabel("Number of Users")
    ax.set_yscale("log")
    fig.savefig(FIG_DIR / f"{dataset_name}_{''.join(description.split())}.pdf")


def user_contributions(data):
    user_rating_counts = data[["user_id"]].value_counts().reset_index(name="ratings_per_user")
    counts = user_rating_counts[["ratings_per_user"]].value_counts().reset_index(name="num_users")
    counts["num_ratings"] = counts["ratings_per_user"] * counts["num_users"]
    counts.sort_values(by="ratings_per_user", inplace=True)
    fraction = pd.DataFrame()
    fraction["Max Ratings per User"] = counts["ratings_per_user"]
    fraction["Fraction of Ratings"] = counts["num_ratings"].cumsum() / counts["num_ratings"].sum()
    print(fraction.to_string(index=False))


def produce_summary_statistics():
    modcloth = pd.read_csv("datasets/modcloth/raw/df_modcloth.csv")
    electronics = pd.read_csv("datasets/electronics/raw/df_electronics.csv")
    datasets = {
        "Modcloth": modcloth,
        "Electronics": electronics
    }
    for dataset_name, data in datasets.items():
        print(dataset_name, '\n---------------------')
        user_contributions(data)
        print()
        ratings_by_user_rating_counts(dataset_name, data)
        hist_of_train_rating_counts_for_users_in_test(dataset_name, data)
        hist_of_rating_counts_for_full_data(dataset_name, data)
    # plt.show()


if __name__ == '__main__':
    produce_summary_statistics()
