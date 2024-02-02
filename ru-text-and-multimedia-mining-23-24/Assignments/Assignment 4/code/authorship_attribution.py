import os
from copy import deepcopy
import json

import pandas as pd
import matplotlib.pyplot as plt

from featurizer import Featurizer
from classifier import Classifier


RERUN_FEATURES = False


def main():
    featurizer = Featurizer()

    full_train_set = get_features(
        featurizer,
        RERUN_FEATURES,
        feature_filename="data/train_features.csv",
        filename="data/pan2324_train_data.csv",
    )
    full_dev_set = get_features(
        featurizer,
        RERUN_FEATURES,
        feature_filename="data/dev_features.csv",
        filename="data/pan2324_dev_data.csv",
    )
    full_test_set = get_features(
        featurizer,
        RERUN_FEATURES,
        feature_filename="data/test_features.csv",
        filename="data/pan2324_test_data.csv",
    )

    print(full_train_set['author'].value_counts())

    exit()
    print("Running classifier...")
    classifier = Classifier()

    colnames = [col for col in full_train_set.columns if col.startswith("f_")]

    train_x = full_train_set[colnames].to_numpy()
    train_y = full_train_set["author"].to_numpy()
    classifier.fit(train_x, train_y)

    print("Evaluating classifier...")
    train_report = classifier.evaluate(train_x, train_y)

    dev_x = full_dev_set[colnames].to_numpy()
    dev_y = full_dev_set["author"].to_numpy()
    dev_report = classifier.evaluate(dev_x, dev_y)

    test_x = full_test_set[colnames].to_numpy()
    test_y = full_test_set["author"].to_numpy()
    test_report = classifier.evaluate(test_x, test_y)

    report = {
        "train_report": train_report,
        "dev_report": dev_report,
        "test_report": test_report,
    }
    with open(f"classifier_results.json", "w") as outfile:
        json.dump(report, outfile, indent=4)

    print("Doing ablation study")
    perform_ablation_study(colnames, full_train_set, full_dev_set, full_test_set)
    perform_ablation_study(
        colnames, full_train_set, full_dev_set, full_test_set, reverse=True
    )
    perform_ablation_study(
        colnames, full_train_set, full_dev_set, full_test_set, groups=True
    )
    perform_ablation_study(
        colnames, full_train_set, full_dev_set, full_test_set, groups=True, reverse=True
    )


def get_features(
    featurizer: Featurizer, rerun_features: bool, feature_filename: str, filename: str
) -> pd.DataFrame:
    """Get the features for texts and return a dataframe with the text and the features

    Args:
        featurizer (Featurizer): Object that performs featurization
        rerun_features (bool): If the features should be re-calculated
        feature_filename (str): Filename for the calculated features
        filename (str): Filename containing the texts and authors

    Returns:
        pd.Dataframe: All texts, their features and their author
    """
    data_set = pd.read_csv(filename, index_col=0)
    data_set.index.names = ["index"]

    if rerun_features or not os.path.exists(feature_filename):
        print("Calculating features...")
        features = featurizer.featurize(data_set)
        features.to_csv(feature_filename)
        summary = features.describe(include="all")
        summary.to_csv(f"{feature_filename.split('.')[0].split('_')[0]}_summary.csv")
    else:
        print("Loading features...")
        features = pd.read_csv(feature_filename, index_col="index")

    return pd.merge(data_set, features, on="index", how="inner")


def perform_ablation_study(
    colnames: list,
    full_train_set: pd.DataFrame,
    full_dev_set: pd.DataFrame,
    full_test_set: pd.DataFrame,
    groups: bool = False,
    reverse: bool = False,
) -> None:
    """Perform an ablation study

    Args:
        colnames (list): Names of the feature columns in the datasets
        full_train_set (pd.DataFrame): Train set
        full_dev_set (pd.DataFrame): Dev set
        full_test_set (pd.DataFrame): Test set
        groups (bool, optional): If the features should be grouped. Defaults to False.
        reverse (bool, optional): When true, the classifiers are trained for each individual feature(group), 
            insead of leaving out that feature(group). Defaults to False.
    """
    train_results = {}
    dev_results = {}
    test_results = {}
    train_y = full_train_set["author"].to_numpy()
    dev_y = full_dev_set["author"].to_numpy()
    test_y = full_test_set["author"].to_numpy()
    features = ["d", "c", "p", "g", "i", "o"] if groups else colnames
    for feature in features:
        if groups:
            if reverse:
                ablation_colnames = [
                    colname for colname in colnames if colname[2] == feature
                ]
            else:
                ablation_colnames = [
                    colname for colname in colnames if colname[2] != feature
                ]
        else:
            if reverse:
                ablation_colnames = [feature]
            else:
                ablation_colnames = deepcopy(colnames)
                ablation_colnames.remove(feature)

        train_x = full_train_set[ablation_colnames].to_numpy()
        dev_x = full_dev_set[ablation_colnames].to_numpy()
        test_x = full_test_set[ablation_colnames].to_numpy()

        classifier = Classifier()
        classifier.fit(train_x=train_x, train_y=train_y)

        train_results[feature] = classifier.get_f1_score(eval_x=train_x, eval_y=train_y)
        dev_results[feature] = classifier.get_f1_score(eval_x=dev_x, eval_y=dev_y)
        test_results[feature] = classifier.get_f1_score(eval_x=test_x, eval_y=test_y)

    settings_str = f"{'_groups' if groups else ''}{'_reverse' if reverse else ''}"
    results = {
        "train_results": train_results,
        "dev_results": dev_results,
        "test_results": test_results,
    }
    with open(f"results{settings_str}.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    plot_ablation_results(train_results, name="Train", settings_str=settings_str)
    plot_ablation_results(dev_results, name="Dev", settings_str=settings_str)
    plot_ablation_results(test_results, name="Test", settings_str=settings_str)


def plot_ablation_results(
    ablation_results: dict, name: str = "", settings_str: str = ""
) -> None:
    """Plot the results of an ablation study

    Args:
        ablation_results (dict): Results of the ablation study. Keys are feature(group) names. Values are F1 scores
        name (str, optional): Name of the dataset the classifiers were evaluated on. Defaults to "".
        settings_str (str, optional): Settings of the ablation study. Defaults to "".
    """

    if "groups" in settings_str:
        x_ticks = [
            "Default Counts",
            "Complexity",
            "POS tags",
            "Grammar/spelling",
            "Punctuation",
            "Other",
        ]
        rotation=45
        bottom=0.3
        figsize = (7, 5)
    else:
        x_ticks = [i[4:] for i in ablation_results.keys()]
        rotation=90
        bottom=0.45
        figsize = (10, 5)

    plt.figure(figsize=figsize)
    plt.bar(x_ticks, ablation_results.values())
    plt.xticks(rotation=rotation)
    plt.xlabel("Missing feature")
    plt.ylabel("F1 score")
    plt.title(
        f"{name} - Ablation study for authorship attribution ({' '.join(settings_str.split('_')).strip()})"
    )
    plt.subplots_adjust(top=0.9, bottom=bottom)
    plt.savefig(f"ablation_plot_{name.lower()}{settings_str}.png")
    plt.close()


if __name__ == "__main__":
    main()
