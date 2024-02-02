import torch
import datasets
import pandas as pd
import numpy as np
import random
import math
import os
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
class CovidDataLoader:
    def __init__(self) -> None:
        self.dataset = None

    def load_data(self, trec_covid_csv_path):
        if os.path.exists(trec_covid_csv_path):
            self.dataset = pd.read_csv(trec_covid_csv_path, index_col="doc_query_key")
            self.dataset = self.dataset[self.dataset["qrel_score"].isin([0, 1, 2])]
        else:
            trec_covid_documents = pd.DataFrame(
                datasets.load_dataset(
                    "BeIR/trec-covid", "corpus", cache_dir="./data/cache"
                )["corpus"]
            )
            trec_covid_queries = pd.DataFrame(
                datasets.load_dataset(
                    "BeIR/trec-covid-generated-queries",
                    "train",
                    cache_dir="./data/cache",
                )["train"]
            )
            trec_covid_qrels = pd.DataFrame(
                datasets.load_dataset(
                    "BeIR/trec-covid-qrels", "test", cache_dir="./data/cache"
                )["test"]
            )

            trec_covid_query_document_pairs = pd.merge(
                trec_covid_queries,
                trec_covid_documents,
                on=["_id", "text", "title"],
                how="inner",
            )

            self.dataset = pd.merge(
                trec_covid_query_document_pairs,
                trec_covid_qrels,
                left_on="_id",
                right_on="corpus-id",
                how="inner",
            )

            self.dataset.drop("corpus-id", axis=1, inplace=True)
            self.dataset.index.name = "doc_query_key"
            self.dataset.rename(
                columns={
                    "_id": "doc_id",
                    "title": "doc_title",
                    "text": "doc",
                    "query-id": "query_id",
                    "score": "qrel_score",
                },
                inplace=True,
            )
            self.dataset["doc"] = self.dataset.apply(
                lambda row: row["doc_title"] + " " + row["doc"]
                if pd.notnull(row["doc_title"]) and pd.notnull(row["doc"])
                else row["doc_title"]
                if pd.notnull(row["doc_title"])
                else row["doc"],
                axis=1,
            )
            self.dataset.dropna(subset=["doc", "query", "qrel_score"])
            self.dataset = self.dataset[self.dataset["doc"] != ""]
            self.dataset = self.dataset[self.dataset["query"] != ""]
            self.dataset = self.dataset[self.dataset["qrel_score"].isin([0, 1, 2])]
            if not os.path.exists("/".join(trec_covid_csv_path.split("/")[:-1])):
                os.makedirs("/".join(trec_covid_csv_path.split("/")[:-1]))
            self.dataset.to_csv(trec_covid_csv_path)

    def split_data(self, train=0.6, val=0.2, test=0.2, balanced=False):
        query_id_set = list(self.dataset["query_id"].unique())
        train_interval_queries = [0, math.floor(len(query_id_set) * train)]
        val_interval_queries = [
            math.floor(len(query_id_set) * train),
            math.floor(len(query_id_set) * (val + train)),
        ]
        test_interval_queries = [math.floor(len(query_id_set) * (val + train)), 0]
        random.shuffle(query_id_set)
        train_queries = query_id_set[: train_interval_queries[1]]
        val_queries = query_id_set[val_interval_queries[0] : val_interval_queries[1]]
        test_queries = query_id_set[test_interval_queries[0] :]

        text_set = list(self.dataset["doc_id"].unique())
        train_interval_texts = [0, math.floor(len(text_set) * train)]
        val_interval_texts = [
            math.floor(len(text_set) * train),
            math.floor(len(text_set) * (val + train)),
        ]
        test_interval_texts = [math.floor(len(text_set) * (val + train)), 0]
        random.shuffle(text_set)
        train_texts = text_set[: train_interval_texts[1]]
        val_texts = text_set[val_interval_texts[0] : val_interval_texts[1]]
        test_texts = text_set[test_interval_texts[0] :]

        train_set = self.dataset[self.dataset["query_id"].isin(train_queries)][
            self.dataset["doc_id"].isin(train_texts)
        ]
        val_set = self.dataset[self.dataset["query_id"].isin(val_queries)][
            self.dataset["doc_id"].isin(val_texts)
        ]
        test_set = self.dataset[self.dataset["query_id"].isin(test_queries)][
            self.dataset["doc_id"].isin(test_texts)
        ]

        if balanced:
            return self.balance_splits(train_set, val_set, test_set)

        return train_set, val_set, test_set

    def move_entries(self, over_set, under_set, check_set, class_):
        entries = over_set[over_set["qrel_score"] == class_]
        original_ratios = self.dataset["qrel_score"].value_counts(normalize=True)
        num_to_move = int((over_set["qrel_score"].value_counts(normalize=True)[class_] - original_ratios[class_]) * len(over_set))

        over_set_rows_class_before = over_set[over_set["qrel_score"] == class_].shape[0]
        
        for _ in range(num_to_move):
            entry = entries.sample(1)
            entries = entries.drop(entry.index)

            if entry["query_id"].values[0] not in check_set["query_id"].values and entry["doc_id"].values[0] not in check_set["doc_id"].values:
                over_set = over_set[~(over_set["query_id"].isin([entry["query_id"].values[0]]) & over_set["doc_id"].isin([entry["doc_id"].values[0]]))]
                if entry["query_id"].values[0] not in over_set["query_id"].values and entry["doc_id"].values[0] not in over_set["doc_id"].values:
                    under_set = pd.concat([under_set, entry], ignore_index=True)
                else:
                    over_set = pd.concat([over_set, entry], ignore_index=False)


        over_set_rows_class_after = over_set[over_set["qrel_score"] == class_].shape[0]
        
        print(f"Moved {over_set_rows_class_before - over_set_rows_class_after} rows from over_set to under_set for class {class_}.")
        return over_set, under_set


    def balance_splits(self, train_set, val_set, test_set):
      print("Balancing splits...")
      original_ratios = self.dataset["qrel_score"].value_counts(normalize=True)
      print("Class ratios from combined dataset:")
      print(original_ratios)

      print("\n\nStarting with balancing process...\n\n")

      fig, ax = plt.subplots()

      train_ratios_history = [[], [], []]
      val_ratios_history = [[], [], []]
      test_ratios_history = [[], [], []]

      for i in range(20):
          print(f"Starting iteration {i+1}...")
          # Calculate class ratios in each split
          train_ratios = train_set['qrel_score'].value_counts(normalize=True)
          val_ratios = val_set['qrel_score'].value_counts(normalize=True)
          test_ratios = test_set['qrel_score'].value_counts(normalize=True)

           # Calculate average class ratios]
          train_ratios = train_ratios.reindex(original_ratios.index, fill_value=0)
          val_ratios = val_ratios.reindex(original_ratios.index, fill_value=0)
          test_ratios = test_ratios.reindex(original_ratios.index, fill_value=0)
          for a in [0, 1, 2]:
              train_ratios_history[a].append(train_ratios[a])
              val_ratios_history[a].append(val_ratios[a])
              test_ratios_history[a].append(test_ratios[a])
          # Plot class ratios
          ax.clear()
          ax.plot(train_ratios_history[0], label='Train 0')
          ax.plot(train_ratios_history[1], label='Train 1')
          ax.plot(train_ratios_history[2], label='Train 2')
          ax.plot(val_ratios_history[0], label='Val 0')
          ax.plot(val_ratios_history[1], label='Val 1')
          ax.plot(val_ratios_history[2], label='Val 2')
          ax.plot(test_ratios_history[0], label='Test 0')
          ax.plot(test_ratios_history[1], label='Test 1')
          ax.plot(test_ratios_history[2], label='Test 2')
          ax.axhline(y=original_ratios[0], label='Optimal', color='red', linestyle='--')
          ax.legend()
          ax.set_title('Class Ratios Over Time')
          ax.set_xlabel('Iteration')
          ax.set_ylabel('Class Ratio')
          plt.pause(0.01)
          # Identify overrepresented and underrepresented classes in each split
          combined_ratios = (train_ratios + val_ratios + test_ratios) / 3
          overrepresented_train = train_ratios[(train_ratios > original_ratios) & (train_ratios > combined_ratios)].index
          overrepresented_val = val_ratios[(val_ratios > original_ratios) & (val_ratios > combined_ratios)].index
          overrepresented_test = test_ratios[(test_ratios > original_ratios) & (test_ratios > combined_ratios)].index
          for class_ in overrepresented_train:
              print(f"\n\nClass {class_} is overrepresented in train set.")
              print("\nTrying to move entries to val set...")
              train_set, val_set = self.move_entries(train_set, val_set, test_set, class_)
              print("Trying to move entries to test set...")
              train_set, test_set = self.move_entries(train_set, test_set, val_set, class_)

          for class_ in overrepresented_val:
              print(f"\n\nClass {class_} is overrepresented in val set.")
              print("\nTrying to move entries to train set...")
              val_set, train_set = self.move_entries(val_set, train_set, test_set, class_)
              print("Trying to move entries to test set...")
              val_set, test_set = self.move_entries(val_set, test_set, train_set, class_)

          for class_ in overrepresented_test:
              print(f"\n\nClass {class_} is overrepresented in test set.")
              print("\nTrying to move entries to train set...")
              test_set, train_set = self.move_entries(test_set, train_set, val_set, class_)
              print("Trying to move entries to val set...")
              test_set, val_set = self.move_entries(test_set, val_set, test_set, class_)

      return train_set, val_set, test_set

    def make_dataloader(self, df: pd.DataFrame, shuffle=True, batch_size=16):
        input_examples = []
        for query, doc, label in zip(df["query"], df["doc"], df["qrel_score"]):
            # input_examples.append(InputExample(texts=[query, doc]))
            input_examples.append(InputExample(texts=[query, doc], label=label))
        return DataLoader(input_examples, shuffle=shuffle, batch_size=batch_size)

    def check_unique(self, df_1, df_2):
        queries_1 = set(df_1["query_id"])
        queries_2 = set(df_2["query_id"])
        texts_1 = set(df_1["doc_id"])
        texts_2 = set(df_2["doc_id"])
        query_intersect = queries_1.intersection(queries_2)
        text_intersect = texts_1.intersection(texts_2)
        return len(query_intersect), len(text_intersect)

    def make_dataloaders(
        self, shuffle=True, batch_size=32, train=0.6, val=0.2, test=0.2, balanced=False
    ):
        train_set, val_set, test_set = self.split_data(train, val, test, balanced)

        assert (
            self.check_unique(train_set, val_set)[0] == 0
        ), "Train and val set should not have any queries in common."
        assert (
            self.check_unique(val_set, test_set)[0] == 0
        ), "Val and test set should not have any queries in common."
        assert (
            self.check_unique(train_set, test_set)[0] == 0
        ), "Train and test set should not have any queries in common."
        
        assert (
            self.check_unique(train_set, val_set)[1] == 0
        ), "Train and val set should not have any texts in common."
        
        assert (
            self.check_unique(val_set, test_set)[1] == 0
        ), "Val and test set should not have any texts in common."
        
        assert (
            self.check_unique(train_set, test_set)[1] == 0
        ), "Train and test set should not have any texts in common."
        
        print ("\n\nNo queries or texts in common between splits!")
        
        print("\n\nTrain set qrel class counts:")
        print(self.get_class_counts(train_set))
        print("Val set qrel class counts:")
        print(self.get_class_counts(val_set))
        print("Test set qrel class counts:")
        print(self.get_class_counts(test_set))

        print(
            "\n\nTotal amount of rows lost in the split: ",
            len(self.dataset) - len(train_set) - len(val_set) - len(test_set),
        )

        train_dataloader = self.make_dataloader(train_set, shuffle, batch_size)
        val_dataloader = self.make_dataloader(val_set, shuffle, batch_size)
        test_dataloader = self.make_dataloader(test_set, shuffle, batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def get_class_counts(self, df: pd.DataFrame = None):
        if df is None:
            return self.dataset["qrel_score"].value_counts().to_dict()
        else:
            return df["qrel_score"].value_counts().to_dict()

    def balance_classes_by_removing_samples(self):
        # Function that balances the classes by removing samples from the majority classes, makes sure all classes have the same amount of samples
        class_counts = self.get_class_counts()
        num_samples_per_class = min(class_counts.values())
        rows_before = self.dataset.shape[0]
        self.dataset = (
            self.dataset.groupby("qrel_score")
            .apply(lambda x: x.sample(num_samples_per_class, replace=False))
            .reset_index(drop=True)
        )
        rows_after = self.dataset.shape[0]
        print(
            f"Removed {rows_before - rows_after} samples to balance the classes. New class counts: {self.get_class_counts()}"
        )