
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

def split_in_sets(df, target_train_frac, target_val_frac):
    '''
    Splits a DataFrame into train, test, and validation sets, ensuring no overlap
    of doc_id or query_id across these sets.

    Parameters:
    df (DataFrame): The dataset to split.
    target_train_frac (float): Target fraction of the train set.
    target_val_frac (float): Target fraction of the validation set.
    target_test_frac (float): Target fraction of the test set.

    Returns:
    tuple: Three DataFrames representing train, test, and validation sets.
    '''
    print('Initial DataFrame size:', df.shape)
    running_splits = pd.DataFrame(columns=['train', 'val', 'test', 'rows_lost'])
    
    for i in range(50):
      train_df, val_df, test_df = shuffle_and_split(df, target_train_frac, target_val_frac)
      rows_lost = df.shape[0] - train_df.shape[0] - val_df.shape[0] - test_df.shape[0]
      row = pd.DataFrame([[train_df, val_df, test_df, rows_lost]], columns=['train', 'val', 'test', 'rows_lost'])
      running_splits = running_splits._append(row, ignore_index=True)
      print('Amount of rows lost:', df.shape[0] - train_df.shape[0] - val_df.shape[0] - test_df.shape[0])
    running_splits = running_splits.sort_values(by='rows_lost')
    train_df = running_splits.iloc[0]['train']
    test_df = running_splits.iloc[0]['test']
    val_df = running_splits.iloc[0]['val']

    return train_df, val_df, test_df


def split_list(lst, train_frac, val_frac):
    """
    Splits a list into three parts based on specified fractions.

    Parameters:
    lst (list): List to be split.
    train_frac (float): Fraction for the train set.
    val_frac (float): Fraction for the validation set.
    test_frac (float): Fraction for the test set.

    Returns:
    tuple: Three lists representing train, test, and validation parts.
    """
    print("Starting split_list function.")

    total_size = len(lst)
    print(f"Total size of the list: {total_size}")

    train_size = int(total_size * train_frac)
    val_size = int(total_size * val_frac)
    test_size = total_size - train_size - val_size  # Calculated to ensure total coverage of the list

    print(f"Calculated sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")

    train_lst = lst[:train_size]
    val_lst = lst[train_size : train_size + val_size]
    test_lst = lst[train_size + val_size :]

    print(f"Final split sizes - Train: {len(train_lst)}, Validation: {len(val_lst)}, Test: {len(test_lst)}")
    return train_lst, val_lst, test_lst
  
def shuffle_and_split(df, target_train_frac, target_val_frac):
    """
    Shuffles and splits a DataFrame into train, test, and validation sets based on the target fractions.

    Parameters:
    df (DataFrame): The DataFrame to be split.
    target_train_frac, target_val_frac, target_test_frac (float): Target fractions for the train, validation, and test sets.

    Returns:
    tuple: Three DataFrames representing train, test, and validation sets.
    """
    print("Starting shuffle_and_split function.")

    # Create and shuffle lists of unique documents and queries
    documents, queries = list(set(df["doc_id"])), list(set(df["query_id"]))
    random.shuffle(documents)
    random.shuffle(queries)
    print("Shuffled documents and queries.")

    # Split documents and queries into train, test, validation sets
    train_docs, val_docs, test_docs = split_list(
        documents, target_train_frac, target_val_frac
    )
    print(f"Split documents into - Train: {len(train_docs)}, Validation: {len(val_docs)}, Test: {len(test_docs)}")

    train_queries, val_queries, test_queries = split_list(
        queries, target_train_frac, target_val_frac
    )
    print(f"Split queries into - Train: {len(train_queries)}, Validation: {len(val_queries)}, Test: {len(test_queries)}")

    # Assign pairs to train, test, and validation sets
    train_pairs = df[df["doc_id"].isin(train_docs) & df["query_id"].isin(train_queries)]
    val_pairs = df[df["doc_id"].isin(val_docs) & df["query_id"].isin(val_queries)]
    test_pairs = df[df["doc_id"].isin(test_docs) & df["query_id"].isin(test_queries)]

    print(f"Final dataset sizes - Train: {train_pairs.shape}, Validation: {val_pairs.shape}, Test: {test_pairs.shape}")
    return train_pairs, val_pairs, test_pairs
  

def remove_faulty(train_pairs, test_pairs, val_pairs):
    """
    Removes faulty pairs from train, test, and validation sets to ensure no overlap.

    Parameters:
    train_pairs, test_pairs, val_pairs (DataFrame): The initial train, test, and validation sets.

    Returns:
    tuple: Cleaned train, test, and validation DataFrames.
    """
    print("Starting removal of faulty pairs.")

    # Identify faulty pairs in the train set
    faulty_train_pairs = train_pairs[
        train_pairs["doc_id"].isin(test_pairs["doc_id"])
        | train_pairs["doc_id"].isin(val_pairs["doc_id"])
        | train_pairs["query_id"].isin(test_pairs["query_id"])
        | train_pairs["query_id"].isin(val_pairs["query_id"])
    ]
    print(f"Faulty pairs identified in train set: {faulty_train_pairs.shape[0]}")

    # Identify faulty pairs in the test set
    faulty_test_pairs = test_pairs[
        test_pairs["doc_id"].isin(train_pairs["doc_id"])
        | test_pairs["doc_id"].isin(val_pairs["doc_id"])
        | test_pairs["query_id"].isin(train_pairs["query_id"])
        | test_pairs["query_id"].isin(val_pairs["query_id"])
    ]
    print(f"Faulty pairs identified in test set: {faulty_test_pairs.shape[0]}")

    # Identify faulty pairs in the validation set
    faulty_val_pairs = val_pairs[
        val_pairs["doc_id"].isin(train_pairs["doc_id"])
        | val_pairs["doc_id"].isin(test_pairs["doc_id"])
        | val_pairs["query_id"].isin(train_pairs["query_id"])
        | val_pairs["query_id"].isin(test_pairs["query_id"])
    ]
    print(f"Faulty pairs identified in validation set: {faulty_val_pairs.shape[0]}")

    # Remove faulty pairs from each set
    train_pairs_cleaned = train_pairs.drop(faulty_train_pairs.index)
    print(f"Cleaned train set size: {train_pairs_cleaned.shape}")

    test_pairs_cleaned = test_pairs.drop(faulty_test_pairs.index)
    print(f"Cleaned test set size: {test_pairs_cleaned.shape}")

    val_pairs_cleaned = val_pairs.drop(faulty_val_pairs.index)
    print(f"Cleaned validation set size: {val_pairs_cleaned.shape}")

    # Concatenate all faulty pairs
    all_faulty_pairs = pd.concat([faulty_train_pairs, faulty_val_pairs, faulty_test_pairs])
    print(f"Total faulty pairs removed: {all_faulty_pairs.shape[0]}")

    print("Finished removing faulty pairs.")
    return train_pairs_cleaned, test_pairs_cleaned, val_pairs_cleaned




def balance_sets(
    train_pairs,
    val_pairs,
    test_pairs,
    df,
    target_train_frac,
    target_val_frac,
    target_test_frac,
):
    """
    Balances the train, test, and validation sets based on target fractions.

    Parameters:
    train_pairs, val_pairs, test_pairs (DataFrame): The initial train, test, and validation sets.
    df (DataFrame): The original dataset.
    target_train_frac, target_val_frac, target_test_frac (float): Target fractions for the train, validation, and test sets.

    Returns:
    tuple: Balanced train, test, and validation DataFrames.
    """
    print("Starting balance_sets function.")

    total_pairs = len(df)
    current_train_frac = len(train_pairs) / total_pairs
    current_val_frac = len(val_pairs) / total_pairs
    current_test_frac = len(test_pairs) / total_pairs

    print(f"Initial fractions - Train: {current_train_frac}, Validation: {current_val_frac}, Test: {current_test_frac}")

    # Function to move pairs from one set to another to balance
    def move_pairs(source_set, target_set, num_pairs_to_move):
        num_pairs_to_move = min(num_pairs_to_move, len(source_set))  # Ensure we do not exceed source set size
        pairs_moved = source_set.sample(n=num_pairs_to_move)
        source_set = source_set.drop(pairs_moved.index)
        target_set = pd.concat([target_set, pairs_moved])
        return source_set, target_set

    # Adjust train set size
    if current_train_frac < target_train_frac:
        num_pairs_to_move = int((target_train_frac - current_train_frac) * total_pairs)
        print(f"Moving {num_pairs_to_move} pairs to balance the train set.")
        if len(val_pairs) >= num_pairs_to_move:
            val_pairs, train_pairs = move_pairs(val_pairs, train_pairs, num_pairs_to_move)
        else:
            test_pairs, train_pairs = move_pairs(test_pairs, train_pairs, num_pairs_to_move)

    # Adjust validation set size
    if current_val_frac < target_val_frac:
        num_pairs_to_move = int((target_val_frac - current_val_frac) * total_pairs)
        print(f"Moving {num_pairs_to_move} pairs to balance the validation set.")
        if len(test_pairs) >= num_pairs_to_move:
            test_pairs, val_pairs = move_pairs(test_pairs, val_pairs, num_pairs_to_move)
        else:
            train_pairs, val_pairs = move_pairs(train_pairs, val_pairs, num_pairs_to_move)

    # Adjust test set size
    if current_test_frac < target_test_frac:
        num_pairs_to_move = int((target_test_frac - current_test_frac) * total_pairs)
        print(f"Moving {num_pairs_to_move} pairs to balance the test set.")
        if len(val_pairs) >= num_pairs_to_move:
            val_pairs, test_pairs = move_pairs(val_pairs, test_pairs, num_pairs_to_move)
        else:
            train_pairs, test_pairs = move_pairs(train_pairs, test_pairs, num_pairs_to_move)

    print("Balancing complete.")
    print(f"Final dataset sizes - Train: {len(train_pairs)}, Validation: {len(val_pairs)}, Test: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs
