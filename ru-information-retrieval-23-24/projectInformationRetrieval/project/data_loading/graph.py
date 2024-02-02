import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from random import uniform

# Assuming your dataset is in a CSV file with columns 'document_id' and 'query_id'
df = pd.read_csv('./project/data/dummy_data.csv')
G = nx.Graph()

# Adding nodes and edges
for _, row in df.iterrows():
    G.add_node(row['doc_id'], type='document')
    G.add_node(row['query_id'], type='query')
    G.add_edge(row['doc_id'], row['query_id'])
    
    
from networkx.algorithms import community

# Using a simple community detection algorithm
communities = list(community.label_propagation_communities(G))
train_communities, test_communities = train_test_split(communities, test_size=0.4, random_state=42)
test_communities, val_communities = train_test_split(test_communities, test_size=0.5, random_state=42)

def extract_data_from_communities(communities, df):
    data = []
    for community in communities:
        for node in community:
            if G.nodes[node]['type'] == 'document':
                related_queries = G[node]
                for query in related_queries:
                    if query in community:
                        data.append({'doc_id': node, 'query_id': query})
    return pd.DataFrame(data)

# Example function to categorize clusters (adjust based on your data)
def categorize_clusters(clusters):
    categories = {'small': [], 'medium': [], 'large': []}
    for cluster in clusters:
        if len(cluster) < 10:  # Example condition for small clusters
            categories['small'].append(cluster)
        elif len(cluster) < 50:  # Example condition for medium clusters
            categories['medium'].append(cluster)
        else:
            categories['large'].append(cluster)
    return categories

# Categorize your clusters
cluster_categories = categorize_clusters(communities)

def calculate_cluster_representation(split, cluster_categories, G):
    representation_score = 0
    total_clusters = sum([len(c) for c in cluster_categories.values()])

    split_docs = set(split['doc_id'])
    split_queries = set(split['query_id'])

    for category, clusters in cluster_categories.items():
        if len(clusters) == 0:  # Check to avoid division by zero
            continue

        expected_proportion = len(clusters) / total_clusters
        represented_clusters = 0
        for cluster in clusters:
            if any((doc in split_docs or query in split_queries) for doc in cluster for query in G[doc]):
                represented_clusters += 1

        actual_proportion = represented_clusters / len(clusters)
        representation_score += abs(expected_proportion - actual_proportion)  # Lower is better

    return 1 - representation_score  # Higher score indicates better representation


# Update evaluate_split_criteria function to use cluster_representation_score
def evaluate_split_criteria(splits, cluster_categories):
    # Initialize metrics
    metrics = {
        'balance_score': 0,
        'total_rows': 0,
        'diversity_score': 0,
        'cluster_representation': 0,
        'rows_lost_percentage': 0,
    }

    # Iterate over each split
    for split_name, split_data in splits.items():
        # Calculate balance between documents and queries
        num_docs = len(split_data['doc_id'].unique())
        num_queries = len(split_data['query_id'].unique())
        balance_score = min(num_docs, num_queries) / max(num_docs, num_queries)  # Ratio of smaller to larger count

        # Count total rows
        total_rows = len(split_data)

        # Assess diversity (this is a placeholder, replace with your own logic)
        # For example, the diversity of query types or document categories
        diversity_score =   len(split_data['doc_id'].unique()) / len(split_data)  # Ratio of unique documents to total rows

        # Update metrics
        metrics['balance_score'] += balance_score
        metrics['total_rows'] += total_rows
        metrics['diversity_score'] += diversity_score

    # Cluster representation (if applicable)
    # This part depends on how you classify or score your clusters
    # For example, check if all cluster types/sizes are represented in each split
    
    cluster_representation_score = sum([calculate_cluster_representation(split, cluster_categories, G) for split in splits.values()])

    total_rows = len(df)
    rows_lost = total_rows - len(splits['train']) - len(splits['test']) - len(splits['val'])
    metrics['rows_lost_percentage'] = (rows_lost / total_rows) * 100 * -1

    metrics['cluster_representation'] = cluster_representation_score

    # Combine metrics into a final score or return them as-is
    # The combination can be a weighted sum or any other method that makes sense for your scenario
    metrics_values = list(metrics.values())
    scaler = MinMaxScaler()
    scaled_metrics = scaler.fit_transform(np.array(metrics_values).reshape(-1, 1))

    # Assigning weights
    weights = [0.1, 0.1, 0.1, 0.1, 0.6]  # Adjust these values as needed

    final_score = sum(scaled_metric * weight for scaled_metric, weight in zip(scaled_metrics, weights))
    return final_score

best_split = None
best_evaluation = None

for i in range(999):  # Number of iterations can be adjusted
    # Adjust test_size and random_state for different splits
    train_communities, test_communities = train_test_split(communities, test_size=0.4, random_state=42+i)
    test_communities, val_communities = train_test_split(test_communities, test_size=0.5, random_state=42+i)

    # Extract data
    train_data = extract_data_from_communities(train_communities, df)
    test_data = extract_data_from_communities(test_communities, df)
    val_data = extract_data_from_communities(val_communities, df)

    # Evaluate split
    evaluation = evaluate_split_criteria({'train': train_data, 'test': test_data, 'val': val_data}, cluster_categories)

    # Compare and store the best split
    # In your main loop
    if best_evaluation is None or evaluation > best_evaluation:
        print(evaluation)
        best_evaluation = evaluation
        best_split = (train_data, test_data, val_data)

from multiprocessing import Pool

def evaluate_split(i):
    test_size_factor = uniform(0.3, 0.5)   # Adjust this range as needed
    
    train_communities, test_communities = train_test_split(communities, test_size=test_size_factor, random_state=42+i)
    test_communities, val_communities = train_test_split(test_communities, test_size=0.5, random_state=42+i)

    train_data = extract_data_from_communities(train_communities, df)
    test_data = extract_data_from_communities(test_communities, df)
    val_data = extract_data_from_communities(val_communities, df)

    evaluation = evaluate_split_criteria({'train': train_data, 'test': test_data, 'val': val_data}, cluster_categories)

    return evaluation, (train_data, test_data, val_data)

if __name__ == '__main__':
    with Pool() as p:
        results = p.map(evaluate_split, range(100))

    best_evaluation, best_split = max(results, key=lambda x: x[0])

    train_data, test_data, val_data = best_split

    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)


    def check_unique(df_1, df_2):
        queries_1 = set(df_1["query_id"])
        queries_2 = set(df_2["query_id"])
        texts_1 = set(df_1["doc_id"])
        texts_2 = set(df_2["doc_id"])
        query_intersect = queries_1.intersection(queries_2)
        text_intersect = texts_1.intersection(texts_2)
        print(len(query_intersect))
        print(len(text_intersect))
        
    check_unique(train_data, val_data)
    check_unique(val_data, test_data)
    check_unique(train_data, test_data)
    print(len(df))
    print(len(train_data))
    print(len(test_data))
    print(len(val_data))
    print(
        f"lost rows 😦 ): {len(df) - len(train_data) - len(test_data) - len(val_data)}"
    )

