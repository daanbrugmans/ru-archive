# This code is provided by Theo Kent.
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

def contains_sentence(auto_df, sentence):
    """

    :param auto_df:
    :param sentence:
    :return:
    """
    return any([sentence == a for a in auto_df.Sentence])


def contains_date(auto_df, date):
    return any([pd.to_datetime(date) == a for a in auto_df.index])
 
    
def get_sentence(auto_df, sentence):
    groups = auto_df.groupby('Sentence')
    return groups.get_group(sentence)
    

def calculate_confusion_matrix(manual_results, automatic_results):
    # True Positive, True Negative, False Positive, False Negative
    TP, TN, FP, FN = 0, 0, 0, 0

    # Calculate our full Confusion Matrix:
    for row in manual_results:
        # Contains 0 dates
        if len(row["Dates"]) == 0:
            # This row should not exist in the automatic results.
            if contains_sentence(automatic_results, row["Sentence"]):
                FP += 1
            else:
                TN += 1
        # Row contains dates
        else:
            # Should exist in auto results
            if contains_sentence(automatic_results, row["Sentence"]):
                sentence_group = get_sentence(automatic_results, row["Sentence"])
                if len(sentence_group) > len(row["Dates"]):
                    FP += len(sentence_group) - len(row["Dates"])
                for dt in row["Dates"]:
                    if contains_date(sentence_group, dt):
                        TP += 1
                    else:
                        FN += 1
                        
                    
            else:
                FN += 1

    # Return 2D matrix of results
    confusion_matrix = [[TP, FN],
                        [FP, TN]]

    return np.array(confusion_matrix)


def plot_confusion_matrix(manual_labels,
                          sorted_date_df,
                          title_names,
                          cmap=None,
                          normalize=True):
    cm = calculate_confusion_matrix(manual_labels, sorted_date_df)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    if title_names is not None:
        tick_marks = np.arange(len(title_names))
        plt.xticks(tick_marks, title_names, rotation=45)
        plt.yticks(tick_marks, title_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.show()
