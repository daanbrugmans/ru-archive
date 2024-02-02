"""
SVM: https://arxiv.org/abs/1310.4909
logistic regression: https://ieeexplore.ieee.org/document/8424720
"""
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Classifier:
    def __init__(self) -> None:
        self.clf = make_pipeline(StandardScaler(), SVC(gamma="auto", random_state=42))

    def fit(self, train_x: pd.DataFrame, train_y):
        self.clf.fit(train_x, train_y)

    def evaluate(self, eval_x: pd.DataFrame, eval_y):
        predictions = self.clf.predict(eval_x)
        accuracy = metrics.accuracy_score(y_true=eval_y, y_pred=predictions)
        print(f"Accuracy: {accuracy}")
        f1_score = metrics.f1_score(y_true=eval_y, y_pred=predictions,average="weighted")
        print(f"F1 score: {f1_score}")
        classification_report = metrics.classification_report(y_true=eval_y, y_pred=predictions)
        print(f"Classification report: {classification_report}")
        return classification_report

    def get_f1_score(self, eval_x, eval_y):
        predictions = self.clf.predict(eval_x)
        return metrics.f1_score(y_true=eval_y, y_pred=predictions,average="weighted")