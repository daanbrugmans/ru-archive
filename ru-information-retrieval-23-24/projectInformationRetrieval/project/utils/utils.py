
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_precision_recall_f1(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return precision, recall, f1

def confusion_matrix_visual(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    return cm

def accuracy(y_true, y_pred):
    correct = (y_true == y_pred).float() 
    accuracy = correct.sum() / len(correct)
    return accuracy
