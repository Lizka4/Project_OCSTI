import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='weighted'):
    return precision_score(y_true, y_pred, average=average, zero_division=0)

def calculate_recall(y_true, y_pred, average='weighted'):
    return recall_score(y_true, y_pred, average=average, zero_division=0)

def calculate_f1_score(y_true, y_pred, average='weighted'):
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def calculate_all_metrics(y_true, y_pred):
    return {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1_score(y_true, y_pred),
        'confusion_matrix': calculate_confusion_matrix(y_true, y_pred).tolist()
    }

def tensor_to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    return tensor