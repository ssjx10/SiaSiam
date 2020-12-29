import os
import numpy as np

'author seung-wan.J'

def evaluation_metrics (label, pred):
    metrics = get_metrics (label, pred)
    return np.round(np.mean(list(metrics.values())), 4)

def _confusion_matrix(label, pred):  
    ### TN (0,0) / FN (0,1)/ FP (1,0) / TP (1,1)
    TN, FN, FP, TP = 0, 0, 0, 0

    for y_hat,y in zip(pred,label):
        if y == 0:
            if y_hat ==0:
                    TN = TN + 1
            else:
                    FN = FN + 1
        elif y == 1:
            if y_hat == 0:
                FP = FP +1
            else:
                TP = TP +1
    return TN, FN, FP, TP

def get_metrics (label, pred):
    metrics = dict()
    SMOOTH = 1e-4
    
    pred = pred > 0.5
    num_P, num_N = np.sum(label==1), np.sum(label==0)
    TN, FN, FP, TP = _confusion_matrix(label, pred)


    metrics['acc'] = (TP + TN) / (TP + FN + FP + TN + SMOOTH)
    metrics['recall'] = TP / (TP+FN + SMOOTH) ## sensitivive
    metrics['spec'] = TN / (TN + FP + SMOOTH) ## 

    return metrics