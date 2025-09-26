import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def binary_classification(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def ndcg_at_k(scores, labels, k: int = 10):
    order = np.argsort(scores)[::-1][:k]
    gains = (2 ** np.array(labels)[order]) - 1
    discounts = np.log2(np.arange(len(order)) + 2)
    dcg = np.sum(gains / discounts)
    sorted_labels = sorted(labels, reverse=True)[:k]
    ideal_gains = (2 ** np.array(sorted_labels)) - 1
    idcg = np.sum(ideal_gains / discounts)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)
