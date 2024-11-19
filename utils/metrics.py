import copy
import torch
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

GLOBAL_METRICS = {
    'auc': roc_auc_score,
    'logloss': log_loss,
    'accuracy': accuracy_score,
}

def sigmoid(x):
    s = np.where(
        x > 0, 
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
        )
    return s

def compute_metrics_for_ctr(p: EvalPrediction, neg_sampling_ratio=1):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = sigmoid(preds)
    result = {
                'auc': GLOBAL_METRICS['auc'](p.label_ids, preds),
            }
    
    if neg_sampling_ratio == 1:
        result['logloss'] = GLOBAL_METRICS['logloss'](p.label_ids, preds)
    else:
        ori_len = len(preds)
        preds /= (preds + (1 - preds) / neg_sampling_ratio)

        idx = torch.where((preds > 0) & (preds < 1))[0]
        preds = preds[idx]
        p.label_ids = p.label_ids[idx]

        after_len = len(preds)
        result['logloss'] = GLOBAL_METRICS['logloss'](p.label_ids, preds) * after_len / ori_len
    return result


def compute_metrics_for_clm(p: EvalPrediction):
    metrics = ['accuracy']
    
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = preds[:, :-1].reshape(-1)
    labels = p.label_ids[:, 1:].reshape(-1)
    result = {}
    for metric in metrics:
        result[metric] = GLOBAL_METRICS[metric](labels, preds)
    return result



def compute_hyperparam_search_objective_for_ctr(metrics):
    """
    The  objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the [`Trainer`], the sum of all metrics otherwise.

    Args:
        metrics (`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        `float`: The objective to minimize or maximize
    """
    return  metrics['eval_auc']