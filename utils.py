import numpy as np

def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (exp + 1)

def sigmoid(x):
    '''
        Numerically stable sigmoid.
    '''
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result

def mean_reverse_height(y_truth, y_pred):
    # y_pred = predicted proba
    y_pred = np.array(y_pred)
    y_truth = np.array(y_truth).astype(np.uint8)
    pos = y_pred[y_truth == 1]
    neg = y_pred[y_truth == 0] 
    proba_diff = (pos[:,None] - neg[None,:]) # pos*neg
    proba_indicator = np.sum(proba_diff <= 0, axis=1) # neg
    return np.mean(proba_indicator) # the smaller the better

def mean_height(y_truth, y_pred):
    y_pred = np.array(y_pred)
    y_truth = np.array(y_truth).astype(np.uint8)
    pos = y_pred[y_truth == 1]
    neg = y_pred[y_truth == 0] 
    proba_diff = (pos[:,None] - neg[None,:])
    proba_indicator = np.sum(proba_diff <= 0, axis=0) # pos
    return np.mean(proba_indicator)