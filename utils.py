import numpy as np

def weightBCE(y, p, pos_weight=50):
    '''
        y: truth label
        p: predicted probability
        pos_weight: weight on Y = 1 to penalize false negatives
    '''
    # weighted binary cross entropy
    p = np.clip(p, 1e-3, 1-1e-3)
    return  np.sum(y * -1 * np.log(p) * pos_weight + (1 - y) * -1 * np.log(1 - p)) / len(p)