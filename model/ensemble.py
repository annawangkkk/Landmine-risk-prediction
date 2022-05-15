import numpy as np
from utils import weightBCE
from sklearn.metrics import roc_auc_score

class LossVotingClassifier(object):
    # use soft labels
    def __init__(self, estimators):
        self.estimators = estimators
        self.weights = None
        self.fitted_ = False

    def fit(self, X, y):
        predicted_prob = []
        raw_weights = []
        for (name, clf) in self.estimators:
            if name == 'TabNet':
                clf.fit(X, y,
                max_epochs=150 , patience=50,
                batch_size=100, virtual_batch_size=50,
                weights=1,
                drop_last=False
                )
            else:
                clf.fit(X, y)
            p = clf.predict_proba(X)[:,1]
            predicted_prob.append(p)
            # calculate wBCE
            raw_weights.append(weightBCE(y, p, pos_weight=50))
        raw_weights = np.array(raw_weights)
        raw_weights = 1/raw_weights # weight higher on the model with smaller BCE
        raw_weights /= max(raw_weights)
        self.weights = raw_weights
        self.fitted_ = True

    def predict_proba(self, X):
        res = np.zeros((X.shape[0],2))
        predicted_prob = []
        for name, clf in self.estimators:
            p = clf.predict_proba(X)[:,1]
            predicted_prob.append(p)
        for w in self.weights:
            res[:,1] += w * p
        res[:,1] /= sum(self.weights)
        res[:,0] = 1 - res[:, 1]
        return res
    
    def score(self, X, y):
        y_prob = np.zeros((X.shape[0],))
        predicted_prob = []
        for name, clf in self.estimators:
            p = clf.predict_proba(X)[:,1]
            predicted_prob.append(p)
        for w in self.weights:
            y_prob += w * p
        return roc_auc_score(y, y_prob)
