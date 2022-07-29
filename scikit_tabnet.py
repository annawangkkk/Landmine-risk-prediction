from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import numpy as np


class ScikitTabNet(BaseEstimator, ClassifierMixin):
    def __init__(self, seed, n_d, momentum):
        self.seed = seed
        self.n_d = n_d
        self.momentum = momentum
        self.estimator = TabNetClassifier(seed=seed, n_d=n_d, momentum=momentum)
        self.fitted_ = False
        self.stacking_split = None
        self.classes_ = None
    
    def fit(self, X, y, X_val=None, y_val=None, early_stopping=True):
        self.fitted_ = True
        if early_stopping == False:
            return self.estimator.fit(X,y,patience=200,max_epochs=500)
        else:
            return self.estimator.fit(X,y, 
                                      eval_set=[(X_val, y_val)],
                                      patience=200,
                                      max_epochs=500)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    
class ScikitTabNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seed, n_d, momentum):
        self.seed = seed
        self.n_d = n_d
        self.momentum = momentum 
        self.estimator = TabNetRegressor(seed=seed, n_d=n_d, momentum=momentum)
        self.fitted_ = False
        self.stacking_split = None
        self.classes_ = None
    
    def fit(self, X, y, X_val=None, y_val=None, early_stopping=True):
        y = np.array(y).reshape(-1,1)
        y_val = np.array(y_val).reshape(-1,1)
        self.fitted_ = True
        if early_stopping == False:
            return self.estimator.fit(X,y,patience=200,max_epochs=500)
        else:
            return self.estimator.fit(X,y, 
                                      eval_set=[(X_val, y_val)],
                                      patience=200,
                                      max_epochs=500)
        
    
    def predict(self, X):
        return self.estimator.predict(X)
