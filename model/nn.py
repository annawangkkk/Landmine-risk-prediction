import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, AdamW

def sigmoid(x):
    sig = 1 / (1 + torch.exp(-x))     
    sig = torch.minimum(sig, torch.Tensor([0.999999] * x.shape[0]).reshape(-1,1))  # restrict upper bound
    sig = torch.maximum(sig, torch.Tensor([0.000001] * x.shape[0]).reshape(-1,1)) # restrict lower bound
    return sig

class LandmineFeature(Dataset):
    def __init__(self,X,y=None): # allow no targets for feature extraction/unsupervised learning
        self.X = X
        self.y = y
        if isinstance(X, pd.DataFrame):
            self.X = np.array(X)
            if not (y is None):
                self.y = np.array(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        if self.y is None:
            return torch.from_numpy(self.X[idx]).float()
        else:
            return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(np.array(self.y[idx])).float()

class NN_torch(nn.Module): # standard MLP
    def __init__(self,input_dim):
        super(NN_torch,self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = nn.Dropout(p=0.5)(F.leaky_relu(self.fc1(x)))
        return self.fc2(x)


class NN(BaseEstimator, ClassifierMixin):  

    def __init__(self, batchsize=100, epochs=30, lr=1e-2, omega=50, lambda1=0, lambda2=0, step=-1):
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.omega = omega
        self.model = None
        self.fitted_ = False
        self.classes_ = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_BCE_batch_weight(self, target):
        # 1 => positive
        # 0 => negative
        weight = []
        for i in target:
            if i == 1:
                weight.append(self.omega)
            else:
                weight.append(1)
        return weight

    def fit(self, X, y, X_test=None, y_test=None): # train
        train_set = LandmineFeature(X, y)
        if (not (X_test is None)) and (not (y_test is None)):
            val_set = LandmineFeature(X_test, y_test)
            val_loader = DataLoader(val_set, batch_size=self.batchsize, shuffle=False)
        
        self.model = nn.DataParallel(NN_torch(X.shape[1]))
        self.model.to(self.device)
        
        train_loader = DataLoader(train_set, batch_size=self.batchsize, shuffle=True) # TODO: sampler
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4) 

        for epoch in range(self.epochs):
            loss_list = []
            train_auc_list = []
            test_auc_list = []
            self.model.train()
            for idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                out = self.model(data)
                y_prob = torch.sigmoid(out).squeeze(-1) 
                loss = BCEWithLogitsLoss()
                output_loss = loss(out.squeeze(-1),target)   
                output_loss.backward()
                optimizer.step()
                target = target.detach().cpu().numpy() 
                y_prob = y_prob.detach().cpu().numpy()
                if idx % 100 == 1:
                    print(f"Epoch: {epoch}, Train Loss: {output_loss.item()}, Train Auc: {roc_auc_score(target, y_prob)}")
            loss_list.append(output_loss.item())   
            train_auc_list.append(roc_auc_score(target, y_prob))
            if (not (X_test is None)) and (not (y_test is None)):
                self.model.eval()
                val_loss = []
                y_val_prob = []
                y_val_truth = []
                with torch.no_grad():
                    for idx, (data, target) in enumerate(val_loader):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        out = self.model(data)
                        y_prob = torch.sigmoid(out).squeeze(-1) 
                        loss = BCEWithLogitsLoss()
                        output_loss = loss(out.squeeze(-1),target)  
                        target = target.cpu().numpy() 
                        y_val_prob.append(y_prob.detach().cpu().numpy())
                        y_val_truth.append(target)
                        val_loss.append(output_loss.item())
                print(f"Epoch: {epoch}, Val Loss: {sum(val_loss)/len(val_loss)}, Val Auc: {roc_auc_score(y_val_truth, y_val_prob)}")
                test_auc_list.append(roc_auc_score(y_val_truth, y_val_prob))
                
        self.fitted_ = True
        self.classes_ = range(2) 
        return self

    def predict_proba(self, X):
        if not self.fitted_:
            print("Warning: Did you train the classifier before predicting data?")
        self.model.eval()
        val_set = LandmineFeature(X)
        val_loader = DataLoader(val_set, batch_size=self.batchsize, shuffle=False)
        probabilities = [[],[]]
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                y_prob = torch.sigmoid(self.model(data)).squeeze(-1) 
                probabilities[1].extend(list(y_prob.detach().cpu().numpy()))
                probabilities[0].extend(list((1-y_prob).detach().cpu().numpy()))
        return np.array(probabilities).T

    def predict(self, X, threshold=0.5):
        if not self.fitted_:
            print("Warning: Did you train the classifier before predicting data?")
        self.model.eval()
        val_set = LandmineFeature(X)
        val_loader = DataLoader(val_set, batch_size=self.batchsize, shuffle=False)
        y_preds = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                y_prob = torch.sigmoid(self.model(data)).squeeze(-1) 
                y_pred = [0 if y < threshold else 1 for y in y_prob]
                y_preds.extend(y_pred)
        return np.array(y_preds)

    def score(self, X, y):
        if not self.fitted_:
            print("Warning: Did you train the classifier before predicting data?")
        self.model.eval()
        val_set = LandmineFeature(X, y)
        val_loader = DataLoader(val_set, batch_size=self.batchsize, shuffle=False)
        val_loss = []
        y_val_prob = []
        y_val_truth = []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                out = self.model(data)
                y_prob = torch.sigmoid(out).squeeze(-1) 
                loss = BCEWithLogitsLoss()
                output_loss = loss(out.squeeze(-1),target)  
                target = target.cpu().numpy() 
                y_val_prob.extend(list(y_prob.detach().cpu().numpy()))
                y_val_truth.extend(list(target.detach().cpu().numpy()))
                val_loss.append(output_loss.item())
        return {'AUC':roc_auc_score(y_val_truth, y_val_prob), 'wBCE': sum(val_loss)/len(val_loss)}
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
class CurriculumMLP(NN):
    # TODO
    def __init__():
        pass

    def fit():
        pass
