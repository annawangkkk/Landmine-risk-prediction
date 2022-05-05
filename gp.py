import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
from torch.optim import Adam
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DiriGPC(BaseEstimator, ClassifierMixin):  

    def __init__(self, epochs=200, lr=5e-1, learn_additional_noise=True, verbose=1):
        self.epochs = epochs
        self.lr = lr
        self.learn_additional_noise = learn_additional_noise
        self.verbose = verbose
        self.likelihood = None
        self.model = None
        self.optimizer = None
        self.fitted_ = False
        self.classes_ = None


    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
            y = np.array(y)
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        
        self.likelihood = DirichletClassificationLikelihood(y, learn_additional_noise=self.learn_additional_noise)
        self.model = DirichletGPModel(X, self.likelihood.transformed_targets, 
                                 self.likelihood, num_classes=self.likelihood.num_classes)
        self.model.train()
        self.likelihood.train()
        self.optimizer = Adam(self.model.parameters(), lr=5e-1) 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, self.likelihood.transformed_targets).sum()
            loss.backward()
            if self.verbose and (i % 5 == 0):
                if self.learn_additional_noise:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                        i + 1, self.epochs, loss.item(),
                        self.model.covar_module.base_kernel.lengthscale.mean().item(),
                        self.model.likelihood.second_noise_covar.noise.mean().item()
                    ))
                else:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                        i + 1, self.epochs, loss.item(),
                        self.model.covar_module.base_kernel.lengthscale.mean().item()
                    ))
            self.optimizer.step()
        self.fitted_ = True
        self.classes_ = range(2) 
        return self

    def predict_proba(self, X):
        if not self.fitted_:
            raise RuntimeError("You must train classifer before predicting data!")
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
        X = torch.from_numpy(X).float()
        self.model.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            test_dist = self.model(X)
        pred_samples = test_dist.sample(torch.Size((X.shape[0]//2,))).exp()
        probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
        return probabilities.detach().cpu().numpy().T
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
