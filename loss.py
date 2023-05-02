import torch
import torch.nn as nn
from torch import autograd
from utils import sigmoid
import numpy as np

class RankLoss(nn.Module):
    '''
    Empirical PNorm Loss with logistic loss surrogate. 
    Implementation of: The P-Norm Push: A Simple Convex Ranking Algorithm that Concentrates at the Top of the List
    '''
    def __init__(self, p=4):
        super(RankLoss, self).__init__()
        self.p = p

    @staticmethod
    def grad_hess(target, out): # for LGBM
        p = 4
        y_pred = sigmoid(out) # pred
        
        grad = y_pred
        hess = y_pred*(1-y_pred)
        neg_index = np.argwhere(target == 0)
        pos_index = np.argwhere(target == 1)
        neg_hess = hess[neg_index].flatten() # init

        pos_pred = y_pred[target == 1]
        neg_pred = y_pred[target == 0] 
        proba_diff = (pos_pred[:,None] - neg_pred[None,:])
        surrogate_loss = np.log(1 + np.exp(-proba_diff))
        sigmoid_diff = sigmoid(-proba_diff)
        L = np.mean(surrogate_loss, axis=0)**p 
        D = np.mean(sigmoid_diff,axis=0) * neg_hess
        grad[neg_index] = grad[neg_index] + ((p*(L)**(p-1)) * D).reshape(-1,1)
        grad[pos_index] = grad[pos_index] - 1
        
        H = np.mean(sigmoid_diff*(sigmoid_diff*(neg_hess)**2 + 1),axis=0)
        hess[neg_index] = (neg_hess + p*L**(p-2)*((p-1)*D**2+L*H)).reshape(-1,1)
        
        return grad, hess

    def forward(self, out, target):
        '''
        Args:
            out : predicted probabilities
            target : ground truth labels
        '''
        pos = out[target == 1]
        neg = out[target == 0] 
        if (len(pos) == 0) or (len(neg) == 0):
            return torch.tensor(0., device=target.device) # avoid nan
        else:
            # exp loss can be quite large, switch to logistic loss
            proba_diff = (pos[:,None] - neg[None,:]) 
            surrogate_loss = torch.log(1 + torch.exp(-proba_diff))
            pos_sum = torch.mean(surrogate_loss, axis=0)**self.p
            neg_sum = torch.mean(pos_sum, axis=0)
            return neg_sum**(1/self.p)

class IRMLoss(nn.Module):
    def __init__(self):
        super(IRMLoss, self).__init__()
  
    def mean_nll(self, logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def penalty(self, logits, y):
        # penalty use full dataset
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = self.mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def make_environment(self, hist_mine, data, target):
        hard = torch.arange(len(hist_mine), device=data.device)[((hist_mine > 0) & (target == 0)) | ((hist_mine == 0) & (target == 1))]
        easy_all = torch.arange(len(hist_mine), device=data.device)[~(((hist_mine > 0) & (target == 0)) | ((hist_mine == 0) & (target == 1)))]
        ratio = len(easy_all) // len(hard)
        res = [(data[hard], target[hard])]
        if len(hard) == 0: # use all easy tasks
            res = [(data[easy_all], target[easy_all])]
        if len(easy_all) == 0: # use all hard tasks
            res = [(data[hard], target[hard])]
        for r in range(ratio-1): # drop last easy batch
            easy_slice = easy_all[r*ratio:(r+1)*ratio]
            res.append((data[easy_slice], target[easy_slice]))
        return res
    
    def forward(self, model, hist_mine, data, target):
        envs = self.make_environment(hist_mine, data, target)
        loss_ce_envs = []
        loss_irm_envs = []
        for env in envs:
            micro_data, micro_target = env
            out_micro = model(micro_data).squeeze(-1)
            loss_ce_micro = self.mean_nll(out_micro, micro_target)
            loss_irm_micro = self.penalty(out_micro, micro_target)
            loss_ce_envs.append(loss_ce_micro)
            loss_irm_envs.append(loss_irm_micro)
        loss_ce = torch.stack(loss_ce_envs).mean()
        loss_irm = torch.stack(loss_irm_envs).mean()
        return loss_ce + loss_irm