import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from loss import RankLoss, IRMLoss
from model import TabCmpt, MLP

from datetime import datetime

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils import mean_reverse_height, mean_height



class RELand():
    def __init__(self, num_features, args):
        self.device = args.device
        self.objective = args.objective
        self.model_name = args.model
        
        if self.model_name == 'TabCmpt': # ours
            self.model = TabCmpt(num_features, args.n_step).to(self.device)
        elif self.model_name == 'MLP':
            self.model = MLP(num_features, 20, classifier=True).to(self.device)
        else:
            raise NotImplementedError
        
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        
        self.config = vars(args)
        self.epochs = args.epochs
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.lambda_l2 = args.lambda_l2
        self.timestamp = args.timestamp
        
        # store best metrics
        self.auc = 0
        self.pr = 0
        self.height = 1000000
        self.rheight = 1000000
        self.fin_proba = []
        self.importance = []
    
    def fit(self, train_data, val_data):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        self.model.train()
        for epoch in range(self.epochs):
            train_roc = []
            num_examples_train = 0
            t_start = datetime.now()
            
            for idx, (data, target, lon_lat, hist_mine) in enumerate(train_loader):
                
                data = data.to(self.device)
                target = target.to(self.device)
                lon_lat = lon_lat.to(self.device)
                hist_mine = hist_mine.to(self.device)
                self.optimizer.zero_grad()

                weight_norm = torch.tensor(0.).cuda()
                for k, param in self.model.state_dict().items():
                    if (param.requires_grad) \
                        and (k != 'attentive_transformer.0.weights') \
                        and (k != 'attentive_transformer.0.bias') :
                        weight_norm += param.norm().pow(2)
                logits_envs = self.model(data)
                logits_envs = logits_envs.squeeze(-1)
                
                if self.objective == 'erm': 
                    output = self.lambda_l2 * weight_norm + nn.BCEWithLogitsLoss()(logits_envs, target)
                elif self.objective == 'pnorm':
                    output = self.lambda_l2 * weight_norm \
                            + nn.BCEWithLogitsLoss()(logits_envs, target) \
                            + RankLoss()(torch.sigmoid(logits_envs), target)
                elif self.objective == 'irm':
                    output = self.lambda_l2 * weight_norm + IRMLoss()(self.model, hist_mine, data, target)

                output.backward()
                self.optimizer.step()
                num_examples_train += len(data)
                try:
                    rocauc = roc_auc_score(target.detach().cpu().numpy(),torch.sigmoid(logits_envs).detach().cpu().numpy())
                    train_roc.append(rocauc)
                except:
                    continue
                if idx % 100 == 1:
                    print(f"Train Epoch: {epoch}, Trained Examples: {num_examples_train}, Loss: {output.item()}")
            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            
            if len(train_roc) > 0:
                print(f"Train one epoch takes: {t_delta} sec, roc: {sum(train_roc)/len(train_roc)}")

            eval_rocauc, pr, height, rheight, fin_prob, importance = self.predict_proba(val_loader=val_loader)
            
            if (eval_rocauc > self.auc):
                self.auc = eval_rocauc
                self.pr = pr
                self.height = height
                self.rheight = rheight
                self.importance = importance
                self.fin_proba = fin_prob
                print("best model updated.")
                torch.save(self.model.state_dict(), f"./experiments/{self.timestamp}/{train_loader.dataset.val_municipio}.pth")
            print("val_auc:",eval_rocauc,"curr_pr:",pr)
        
        ckpt = {'roc':self.auc,
                'pr':self.pr,
                'height':self.height,
                'rheight':self.rheight,
                'importance':self.importance,
                'prob':self.fin_proba}
        
        return ckpt
    
    def predict_proba(self, val_loader=None, val_dataset=None, test_dataset=None):
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if test_dataset is not None:
            self.model.eval()
            fin_prob = []
            with torch.no_grad():
                for data, _, _, _ in test_loader:
                    data = data.to(self.device)
                    out = self.model(data)
                    out = out.squeeze(-1)
                    fin_prob.extend(list(torch.sigmoid(out).detach().cpu().numpy()))
                return None, None, None, None, fin_prob, None
        else:
            self.model.eval()
            fin_prob = []
            fin_targets = []
            with torch.no_grad():
                for data, target, _, _ in val_loader:
                    data = data.to(self.device)
                    out = self.model(data)
                    out = out.squeeze(-1)
                    fin_prob.extend(list(torch.sigmoid(out).detach().cpu().numpy()))
                    fin_targets.extend(list(target.detach().cpu().numpy()))
                    if isinstance(self.model, TabCmpt):
                        importance = self.model.get_importance(data).detach().cpu().numpy()
                    else:
                        importance = []
                try:
                    rocauc = roc_auc_score(fin_targets, fin_prob)
                    precision, recall, _ = precision_recall_curve(fin_targets, fin_prob)
                    pr = auc(recall, precision)
                    height = mean_height(fin_targets, fin_prob)
                    rheight = mean_reverse_height(fin_targets, fin_prob)
                except:
                    rocauc = 0
                    pr = 0
                    height = 0
                    rheight = 0
                print(f"Valid roc: {rocauc:4f}, pr {pr:4f}")
                return rocauc, pr, height, rheight, fin_prob, importance