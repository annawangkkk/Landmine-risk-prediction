import numpy as np
import pandas as pd

import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from model import PushedLR
import lightgbm as lgb
import pytorch_tabnet.tab_model as erm_tab_model
import pytorch_tabnet_irm.tab_model as irm_tab_model

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from dataset import Event
from reland import RELand
from loss import RankLoss
from utils import mean_reverse_height, mean_height, sigmoid

import os
import json
import shutil
import glob
import argparse
import pickle
from typing import *

import matplotlib.pyplot as plt

def main(timestamp : str, train_val_stream : List):
    # TODO: ood bench
    # TODO: test set
    if os.path.exists(f'./experiments/{timestamp}/predicted_proba.csv'):
        with open(f"./experiments/{timestamp}/metrics.json") as f:
            res = json.load(f)
        predicted_proba = [pd.read_csv(f"./experiments/{timestamp}/predicted_proba.csv")]
        with open(f"./experiments/{timestamp}/feature_importance.json") as f:
            feature_importance = json.load(f)
    else:
        res = {'Municipio':[],
               'roc':[],'pr':[],'height':[],'rheight':[],
               'mean/std_roc':[],
               'mean/std_pr':[],
               'mean/std_height':[],
               'mean/std_rheight':[],}
        
        feature_importance = {'Municipio':[],
                              'feature_importance':[],
                              'mean':[],
                              'std':[]}
        predicted_proba = []


    for i, (train_municipios, val_municipio) in enumerate(train_val_stream):
        mpio = val_municipio
        
        if mpio in res['Municipio']:
            print(f"Skip training at {mpio}")
            continue
        else:
            print(f"Validate at {mpio}.")

            train_data = Event(train_municipios, val_municipio, subset, split='train')
            val_data = Event(train_municipios, val_municipio, subset, split='val')
            if args.municipio == 'puerto':
                test_data = Event(train_municipios, 'PUERTO LIBERTADOR', subset, split='val')

            feature_importance['features'] = val_data.features

            # numpy tabular data for scikit learn model
            X_train = train_data.tabX
            y_train = train_data.y
            X_val = val_data.tabX
            y_val = val_data.y
            lat_lon_val = val_data.locations
            lat_lon_train = train_data.locations
            hist_train = train_data.hist_mine 

            if args.municipio == 'puerto':
                X_test = test_data.tabX
                y_test = test_data.y
                lat_lon_test = test_data.locations
            

            # create model objects
            if model_name == 'TabCmpt' or model_name == 'MLP':
                if args.municipio == 'transfer':
                    args.lr /= 2
                    model = RELand(X_train.shape[1], args)
                    model.model.load_state_dict(torch.load(warm_start+f'/{mpio}.pth'))
                    # freeze encoder
                    for k, param in model.model.state_dict().items():
                        if (k != 'fc.weight') and (k != 'fc.bias'):
                            param.requires_grad = False
                else:
                    model = RELand(X_train.shape[1], args)
            elif model_name == 'LR' and objective == 'erm':
                params = {'penalty':'l1','C':1.8791083362131904}
                model = LogisticRegression(solver='saga', max_iter=1000, random_state=737, **params)
            elif model_name == 'LR' and objective == 'pnorm':
                model = LogisticRegression()
                # warm start
                with open(warm_start+f'/{mpio}.pkl', 'rb') as M:
                    model = pickle.load(M)
                beta_init = np.concatenate([model.intercept_,model.coef_.flatten()])
                model = PushedLR(X_train, y_train, beta_init=beta_init, p=2.0, max_iter=10)
            elif model_name == 'RF' and objective == 'erm': # not really erm, default split by gini
                params = {'max_depth':3}
                model = RandomForestClassifier(**params)
            elif model_name == 'RF' and objective == 'pnorm':
                model = RandomForestClassifier(**params)
                with open(warm_start+f'/{mpio}.pkl', 'rb') as M:
                    model = pickle.load(M)
                model.set_params(warm_start=True, n_estimators=200, criterion='pnorm',
                          random_state=737, verbose=1,n_jobs=8)
            elif model_name == 'LGBM' and objective == 'erm':
                params = {'learning_rate':0.2955060530944092,'num_leaves':1640,'max_depth':9}
                model = lgb.LGBMClassifier(objective="binary", n_estimators=1000 , **params)
            elif model_name == 'LGBM' and objective == 'pnorm':
                params = {'verbosity':1,'num_leaves':31,'max_depth':9}
                model = lgb.LGBMClassifier(objective=RankLoss.grad_hess, n_estimators=1000,**params)
            elif model_name == 'SVM':
                params = {'C':0.00001,'probability':True}
                model = SVC(**params)
            elif model_name == 'TabNet' and objective == 'erm':
                params = {'seed':737, 'n_steps':n_step}
                if args.municipio == 'transfer':
                    params['optimizer_params'] = {'lr':1e-2} # default lr = 2e-2
                model = erm_tab_model.TabNetClassifier(**params)
            elif model_name == 'TabNet' and objective == 'irm':
                params = {'seed':737, 'n_steps':n_step}
                model = irm_tab_model.TabNetClassifier(**params)

            # fit
            if model_name == 'TabCmpt' or model_name == 'MLP':
                ckpt = model.fit(train_data, val_data) # get validation result
                if args.municipio == 'puerto':
                    _, _, _, _, test_pred, _ = model.predict_proba(test_dataset = test_data)
            elif model_name == 'LGBM' and objective == 'pnorm':
                model.fit(X_train,y_train,eval_set=[(X_val, y_val)],early_stopping_rounds=epochs,eval_metric="auc")
                val_pred = sigmoid(model.predict(np.array(X_val), raw_score=True))
                if args.municipio == 'puerto':
                    test_pred = sigmoid(model.predict(np.array(X_test), raw_score=True))
            elif model_name == 'TabNet' and objective == 'erm':
                if args.municipio == 'transfer':
                    
                    model.fit(X_train,y_train, 
                        eval_set=[(X_val, y_val)], 
                        eval_metric=['auc'], max_epochs=epochs, 
                        patience=epochs, batch_size=batch_size, num_workers=num_workers,
                        warm_start=True, from_supervised=(warm_start+f'/{mpio}.pkl'))
                else:
                    model.fit(X_train,y_train, 
                            eval_set=[(X_val, y_val)], 
                            eval_metric=['auc'], max_epochs=epochs, 
                            patience=epochs, batch_size=batch_size, num_workers=num_workers)
                val_pred = model.predict_proba(np.array(X_val))[:,1]
                if args.municipio == 'puerto':
                    test_pred = model.predict_proba(np.array(X_test))[:,1]
            elif model_name == 'TabNet' and objective == 'irm':
                model.fit(X_train,y_train, hist_train,
                          eval_set=[(X_val, y_val)], 
                          eval_metric=['auc'], max_epochs=epochs, 
                          patience=epochs, batch_size=batch_size, num_workers=num_workers)
                val_pred = model.predict_proba(np.array(X_val))[:,1]
                if args.municipio == 'puerto':
                    test_pred = model.predict_proba(np.array(X_test))[:,1]
            else:
                model.fit(X_train,y_train)
                val_pred = model.predict_proba(np.array(X_val))[:,1]
                if args.municipio == 'puerto':
                    test_pred = model.predict_proba(np.array(X_test))[:,1]
            
            if model_name != 'TabCmpt' and model_name != 'MLP':
                roc = roc_auc_score(y_val, val_pred)
                precision, recall, _ = precision_recall_curve(y_val, val_pred)
                pr = auc(recall, precision)
                height = mean_height(y_val, val_pred)
                rheight = mean_reverse_height(y_val, val_pred)
                
                ckpt = dict()
                ckpt['roc'] = roc
                ckpt['pr'] = pr
                ckpt['height'] = height
                ckpt['rheight'] = rheight
                ckpt['prob'] = val_pred

                with open(f'./experiments/{timestamp}/{mpio}.pkl','wb') as f:
                    pickle.dump(model,f)
            
            roc = ckpt['roc']
            pr = ckpt['pr']
            height = ckpt['height']
            rheight = ckpt['rheight']
            val_prob = ckpt['prob']

            res['Municipio'].append(mpio)
            res['roc'].append(roc)
            res['pr'].append(pr)
            res['height'].append(height)
            res['rheight'].append(rheight)

            with open(f"./experiments/{timestamp}/metrics.json", 'w') as outfile:
                json.dump(res, outfile, indent=4)

            
            if model_name == 'TabCmpt':
                feature_importance['Municipio'].append(mpio)
                feature_importance['feature_importance'].append([float(num) for num in ckpt['importance']])
                with open(f"./experiments/{timestamp}/feature_importance.json", 'w') as outfile:
                    json.dump(feature_importance, outfile, indent=4)
            
            lat_lon = val_data.locations
            
            predicted_proba.append(pd.DataFrame({'LONGITUD_X':lat_lon[:,0], 
                                                 'LATITUD_Y':lat_lon[:,1],
                                                 'predicted_proba':val_prob}))
            
            predicted_proba_df = pd.concat(predicted_proba, axis=0)
            predicted_proba_df.to_csv(f'./experiments/{timestamp}/predicted_proba.csv',index=False)

            _, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
            mappable = axes[0].scatter(lat_lon[:,0], lat_lon[:,1], c=val_prob)
            axes[0].set_title(f'{mpio}\n'
                            f'roc-{roc:.3f}-pr-{pr:.3f}\n'
                            f'height-{height:.3f}-rheight-{rheight:.3f}')

            y_truth = np.array([int(y) for y in val_data.y])
            colors = ['yellow','navy']
            pos_idx = y_truth == 1
            neg_idx = y_truth == 0
            neg = axes[1].scatter(lat_lon[:,0][neg_idx], lat_lon[:,1][neg_idx], c=colors[1])
            pos = axes[1].scatter(lat_lon[:,0][pos_idx], lat_lon[:,1][pos_idx], c=colors[0])
            axes[1].set_title('truth')
            axes[1].legend((pos, neg),('pos','neg'))
            
            plt.tight_layout()
            plt.colorbar(mappable,ax=axes[0])
            plt.savefig(f'./experiments/{timestamp}/{mpio}.png')
            plt.clf()

        res['mean/std_roc'] = [np.mean(res['roc']), np.std(res['roc'])]
        res['mean/std_pr'] = [np.mean(res['pr']), np.std(res['pr'])]
        res['mean/std_height'] = [np.mean(res['height']), np.std(res['height'])]
        res['mean/std_rheight'] = [np.mean(res['rheight']), np.std(res['rheight'])]
        with open(f"./experiments/{timestamp}/metrics.json", 'w') as outfile:
            json.dump(res, outfile, indent=4)  

        predicted_proba_df = pd.concat(predicted_proba, axis=0)
        predicted_proba_df.to_csv(f'./experiments/{timestamp}/predicted_proba.csv',index=False)
        
        if model_name == 'TabCmpt':
            feature_importance['mean'] = np.mean(np.array(feature_importance['feature_importance']), axis=0).tolist()
            feature_importance['std'] = np.std(np.array(feature_importance['feature_importance']), axis=0).tolist()
        else:
            feature_importance['mean'] = -1
            feature_importance['std'] = -1 
        with open(f"./experiments/{timestamp}/feature_importance.json", 'w') as outfile:
            json.dump(feature_importance, outfile, indent=4)  
        feature_importance_df = pd.DataFrame({'features':val_data.features,
                                              'weights_mean':feature_importance['mean'],
                                              'weights_std':feature_importance['std']
                                              })
        feature_importance_df.to_csv(f'./experiments/{timestamp}/feature_importance.csv',index=False)
    
        # only train once for test set
        if args.municipio == 'puerto':
            lat_lon_test = test_data.locations
            test_df = (pd.DataFrame({'LONGITUD_X':lat_lon_test[:,0], 
                                    'LATITUD_Y':lat_lon_test[:,1],
                                    'predicted_proba':test_pred}))
            test_df.to_csv(f'./experiments/{timestamp}/test_results.csv',index=False)
            _, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
            mappable = plt.scatter(lat_lon_test[:,0], lat_lon_test[:,1], c=test_pred)
            axes.set_title('PUERTO LIBERTADOR')
            plt.colorbar(mappable)
            plt.savefig(f'./experiments/{timestamp}/test_results.png')
            plt.clf()


    return res


if __name__ == "__main__":
    # python main.py 2>&1 | tee ./experiments/[your timestamp]/log.txt

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", help=r'unique experiment id, hint: datetime.now().strftime("%m%d%Y%H%M%S")') 
    parser.add_argument("--municipio", required=True, help='directory with train test split info')
    parser.add_argument("--subset", required=True, help='single | geo | full')
    parser.add_argument("--model", required=True, help='TabCmpt | MLP | TabNet | LR | RF | SVM | LGBM')
    parser.add_argument("--objective", required=True, help='irm | erm | pnorm')
    parser.add_argument("--n_step", type=int, help='number of decision blocks')
    parser.add_argument("--warm_start", help='directory for checkpoints for warm start')


    args = parser.parse_args()

    timestamp = args.timestamp
    minimization_type = args.objective
    n_step = args.n_step
    subset = args.subset
    model_name = args.model
    objective = args.objective
    warm_start = args.warm_start
    
    train_val_folder = glob.glob(f'./train_val_stream/{args.municipio}/**.txt')
    folds = len(train_val_folder)//2
    train_val_stream = []
    for fold in range(folds):
        train_municipio = []
        val_municipio = []
        with open(f'./train_val_stream/{args.municipio}/train-{fold}.txt', 'r') as file:
            for line in file:
                train_municipio.append(line.strip())
        with open(f'./train_val_stream/{args.municipio}/val-{fold}.txt', 'r') as file:
            for line in file:
                val_municipio.append(line.strip())
        train_val_stream.append((train_municipio, val_municipio[0]))

    epochs = 500
    batch_size = 2048
    num_workers = 4
    lr = 0.01
    step_size = 75
    gamma = 0.1
    lambda_l2 = 5e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args.epochs = epochs
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.lr = lr
    args.step_size = step_size
    args.gamma = gamma
    args.lambda_l2 = lambda_l2
    args.device = device

    config = vars(args)
    with open(f"./experiments/{args.timestamp}/config.json", 'w') as outfile:
        if config['device'].type == 'cuda':
            config['device'] = 'cuda'
        else:
            config['device'] = 'cpu'
        json.dump(config, outfile, indent=4)
    
    print(f"Exp {timestamp} continued.")
    # current code backup
    os.makedirs(f"./experiments/{timestamp}/code", exist_ok=True)
    python_files = glob.glob('./**.py',recursive=False)
    for py in python_files:
        fname = py.split('/')[-1]
        shutil.copy(py, f'./experiments/{timestamp}/code/{fname}') 
    
    main(timestamp, train_val_stream)

    
    