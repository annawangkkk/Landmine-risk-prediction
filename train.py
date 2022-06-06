import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn import tree
import lightgbm as lgb
from model.gp import DiriGPC
from model.nn import NN
from model.ensemble import LossVotingClassifier
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from visualize_utils import *
from utils import weightBCE

import pickle
import os
import argparse
import json
import sys
import copy


def restart(model_name):
    new_models = {'SVM': NuSVC(kernel ='rbf', nu=0.31, probability = True),
                          'LGBM': lgb.LGBMClassifier(num_leaves=args.num_leaves, random_seed=seed),
                          'GP': DiriGPC(epochs=5, verbose=1),
                          'LR': LogisticRegression(penalty='none', solver = 'saga'),
                          'RF': RandomForestClassifier(max_depth=3),
                          'NN': NN(batchsize=1024, epochs=600, lr=5e-3, omega=1),
                          'TabNet': TabNetClassifier(optimizer_fn=torch.optim.AdamW,
                                    optimizer_params=dict(lr=1e-2,weight_decay=5e-4),
                                    scheduler_params={"step_size":10, "gamma":0.9},
                                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                    mask_type='entmax'),
                          'KNN': KNeighborsClassifier(n_neighbors=15, weights='distance', metric='cosine')}
    m = model_name
    if m.find('+') == -1:
        models = new_models[m]
    else: # ensembles
        base_models = []
        base_model_names = m.split("+")
        for base_model_name in base_model_names:
            base_models.append((base_model_name,new_models[base_model_name]))
        models = LossVotingClassifier(estimators=base_models)
    return models

def preprocess_X(X_train_init, X_test_init):
    '''
        X_train: numpy array
        X_test: numpy array
    '''
    # fit scaler and imputer on X_train
    # transform both X_train and X_test
    X_train = copy.deepcopy(X_train_init)
    X_test = copy.deepcopy(X_test_init)
    train_imputer = imputer.fit(X_train)
    idx = np.argwhere(np.isnan(X_train).any(axis=0)).flatten() # find the rwi column
    if len(idx) > 0:
        X_train[:,idx] = train_imputer.transform(X_train)[:,idx]
        X_test[:,idx] = train_imputer.transform(X_test)[:,idx]
    train_scaler = scaler.fit(X_train[:, 0:len(numeric_cols)])
    X_train[:, 0:len(numeric_cols)] = train_scaler.transform(X_train[:, 0:len(numeric_cols)])
    X_test[:, 0:len(numeric_cols)] = train_scaler.transform(X_test[:, 0:len(numeric_cols)])
    return X_train, X_test

def generate_geo_folds(X):
    '''
        X(pd.Dataframe): all labeled features
    '''
    fold_indices = []
    all_cities = np.unique(X["Municipio"])
    balanced_num = len(all_cities) // 5
    np.random.seed(seed)
    np.random.shuffle(all_cities)
    city_groups = []
    for i in range(5):
        city_groups.append(all_cities[balanced_num*i:balanced_num*(i+1)])
        val_idx = []
        for j in range(len(city_groups[i])):
            val_idx.extend(list(X.loc[X["Municipio"] == city_groups[i][j], "Municipio"].index.values))
        train_idx = list(set(list(range(X.shape[0]))) - set(val_idx))
        fold_indices.append((train_idx, val_idx))
    return fold_indices

def select_best_C(model, model_name, X, y, reg_params):
    '''
        X: all labeled features
        y: all labeled ground truth
        reg_params(list): candidate list of regularization parameter
        return:
            best regularization parameter C
    '''
    # find the best LR regularizer through the intersection of BCE curve and 
    # ROC_AUC curve.

    print("Call select_best_C...",file=f)
    if split == 'random':
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        kf_idx = kf.split(X, y)
    else:
        kf_idx = generate_geo_folds(grid_clean['Municipio'])

    train_bce_cv = []
    test_bce_cv = []
    train_auc_cv = []
    test_auc_cv = []
    num_features_cv = []

    for train_index, test_index in kf_idx:
        train_bce = []
        test_bce = []
        num_features = []
        train_auc = []
        test_auc = []
            
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test)

        if args.reg_parameters is None:
            print("Learning best candidate parameters...", file=f)
            start = 3e-3
            end = 1e2
            clf = tree.DecisionTreeRegressor()
            candidate_reg_params = np.random.uniform(start, end, size=500)
            y_num_features = []
            for reg_param in candidate_reg_params:
                model_ftrs = LogisticRegression(penalty = 'l1', C = reg_param, solver = 'saga')
                model_ftrs.fit(X_train,y_train)
            y_num_features.append(len(np.where(model_ftrs.coef_ != 0)[1]))
            clf = clf.fit(candidate_reg_params, y_num_features)
            reg_params = clf.predict(list(range(1,num_features + 1))) # covers 1 to the full model
            print("Best candidate paramers...", reg_params, file=f)
        
        for reg_param in reg_params:
            model_ftrs = LogisticRegression(penalty = 'l1', C = reg_param, solver = 'saga')
            model_ftrs.fit(X_train,y_train)
            
            X_train_red = X_train[:, np.where(model_ftrs.coef_ != 0)[1]]
            X_test_red = X_test[:, np.where(model_ftrs.coef_ != 0)[1]]
            selected_features = np.where(model_ftrs.coef_ != 0)[1]
            
            model = restart(model_name)
            if model_name == 'TabNet':
                model.fit(X_train_red, y_train,
                max_epochs=150 , patience=50,
                batch_size=100, virtual_batch_size=50,
                weights=1,
                drop_last=False
                )
            elif model_name == 'NN':
                model.fit(X_train_red,y_train,X_test_red,y_test)
            else:
                model.fit(X_train_red,y_train)
            X_train_proba = model.predict_proba(X_train_red)[:,1]
            X_test_proba = model.predict_proba(X_test_red)[:,1]
            '''if calibrated:
                calibrated_clf = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=3)
                calibrated_clf.fit(X_test_red, y_test)
                X_train_proba = calibrated_clf.predict_proba(X_train_red)[:,1]
                X_test_proba = calibrated_clf.predict_proba(X_test_red)[:,1]'''
            train_bce.append(weightBCE(y_train,X_train_proba, pos_weight=50))
            train_auc.append(roc_auc_score(y_train,X_train_proba))
            test_bce.append(weightBCE(y_test,X_test_proba, pos_weight=50))
            test_auc.append(roc_auc_score(y_test,X_test_proba))
            num_features.append(X_train_red.shape[1])
            if verbose:
                print("train_roc_auc:",train_auc[-1], "test_roc_auc:",test_auc[-1],file=f)
                print("train_wBCE:",train_bce[-1], "test_wBCE:",test_bce[-1],file=f)
                print(f"LR + {model_name} selected features:", selected_features,file=f)
        
        train_bce_cv.append(train_bce)
        test_bce_cv.append(test_bce)
        train_auc_cv.append(train_auc)
        test_auc_cv.append(test_auc)
        num_features_cv.append(num_features)
    results = pd.DataFrame({'Train wBCE': np.mean(train_bce_cv, axis = 0), 
                    'Test wBCE': np.mean(test_bce_cv, axis = 0),
                    'Train ROC_AUC': np.mean(train_auc_cv, axis = 0), 
                    'Test ROC_AUC': np.mean(test_auc_cv, axis = 0),
                    'Num. features': np.mean(num_features_cv, axis = 0)})
    results.to_csv(save_path + f'/{model_name}_feature_selection.csv',index=False)
    #plot_auc_bce(results, model_name, save_path)

    # find an estimate of the intersection points of two curves
    # make sure wBCE and ROC_AUC are at the same scale
    # let wBCE in the range of [min(ROC), max(ROC)]
    #OldMin, OldMax, NewMin, NewMax = min(results['Test wBCE']), max(results['Test wBCE']), min(results['Test ROC_AUC']), max(results['Test ROC_AUC'])
    #normalizedwBCE = [((x - OldMin) * (NewMax - NewMin)  / (OldMax - OldMin)) + NewMin for x in results['Test wBCE']]
    #idx = np.argwhere(np.diff(np.sign(normalizedwBCE - results['Test ROC_AUC'])) != 0).flatten()[-1]
    estimate_best_C = reg_params[np.argmax(np.mean(test_auc_cv, axis = 0))] # the nearest X value before intersection
    print(f"Best LR Regularizer is {estimate_best_C}", file=f)
    return estimate_best_C

def select_best_fold(model, model_name, X, y, C):
    '''
        X: all labeled features
        y: all labeled ground truth
        C: the best regularization parameter returned from select_best_C()
           if C is None, use full model
        return: 
            best train fold's index in all labeled features X
    '''
    # after we find the best regularizer, use CV again to find the best fold
    # based on AUC and BCE again
    # to determine the best fold to fit imputer and scaler

    print("Call select_best_fold...",file=f)

    if split == 'random':
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        kf_idx = kf.split(X, y)
    else:
        kf_idx = generate_geo_folds(grid_clean[['Municipio']])

    train_bce_cv = []
    test_bce_cv = []
    train_auc_cv = []
    test_auc_cv = []
    train_idxs = []
    for train_index, test_index in kf_idx:
            
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test)
        
        if C is None:
            X_train_red = X_train
            X_test_red = X_test
            selected_features = features
        else:
            model_ftrs = LogisticRegression(penalty = 'l1', C = C, solver = 'saga')
            model_ftrs.fit(X_train,y_train)
            X_train_red = X_train[:, np.where(model_ftrs.coef_ != 0)[1]]
            X_test_red = X_test[:, np.where(model_ftrs.coef_ != 0)[1]]
            selected_features = np.where(model_ftrs.coef_ != 0)[1]
        
        model = restart(model_name)
        if model_name == 'TabNet':
            model.fit(X_train_red, y_train,
            max_epochs=150 , patience=50,
            batch_size=100, virtual_batch_size=50,
            weights=1,
            drop_last=False
            )
        else:
            model.fit(X_train_red,y_train)
        X_train_proba = model.predict_proba(X_train_red)[:,1]
        X_test_proba = model.predict_proba(X_test_red)[:,1]
        # make sure the output probability well-calibrated before calculating AUC
        '''if calibrated:
            calibrated_clf = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=3)
            calibrated_clf.fit(X_test_red, y_test) # fit model on train, calibrate probabilities on test
            X_train_proba = calibrated_clf.predict_proba(X_train_red)[:,1]
            X_test_proba = calibrated_clf.predict_proba(X_test_red)[:,1]'''
        train_bce_cv.append(weightBCE(y_train,X_train_proba, pos_weight=50))
        train_auc_cv.append(roc_auc_score(y_train,X_train_proba))
        test_bce_cv.append(weightBCE(y_test,X_test_proba, pos_weight=50))
        test_auc_cv.append(roc_auc_score(y_test,X_test_proba))
        train_idxs.append(train_index)
        if verbose:
            print("train_roc_auc:",train_auc_cv[-1], "test_roc_auc:",test_auc_cv[-1],file=f)
            print("train_wBCE:",train_bce_cv[-1], "test_wBCE:",test_bce_cv[-1],file=f)
            if C is None:
                print(f"{model_name} selected features:", selected_features,file=f)
            else:
                print(f"LR + {model_name} selected features:", selected_features,file=f)
    res = train_idxs[np.argmax(test_auc_cv)]
    print("5-fold valid avg",
          "wBCE",np.mean(test_bce_cv),
          "roc_auc",np.mean(test_auc_cv),
          file=f)
    print("Best valid fold num", np.argmax(test_auc_cv),
          "wBCE",test_bce_cv[np.argmax(test_auc_cv)],
          "roc_auc",test_auc_cv[np.argmax(test_auc_cv)],
          file=f)
    with open(save_path + f'/{model_name}_best_train_idx.txt', 'w') as r:
        for i in res:
            r.write("%s\n" % i)
    return res

def train(model, model_name, X, y, best_train_idx, C):
    '''
        X: all labeled features
        y: all labeled ground truth
        best_train_idx: best train fold's index
        C: a regularization parameter C
        return:
            (calibrated) model and selected features
    '''
    # if not using calibrated probabilities,
    # fit the model on the whole labeled dataset.
    # Otherwise,
    # fit the model on the best train fold,
    # and calibrate the probabilties on the test fold.
    # Models are saved as pkl files.
    print("Call train...", file=f)
    best_test_idx = list(set(range(X.shape[0])) - set(best_train_idx))
    best_fold = X[best_train_idx, :]
    best_imputer = imputer.fit(best_fold)
    best_scaler = scaler.fit(best_fold)
    idx = np.argwhere(np.isnan(X).any(axis=0)).flatten() # find the rwi column
    X_tmp = copy.deepcopy(X)
    if len(idx) > 0:
        X_tmp[:,idx] = best_imputer.transform(X_tmp)[:,idx]
        X_tmp[:,idx] = best_imputer.transform(X_tmp)[:,idx]
    X_tmp[:, 0:len(numeric_cols)] = best_scaler.transform(X_tmp)[:, 0:len(numeric_cols)]
    if C is None:
        X_test_red = X_tmp[best_test_idx,:]
        y_test_red = y[best_test_idx]
        X_all_red = X_tmp
    else:
        lasso = LogisticRegression(penalty = 'l1', C = C, solver = 'saga')
        lasso.fit(X_tmp,y)
        X_test_red = X_tmp[:, np.where(lasso.coef_ != 0)[1]][best_test_idx,:]
        y_test_red = y[best_test_idx]
        X_all_red = X_tmp[:, np.where(lasso.coef_ != 0)[1]]
    model = restart(model_name)
    if model_name == 'TabNet':
        model.fit(X_all_red, y,
        max_epochs=150 , patience=50,
        batch_size=100, virtual_batch_size=50,
        weights=1,
        drop_last=False
        )
    else:
        model.fit(X_all_red,y)
    X_all_proba = model.predict_proba(X_all_red)[:,1]
    '''if calibrated:
        calibrated_clf = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=3)
        calibrated_clf.fit(X_test_red, y_test_red)
        X_all_proba = calibrated_clf.predict_proba(X_all_red)[:,1]
        bce = weightBCE(y, X_all_proba, pos_weight=50)
        roc_auc = roc_auc_score(y, X_all_proba)
        print("wBCE:",bce, "roc_auc:",roc_auc,
            "selected_labels:",np.array(features)[np.where(lasso.coef_ != 0)[1]],file=f)
        try:
            pickle.dump(model, open(save_path+f'/{model_name}_calibration_base.pkl','wb'))
            pickle.dump(calibrated_clf, open(save_path+f'/{model_name}_calibration.pkl','wb'))
        except:
            print("Warning: some models are not saved successfully.") # 
        return calibrated_clf, np.array(features)[np.where(lasso.coef_ != 0)[1]]
    else:'''
    bce = weightBCE(y, X_all_proba, pos_weight=50)
    roc_auc = roc_auc_score(y, X_all_proba)
    if C is None:
        print("Metrics for model fitted on the full labeled dataset", 
        "wBCE:",bce, "roc_auc:",roc_auc, "selected_labels:", "all features", file=f)
        pickle.dump(model, open(save_path+f'/{model_name}.pkl','wb'))
        return model, np.array(features)
    else:
        print("Metrics for model fitted on the full labeled dataset", 
            "wBCE:",bce, "roc_auc:",roc_auc,
            "selected_labels:",np.array(features)[np.where(lasso.coef_ != 0)[1]],file=f)
        pickle.dump(model, open(save_path+f'/{model_name}.pkl','wb'))
        return model, np.array(features)[np.where(lasso.coef_ != 0)[1]]

def get_five_prob(C, X, y, X_all, save_path):
    '''
        X: all labeled features
        y: all labeled ground truth
        X_all: labeled + unlabeled
        C: the best regularization parameter returned from select_best_C()
           if C is None, use full model
    '''
    X_all_tmp = X_all
    if split == 'random':
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        kf_idx = kf.split(X, y)
    else:
        kf_idx = generate_geo_folds(grid_clean[['Municipio']])
    
    probs = []
    train_indices = []
    for idx, (train_index, _) in enumerate(kf_idx):
        X_all = np.array(X_all_tmp[features])  
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_train, X_all = preprocess_X(X_train, X_all)
        if C is None:
            X_train_red = X_train
            X_all_red = X_all
        else:
            model_ftrs = LogisticRegression(penalty = 'l1', C = C, solver = 'saga')
            model_ftrs.fit(X_train,y_train)
            X_train_red = X_train[:, np.where(model_ftrs.coef_ != 0)[1]]
            X_all_red = X_all[:, np.where(model_ftrs.coef_ != 0)[1]]
        
        model = restart(model_name)
        if model_name == 'TabNet':
            model.fit(X_train_red, y_train,
            max_epochs=150 , patience=50,
            batch_size=100, virtual_batch_size=50,
            weights=1,
            drop_last=False
            )
        else:
            model.fit(X_train_red,y_train)
        X_all_proba = model.predict_proba(X_all_red)[:,1]
        probs.append(X_all_proba)
        train_indices.append(train_index)
        pickle.dump(model, open(save_path+f'/{model_name}_CV{idx}.pkl','wb'))
    X_all_df = X_all_tmp[features]
    X_all_df["mines_outcome"] = grid_all["mines_outcome"]
    for i in range(5):
        X_all_df[f'CVprob{i}'] = probs[i] 
        with open(save_path + f'/{model_name}_CV{i}_idx.txt', 'w') as r:
            for j in train_indices[i]:
                r.write("%s\n" % j)
    X_all_df.to_csv(save_path + f"/{model_name}_CV_mines_proba.csv",index=False)
    return X_all_df

def get_probability(best_model, model_name, best_fold, selected_features, X_all, numeric_cols, save_path):
    '''
        best_model: returned from train()
        best_fold(DataFrame): labeled feature using best_train_idx returned from select_best_fold()
        selected_features: returned from train()
        X_all: labeled and unlabeled features
        numeric_cols: numeric columns
        return:
            DataFrame with predicted probability, Latitude, Longitude, ground truth,
            and all selected features
    '''
    # generate probabilities
    X_all_tmp = pd.DataFrame(X_all)
    X_all = np.array(X_all[selected_features])  
    best_fold = np.array(best_fold[selected_features])
    best_imputer = KNNImputer(n_neighbors = 10, weights = 'distance').fit(best_fold)
    best_scaler = StandardScaler().fit(best_fold)
    idx = np.argwhere(np.isnan(X_all).any(axis=0)).flatten() # find the rwi column
    if len(idx) > 0:
        X_all[:,idx] = best_imputer.transform(X_all)[:,idx]
    X_all[:, 0:len(numeric_cols)] = best_scaler.transform(X_all)[:, 0:len(numeric_cols)]
    prob = best_model.predict_proba(X_all)[:,1]
    X_all_df = pd.DataFrame(X_all,columns=selected_features)
    X_all_df['LATITUD_Y'] = X_all_tmp['LATITUD_Y']
    X_all_df['LONGITUD_X'] = X_all_tmp['LONGITUD_X']
    X_all_df['geometry'] = X_all_tmp['geometry']
    X_all_df['mines_outcome'] = X_all_tmp['mines_outcome']
    X_all_df['prob'] = prob # probability for being positive
    X_all_df.to_csv(save_path + f"/{model_name}_mines_proba.csv",index=False)
    return X_all_df

def load_models(model_name, save_path, suffix=""):
    with open(save_path + f'/{model_name}{suffix}.pkl', 'rb') as m:
        clf = pickle.load(m)
    return clf

def complete_pipeline(full):
    if full:
        estimate_best_C = None
    else:
        estimate_best_C = select_best_C(model, model_name, X, y, args.reg_parameters)
    CV_df = get_five_prob(estimate_best_C, X, y, grid_all, save_path)
    best_train_idx = select_best_fold(model, model_name, X, y, estimate_best_C)
    best_model, selected_features = train(model, model_name, X, y, best_train_idx, estimate_best_C)
    best_fold = pd.DataFrame(X[best_train_idx, :])
    best_fold.columns = features
    output = get_probability(best_model, model_name, best_fold, selected_features, grid_all, numeric_cols, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_json',
        help='Load settings from file in json format.')
    parser.add_argument('--num_leaves', default=45,
        help='num_leaves in LGBM model.')
    parser.add_argument('--CV', default='random',
        help='use random cv or geo-based cv')
    args = parser.parse_args()
    json_dir = args.load_json
    path_not_exist = False
    if not os.path.exists(json_dir):
        time_info = json_dir.split("/")[-1].split(".")[0].split("_")[-1][3:]
        json_dir = '/'.join(json_dir.split("/")[:-1]) + f'/exp_local/{time_info}/params_exp{time_info}.json'
        print(f"Opened json file {json_dir}")
        path_not_exist = True
    with open(json_dir, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    seed = args.seed 
    np.random.seed(seed)
    #calibrated = args.calibrated
    split = args.split
    verbose = args.verbose
    current_time = args.curr_time
    full = args.isfull
    save_path = args.root + f'/exp_local/{current_time}'
    print("All experiment results will be saved in", save_path)
    
    redo = False
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        msg = input(f"Are you sure to OVERWRITE THE PROBABILITY OUTPUT IN {save_path}? ")
        if (msg.lower() == 'yes' or msg.lower() == 'y'):
            redo = True
        else:
            sys.exit("Terminated by user.")
    if not path_not_exist:
        os.replace(args.load_json, args.root + f'/exp_local/{current_time}/' + args.load_json.split("/")[-1])
    grid_all = pd.read_csv(args.root + f'/processed_dataset/grid_features_withimg_res18.csv') 
    grid_clean = grid_all[grid_all['mines_outcome'] != -1].reset_index(drop = True)
    
    numeric_cols = args.numeric_cols 
    binary_cols = args.binary_cols

    features = numeric_cols + binary_cols
    X = np.array(grid_clean[features]) # all labeled X
    y = np.array(grid_clean['mines_outcome']) # all labeled y
    imputer = KNNImputer(n_neighbors = 10, weights = 'distance')
    scaler = StandardScaler()

    
    models = [] 
    # TODO: set default params in config outside of json
    implemented_models = {'SVM': NuSVC(kernel ='rbf', nu=0.31, probability = True),
                          'LGBM': lgb.LGBMClassifier(num_leaves=args.num_leaves, random_seed=seed),
                          'GP': DiriGPC(epochs=5, verbose=1),
                          'LR': LogisticRegression(penalty='none', solver = 'saga'),
                          'RF': RandomForestClassifier(max_depth=3),
                          'NN': NN(batchsize=1024, epochs=600, lr=5e-3, omega=1),
                          'TabNet': TabNetClassifier(optimizer_fn=torch.optim.AdamW,
                                    optimizer_params=dict(lr=1e-2,weight_decay=5e-4),
                                    scheduler_params={"step_size":10, "gamma":0.9},
                                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                    mask_type='entmax'),
                          'KNN': KNeighborsClassifier(n_neighbors=15, weights='distance', metric='cosine')}
    for m in args.models:
        if m.find('+') == -1:
            models.append((m,implemented_models[m]))
        else: # ensembles
            base_models = []
            base_model_names = m.split("+")
            for base_model_name in base_model_names:
                base_models.append((base_model_name,implemented_models[base_model_name]))
            models.append((m, LossVotingClassifier(estimators=base_models)))
    

    for (model_name, model) in models:
        if redo:
            f = open(save_path + "/log.txt", "a")
            print(f"Calculating probability from existing {model_name} model", file=f)
            try:
                print(f"Attempted to load {model_name}.pkl from {save_path}...", file=f)
                best_model = load_models(model_name, save_path)
                best_train_idx = open(save_path + f'/{model_name}_best_train_idx.txt').read().split('\n')
                best_train_idx = [int(x) for x in best_train_idx[:-1]] # remove ending new line
                output_proba = pd.read_csv(save_path + f"/{model_name}_mines_proba.csv")
                selected_features = list(set(output_proba.columns) - set(['LATITUD_Y','LONGITUD_X','geometry','mines_outcome','prob'])) 
                best_fold = pd.DataFrame(X[best_train_idx, :])
                best_fold.columns = features
                overwrite_output = get_probability(best_model, model_name, best_fold, selected_features, grid_all, numeric_cols, save_path)
            except:
                print(f"Incomplete prerequisites. Retrain models.", file=f)
                complete_pipeline(full)
            f.close()
        else:
            f = open(save_path + "/log.txt", "w")
            print("Current model is ", model_name, file=f)
            complete_pipeline(full)
            f.close()
    

    