import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import argparse

from utils import *
from scikit_tabnet import *

def prepare_precision_recall(root='~/social_good/Landmine-risk-prediction'):
    '''
        generate two csvs on whole labeled dataset: 
        prepare_precision_recall.csv, include the probability input for tuning precision and recall
        all_roc.csv, include five fold rocauc for cross validation & five seeds rocauc on test
    '''
    df_proba = pd.read_csv(root + '/processed_dataset/all/all.csv',index_col=0)
    rows = df_proba.shape[0]
    df_proba = df_proba[['mines_outcome','LATITUD_Y','LONGITUD_X']]
    for split in ['random','sonson','caldas']:
        df_proba_split_CV_fold = np.array([-1] * rows)
        for idx in range(5):
            _, cv_val = load_cv(root, split, idx)
            tochange = list(cv_val.index.values)
            df_proba_split_CV_fold[tochange] = idx
        # the fold number of a train set datapoint, if it in test, the value is -1
        df_proba[f'{split}_val_fold'] = df_proba_split_CV_fold 
    df_proba = df_proba[df_proba['mines_outcome'] != -1] # labeled
    df_proba.to_csv(root + '/processed_dataset/results/prepare_precision_recall.csv')
    df_roc = pd.DataFrame()
    for dataset in ['new','old']:
        cols_info = FeatureNames(dataset)
        numeric_cols = cols_info.numeric_cols
        binary_cols = cols_info.binary_cols
        if dataset == 'old': # alternative features in old literature
            # optuna tuned CV parameters
            params_grids_splits = [('random',(('lr',{'penalty':'l1','C':8.509969763106886}),
                                              ('svm',{'kernel':'rbf','C':8.82114075382198}),
                                              )),
                                   ('sonson',(('lr',{'penalty':'l1','C':9.294118711689912}),
                                              ('svm',{'kernel':'poly','degree':4,'C':7.639929372165295}),
                                              )),
                                   ('caldas',(('lr',{'penalty':'l1','C':2.012819254091052}),
                                              ('svm',{'kernel':'rbf','C':4.119111812123697}),
                                              ))]
        else: # all new features
            params_grids_splits = [('random',(('lgb_lasso',{'learning_rate':0.045006696703877115,'num_leaves':1800,'max_depth':12}),
                                              ('rf_lasso',{'max_depth':120,'min_samples_split':3}),
                                              ('tabnet_lasso',{'n_d':33,'momentum':0.12356779981858614}),
                                              ('ensemble_lasso',{"C":1.1008035034975954}),
                                              ('lr',{'penalty':'l2',"C":9.626094309508076}),
                                              ('svm',{'kernel':'rbf','C':8.208800589942525}),
                                              ('mlp',{'alpha' : 0.0050312031367067063465, 'hidden_layer_sizes': 500}),
                                              ('lgb',{'learning_rate':0.045006696703877115,'num_leaves':1800,'max_depth':12}),
                                              ('rf',{'max_depth':120,'min_samples_split':3}),
                                              ('tabnet',{'n_d':33,'momentum':0.12356779981858614}),
                                              ('ensemble',{}),
                                              )),
                                   ('sonson',(('lgb_lasso',{'learning_rate':0.2955060530944092,'num_leaves':1640,'max_depth':9}),
                                              ('rf_lasso',{'max_depth':20,'min_samples_split':2}),
                                              ('tabnet_lasso',{'n_d':48,'momentum':0.017369738500399023}),
                                              ('ensemble_lasso',{'C':4.883929656273532}),
                                              ('lr',{'penalty':'l2','C':1.8791083362131904}),
                                              ('svm',{'kernel':'poly','degree':5,'C':1.869228074183435}),
                                              ('mlp',{'alpha':0.00021443630324341302,'hidden_layer_sizes':100}),
                                              ('lgb',{'learning_rate':0.2955060530944092,'num_leaves':1640,'max_depth':9}),
                                              ('rf',{'max_depth':20,'min_samples_split':2}),
                                              ('tabnet',{'n_d':48,'momentum':0.017369738500399023}),
                                              ('ensemble',{}),
                                              )),
                                   ('caldas',(('lgb_lasso',{'learning_rate':0.07438355074180618,'num_leaves':2500,'max_depth':8}),
                                              ('rf_lasso',{'max_depth':145,'min_samples_split':5}),
                                              ('tabnet_lasso',{'n_d':39,'momentum':0.18714104672552365}),
                                              ('ensemble_lasso',{'C':5.3301293777982215}),
                                              ('lr',{'penalty':'l2','C':0.539215153784205}),
                                              ('svm',{'C':2.5222379005101048,'kernel':'poly','degree':1}),
                                              ('mlp',{'alpha':0.002965571924436114,'hidden_layer_sizes':100}),
                                              ('lgb',{'learning_rate':0.07438355074180618,'num_leaves':2500,'max_depth':8}),
                                              ('rf',{'max_depth':145,'min_samples_split':5}),
                                              ('tabnet',{'n_d':39,'momentum':0.18714104672552365}),
                                              ('ensemble',{}),
                                              ))]
        for split, param_grids in params_grids_splits:
            # save ensemble bases
            clf_cv_list = [[],[],[],[],[]]
            clf_test_list = [[],[],[],[],[]]
            lasso_clf_cv_list = [[],[],[],[],[]]
            lasso_clf_test_list = [[],[],[],[],[]]
            for model_name, param in param_grids:
                X_all_labeled = pd.read_csv(root + '/processed_dataset/all/all_labeled.csv',index_col=0)
                X_all_labeled  = np.array(X_all_labeled.loc[:, numeric_cols + binary_cols])
                X_all_labeled, _ = preprocessX(X_all_labeled, X_all_labeled, numeric_cols)
                cv_scores = np.zeros((5,))
                test_scores = np.zeros((5,))
                print(f'dataset: {dataset}, split: {split}, model: {model_name}')
                for idx in tqdm(range(5)):
                    X_train, y_train, X_test, y_test = init_train_test(root, split, numeric_cols, binary_cols, idx)
                    seed_everything(idx)
                    X_train_cv, y_train_cv, X_val_cv, y_val_cv = init_cv(root, split, idx, numeric_cols, binary_cols)
                    if model_name == 'lr':
                        model_cv = LogisticRegression(solver='saga', max_iter=1000, random_state=idx, **param)
                        model = LogisticRegression(solver='saga', max_iter=1000, random_state=idx, **param)  
                        model_cv.fit(X_train_cv,y_train_cv)
                        model.fit(X_train,y_train)
                        X_all_cv_red = X_all_labeled
                        X_all_red = X_all_labeled
                    elif model_name == 'svm':
                        model_cv = SVC(probability = True, **param)
                        model = SVC(probability = True, **param)
                        model_cv.fit(X_train_cv,y_train_cv)
                        model.fit(X_train,y_train)
                        X_all_cv_red = X_all_labeled
                        X_all_red = X_all_labeled
                    elif model_name == 'mlp':
                        model_cv = MLPClassifier(random_state=idx, max_iter=1000, early_stopping=True, **param)
                        model = MLPClassifier(random_state=idx, max_iter=1000, early_stopping=True, **param)
                        model_cv.fit(X_train_cv,y_train_cv)
                        model.fit(X_train,y_train)
                        X_all_cv_red = X_all_labeled
                        X_all_red = X_all_labeled
                    elif model_name == 'lgb':
                        model_cv = lgb.LGBMClassifier(objective="binary", n_estimators=1000, **param)
                        model = lgb.LGBMClassifier(objective="binary", n_estimators=1000 , **param)
                        model_cv.fit(X_train_cv,y_train_cv,eval_set=[(X_val_cv, y_val_cv)],eval_metric=['auc'],
                                    callbacks=[lgb.early_stopping(200,  first_metric_only=True, verbose=0)])
                        model.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric=['auc'],
                                    callbacks=[lgb.early_stopping(200,  first_metric_only=True, verbose=0)])
                        clf_cv_list[idx].append(model_cv)
                        clf_test_list[idx].append(model)
                        X_all_cv_red = X_all_labeled
                        X_all_red = X_all_labeled
                    elif model_name == 'rf':
                        model_cv = RandomForestClassifier(random_state=idx, **param)
                        model = RandomForestClassifier(random_state=idx, **param)
                        model_cv.fit(X_train_cv,y_train_cv)
                        model.fit(X_train,y_train)
                        clf_cv_list[idx].append(model_cv)
                        clf_test_list[idx].append(model)
                        X_all_cv_red = X_all_labeled
                        X_all_red = X_all_labeled
                    elif model_name == 'tabnet':
                        model_cv = ScikitTabNet(seed=idx,**param)
                        model_cv.fit(X_train_cv,y_train_cv,X_val_cv, y_val_cv)
                        model = ScikitTabNet(seed=idx,**param)
                        model.fit(X_train, y_train, X_test, y_test)
                        clf_cv_list[idx].append(model_cv)
                        clf_test_list[idx].append(model)
                        X_all_cv_red = X_all_labeled
                        X_all_red = X_all_labeled
                    elif model_name == 'ensemble':
                        base_cv = [('lgb', clf_cv_list[idx][0]),
                                   ('rf', clf_cv_list[idx][1]),
                                   ('tabnet', clf_cv_list[idx][2])]
                        model_cv = VotingClassifier(estimators=base_cv, voting='soft')
                        model_cv.estimators_ = clf_cv_list[idx]
                        model_cv.le_ = LabelEncoder().fit(y_train_cv)
                        model_cv.classes_ = model_cv.le_.classes_
                        base = [('lgb', clf_test_list[idx][0]),
                                ('rf', clf_test_list[idx][1]),
                                ('tabnet', clf_test_list[idx][2])]
                        model = VotingClassifier(estimators=base, voting='soft')
                        model.estimators_ = clf_test_list[idx]
                        model.le_ = LabelEncoder().fit(y_train)
                        model.classes_ = model.le_.classes_
                        X_all_cv_red = X_all_labeled
                        X_all_red = X_all_labeled
                    elif model_name == 'lgb_lasso':
                        lasso_param = param_grids[3][-1]
                        X_train_cv, X_val_cv, X_train, X_test, lasso_model_cv, lasso_model = init_lasso(X_train_cv, 
                                                                           y_train_cv, 
                                                                           X_val_cv, 
                                                                           X_train, 
                                                                           y_train, 
                                                                           X_test, 
                                                                           numeric_cols, 
                                                                           lasso_param)
                        model_cv = lgb.LGBMClassifier(objective="binary", n_estimators=1000, **param)
                        model = lgb.LGBMClassifier(objective="binary", n_estimators=1000 , **param)
                        model_cv.fit(X_train_cv,y_train_cv,eval_set=[(X_val_cv, y_val_cv)],eval_metric=['auc'],
                                    callbacks=[lgb.early_stopping(200,  first_metric_only=True, verbose=0)])
                        model.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric=['auc'],
                                    callbacks=[lgb.early_stopping(200,  first_metric_only=True, verbose=0)])
                        lasso_clf_cv_list[idx].append(model_cv)
                        lasso_clf_test_list[idx].append(model)
                        X_all_cv_red = X_all_labeled[:, np.where(lasso_model_cv.coef_ != 0)[1]]
                        X_all_red = X_all_labeled[:, np.where(lasso_model.coef_ != 0)[1]]
                    elif model_name == 'rf_lasso':
                        lasso_param = param_grids[3][-1]
                        X_train_cv, X_val_cv, X_train, X_test, lasso_model_cv, lasso_model = init_lasso(X_train_cv, 
                                                                           y_train_cv, 
                                                                           X_val_cv, 
                                                                           X_train, 
                                                                           y_train, 
                                                                           X_test, 
                                                                           numeric_cols, 
                                                                           lasso_param)
                        model_cv = RandomForestClassifier(random_state=idx, **param)
                        model = RandomForestClassifier(random_state=idx, **param)
                        model_cv.fit(X_train_cv,y_train_cv)
                        model.fit(X_train,y_train)
                        lasso_clf_cv_list[idx].append(model_cv)
                        lasso_clf_test_list[idx].append(model)
                        X_all_cv_red = X_all_labeled[:, np.where(lasso_model_cv.coef_ != 0)[1]]
                        X_all_red = X_all_labeled[:, np.where(lasso_model.coef_ != 0)[1]]
                    elif model_name == 'tabnet_lasso':
                        lasso_param = param_grids[3][-1]
                        X_train_cv, X_val_cv, X_train, X_test, lasso_model_cv, lasso_model = init_lasso(X_train_cv, 
                                                                           y_train_cv, 
                                                                           X_val_cv, 
                                                                           X_train, 
                                                                           y_train, 
                                                                           X_test, 
                                                                           numeric_cols, 
                                                                           lasso_param)
                        model_cv = ScikitTabNet(seed=idx,**param)
                        model_cv.fit(X_train_cv,y_train_cv,X_val_cv, y_val_cv)
                        model = ScikitTabNet(seed=idx,**param)
                        model.fit(X_train, y_train, X_test, y_test)
                        lasso_clf_cv_list[idx].append(model_cv)
                        lasso_clf_test_list[idx].append(model)
                        X_all_cv_red = X_all_labeled[:, np.where(lasso_model_cv.coef_ != 0)[1]]
                        X_all_red = X_all_labeled[:, np.where(lasso_model.coef_ != 0)[1]]
                    elif model_name == 'ensemble_lasso':
                        X_train_cv, X_val_cv, X_train, X_test, lasso_model_cv, lasso_model = init_lasso(X_train_cv, 
                                                                           y_train_cv, 
                                                                           X_val_cv, 
                                                                           X_train, 
                                                                           y_train, 
                                                                           X_test, 
                                                                           numeric_cols, 
                                                                           param)
                        base_cv = [('lgb', lasso_clf_cv_list[idx][0]),
                                   ('rf', lasso_clf_cv_list[idx][1]),
                                   ('tabnet', lasso_clf_cv_list[idx][2])]
                        model_cv = VotingClassifier(estimators=base_cv, voting='soft')
                        model_cv.estimators_ = lasso_clf_cv_list[idx]
                        model_cv.le_ = LabelEncoder().fit(y_train_cv)
                        model_cv.classes_ = model_cv.le_.classes_
                        
                        base = [('lgb', lasso_clf_test_list[idx][0]),
                                ('rf', lasso_clf_test_list[idx][1]),
                                ('tabnet', lasso_clf_test_list[idx][2])]
                        model = VotingClassifier(estimators=base, voting='soft')
                        model.estimators_ = lasso_clf_test_list[idx]
                        model.le_ = LabelEncoder().fit(y_train)
                        model.classes_ = model.le_.classes_
                        X_all_cv_red = X_all_labeled[:, np.where(lasso_model_cv.coef_ != 0)[1]]
                        X_all_red = X_all_labeled[:, np.where(lasso_model.coef_ != 0)[1]]
                        
                    val_preds = model_cv.predict_proba(X_val_cv)
                    cv_scores[idx] = roc_auc_score(y_val_cv,val_preds[:,1]) 
                    val_preds_combined = model_cv.predict_proba(X_all_cv_red)[:,1]
                    df_proba[f'{model_name}_cv{idx}_{split}_{dataset}'] = val_preds_combined
                    test_preds = model.predict_proba(X_test)
                    test_scores[idx] = roc_auc_score(y_test, test_preds[:,1])
                    test_preds_combined = model.predict_proba(X_all_red)[:,1]
                    df_proba[f'{model_name}_seed{idx}_{split}_{dataset}'] = test_preds_combined
                    df_roc[f'{model_name}_cv_{split}_{dataset}'] = list(cv_scores)
                    df_roc[f'{model_name}_test_{split}_{dataset}'] = list(test_scores)            
                df_proba.to_csv(root + '/processed_dataset/results/prepare_precision_recall.csv')
                df_roc.to_csv(root + '/processed_dataset/results/all_roc.csv',index=False)       
    return "Completed"

def prepare_test_base_models(overfit:bool, root='~/social_good/Landmine-risk-prediction'):
    '''
        1. generate prepare_website_underfit.csv on unlabeled & labeled dataset:
        include REAL fitted on the three different train sets (RANDOM/SONSON/CALDAS) 
        if overfit = False, otherwise, fit on the whole labeled data,
        and then predict on all data to prepare for risk map. 
        2. return base models and ensemble
        3. use overfit = False for feature importance, overfit = True for pdp 
        plots and web interface input
    '''
    if overfit:
        fname = 'overfit'
    else:
        fname = 'underfit'
    cols_info = FeatureNames('new')
    numeric_cols = cols_info.numeric_cols
    binary_cols = cols_info.binary_cols
    features = numeric_cols + binary_cols
    params_grids_splits = [('random',(('lgb',{'learning_rate':0.045006696703877115,'num_leaves':1800,'max_depth':12}),
                                      ('rf',{'max_depth':120,'min_samples_split':3}),
                                      ('tabnet',{'n_d':33,'momentum':0.12356779981858614}),
                                      )),
                           ('sonson',(('lgb',{'learning_rate':0.2955060530944092,'num_leaves':1640,'max_depth':9}),
                                      ('rf',{'max_depth':20,'min_samples_split':2}),
                                      ('tabnet',{'n_d':48,'momentum':0.017369738500399023}),
                                      )),
                           ('caldas',(('lgb',{'learning_rate':0.07438355074180618,'num_leaves':2500,'max_depth':8}),
                                      ('rf',{'max_depth':145,'min_samples_split':5}),
                                      ('tabnet',{'n_d':39,'momentum':0.18714104672552365}),
                                      ))]
    all_test_bases = {'random':{'lgb':[],'rf':[],'tabnet':[]},
                 'sonson':{'lgb':[],'rf':[],'tabnet':[]},
                 'caldas':{'lgb':[],'rf':[],'tabnet':[]},}
    df_proba = pd.read_csv(root + '/processed_dataset/all/all.csv',index_col=0)
    df_proba = df_proba[['mines_outcome','LATITUD_Y','LONGITUD_X']]
    df_proba.to_csv(root + f'/processed_dataset/results/prepare_website_{fname}.csv')
    underfit_ensembles = {'random':[],'sonson':[],'caldas':[]}
    for split, param_grids in params_grids_splits:
        for model_name, param in param_grids:
            print(f'split: {split}, model: {model_name}')
            for idx in tqdm(range(5)):
                X_train, y_train, X_test, y_test = init_train_test(root, split, numeric_cols, binary_cols, idx)
                seed_everything(idx)
                X_all_labeled = pd.read_csv(root + f'/processed_dataset/all/all_labeled.csv',index_col=0)
                X_all = np.array(X_all_labeled.loc[:, features])
                y_all = np.array(X_all_labeled.loc[:, 'mines_outcome'])
                X_all, _ = preprocessX(X_all, X_test, numeric_cols)
                if model_name == 'lgb':
                    model = lgb.LGBMClassifier(objective="binary", n_estimators=1000 , **param)
                    model.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric=['auc'],
                                callbacks=[lgb.early_stopping(200,  first_metric_only=True, verbose=0)])                  
                elif model_name == 'rf':
                    model = RandomForestClassifier(random_state=idx, **param)
                    model.fit(X_train,y_train)
                elif model_name == 'tabnet':
                    model = ScikitTabNet(seed=idx,**param)
                    if overfit:
                        model.fit(X_all, y_all, early_stopping=False)
                    else:
                        model.fit(X_train, y_train, X_test, y_test)
                all_test_bases[split][model_name].append(model)
        for seed in range(5):
            X_train, y_train, X_test, y_test = init_train_test(root, split, numeric_cols, binary_cols, seed)
            seed_everything(seed)
            X_all_everything = pd.read_csv(root + f'/processed_dataset/all/all.csv',index_col=0)
            X_all_fin = np.array(X_all_everything.loc[:, features])
            X_all_fin, _ = preprocessX(X_all_fin, X_test, numeric_cols)
            base = [('lgb', all_test_bases[split]['lgb'][seed]),
                    ('rf', all_test_bases[split]['rf'][seed]),
                    ('tabnet', all_test_bases[split]['tabnet'][seed])]
            ensemble_model = VotingClassifier(estimators=base, voting='soft')
            ensemble_model.estimators_ = [m for (name, m) in base]
            ensemble_model.le_ = LabelEncoder().fit(y_test)
            ensemble_model.classes_ = ensemble_model.le_.classes_     
            df_proba[f'{split}_seed{seed}'] = ensemble_model.predict_proba(X_all_fin)[:,1]
            underfit_ensembles[split].append(ensemble_model)
        df_proba.to_csv(root + f'/processed_dataset/results/prepare_website_{fname}.csv')
    return all_test_bases, underfit_ensembles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='root directory of the folder.', default='~/social_good/Landmine-risk-prediction')
    args = parser.parse_args()
    prepare_precision_recall(args.root)
    # Two Optional Lines
    prepare_test_base_models(False, args.root)
    prepare_test_base_models(True, args.root)