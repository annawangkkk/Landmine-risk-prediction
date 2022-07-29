from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import lightgbm as lgb
from scikit_tabnet import ScikitTabNet, ScikitTabNetRegressor

from utils import *

from tqdm import tqdm

def assign_labels(split,seed, ssl_type, features, root='~/social_good/Landmine-risk-prediction'):
    all_proba = pd.read_csv(root + '/processed_dataset/results/prepare_website_underfit.csv', index_col = 0)
    target_proba = all_proba.loc[(all_proba['mines_outcome'] == -1) \
                                 & ((all_proba[f'{split}_seed{seed}'] > 0.9) \
                                    | (all_proba[f'{split}_seed{seed}'] < 0.1)), \
                                 f'{split}_seed{seed}']
    labeled_idx = list(target_proba.index.values)
    if ssl_type == 'soft':
        slicedY = list(target_proba)
    else:
        slicedY = [1 if p > 0.5 else 0 for p in list(target_proba)]
    all_data = pd.read_csv(root + '/processed_dataset/all/all.csv', index_col = 0)[features]
    slicedX = np.array(all_data.iloc[labeled_idx])
    return slicedX, slicedY

def refit_ensemble(ssl_type, root='~/social_good/Landmine-risk-prediction'):
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
    for split, param_grids in params_grids_splits:
        for model_name, param in param_grids:
            print(f'split: {split}, model: {model_name}')
            for idx in tqdm(range(5)):
                seed_everything(idx)
                train_labeled = pd.read_csv(root + f'/processed_dataset/{split}/train/train_labeled.csv', index_col=0)
                test_labeled = pd.read_csv(root + f'/processed_dataset/{split}/test/test_labeled.csv', index_col=0)
                X_unlabeled, y_unlabeled = assign_labels(split,idx,ssl_type,features)
                X_train = np.array(train_labeled.loc[:, features])
                y_train = np.array(train_labeled.loc[:, 'mines_outcome'])
                X_train = np.concatenate([X_unlabeled,X_train])
                y_train = np.concatenate([y_unlabeled,y_train])
                X_test = np.array(test_labeled.loc[:, features])
                y_test = np.array(test_labeled.loc[:, 'mines_outcome'])
                X_train, X_test = preprocessX(X_train, X_test, numeric_cols)
                train = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)
                np.random.seed(idx)
                np.random.shuffle(train)
                X_train = train[:,:-1]
                y_train = train[:,-1]
                if ssl_type == 'hard':
                    if model_name == 'lgb':
                        model = lgb.LGBMClassifier(objective="binary", n_estimators=1000 , **param)
                        model.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric=['auc'],
                                    callbacks=[lgb.early_stopping(200,  first_metric_only=True, verbose=0)])                  
                    elif model_name == 'rf':
                        model = RandomForestClassifier(random_state=idx, **param)
                        model.fit(X_train,y_train)
                    elif model_name == 'tabnet':
                        model = ScikitTabNet(seed=idx,**param)
                        model.fit(X_train, y_train, X_test, y_test)
                else:
                    if model_name == 'lgb':
                        model = lgb.LGBMRegressor(objective="binary", n_estimators=1000 , **param)
                        model.fit(X_train,y_train,eval_set=[(X_test, y_test)],eval_metric=['auc'],
                                    callbacks=[lgb.early_stopping(200,  first_metric_only=True, verbose=0)])                  
                    elif model_name == 'rf':
                        model = RandomForestRegressor(random_state=idx, **param)
                        model.fit(X_train,y_train)
                    elif model_name == 'tabnet':
                        model = ScikitTabNetRegressor(seed=idx,**param)
                        model.fit(X_train, y_train, X_test, y_test)
                all_test_bases[split][model_name].append(model)
        for seed in range(5):
            seed_everything(idx)
            train_labeled = pd.read_csv(root + f'/processed_dataset/{split}/train/train_labeled.csv', index_col=0)
            test_labeled = pd.read_csv(root + f'/processed_dataset/{split}/test/test_labeled.csv', index_col=0)
            X_unlabeled, y_unlabeled = assign_labels(split,seed,ssl_type,features)
            X_train = np.array(train_labeled.loc[:, features])
            y_train = np.array(train_labeled.loc[:, 'mines_outcome'])
            X_train = np.concatenate([X_unlabeled,X_train])
            y_train = np.concatenate([y_unlabeled,y_train])
            X_test = np.array(test_labeled.loc[:, features])
            y_test = np.array(test_labeled.loc[:, 'mines_outcome'])
            X_train, X_test = preprocessX(X_train, X_test, numeric_cols)
            train = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)
            np.random.seed(idx)
            np.random.shuffle(train)
            X_train = train[:,:-1]
            y_train = train[:,-1]
            if ssl_type == 'hard': 
                lgb_proba = all_test_bases[split]['lgb'][seed].predict_proba(X_test)[:,1]
                rf_proba = all_test_bases[split]['rf'][seed].predict_proba(X_test)[:,1]
                tabnet_proba = all_test_bases[split]['tabnet'][seed].predict_proba(X_test)[:,1]
            else:
                lgb_proba = np.clip(all_test_bases[split]['lgb'][seed].predict(X_test),0,1)
                rf_proba = np.clip(all_test_bases[split]['rf'][seed].predict(X_test),0,1)
                tabnet_proba = np.clip(all_test_bases[split]['tabnet'][seed].predict(X_test),0,1)
            ensemble_proba = (lgb_proba + rf_proba + tabnet_proba) / 3
            print(f'rocauc: {roc_auc_score(y_test, ensemble_proba)}, seed: {seed}, ssl_type: {ssl_type}')
    return 

if __name__ == '__main__':
    for ssl_type in ['soft','hard']:
        refit_ensemble(ssl_type)