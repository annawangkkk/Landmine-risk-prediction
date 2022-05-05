# evaluate on unseen data
# Never Ever change your model based on test_features_labels.csv !!!
from train import load_models, get_probability, weightBCE
import argparse
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from visualize_utils import plot_proba_dist, plot_heatmap
import os
import sys
import json
import glob

def split_numeric_binary(selected_features):
    f = open(save_exp_path + f'/params_exp{args.curr_time}.json')
    params = json.load(f)
    train_numeric = params['numeric_cols']
    f.close()
    selected_numeric_features = set(selected_features).intersection(set(train_numeric))
    selected_binary_features = set(selected_features) - selected_numeric_features
    return list(selected_numeric_features), list(selected_binary_features)


def eval(model, best_train_idx, train_grid_clean, test_grid_clean, selected_features):
    # evaluate on labeled test set
    imputer = KNNImputer(n_neighbors = 10, weights = 'distance')
    scaler = StandardScaler()
    best_fold = train_grid_clean[selected_features].iloc[best_train_idx]
    best_imputer = imputer.fit(best_fold)
    best_scaler = scaler.fit(best_fold)
    testX = np.array(test_grid_clean[selected_features])
    testy = np.array(test_grid_clean['mines_outcome'])
    idx = np.argwhere(np.isnan(testX).any(axis=0)).flatten() # find the rwi column
    if len(idx) > 0:
        testX[:,idx] = best_imputer.transform(testX)[:,idx]
    testX[:, 0:len(numeric_cols)] = best_scaler.transform(testX)[:, 0:len(numeric_cols)]
    # temporary:
    # refit the model on common features
    # is this correct? 
    model.fit(np.array(best_fold), np.array(train_grid_clean['mines_outcome'].iloc[best_train_idx]))
    testX_proba = model.predict_proba(testX)[:,1]
    rocauc = roc_auc_score(testy, testX_proba)
    wbce = weightBCE(testy, testX_proba, pos_weight=50)
    with open(save_eval_path + f'/{model_name}_test_metrics.txt', 'w') as f: # overwrite
        print("roc_auc",rocauc, file=f)
        print("wBCE",wbce, file=f)
        print("selected features",selected_features, file=f)  
    return rocauc, wbce

def generate_plots(proba):
    # plot heatmap, distribution
    plot_proba_dist(proba, model_name, save_eval_path)
    plot_heatmap(proba, model_name, save_eval_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/Users/zengzh/Desktop/17737/repo', help='root directory')
    parser.add_argument('--curr_time', help='name of your experiment results folder')
    parser.add_argument('--model_name', help='Any of SVM/LGBM/GP')
    #parser.add_argument('--calibrated', default=1, help='whether calibrate output probabilities by test set labels')
    
    args = parser.parse_args()
    model_name = args.model_name
    save_exp_path = args.root + f'/exp/{args.curr_time}'
    save_eval_path = save_exp_path + '/eval'
    if not os.path.exists(save_eval_path):
        os.makedirs(save_eval_path)
    else:
        file_names = glob.glob(save_eval_path + "/*")
        for file_name in file_names:
            if file_name.split("/")[-1].startswith(model_name):
                msg = input(f"Are you sure to OVERWRITE TEST SET EVALUATION OUTPUT IN {save_eval_path}? ")
                if (msg.lower() == 'yes' or msg.lower() == 'y'):
                    redo = True
                    break
                else:
                    sys.exit("Terminated by user.")
    #calibrated = args.calibrated
    
    print(f"Loading {model_name} results from {save_exp_path}...")
    output_proba = pd.read_csv(save_exp_path + f"/{model_name}_mines_proba.csv")
    selected_features = list(set(output_proba.columns) - set(['LATITUD_Y','LONGITUD_X','geometry','mines_outcome','prob']))
    best_model = load_models(model_name, save_exp_path)
    best_train_idx = open(save_exp_path + f'/{model_name}_best_train_idx.txt').read().split('\n')
    best_train_idx = [int(x) for x in best_train_idx[:-1]] # remove ending new line
    train_grid_all = pd.read_csv(args.root + "/processed_dataset/grid_features_labels.csv")
    train_grid_clean = train_grid_all[train_grid_all['mines_outcome'] != -1].reset_index(drop = True)

    test_grid_all = pd.read_csv(args.root + "/processed_dataset/test_features_labels.csv")
    # not all selected features in train are in test
    test_selected_features = list(set(selected_features).intersection(set(test_grid_all.columns))) 
    test_grid_clean = test_grid_all[test_grid_all['mines_outcome'] != -1].reset_index(drop = True)
    test_grid_clean = test_grid_clean[test_selected_features + ['mines_outcome']]
    
    # get numeric and binary cols from json settings, get intersection
    numeric_cols, _ = split_numeric_binary(test_selected_features)
    eval(best_model, best_train_idx, train_grid_clean, test_grid_clean, test_selected_features)
    best_fold = train_grid_all.iloc[best_train_idx]
    output_test_proba = get_probability(best_model, model_name, best_fold, test_selected_features, 
                                        test_grid_all, numeric_cols, save_eval_path)
    generate_plots(output_test_proba)