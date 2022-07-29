import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay

from utils import *
from train_tabular import prepare_test_base_models 

from tqdm import tqdm

def plot_distribution_shift(root='~/social_good/Landmine-risk-prediction'):
    # PCA
    for dataset in ['new','old']:
        cols_info = FeatureNames(dataset)
        numeric_cols = cols_info.numeric_cols
        binary_cols = cols_info.binary_cols
        features = numeric_cols + binary_cols
        for split in ['random','sonson','caldas']:
            train_labeled = pd.read_csv(root + f'/processed_dataset/{split}/train/train_labeled.csv', index_col=0)
            test_labeled = pd.read_csv(root + f'/processed_dataset/{split}/test/test_labeled.csv', index_col=0)
            X_train = np.array(train_labeled.loc[:, features])
            y_train = np.array(train_labeled.loc[:, 'mines_outcome'])
            X_test = np.array(test_labeled.loc[:, features])
            y_test = np.array(test_labeled.loc[:, 'mines_outcome'])
            X_train, X_test = preprocessX(X_train, X_test, numeric_cols) 

            all_labeled = pd.read_csv(root +f'/processed_dataset/all/all_labeled.csv', index_col=0)
            X_all = np.array(all_labeled.loc[:, features])
            X_all, _ = preprocessX(X_all, X_test, numeric_cols) 

            pca = PCA(n_components=2)
            pca.fit(X_all)

            colors = ["navy", "darkorange", "turquoise", "pink"]
            target_names = ["train(y=0)", "train(y=1)","test(y=0)", "test(y=1)"]

            for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
                if target_name == "train(y=0)":
                    X_train_tsfm = pca.transform(X_train)
                    filter_idx = (y_train == 0).nonzero()[0]
                    plt.scatter(
                        X_train_tsfm[filter_idx, 0], X_train_tsfm[filter_idx, 1], color=color, alpha=0.8, label=target_name
                    )
                elif target_name == "train(y=1)":
                    X_train_tsfm = pca.transform(X_train)
                    filter_idx = (y_train == 1).nonzero()[0]
                    plt.scatter(
                        X_train_tsfm[filter_idx, 0], X_train_tsfm[filter_idx, 1], color=color, alpha=0.8, label=target_name
                    )
                elif target_name == 'test(y=0)':
                    X_test_tsfm = pca.transform(X_test)
                    filter_idx = (y_test == 0).nonzero()[0]
                    plt.scatter(
                        X_test_tsfm[filter_idx, 0], X_test_tsfm[filter_idx, 1], color=color, alpha=0.8, label=target_name
                    )
                elif target_name == 'test(y=1)':
                    X_test_tsfm = pca.transform(X_test)
                    filter_idx = (y_test == 1).nonzero()[0]
                    plt.scatter(
                        X_test_tsfm[filter_idx, 0], X_test_tsfm[filter_idx, 1], color=color, alpha=0.8, label=target_name
                    )
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.legend(loc="best")
            plt.savefig(root + f'/refactor/tabular/results/PCA/{split}_{dataset}_PCA.png') 
            plt.clf()
    return

def get_feature_importance(all_test_bases, root='~/social_good/Landmine-risk-prediction'):
    all_bases = all_test_bases
    cols_info = FeatureNames('new')
    numeric_cols = cols_info.numeric_cols
    binary_cols = cols_info.binary_cols
    all_features = numeric_cols + binary_cols
    orders = {'random':[],'sonson':[],'caldas':[]}
    df = pd.DataFrame()
    df['feature'] = all_features
    for j, split in enumerate(all_bases):
        three_models_avg = np.zeros((len(all_features),))
        for model_name in all_bases[split]:
            models = all_bases[split][model_name]
            if model_name == 'lgb':
                five_fold_avg = np.zeros((len(all_features),))
                # number of splits
                for model in models:
                    five_fold_avg += np.argsort(np.argsort(model.feature_importances_)) # normalized to rank
            elif model_name == 'rf':
                five_fold_avg = np.zeros((len(all_features),))
                # impurity
                for idx, model in enumerate(models):
                    five_fold_avg += np.argsort(np.argsort(model.feature_importances_))
            elif model_name == 'tabnet':
                five_fold_avg = np.zeros((len(all_features),))
                # built-in (masked val)
                for model in models:
                    five_fold_avg += np.argsort(np.argsort(model.estimator.feature_importances_))
            five_fold_avg /= len(models)
            df[f'{split}_{model_name}'] = five_fold_avg
            three_models_avg += five_fold_avg
        three_models_avg /= 3
        df[f'{split}_avg'] = three_models_avg
        sortidx = np.argsort(three_models_avg)
        df[f'{split}_rank'] = np.argsort(sortidx)
        orders[split] = [all_features[i] for i in sortidx]
    df.to_csv(root + f'/refactor/tabular/results/feature_importance/ensemble_base_importance.csv',index=False)
    return orders


def plot_pdp(ensemble_pairs, root='~/social_good/Landmine-risk-prediction'):
    cols_info = FeatureNames('new')
    numeric_cols = cols_info.numeric_cols
    binary_cols = cols_info.binary_cols
    features = numeric_cols + binary_cols
    X_all_labeled = pd.read_csv(root + f'/processed_dataset/all/all_labeled.csv',index_col=0)
    X_all = np.array(X_all_labeled.loc[:, features])
    y_all = np.array(X_all_labeled.loc[:, 'mines_outcome'])
    for fit, ensemble_list in ensemble_pairs: # [underfit * 3 split * 5 seed, overfit * 3 split * 5 seed]
        all_ensemble_base = []
        for seed in range(5):
            for split in ensemble_list:
                X_train, y_train, X_test, y_test = init_train_test(root, split, numeric_cols, binary_cols, seed)
                X_all, _ = preprocessX(X_all, X_test, numeric_cols)
                split_ensemble_model = ensemble_list[split][seed]
                all_ensemble_base.append((f'{split}_{seed}',split_ensemble_model))
        all_ensemble = VotingClassifier(estimators=all_ensemble_base, voting='soft')
        all_ensemble.estimators_ = [m for (name, m) in all_ensemble_base]
        all_ensemble.le_ = LabelEncoder().fit(y_all)
        all_ensemble.classes_ = all_ensemble.le_.classes_  
        for i in tqdm(range(X_all.shape[1])):
            pdp_fig = PartialDependenceDisplay.from_estimator(all_ensemble, 
                                                              X_all, 
                                                              [i], 
                                                              percentiles=[0,1],
                                                              feature_names=features, 
                                                              response_method='predict_proba',
                                                              random_state=i)
            plt.ylabel("Probablity of y = 1")
            plt.savefig(root + f'/refactor/tabular/results/feature_importance/pdp/{fit}/{features[i]}.png')
            plt.clf()           
    return "All pdp plots saved."

if __name__ == '__main__':
    all_test_bases, underfit_ensembles = prepare_test_base_models()
    orders = get_feature_importance(all_test_bases)
    plot_pdp([('underfit',underfit_ensembles)])
    plot_distribution_shift()