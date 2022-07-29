import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from scikit_tabnet import ScikitTabNet

from ..utils import *
from images.dataset import *
from images.model import *



def feature_extraction(split):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = '~/social_good/Landmine-risk-prediction'
    imputer = KNNImputer(n_neighbors = 10, weights = 'distance')
    scaler = StandardScaler()
    cols_info = FeatureNames('new')
    numeric_cols = cols_info.numeric_cols
    binary_cols = cols_info.binary_cols
    features = numeric_cols + binary_cols
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomVerticalFlip(),
                                                      torchvision.transforms.ToTensor(), 
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    five_seeds_res = np.zeros((5,))
    for seed in tqdm(range(5)):
        seed_everything(seed)
        if split == 'random':
            model = nn.DataParallel(ResNet18('unfreeze').to(device))
            model.load_state_dict(torch.load(f'~/social_good/satellite_models/random/pure/res18_unfreeze_satellite_seed{seed}.pth'))
            train_data = Landmine(root, split, 'train_labeled', 'satellite', transform=train_transform)
            test_data = Landmine(root, split, 'test_labeled', 'satellite', transform=test_transform)
        elif split == 'sonson':
            model = nn.DataParallel(ResNet50('freeze').to(device))
            model.load_state_dict(torch.load(f'~/social_good/terrain_models/sonson/pure/res50_freeze_terrain_seed{seed}.pth'))
            train_data = Landmine(root, split, 'train_labeled', 'terrain', transform=train_transform)
            test_data = Landmine(root, split, 'test_labeled', 'terrain', transform=test_transform)
        elif split == 'caldas':
            model = nn.DataParallel(ResNet50('unfreeze').to(device))
            model.load_state_dict(torch.load(f'~/social_good/satellite_models/caldas/pure/res50_unfreeze_satellite_seed{seed}.pth'))
            train_data = Landmine(root, split, 'train_labeled', 'satellite', transform=train_transform)
            test_data = Landmine(root, split, 'test_labeled', 'satellite', transform=test_transform)
        train_loader = DataLoader(train_data, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        model.module.fc = nn.Identity()
        model.eval()
        
        X_train_img = []
        X_test_img = []
        with torch.no_grad():
            for data, target in train_loader:
                data = data.to(device)
                out = model(data).squeeze()
                X_train_img.append(out.detach().cpu().numpy())
        X_train_img = np.concatenate(X_train_img)
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                out = model(data).squeeze()
                X_test_img.append(out.detach().cpu().numpy())
        X_test_img = np.concatenate(X_test_img)
        
        train_labeled = pd.read_csv(root + f'/processed_dataset/{split}/train/train_labeled.csv', index_col=0)
        test_labeled = pd.read_csv(root + f'/processed_dataset/{split}/test/test_labeled.csv', index_col=0)
        X_train = np.array(train_labeled.loc[:, features])
        X_train = np.concatenate([X_train, X_train_img],axis=1)
        y_train = np.array(train_labeled.loc[:, 'mines_outcome'])
        X_test = np.array(test_labeled.loc[:, features])
        X_test = np.concatenate([X_test, X_test_img],axis=1)
        y_test = np.array(test_labeled.loc[:, 'mines_outcome'])
        X_train, X_test = preprocessX(X_train, X_test, numeric_cols)
        
        if split == 'random':
            estimators = [
                ('lgb', lgb.LGBMClassifier(objective="binary", 
                                           learning_rate = 0.045006696703877115, 
                                           n_estimators = 1000,
                                           num_leaves = 1800, max_depth=12)),
                ('rf', RandomForestClassifier(random_state=seed, 
                                              max_depth=120, 
                                              min_samples_split=3)),
                ('tabnet', ScikitTabNet(seed=seed, 
                                        n_d=33, 
                                        momentum=0.12356779981858614))]
        elif split == 'sonson':
            estimators = [
                ('lgb', lgb.LGBMClassifier(objective="binary", 
                                           learning_rate = 0.2955060530944092, 
                                           n_estimators = 1000,
                                           num_leaves = 1640, max_depth=9)),
                ('rf', RandomForestClassifier(random_state=seed, 
                                              max_depth=20, 
                                              min_samples_split=2)),
                ('tabnet', ScikitTabNet(seed=seed, 
                                        n_d=48, 
                                        momentum=0.017369738500399023))]
        elif split == 'caldas':
            estimators = [
                ('lgb', lgb.LGBMClassifier(objective="binary", 
                                           learning_rate = 0.07438355074180618, 
                                           n_estimators = 1000,
                                           num_leaves = 2500, max_depth=8)),
                ('rf', RandomForestClassifier(random_state=seed, 
                                              max_depth=145, 
                                              min_samples_split=5)),
                ('tabnet', ScikitTabNet(seed=seed, 
                                        n_d=39, 
                                        momentum=0.18714104672552365))]
        
        clf_list = []
        for name, clf in estimators:
            if name == 'tabnet':
                clf.fit(X_train,
                        y_train,
                        X_test, 
                        y_test)
            elif name == 'lgb':
                clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=['auc'], 
                        callbacks=[
                                lgb.early_stopping(200, first_metric_only=True, verbose=1)
                            ])
            else:
                clf.fit(X_train,y_train)
            clf_list.append(clf)
            
                    
        model = VotingClassifier(estimators=estimators, voting='soft')
        model.estimators_ = clf_list
        model.le_ = LabelEncoder().fit(y_train)
        model.classes_ = model.le_.classes_ 
        test_preds = model.predict_proba(X_test)
        five_seeds_res[seed] = roc_auc_score(y_test, test_preds[:,1])
        print(f"ROCAUC: {five_seeds_res[seed]}, split: {split}, seed: {seed}")
           
    print(f"Mean ROCAUC: {np.mean(five_seeds_res)}, split: {split}, seed: {seed}")
    return five_seeds_res

if __name__ == '__main__':
    for split in ['random','sonson','caldas']:
        feature_extraction(split)