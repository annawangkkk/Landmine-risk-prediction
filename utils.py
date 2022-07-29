import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression

import torch

import copy
import random
import os

def seed_everything(seed): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def preprocessX(X_train_init:np.ndarray, X_test_init:np.ndarray, numeric_cols):
    # fit scaler and imputer on X_train
    # transform both X_train and X_test
    imputer = KNNImputer(n_neighbors = 10, weights = 'distance')
    scaler = StandardScaler()
    X_train = copy.deepcopy(X_train_init)
    X_test = copy.deepcopy(X_test_init)
    train_imputer = imputer.fit(X_train)
    idx = np.argwhere(np.isnan(X_train).any(axis=0)).flatten()  # find the rwi column
    if len(idx) > 0:
        X_train[:,idx] = train_imputer.transform(X_train)[:,idx]
        X_test[:,idx] = train_imputer.transform(X_test)[:,idx]
    train_scaler = scaler.fit(X_train[:, 0:len(numeric_cols)])
    X_train[:, 0:len(numeric_cols)] = train_scaler.transform(X_train[:, 0:len(numeric_cols)])
    X_test[:, 0:len(numeric_cols)] = train_scaler.transform(X_test[:, 0:len(numeric_cols)])
    return X_train, X_test

def load_cv(root, split, idx):
    if split == 'random':
        cv_train = pd.read_csv(root + f'/processed_dataset/{split}/train/cv{idx}/train.csv',index_col=0)
        cv_val = pd.read_csv(root + f'/processed_dataset/{split}/train/cv{idx}/val.csv',index_col=0)
    else:
        cv_train = pd.read_csv(root + f'/processed_dataset/{split}/train/geoCV/cv{idx}/train.csv',index_col=0)
        cv_val = pd.read_csv(root + f'/processed_dataset/{split}/train/geoCV/cv{idx}/val.csv',index_col=0)
    return cv_train, cv_val

def init_cv(root, split, idx, numeric_cols, binary_cols):
    features = numeric_cols + binary_cols
    cv_train, cv_val = load_cv(root, split, idx)
    X_train_cv = np.array(cv_train.loc[:, features])
    y_train_cv = np.array(cv_train.loc[:, 'mines_outcome'])
    X_val_cv = np.array(cv_val.loc[:, features])
    y_val_cv = np.array(cv_val.loc[:, 'mines_outcome'])
    X_train_cv, X_val_cv = preprocessX(X_train_cv, X_val_cv, numeric_cols)
    return X_train_cv, y_train_cv, X_val_cv, y_val_cv

def init_train_test(root, split, numeric_cols, binary_cols, seed):
    features = numeric_cols + binary_cols
    train_labeled = pd.read_csv(root + f'/processed_dataset/{split}/train/train_labeled.csv', index_col=0)
    test_labeled = pd.read_csv(root + f'/processed_dataset/{split}/test/test_labeled.csv', index_col=0)
    X_train = np.array(train_labeled.loc[:, features])
    y_train = np.array(train_labeled.loc[:, 'mines_outcome'])
    X_test = np.array(test_labeled.loc[:, features])
    y_test = np.array(test_labeled.loc[:, 'mines_outcome'])
    X_train, X_test = preprocessX(X_train, X_test, numeric_cols)
    train = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)
    return X_train, y_train, X_test, y_test

def init_lasso(X_train_cv, y_train_cv, X_val_cv, X_train, y_train, X_test, numeric_cols, param):
    # use lasso for feature selection
    X_train_cv_preprocess, _ = preprocessX(X_train_cv, X_val_cv, numeric_cols)
    model_ftrs_cv = LogisticRegression(solver = 'saga', **param)
    model_ftrs_cv.fit(X_train_cv_preprocess, y_train_cv)
    X_train_cv_red = X_train_cv[:, np.where(model_ftrs_cv.coef_ != 0)[1]]
    X_val_cv_red = X_val_cv[:, np.where(model_ftrs_cv.coef_ != 0)[1]]
    X_train_cv, X_val_cv = preprocessX(X_train_cv_red, X_val_cv_red, numeric_cols)

    X_train_preprocess, _ = preprocessX(X_train, X_test, numeric_cols)
    model_ftrs = LogisticRegression(solver = 'saga', **param)
    model_ftrs.fit(X_train_preprocess,y_train)
    X_train_red = X_train[:, np.where(model_ftrs.coef_ != 0)[1]]
    X_test_red = X_test[:, np.where(model_ftrs.coef_ != 0)[1]]
    X_train, X_test = preprocessX(X_train_red, X_test_red, numeric_cols)             
    return X_train_cv, X_val_cv, X_train, X_test, model_ftrs_cv, model_ftrs

class FeatureNames():
    def __init__(self, dataset):
        if dataset == 'old':
            self.numeric_cols = ['elevation',
                            'settlement_dist','edu_dist', 'buildings_dist', 'railways_dist',
                            'waterways_dist', 'roads_dist', 'coca_dist']
            self.binary_cols =  ['Vocacion_Agroforestal', 'Vocacion_Agrícola',
                            'Vocacion_Conservación de Suelos', 'Vocacion_Cuerpo de agua',
                            'Vocacion_Forestal', 'Vocacion_Ganadera', 'Vocacion_Zonas urbanas']
        elif dataset == 'new':
            self.numeric_cols = ['elevation',
                        'population_2020', 
                        'No. Víctimas por Declaración', 'No. Sujetos Atención ',
                        'Acto terrorista, Atentados, Combates, Enfrentamientos, Hostigamientos',
                        'Amenaza',
                        'Delitos contra la Libertad y la Integridad sexual en desarrollo del conflicto Armado',
                        'Desaparición forzada', 'Desplazamiento Forzado', 'Homicidio, Masacre',
                        'Vincuación de Niños, Niñas y Adolescentes a actividades relacionadas con grupos armados',
                        'Secuestro', 'Tortura', 'Perdida de Bienes Muebles o Inmuebles ',
                        'Lesiones Personales Fisicas', 'Lesiones Personales Psicologicas',
                        'comimo_dist', 'airports_dist', 'seaport_dist', 'settlement_dist',
                        'finance_dist', 'edu_dist', 'buildings_dist', 'railways_dist',
                        'waterways_dist', 'roads_dist', 'coca_dist', 'rwi']
            self.binary_cols =  ['Vocacion_Agroforestal', 'Vocacion_Agrícola',
                            'Vocacion_Conservación de Suelos', 'Vocacion_Cuerpo de agua',
                            'Vocacion_Forestal', 'Vocacion_Ganadera', 'Vocacion_Zonas urbanas',
                            'CLIMA_Cuerpo de agua', 'CLIMA_Cálido húmedo',
                            'CLIMA_Cálido húmedo a muy húmedo', 'CLIMA_Cálido seco',
                            'CLIMA_Cálido seco a húmedo', 'CLIMA_Frío húmedo a muy húmedo',
                            'CLIMA_Frío húmedo y frío muy húmedo', 'CLIMA_Frío muy húmedo',
                            'CLIMA_Muy frío y muy húmedo', 'CLIMA_Templado húmedo a muy húmedo',
                            'CLIMA_Zona urbana', 'TIPO_RELIE_Cuerpo de agua',
                            'TIPO_RELIE_Espinazos', 'TIPO_RELIE_Filas y vigas',
                            'TIPO_RELIE_Glacís coluvial y coluvios de remoción',
                            'TIPO_RELIE_Glacís y coluvios de remoción',
                            'TIPO_RELIE_Lomas y colinas', 'TIPO_RELIE_Plano de inundación',
                            'TIPO_RELIE_Terrazas', 'TIPO_RELIE_Terrazas y abanicos terrazas',
                            'TIPO_RELIE_Vallecitos', 'TIPO_RELIE_Vallecitos coluvio-aluviales',
                            'TIPO_RELIE_Zona urbana', 'CLIMA_Cálido muy húmedo',
                            'CLIMA_Frío húmedo', 'CLIMA_Medio húmedo', 'CLIMA_Medio muy húmedo',
                            'CLIMA_Muy frío, pluvial',
                            'CLIMA_Tierras misceláneas, con pendientes mayores del 75%, relieve muy escarpado',
                            'TIPO_RELIE_Desde ligeramente ondulado hasta fuertemente quebrado',
                            'TIPO_RELIE_Escarpado',
                            'TIPO_RELIE_Fuertemente ondulado a fuertemente quebrado',
                            'TIPO_RELIE_Fuertemente quebrado a escarpado',
                            'TIPO_RELIE_Ligeramente ondulado a fuertemente quebrado y escarpado',
                            'TIPO_RELIE_Plano', 'TIPO_RELIE_Quebrado',
                            'TIPO_RELIE_Quebrado a escarpado',
                            'TIPO_RELIE_Quebrado a fuertemente quebrado',
                            'TIPO_RELIE_Tierras misceláneas, con pendientes mayores del 75%, relieve muy escarpado']