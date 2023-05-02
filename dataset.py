import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

import torch
from torch.utils.data import Dataset

from typing import *

class Event(Dataset):
    def __init__(self, 
                 train_municipios : List[str], val_municipio : str, 
                 subset : str, split : str):
        """
        Landmine dataset class.

        Args:
            train_municipios (List[str]) : training municipalities. If split != 'train', load for normalization.
            val_munipio (str) : the validation municipality.
            subset (str): full | geo | single
            split (str) : train | val
        """ 
        self.split = split
        self.val_municipio = val_municipio
        
        data_path = './processed_dataset/resolution_0.5.csv' # combined full dataset
        data = pd.read_csv(data_path)
        
        all_locations = data[['LONGITUD_X','LATITUD_Y']].to_numpy()
        all_hist_mine = data['0.5km_hist_mines'].to_numpy()
        
        if subset == 'full':

            self.numeric_cols = ['airports_dist','seaport_dist', 'settlement_dist', 'finance_dist', 'edu_dist',
                                'buildings_dist', 'waterways_dist', 'rwi', 'elevation',
                                'roads_dist', 'No. Víctimas por Declaración',
                                'retro_pobl_tot', 'indrural', 'dist_old_mine',
                                'areaoficialkm2', 'altura', 'discapital', 'pib_percapita_cons', 'hist_mines',
                                'rainfall', 'temperature', 'dist_roads_t1', 'dist_roads_t2', 'dist_roads_t3',
                                'population_2012', 'coca_dist', 'soil_texture15_trans1', 'soil_texture15_trans2',
                                'nighttime_lights_2012', 'dist_pipeline','dist_powerline','dist_telecom','dist_mining',
                                'forest_gain','forest_loss',
                                '0.5km_hist_mines','2.0km_hist_mines','8.0km_hist_mines','16.0km_hist_mines','32.0km_hist_mines',
                                'animal_inc','animal_dec'] 

            self.binary_cols = ['binary_hist_mine',
                                'land_use_Agroforestal', 'land_use_Agrícola',
                                'land_use_Conservación de Suelos', 'land_use_Cuerpo de agua',
                                'land_use_Forestal', 'land_use_Ganadera', 'land_use_Zonas urbanas',
                                'weather_Cuerpo de agua', 'weather_Cálido húmedo',
                                'weather_Cálido húmedo a muy húmedo', 'weather_Cálido seco a húmedo',
                                'weather_Frío húmedo a muy húmedo',
                                'weather_Frío húmedo y frío muy húmedo', 'weather_Frío muy húmedo',
                                'weather_Muy frío y muy húmedo', 'weather_Templado húmedo a muy húmedo',
                                'weather_Zona urbana', 'relief_Cuerpo de agua', 'relief_Espinazos',
                                'relief_Filas y vigas', 'relief_Glacís coluvial y coluvios de remoción',
                                'relief_Glacís y coluvios de remoción', 'relief_Lomas y colinas',
                                'relief_Terrazas y abanicos terrazas', 'relief_Vallecitos',
                                'relief_Vallecitos coluvio-aluviales', 'relief_Zona urbana']
        elif subset == 'geo':

            self.numeric_cols = ['airports_dist','seaport_dist', 'settlement_dist', 'finance_dist', 'edu_dist',
                                'buildings_dist', 'waterways_dist', 'rwi', 'elevation',
                                'roads_dist', 'No. Víctimas por Declaración',
                                'retro_pobl_tot', 'indrural', 
                                'areaoficialkm2', 'altura', 'discapital', 'pib_percapita_cons', 'hist_mines',
                                'rainfall', 'temperature', 'dist_roads_t1', 'dist_roads_t2', 'dist_roads_t3',
                                'population_2012', 'coca_dist', 'soil_texture15_trans1', 'soil_texture15_trans2',
                                'nighttime_lights_2012', 'dist_pipeline','dist_powerline','dist_telecom','dist_mining',
                                'forest_gain','forest_loss',
                                'animal_inc','animal_dec'] 

            self.binary_cols = ['land_use_Agroforestal', 'land_use_Agrícola',
                                'land_use_Conservación de Suelos', 'land_use_Cuerpo de agua',
                                'land_use_Forestal', 'land_use_Ganadera', 'land_use_Zonas urbanas',
                                'weather_Cuerpo de agua', 'weather_Cálido húmedo',
                                'weather_Cálido húmedo a muy húmedo', 'weather_Cálido seco a húmedo',
                                'weather_Frío húmedo a muy húmedo',
                                'weather_Frío húmedo y frío muy húmedo', 'weather_Frío muy húmedo',
                                'weather_Muy frío y muy húmedo', 'weather_Templado húmedo a muy húmedo',
                                'weather_Zona urbana', 'relief_Cuerpo de agua', 'relief_Espinazos',
                                'relief_Filas y vigas', 'relief_Glacís coluvial y coluvios de remoción',
                                'relief_Glacís y coluvios de remoción', 'relief_Lomas y colinas',
                                'relief_Terrazas y abanicos terrazas', 'relief_Vallecitos',
                                'relief_Vallecitos coluvio-aluviales', 'relief_Zona urbana']
        
        elif subset == 'single':
            self.numeric_cols = ['dist_old_mine']
            self.binary_cols = []
        
        self.features = self.numeric_cols + self.binary_cols
        
        tabX = pd.get_dummies(columns=['land_use', 'weather', 'relief'], data = data)[self.features]
        if val_municipio == 'RANDOM' or val_municipio == 'PUERTO LIBERTADOR': # train_val split + test
            train_tabX_combined = tabX.loc[list(data[data['Municipio'].isin(train_municipios)].index),self.features]
            np.random.RandomState(737)
            train_idx = pd.Index(np.random.choice(train_tabX_combined.index, int(len(train_tabX_combined)*0.7), replace=False))
            train_tabX = train_tabX_combined.loc[train_idx]
            if val_municipio == 'RANDOM':
                val_tabX = train_tabX_combined.loc[~train_tabX_combined.index.isin(train_idx)]
            elif val_municipio == 'PUERTO LIBERTADOR':
                val_tabX = tabX.loc[list(data[data['Municipio'] == val_municipio].index),self.features]
        else:
            train_tabX = tabX.loc[list(data[data['Municipio'].isin(train_municipios)].index),self.features]
            val_tabX = tabX.loc[list(data[data['Municipio'] == val_municipio].index),self.features]
        
        imputer = KNNImputer(n_neighbors = 4, weights = 'distance')
        train_imputer = imputer.fit(train_tabX)
        if len(np.where(np.isnan(train_tabX).any(axis=0))[0]) != 0:
            idx = np.where(np.isnan(train_tabX).any(axis=0))[0][0]  # find the nan column
            train_tabX.iloc[:,idx] = train_imputer.transform(train_tabX)[:,idx]
            val_tabX.iloc[:,idx] = train_imputer.transform(val_tabX)[:,idx]
        
        scaler = StandardScaler()
        train_scaler = scaler.fit(train_tabX[self.numeric_cols])
        train_tabX[self.numeric_cols] = train_scaler.transform(train_tabX[self.numeric_cols])
        val_tabX[self.numeric_cols] = train_scaler.transform(val_tabX[self.numeric_cols])
    
        if self.split == 'val':
            if val_municipio == 'RANDOM':
                # all train - train_idx
                val_idx = train_tabX_combined.index[~train_tabX_combined.index.isin(train_idx)]
                self.locations = all_locations[list(val_idx)]
                self.y = data.loc[val_idx,'mines_outcome'].to_numpy()
                self.tabX = val_tabX.to_numpy()
                self.hist_mine = all_hist_mine[list(val_idx)]
            else:
                self.locations = all_locations[list(data[data['Municipio'] == val_municipio].index)]
                self.y = (data.loc[data['Municipio'] == val_municipio,'mines_outcome']).to_numpy()
                self.tabX = val_tabX.to_numpy()
                self.hist_mine = all_hist_mine[list(data[data['Municipio'] == val_municipio].index)]
        elif self.split == 'train': 
            if val_municipio == 'RANDOM' or val_municipio == 'PUERTO LIBERTADOR':
                self.locations = all_locations[list(train_idx)]
                self.y = data.loc[train_idx,'mines_outcome'].to_numpy()
                self.tabX = train_tabX.to_numpy()
                self.hist_mine = all_hist_mine[list(train_idx)]
            else:
                self.locations = all_locations[list(data[data['Municipio'].isin(train_municipios)].index)]
                self.y = (data.loc[data['Municipio'].isin(train_municipios),'mines_outcome']).to_numpy()
                self.tabX = train_tabX.to_numpy()
                self.hist_mine = all_hist_mine[list(data[data['Municipio'].isin(train_municipios)].index)]
       
        # for ood bench
        self.samples = [(self.tabX[i], self.y[i])for i in range(len(self.y))]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        lon, lat = self.locations[idx]
        hist_mine = self.hist_mine[idx]
        label = self.y[idx]
        tab_data = self.tabX[idx,:]
        return  (torch.tensor(tab_data).float(),\
                torch.tensor(label).float(), \
                torch.tensor((lon, lat)).float(), \
                torch.tensor(hist_mine).float())
