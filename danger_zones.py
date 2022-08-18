import pandas as pd
import numpy as np

import geopandas as gpd
import libpysal as lps
import esda

from prepare_data import create_grid_buffer


def get_danger_zones(df, col_pred, significance = 0.01):

    grid_buffer = create_grid_buffer(df)
    grid_buffer['pred'] = df[col_pred]
    
    wq = lps.weights.KNN.from_dataframe(grid_buffer, k=8)
    wq.transform = 'r'
    pred_lag = lps.weights.lag_spatial(wq, grid_buffer['pred'])

    local_moran = esda.moran.Moran_Local(grid_buffer['pred'], wq)
    hotspots = (local_moran.q == 1) & (local_moran.p_sim <= significance)
    coldspots = (local_moran.q == 3) & (local_moran.p_sim <= significance)
    df['cluster'] = [(hotspots*1)[j] + (coldspots*2)[j] for j in range(len(local_moran.q))]
    
    return df
    

if __name__ == '__main__':
    root='~/social_good/Landmine-risk-prediction'
    final_pred = pd.read_csv(root + f'/processed_dataset/results/prepare_website_overfit.csv')
    final_pred['final_pred'] = final_pred[[f'sonson_seed{x}' for x in range(5)]].mean(axis=1)
    df = get_danger_zones(df, 'final_pred')
    df.to_csv(root + f'/processed_dataset/results/prepare_website_overfit.csv')