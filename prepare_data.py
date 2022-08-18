import geopandas as gpd
import pandas as pd
import numpy as np

from shapely import ops
from shapely.geometry import Point, Polygon

from itertools import product
from sklearn.neighbors import NearestNeighbors



def get_grid_centers(polygon, side_size = 1, fmin_y=None):
    puntos = []
    puntos_lon, puntos_lat = [], []
    min_x, min_y, max_x, max_y = polygon.bounds
    
    x = min_x
    if fmin_y:
        min_y = fmin_y
    y = min_y
    
    delta = side_size * 0.007 / 0.78
    while x <= max_x:
        while y <= max_y:
            next_point = Point(x, y)
            
            if next_point.within(polygon):
                puntos.append(next_point)
                puntos_lon.append(next_point.x)
                puntos_lat.append(next_point.y)
            y = y + delta
        y = min_y
        x = x + delta
    grilla = gpd.GeoDataFrame(
        {"LONGITUD_X": puntos_lon, "LATITUD_Y": puntos_lat, "geometry": puntos},
        crs="EPSG:4326",
    )

    return grilla


def create_grid_buffer(df, distance = 500):
    
    mines_grid = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['LONGITUD_X'], df['LATITUD_Y']), crs = 'EPSG:4326')
    
    grid_buffer = (mines_grid.to_crs(epsg=3395).buffer(distance, cap_style=3).to_crs(epsg=4326))
    grid_buffer = gpd.GeoDataFrame(grid_buffer)
    grid_buffer = grid_buffer.set_geometry(0)
    
    return grilla_buffer


def create_area(our_area):
    
    mpios = gpd.read_file(root + "../datos/mpio.zip")
    mpios = mpios.to_crs('EPSG:4326')
    our_mpios = mpios[mpios['NOMBRE_DPT'].isin(['ANTIOQUIA', 'CALDAS'])]
    
    our_mpios = our_mpios[our_mpios['NOMBRE_MPI'].isin(our_area)].to_crs('EPSG:4326')
    polygon = ops.unary_union(our_mpios['geometry'])
    
    return our_mpios, get_grid_centers(polygon, side_size = 1)
    
    
def assign_events(grid, events):
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric='euclidean')
    nbrs = nbrs.fit(grid[['LATITUD_Y', 'LONGITUD_X']])
    
    events.rename(columns = {'longitud_c': 'LONGITUD_X', 'latitud_ca': 'LATITUD_Y'}, inplace = True)
    cells = nbrs.kneighbors(events[['LATITUD_Y', 'LONGITUD_X']], return_distance=False)[:, 0]
    
    grid = grid.merge(pd.Series(celdas).value_counts().rename('mines_events'), left_index=True, right_index=True, how='left')
    grid['mines_events'] = grid['mines_events'].fillna(0)
    
    return grid


def assign_hazard_areas(grid, hazard):
    
    grid_buffer = create_grid_buffer(grid)
    celdas_hazard = grilla_buffer.reset_index().overlay(hazard)['index']
    
    grid = grid.merge(celdas_hazard.value_counts().rename('hazard'), left_index=True, right_index=True, how='left')
    grid['hazard'] = grid['hazard'].fillna(0)
    
    return grid


def assign_cleared_areas(grid, cleared):
    
    grid_buffer = create_grid_buffer(grid)
    celdas_cleaned = grilla_buffer.reset_index().overlay(cleared)['index']
    
    grid = grid.merge(celdas_cleaned.value_counts().rename('cleared'), left_index=True, right_index=True, how='left')
    grid['cleared'] = grid['cleared'].fillna(0)
    
    return grid


def assign_mine_free(grid, mine_free):
    
    grid_buffer = create_grid_buffer(grid)
    celdas_sectores = grilla_buffer.reset_index().overlay(our_sectores)['index']
    
    grid = grid.merge(celdas_sectores.value_counts().rename('sectores_done'), left_index=True, right_index=True, how='left')
    grid['sectores_done'] = grid['sectores_done'].fillna(0)
    
    return grid


def assign_labels(grid, events, hazard, cleared, mine_free):
    
    grid = assign_events(grid, events)
    grid = assign_hazard_areas(grid, hazard)
    grid = assign_cleared_areas(grid, cleared)
    grid = assign_mine_free(grid, mine_free)
    
    grid['mines_outcome'] = -1*((grid['mines_events'] == 0) & (grid['hazard'] == 0) & (grid['cleaned'] == 0) & (grid['sectores_done'] == 0))
    grid.loc[grid['mines_outcome'] != -1, 'mines_outcome'] = 1*((grid.loc[grid['mines_outcome'] != -1, 'mines_events'] > 0) | (grid.loc[grid['mines_outcome'] != -1, 'hazard'] > 0) | (grid.loc[grid['mines_outcome'] != -1, 'cleaned'] > 0))
    
    return grid
    
    
def prepare_antioquia():

    our_area = ['SAN CARLOS', 'SAN RAFAEL', 'GRANADA', 'SAN LUIS', 'GUATAPE' , 'SAN FRANCISCO', 
            'SONSON', 'EL CARMEN DE VIBORAL', 'COCORNA', 'ALEJANDRIA', 'ARGELIA',
            'EL SANTUARIO', 'PE¥OL', 'SAN VICENTE', 
            'NARI¥O', 'PUERTO TRIUNFO', 'PUERTO NARE', 'ABEJORRAL', 'LA UNION'] 
    
    our_mpios, grid = create_area(our_area)
    
    events = gpd.read_file(root + '../datos/eventos_map/geo_export_4cd0f640-76d8-4ba9-abe1-084455ba1d7b.shp')
    events = events[(events['municipio'].isin(['SAN CARLOS', 'SAN RAFAEL', 'GRANADA', 'SAN LUIS', 'GUATAP?' , 'SAN FRANCISCO', 'SONS?N', 'CARMEN DE VIBORAL', 'COCORN?', 'ALEJANDR?A', 'ARGELIA','SANTUARIO', 'PE?OL', 'SAN VICENTE', 'NARI?O', 'PUERTO TRIUNFO', 'PUERTO NARE', 'ABEJORRAL', 'LA UNIÓN'])) & (events['departamen'] == 'ANTIOQUIA')].to_crs('EPSG:4326')
    events = our_mpios.overlay(events, how = 'intersection', keep_geom_type = False)
    
    
    hazard = gpd.read_file(root + "../datos/areas_peligrosas_ant.geojson")
    hazard = hazard[hazard['Municipio'].isin(['SAN CARLOS', 'SAN RAFAEL', 'GRANADA', 'SAN LUIS', 'GUATAPE', 'SAN FRANCISCO', 'SONSÓN', 'CARMEN DE VIBORAL', 'COCORNÁ', 'ALEJANDRIA', 'ARGELIA', 'NARIÑO', 'ABEJORRAL', 'SAN VICENTE'])].to_crs('EPSG:4326')
    hazard = our_mpios.overlay(hazard, how = 'intersection', keep_geom_type = False)
    
    cleared = gpd.read_file(root + "../datos/estudiot_despeje_ant.geojson")
    cleared['Municipio'] = cleared['Municipio'].str.strip()
    cleared = cleared[(cleared['map'] > 0) | (cleared['muse'] > 0)]
    cleared = cleared[cleared['Municipio'].isin(['SAN CARLOS', 'SAN RAFAEL', 'GRANADA', 'SAN LUIS', 'GUATAPE', 'SAN FRANCISCO', 'SONSÓN', 'CARMEN DE VIBORAL', 'COCORNÁ', 'ALEJANDRÍA', 'ARGELIA', 'NARIÑO', 'SAN VICENTE', 'ABEJORRAL'])].to_crs('EPSG:4326')
    cleared = our_mpios.overlay(cleared, how = 'intersection', keep_geom_type = False)
    
    mine_free = gpd.read_file(root + "../datos/sectores_desminado.geojson")
    mine_free = mine_free[(mine_free['Departamento'] == 'ANTIOQUIA') & (sectores['Status'] == 'Finalizada')]
    
    return assign_labels(grid, events, hazard, cleared, mine_free)


def prepare_caldas():

    our_area = ['NORCASIA', 'SAMANA', 'PENSILVANIA'] 
    
    our_mpios, grid = create_area(our_area)
    
    events = gpd.read_file(root + '../datos/eventos_map/geo_export_4cd0f640-76d8-4ba9-abe1-084455ba1d7b.shp')
    events = events[(events['municipio'].isin(['NORCASIA', 'SAMAN?', 'PENSILVANIA'])) & (events['departamen'] == 'CALDAS')].to_crs('EPSG:4326')
    events = our_mpios.overlay(events, how = 'intersection', keep_geom_type = False)
    
    hazard = gpd.read_file(root + "../datos/test_caldas/peligrosas_caldas.geojson")
    hazard = our_mpios.overlay(hazard, how = 'intersection', keep_geom_type = False)
    
    cleared = gpd.read_file(root + "../datos/test_caldas/estudiot_despeje_caldas.geojson")
    cleared['Municipio'] = cleared['Municipio'].str.strip()
    cleared = cleared[(cleared['map'] > 0) | (cleared['muse'] > 0)]
    cleared = cleared[cleared['Municipio'].isin(['NORCASIA', 'SAMANÁ', 'PENSILVANIA'])].to_crs('EPSG:4326')
    cleared = our_mpios.overlay(cleared, how = 'intersection', keep_geom_type = False)
    
    mine_free = gpd.read_file(root + "../datos/test_caldas/sectores_caldas.geojson")
    mine_free = mine_free[(mine_free['Departamento'] == 'CALDAS') & (sectores['Status'] == 'Finalizada')]
    
    return assign_labels(grid, events, hazard, cleared, mine_free)





if __name__ == '__main__':
    root = '~/social_good/Landmine-risk-prediction'
    
    grid_antioquia = prepare_antioquia()
    grid_caldas = prepare_caldas()
    
    grid_all = grid_antioquia.merge(grid_caldas)
    
    grid_all.to_csv(root + '/processed_dataset/all/all.csv', index = False)