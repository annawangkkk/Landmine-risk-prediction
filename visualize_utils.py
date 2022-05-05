import matplotlib.pyplot as plt
from pdpbox import pdp
import geopandas as gpd

def plot_PDP(model, model_name, df, selected_feature_names, save_path): 
    '''
        df: output DataFrame with predicted probabilities and all model selected features
    '''
    # make sure your matplotlib version == 3.2.2
    for ii in range(len(selected_feature_names)):
        pdp_model = pdp.pdp_isolate(
            model=model, dataset=df[selected_feature_names], model_features=selected_feature_names, 
            feature=selected_feature_names[ii]
        )
        try:
            pdp.pdp_plot(
                pdp_model, selected_feature_names[ii], plot_pts_dist=True,
                frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True
            )
            pdp.plt.savefig(save_path + f"/{model_name}_{selected_feature_names[ii]}_pdp.png")
            pdp.plt.clf()
        except: # some bugs in pdp package, sometimes needs to rerun the code again
            print(f"Something went wrong while plotting {selected_feature_names[ii]}")
            continue

def plot_heatmap(df, model_name, save_path): 
    '''
        df: output DataFrame with predicted probabilities and all model selected features
    '''
    mines_grid = gpd.GeoDataFrame(df.drop(columns = 'geometry'), 
                                  geometry = gpd.points_from_xy(df['LONGITUD_X'], df['LATITUD_Y']), 
                                  crs = 'EPSG:4326')
    distancia = 500 # esto es lado/2 (en metros)
    grilla_buffer = (mines_grid.to_crs(epsg=3395).buffer(distancia, cap_style=3).to_crs(epsg=4326))
    grilla_buffer = gpd.GeoDataFrame(grilla_buffer)
    grilla_buffer = grilla_buffer.set_geometry(0)
    grilla_buffer['pred'] = df['prob']
    grilla_buffer.plot(column = 'pred', figsize = (12,8), cmap = 'YlOrRd')
    plt.savefig(save_path + f"/{model_name}_heatmap.png")
    plt.clf()

def plot_auc_bce(table, model_name, save_path):
    '''
        table: dataframe generated from select_best_C()
    '''
    _, ax = plt.subplots(figsize=(10,8))
    ax.plot(table['Num. features'], table['Train wBCE'], label = "Train wBCE")
    ax.plot(table['Num. features'], table['Test wBCE'], label = "Test wBCE")
    ax.set_title(f'Cross validation Metrics - Logit + {model_name} model', fontsize = 15)
    ax.set_ylabel('weighted BCE (w = 50)', fontsize = 13)
    ax.set_xlabel('# features used', fontsize = 13)
    ax.legend()
    ax2 = ax.twinx()
    roc_auc = []
    roc_auc.append(ax2.plot(table['Num. features'], table['Train ROC_AUC'], "r-", label = "Train ROC_AUC"))
    roc_auc.append(ax2.plot(table['Num. features'], table['Test ROC_AUC'], "g-", label = "Test ROC_AUC"))
    ax2.set_ylabel('ROC_AUC', fontsize = 13)
    ax2.legend(loc=0)
    plt.savefig(save_path + f'/{model_name}_feature_selection.png')
    plt.clf()

def plot_proba_dist(df, model_name, save_path):
    '''
        df: output DataFrame with predicted probabilities and all model selected features
    '''
    ax = df["prob"].hist()
    fig = ax.get_figure()
    ax.set_xlabel('Probability')
    ax.set_ylabel('Counts')
    ax.set_title(model_name)
    fig.savefig(save_path + f'/{model_name}_distribution.jpg')
    plt.clf()