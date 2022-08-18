import pandas as pd
import plotly.graph_objects as go
# pip install dash (version 2.0.0 or higher)
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import json
from dash.exceptions import PreventUpdate

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'RELand'
# -- Import and clean data (importing csv into pandas)
# -- Import and clean data (importing csv into pandas)
df = pd.read_csv(
    'https://raw.githubusercontent.com/annawangkkk/minesWeb/main/website_table.csv')
# df = pd.read_csv('website_table.csv')
area_list = df['Municipio'].unique()

# ------------------------------------------------------------------------------
# App layout
app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(html.H1("RELand: Risk Estimator of Landmines",
                className='text-center text-primary mb-4'), width=12)
    ),

    html.Hr(),

    dbc.Row([
        # dbc.Col(html.H5("Choose The Model")),
        dbc.Col(html.H5("Choose The Map Style")),
        dbc.Col(html.H5("Choose The Areas"))

    ]),


    # sonson_avg
    dbc.Row([

        dbc.Col([
            dcc.Dropdown(id="layer",
                         options=[
                             {"label": "Street", "value": 'streets'},
                             {"label": "Satellite Streets", "value": 'satellite'},
                             {"label": "Outdoors", "value": 'outdoors'}],
                         multi=False,
                         value='streets')],
                ),
        dbc.Col([
            dcc.Dropdown(id="area",
                         value=[x for x in area_list],
                         placeholder="Select a region",
                         options=[{'value': x, 'label': str(x)} for x in area_list], multi=True),
            html.Div([dcc.Checklist(id='select-all-regions',
                                    options=[{'label': 'Show All Regions', 'value': 1}], value=[1])], id='checklist-container')

        ]),

    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col(html.H5("Risk Prediction")),
        # search bar for searching address
        dbc.Col([dbc.Row([dbc.Textarea(id='address-search-tab1',
                                       placeholder='Search for the street or area'),
                 dbc.Button(
                     'Find', id='search-address-button-tab1', n_clicks=0),
                 # html.P(id='no-result-alert')
                 ])
                 ], width=7)]),
    dbc.Row([
        dbc.Col([
            dbc.Row(dbc.Checklist(
                options=[
                    {
                        "label": "Show the landmine risk prediction by RELand system",
                        "value": "sonson_avg",
                        "label_id": "showPrediction"
                    }, ],
                value=['sonson_avg'],
                id="model"
            )),
            html.Hr(),
            dbc.Row(html.H5("Historical Events")),
            dbc.Row([
                    dbc.Checklist(
                        id="all-or-none",
                        options=[{"label": "Select All", "value": "All"}],
                        value=[],
                        labelStyle={"display": "inline-block"},
                    ),

                    dbc.Checklist(id="recycling_type",
                                  value=[''],
                                  options=[
                                      {'value': -1, 'label': 'Areas with no historical data',
                                       'label_id': 'unknown'},
                                      {'value': 0, 'label': 'Areas declared mine-free',
                                       'label_id': 'negative'},
                                      {'value': 1, 'label': 'Areas affected by landmines',
                                       'label_id': 'positive'},
                                  ],)
                    ]),
            html.Hr(),
            dbc.Row(html.H5("Danger Zones")),
            dbc.Row(html.P('For the areas with no historical data.'),
                    style={'font-size': '13px'}),
            dbc.Row([
                dbc.Checklist(
                    id="all-or-none-clusters",
                    options=[{"label": "Select All", "value": "All"}],
                    value=[],
                    labelStyle={"display": "inline-block"},
                ),
                dbc.Checklist(id="risk_clusters", value=[''],
                              options=[
                              {'value': 2, 'label': 'Low Risk region predicted by RELand system',
                                  'label_id': 'low'},
                              {'value': 0, 'label': 'Medium Risk region predicted by RELand system',
                                  'label_id': 'medium'},
                              {'value': 1, 'label': 'High Risk region predicted by RELand system',
                                  'label_id': 'high'},
                              ]),


            ]),

        ], width=5),


        dbc.Col(dcc.Graph(id='map', figure={}), width=7)
    ]),

    html.Hr(),

    dbc.Row(html.P('Disclaimer: The information presented here should be taken only as an estimate resulting from academic research and should be analyzed with other expert knowledge to determine the actual risk of landmines in the studied area in a human-in-the-loop approach.'),
            style={'font-size': '13px'}),
    html.Br(),

])


@ app.callback(
    Output('area', 'value'),
    [Input('select-all-regions', 'value')],
    [State('area', 'options')])
def showAllRegions(selected, options):
    if len(selected) > 0:
        return [i['value'] for i in options]
    elif len(selected) == 0:
        return []
    raise PreventUpdate()

# ------------------------------------------------------------------------------


@ app.callback(
    Output(component_id='map', component_property='figure'),
    [Input('area', 'value'),
     Input(component_id='model', component_property='value'),
     Input(component_id='layer', component_property='value'),
     Input('recycling_type', 'value'),
     Input('search-address-button-tab1', 'n_clicks'),
     Input('risk_clusters', 'value')],
    State('address-search-tab1', 'value')
)
def update_graph(area, option_slctd, layer, chosen_label, n_clicks, chosen_cluster, address_search_1):
    print(option_slctd)
    mapbox_access_token = 'pk.eyJ1IjoicWl3YW5nYWFhIiwiYSI6ImNremtyNmxkNzR5aGwyb25mOWxocmxvOGoifQ.7ELp2wgswTdQZS_RsnW1PA'

    df_1 = df[(df['Municipio'].isin(area))]
    df_1_prediction = df_1[(df_1['option'].isin(option_slctd))]

    colorList = ['lightgrey', 'lime', 'red']
    labelList = [-1, 0, 1]
    # -1: unknown; 0:negative; 1:positive
    df_1_unknow = df_1.loc[df_1['mines_outcome'] == -1]

    for item in zip(labelList, colorList):
        df_1.loc[df_1['mines_outcome'] == item[0],
                 'colorBasedLabel'] = item[1]

    colorList_custer = ['cyan', 'rgb(245, 221, 43)', 'orangered']
    clusterList = [2, 0, 1]

    for item in zip(clusterList, colorList_custer):
        df_1_unknow.loc[df_1_unknow['cluster'] == item[0],
                        'colorBasedCluster'] = item[1]

    df_sub = df_1[(df_1['mines_outcome'].isin(chosen_label))]

    # df.loc[df['col1'] == value]

    df_sub_cluster = df_1_unknow[(
        df_1_unknow['cluster'].isin(chosen_cluster))]

    # scl = [0, "rgb(150,0,90)"], [0.125, "rgb(0, 0, 200)"], [0.25, "rgb(0, 25, 255)"],\
    #     [0.375, "rgb(0, 152, 255)"], [0.5, "rgb(44, 255, 150)"], [0.625, "rgb(151, 255, 0)"],\
    #     [0.75, "rgb(255, 234, 0)"], [0.875, "rgb(255, 111, 0)"], [
    #     1, "rgb(255, 0, 0)"]

    scl = [(0, "red"), (0.5, "yellow"), (1, "rgb(28,238,238)")]
    # Plotly Express
    locations = [

        go.Scattermapbox(
            lat=df_1_prediction['LATITUD_Y'],
            lon=df_1_prediction['LONGITUD_X'],
            customdata=df_1['sonson_avg'],
            hovertext=df_1['sonson_avg'],

            hovertemplate='<br>Locations: (%{lat},%{lon})</br>Prediction Risk: %{customdata} <extra></extra>',
            marker=dict(
                color=df_1['sonson_avg'],
                colorscale=scl,
                reversescale=True,
                size=8,
                colorbar=dict(
                    titleside="right",
                    outlinecolor="rgba(68, 68, 68, 0)",
                    ticks="outside",
                    showticksuffix="last",
                    dtick=0.1
                )
            )
        ),
        # the layer of ground truth lable
        go.Scattermapbox(
            lat=df_sub['LATITUD_Y'],
            lon=df_sub['LONGITUD_X'],
            hovertemplate='<br>True Label Layer</br>Locations: (%{lat},%{lon})<extra></extra>',
            marker={'size': 10, 'color': df_sub['colorBasedLabel']}
        ),
        # the layer of cluster
        go.Scattermapbox(
            lat=df_sub_cluster['LATITUD_Y'],
            lon=df_sub_cluster['LONGITUD_X'],
            hovertemplate='<br>Risk Cluster</br>Locations: (%{lat},%{lon})<extra></extra>',
            marker={'size': 10, 'color': df_sub_cluster['colorBasedCluster']}
        ),

    ]

    if n_clicks == 0:
        layout = go.Layout(
            uirevision='foo',  # preserves state of figure/map after callback activated
            # clickmode='event+select',
            hovermode='closest',
            hoverdistance=2,
            showlegend=False,
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),

            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=25,
                style='{}'.format(layer),
                center=dict(
                    lat=5.920689177,
                    lon=-75.10525796
                ),
                pitch=40,
                zoom=9
            ),
        )

        return{
            'data': locations,
            'layout': layout
        }

    else:
        # Geocode the lat-lng using Google Maps API
        google_api_key = 'AIzaSyDitOkTVs4g0ibg_Yt04DQqLaUYlxZ1o30'

        # Adding Uniontown PA to make the search more accurate (to generalize)
        address_search = address_search_1 + ' Antioquia, Colombia'

        params = {'key': google_api_key,
                  'address': address_search}

        url = 'https://maps.googleapis.com/maps/api/geocode/json?'

        response = requests.get(url, params)
        result = json.loads(response.text)

        if result['status'] not in ['INVALID_REQUEST', 'ZERO_RESULTS']:

            lat = result['results'][0]['geometry']['location']['lat']
            lon = result['results'][0]['geometry']['location']['lng']

            layout = go.Layout(
                uirevision=address_search,
                hovermode='closest',
                hoverdistance=2,
                showlegend=False,
                autosize=True,
                margin=dict(l=0, r=0, t=0, b=0),

                mapbox=dict(
                    accesstoken=mapbox_access_token,
                    bearing=25,
                    style='{}'.format(layer),
                    center=dict(
                        lat=lat,
                        lon=lon
                    ),
                    pitch=40,
                    zoom=15
                ),
            )
            return {
                'data': locations,
                'layout': layout
            }

        else:
            layout = go.Layout(
                uirevision='foo',  # preserves state of figure/map after callback activated
                # clickmode='event+select',
                hovermode='closest',
                hoverdistance=2,
                showlegend=False,
                autosize=True,
                margin=dict(l=0, r=0, t=0, b=0),

                mapbox=dict(
                    accesstoken=mapbox_access_token,
                    bearing=25,
                    style='{}'.format(layer),
                    center=dict(
                        lat=5.920689177,
                        lon=-75.10525796
                    ),
                    pitch=40,
                    zoom=9
                ),
            )

        return{
            'data': locations,
            'layout': layout
        }

# select all for labels


@app.callback(
    Output("recycling_type", "value"),
    [Input("all-or-none", "value")],
    [State("recycling_type", "options")],
)
def select_all_none(all_selected, options):
    all_or_none = []
    all_or_none = [option["value"] for option in options if all_selected]
    return all_or_none

# select all for cluster


@app.callback(
    Output("risk_clusters", "value"),
    [Input("all-or-none-clusters", "value")],
    [State("risk_clusters", "options")],
)
def select_all_none_clusters(all_selected, options):
    all_or_none = []
    all_or_none = [option["value"] for option in options if all_selected]
    return all_or_none


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
