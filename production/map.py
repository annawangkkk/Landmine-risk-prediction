# Deprecated: use javascript leaflet to generate dynamic maps


import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle
import os
import requests

from qgis.core import *
from qgis.utils import iface
from qgis.gui import *
from qgis.PyQt.QtWidgets import QAction, QMainWindow
from qgis.PyQt.QtCore import Qt



app = Flask(__name__)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/map',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    df = pd.read_csv("mines_proba.csv")
    prediction = df["prob"]
    return render_template('index.html', prediction_text=f'{prediction}')


class MyWnd(QMainWindow):
    def __init__(self, layers):
        QMainWindow.__init__(self)

        self.canvas = QgsMapCanvas()
        self.canvas.setCanvasColor(Qt.white)
        for i in range(len(layers)):
            self.canvas.setExtent(layers[i].extent())
        self.canvas.setLayers(layers)

        self.setCentralWidget(self.canvas)

        self.actionZoomIn = QAction("Zoom in", self)
        self.actionZoomOut = QAction("Zoom out", self)
        self.actionPan = QAction("Pan", self)

        self.actionZoomIn.setCheckable(True)
        self.actionZoomOut.setCheckable(True)
        self.actionPan.setCheckable(True)

        self.actionZoomIn.triggered.connect(self.zoomIn)
        self.actionZoomOut.triggered.connect(self.zoomOut)
        self.actionPan.triggered.connect(self.pan)

        self.toolbar = self.addToolBar("Canvas actions")
        self.toolbar.addAction(self.actionZoomIn)
        self.toolbar.addAction(self.actionZoomOut)
        self.toolbar.addAction(self.actionPan)

        # create the map tools
        self.toolPan = QgsMapToolPan(self.canvas)
        self.toolPan.setAction(self.actionPan)
        self.toolZoomIn = QgsMapToolZoom(self.canvas, False) # false = in
        self.toolZoomIn.setAction(self.actionZoomIn)
        self.toolZoomOut = QgsMapToolZoom(self.canvas, True) # true = out
        self.toolZoomOut.setAction(self.actionZoomOut)

        self.pan()

    def zoomIn(self):
        self.canvas.setMapTool(self.toolZoomIn)

    def zoomOut(self):
        self.canvas.setMapTool(self.toolZoomOut)

    def pan(self):
        self.canvas.setMapTool(self.toolPan)


if __name__ == "__main__":
    #Initialize the flask App
    #model = pickle.load(open('./models/lgbm_fold4_auc0.9037129745515227.pkl', 'rb'))
    #app.run(debug=True)
    qgs = QgsApplication([], False)
    qgs.initQgis()
    

    vlayers = []
    # load all shp files
    names = ["comimo","airports","buildings","edu","finance","settlement","railway","road","seaport","water"]
    paths = ["./raw_dataset/comimo.first.zip","./raw_dataset/hotosm_col_airports_points_shp.zip",
            "./raw_dataset/hotosm_col_buildings_polygons_shp.zip","./raw_dataset/hotosm_col_education_facilities_points_shp.zip",
            "./raw_dataset/hotosm_col_financial_services_points_shp.zip","./raw_dataset/hotosm_col_populated_places_points_shp.zip",
            "./raw_dataset/hotosm_col_railways_lines_shp.zip","./raw_dataset/hotosm_col_roads_lines_shp.zip",
            "./raw_dataset/hotosm_col_sea_ports_points_shp.zip","./raw_dataset/hotosm_col_waterways_lines_shp.zip"]
    for i, name in enumerate(names):
        vlayer = QgsVectorLayer(paths[i], names[i], "ogr")
        vlayers.append(vlayer)
        if not vlayer.isValid():
            print("Layer failed to load!")
        else:
            QgsProject.instance().addMapLayer(vlayer)
            #canvas.setExtent(vlayer.extent())
    # load probability
    # load pos/neg/unlabelled
    # load rwi, pop
    proba_uri = f"file://{os.getcwd()}/processed_dataset/mines_proba.csv?delimiter=,&xField=LATITUD_X&yField=LONGITUD_Y"
    vlayer_proba = QgsVectorLayer(proba_uri, "prediction", "delimitedtext")
    QgsProject.instance().addMapLayer(vlayer_proba)
    
    # load Google Earth background
    ee_url = "mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}" 
    ee_uri = "type=xyz&zmin=0&zmax=21&url=https://"+requests.utils.quote(ee_url)
    ee_layer = QgsRasterLayer(ee_uri, 'GoogleSatellite','wms')
    tms_layer = QgsProject.instance().addMapLayer(ee_layer)

    # load OpenStreetMap
    osm_uri = 'type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0'
    osm_layer = QgsRasterLayer(osm_uri,'OSM', 'wms')
    QgsProject.instance().addMapLayer(osm_layer)

    # load terrain
    terrain_url = "mt1.google.com/vt/lyrs=t&x=%7Bx%7D&y=%7By%7D&z=%7Bz%7D"
    terrain_uri = "type=xyz&url=https://"+requests.utils.quote(terrain_url)
    terrain_layer = QgsRasterLayer(terrain_uri,'terrain', 'wms')
    QgsProject.instance().addMapLayer(terrain_layer)

    #import pdb; pdb.set_trace()
    qgs.exitQgis()