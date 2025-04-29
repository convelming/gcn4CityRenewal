import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd


gdf_node = gpd.read_file("../data/base_data/guangzhou_drive.gpkg",layer='nodes')
gdf_edge = gpd.read_file("../data/base_data/guangzhou_drive.gpkg",layer='edges')
df_pop = pd.read_csv('../data/base_data/base_pop.csv')
df_net = gpd.read_file('../data/base_data/voronoi_gz.shp')
df_landuse = pd.read_csv('../data/base_data/base_landuse.csv')
df_road = pd.read_csv('../data/base_data/base_road.csv')
df_poi = pd.read_csv('../data/base_data/base_poi.csv')
df_facility = pd.read_csv('../data/base_data/base_facility_area.csv')


node_id_col_name = 'osmid'
df_landuse = df_landuse[[node_id_col_name,'traffic','public','business','resident','industry']]
df_input = pd.merge(gdf_node,df_pop,on=[node_id_col_name],how='left')
df_input = df_input.fillna(0.00001)
df_input = pd.merge(df_input,df_landuse,on=[node_id_col_name],how='left')
df_input = pd.merge(df_input,df_road,on=[node_id_col_name],how='left')
df_input = pd.merge(df_input,df_poi,on=[node_id_col_name],how='left')

df_input = pd.merge(df_input,df_facility,on=[node_id_col_name],how='left')
df_input = df_input.fillna(0)
df_input.to_csv('../data/base_data/InputFeature.csv',index=False)

df_input = pd.read_csv('../data/base_data/InputFeature.csv')
gdf_node = gpd.read_file("../data/base_data/guangzhou_drive.gpkg",layer='nodes')
list_feature_col = df_input.columns.to_list()[len(gdf_node.columns.to_list()):]

gr = ox.load_graphml('../data/base_data/guangzhou_drive.graphml')
for i in range(len(gdf_node)):
    if i % 1000 == 0:
        print(i,len(gdf_node))
    nx.set_node_attributes(gr,{df_input[node_id_col_name][i]:{"features":list(df_input[list_feature_col][i:i+1].values[0])}} )
ox.save_graph_geopackage(gr, "../data/base_data/guangzhou_drive_feature.gpkg")
ox.save_graphml(gr, "../data/base_data/guangzhou_drive_feature.graphml")