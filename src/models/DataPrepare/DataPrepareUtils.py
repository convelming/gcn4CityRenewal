import ast
import random
import geopandas as gpd

import osmnx as ox
import pandas as pd
import math
import pickle

from sklearn.cluster import KMeans

def count_points_in_polygons(gpd_points,gpd_polygons,polygon_col,new_col_name):
    gpd_points = gpd_points[['geometry']].drop_duplicates()
    gpd_points = gpd_points.to_crs(gpd_polygons.crs)
    df_join = gpd.sjoin(gpd_points, gpd_polygons, how='left', predicate='within')
    df_vc = pd.DataFrame()
    df_vc[polygon_col] = df_join[polygon_col].value_counts().index
    df_vc[new_col_name] = df_join[polygon_col].value_counts().values
    return(df_vc)

def count_number_in_polygons(gpd_points,gpd_polygons,polygon_col,cal_num_col,new_col_name):
    gpd_points = gpd_points[['geometry',cal_num_col]].drop_duplicates()
    gpd_points = gpd_points.to_crs(gpd_polygons.crs)
    df_join = gpd.sjoin(gpd_points, gpd_polygons, how='left', predicate='within')
    df_vc = df_join.groupby([polygon_col])[cal_num_col].sum().reset_index()
    df_vc.columns = [polygon_col,new_col_name]
    return(df_vc)



