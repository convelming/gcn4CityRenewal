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



def cluster_dis(dis):
    """每5km的区域进行分类"""
    if dis <= 5000:
        return (1)
    elif dis <= 10000:
        return (2)
    elif dis <= 15000:
        return (3)
    elif dis <= 20000:
        return (4)
    elif dis <= 30000:
        return (5)
    else:
        return (6)

def sum_feature(df_data, clus_col='o_id', feature_col='features'):
    """将区域内的feature进行求和"""
    df_feature = df_data.copy()
    if type(df_feature[feature_col].values[0]) != list:
        df_feature[feature_col] = df_feature[feature_col].apply(ast.literal_eval)
    df_sum = (
        df_feature.groupby(clus_col)[feature_col]
        .apply(lambda x: pd.DataFrame(x.tolist()).sum().tolist())
        .reset_index(name=feature_col)
    )
    return (df_sum)


def get_o_od(clu_id_o, clu_id_d, df_od , o_id_col = 'source_id', d_id_col = 'target_id',uv_col = 'car_uv'):
    """获取起点在clu_id_o中，终点在clu_id_d的OD量"""
    df_od_num = df_od[((df_od[o_id_col].isin(clu_id_o)) &
                       (df_od[d_id_col].isin(clu_id_d)))
    ]
    return (df_od_num[uv_col].sum())        