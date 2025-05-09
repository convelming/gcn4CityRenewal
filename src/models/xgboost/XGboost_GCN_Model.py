import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
# import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import KMeans

import geopandas as gpd

import osmnx as ox
import math

from src.utils.graphUtils import getSubGraphInPoly, get_graph_central_node, get_bi_dir_depth_info, \
    get_bi_avg_graph_depth, bidirectional_search
from src.models.DataPrepare.InputPrepareUtils import cluster_dis,sum_feature,clu_osmid,creat_input
from src.models.DataPrepare.ModelPredictUtils import fu_to_zero,calculate_rmse,calculate_std_dev,calculate_cpc,prepare_data,train_xgboost



def predict_with_model(model, new_data):
    """
    使用训练好的模型进行预测
    
    参数:
        model: 训练好的XGBoost模型
        new_data: 新数据DataFrame，包含input_x列
        
    返回:
        predictions: 预测结果数组
    """
    if isinstance(new_data, pd.DataFrame):
        if type(new_data['input_x'].values[0]) != list:
            new_data['input_x'] =  new_data['input_x'].apply(ast.literal_eval)        

        X_new = np.array(new_data['input_x'].tolist())
    else:
        X_new = np.array(new_data)
    
    dnew = xgb.DMatrix(X_new)
    predictions = model.predict(dnew)
    return predictions




middle_data_floder = './src/models/middle_data/'
TrainData_floder = './src/models/TrainData/'
#镇龙ID
zl_net_id = [9444,9445,9600,9601,9602,9756,9757,9758]
# zl_TAZ = gpd.read_file('./镇龙/交通小区.shp')    
# zl_TAZ_id = list(set(zl_TAZ['FID_1'])-set([2603,1868,659,3190,1764,533,375]))
gr = ox.load_graphml(f'{middle_data_floder}guangzhou_drive_feature_node&edge.graphml')
poly_coords = [(113.557801, 23.287852),
(113.555640, 23.284914),
(113.554069, 23.282053),
(113.556145, 23.279295),
(113.558951, 23.277980),
(113.564731, 23.277516),
(113.567201, 23.277233),
(113.570540, 23.277310),
(113.573907, 23.278264),
(113.579407, 23.279269),
(113.582381, 23.281847),
(113.579351, 23.284733),
(113.576713, 23.287053),
(113.579106, 23.288529),
(113.577815, 23.291158),
(113.574897, 23.289663),
(113.571810, 23.288735),
(113.568443, 23.287446),
(113.566703, 23.288684),
(113.563448, 23.287292),
(113.563055, 23.288529),
(113.557801, 23.287852),]
G_sub = getSubGraphInPoly(gr, poly_coords)
sub_g_avg_depth = get_bi_avg_graph_depth(G_sub, get_graph_central_node(G_sub))
zl_graph_id = bidirectional_search(gr, get_graph_central_node(G_sub), sub_g_avg_depth)

r_df_od = pd.read_csv(f'{middle_data_floder}base_od.csv')
list_cpc_graph,list_rmse_graph = [],[]
#graph_Kmeans_prepare
for conv_num in range(1,21):
    if conv_num == 1:
        r_gdf_node = gpd.read_file(f'{middle_data_floder}guangzhou_drive_feature_node&edge.gpkg', layer='nodes') 
        r_gdf_node = r_gdf_node.to_crs('EPSG:4526')
        r_gdf_node['x'] = r_gdf_node.apply(lambda z: z.geometry.x, axis=1)
        r_gdf_node['y'] = r_gdf_node.apply(lambda z: z.geometry.y, axis=1) 
    else:
        r_gdf_node = pd.read_csv(f'{middle_data_floder}gcn_features/features_conv_{conv_num}.csv') 


    df_graph_Kmeans_train = pd.read_csv(f'{TrainData_floder}graph_kmeans_input_data_{conv_num}.csv')
    model, X_test, y_test, eval_results = train_xgboost(df_graph_Kmeans_train.copy())

    input_df = creat_input(r_gdf_node,r_df_od, zl_graph_id)
    predictions = predict_with_model(model, input_df)
    input_df['predict'] = predictions
    input_df['predict'] = input_df.apply(lambda z:fu_to_zero(z.predict),axis=1)
    input_df['min_v'] = input_df.apply(lambda z:min(z.predict,z.flowCounts),axis=1)
    cpc = 2*(input_df['min_v'].sum()) / (input_df['flowCounts'].sum()+input_df['predict'].sum())
    df_zero = input_df[(input_df['area_clus']=='0_0')|(input_df['area_clus_d']=='0_0')][['area_clus','area_clus_d','flowCounts','predict']]
    rmse = calculate_rmse(df_zero['predict'], df_zero['flowCounts'])
    print(conv_num,'cpc:',cpc,'rmse',rmse)
    list_cpc_graph.append(cpc)
    list_rmse_graph.append(rmse)
    

