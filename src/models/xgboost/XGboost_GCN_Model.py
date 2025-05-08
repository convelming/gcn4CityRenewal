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
import random
import geopandas as gpd
import networkx as nx
import osmnx as ox
import math
from src.subGraphSearh.heuristicSearch import heuristic_search
from src.subGraphSearh.similarityCals import cal_graph_degree_distribution, cal_KL_divergence, cal_cluster_coe_diff, \
    cal_shortest_path_length_ratio, cal_edge_similarity, \
    cal_total_weighted_similarity, cal_graph_cosine_similarity
from src.utils.graphUtils import getSubGraphInPoly, get_graph_central_node, get_bi_dir_depth_info, \
    get_bi_avg_graph_depth, bidirectional_search


def cluster_dis(dis):
    if dis<=5000:
        return(1)
    elif dis<=10000:
        return(2)
    elif dis<=15000:
        return(3)
    elif dis<=20000:
        return(4)
    elif dis<=30000:
        return(5)
    else:
        return(6)      
def sum_feature(df_data,clus_col,feature_col):
    df_feature = df_data.copy()
    if type(df_feature[feature_col].values[0]) != list:
        df_feature[feature_col] = df_feature[feature_col].apply(ast.literal_eval)
    df_sum = (
        df_feature.groupby(clus_col)[feature_col]
        .apply(lambda x: pd.DataFrame(x.tolist()).sum().tolist())
        .reset_index(name=feature_col)
    )
    return(df_sum)
def clu_osmid(df_data,clus_name):
    df = df_data[df_data['area_clus']==clus_name]
    return(str(list(df['osmid']))) 
def fu_to_zero(value_data):
    if value_data<0:
        return(0)
    else:
        return(value_data)
def calculate_rmse(y_true, y_pred):
    n = len(y_true)
    squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
    mse = sum(squared_errors) / n  # 均方误差（MSE）
    rmse = math.sqrt(mse)          # 开平方根
    return rmse    
def calculate_std_dev(data):
    n = len(data)
    mean = sum(data) / n
    squared_diff = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diff) / n  # 总体方差
    std_dev = math.sqrt(variance)     # 标准差
    return std_dev
def calculate_cpc(y_true, y_pred):
    """计算CPC指标"""
    min_sum = np.sum(np.minimum(y_true, y_pred))
    total_sum = np.sum(y_true) + np.sum(y_pred)
    cpc = (2 * min_sum) / total_sum if total_sum != 0 else 0
    return cpc


def prepare_data(df):
    """准备数据，将input_x从字符串转换为数组"""
    df = df.copy()
    df['input_x'] = df['input_x'].apply(ast.literal_eval)
    X = np.array(df['input_x'].tolist())
    y = df['flowCounts'].values
    return X, y

def train_xgboost(df, test_size=0.2, random_state=42, params=None):
    """
    训练XGBoost模型并输出训练过程的损失函数
    
    参数:
        df: 包含o_id, input_x, flowCounts的DataFrame
        test_size: 测试集比例
        random_state: 随机种子
        params: XGBoost参数字典
        
    返回:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集真实值
        eval_results: 训练评估结果
    """
    # 准备数据
    X, y = prepare_data(df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 默认参数
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': random_state,
            'nthread': -1
        }
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 训练模型
    eval_results = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=500,
        evals_result=eval_results,
        verbose_eval=False
    )
    
    
    return model, X_test, y_test, eval_results

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


def creat_input(gdf_node,df_od_data,candidate_subgraph_node_list):      
    gpd_sub = gdf_node[gdf_node['osmid'].isin(candidate_subgraph_node_list)].reset_index(drop=True)
    gpd_sub_x = gpd_sub['x'].mean()
    gpd_sub_y = gpd_sub['y'].mean()    
    gpd_rest = gdf_node[~gdf_node['osmid'].isin(candidate_subgraph_node_list)].reset_index(drop=True)
    gpd_rest['dis'] = gpd_rest.apply(lambda z:math.sqrt((z.y-gpd_sub_y)**2+(z.x-gpd_sub_x)**2),axis=1)
    gpd_rest['dis_clus'] = gpd_rest.apply(lambda z: cluster_dis(z.dis),axis=1)
    df_all_data = pd.DataFrame()
    for c in range(1,7):
        df_clu = gpd_rest[gpd_rest['dis_clus']==c]
        n_c = (9-c)*2
        if len(df_clu)<n_c:
            n_c = len(df_clu)
        kmeans = KMeans(n_clusters=int(n_c))
        kmeans.fit(df_clu[['x', 'y']])
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        df_clu['cluster'] = labels
        df_all_clu = pd.DataFrame()
        for la in range(labels.max()+1):
            df_clu_la = df_clu[df_clu['cluster']==la]
            df_clu_la['clu_x'] = centroids[la][0] 
            df_clu_la['clu_y'] = centroids[la][1] 
            df_all_clu = pd.concat([df_all_clu,df_clu_la])
        df_all_data = pd.concat([df_all_data,df_all_clu])
    df_all_data['area_clus'] = df_all_data.apply(lambda z:str(int(z.dis_clus))+'_'+str(int(z.cluster)),axis=1)
    df_all_clus = df_all_data[['area_clus','clu_x','clu_y']].drop_duplicates()
    df_all_clus['clus_osmid'] = df_all_clus.apply(lambda z:clu_osmid(df_all_data,z.area_clus),axis=1)
#     return(df_all_clus)
    df_d = pd.merge(df_all_clus,sum_feature(df_all_data,'area_clus','features'))
    df_d['clus_osmid'] =  df_d['clus_osmid'].apply(ast.literal_eval)
    gpd_sub['o_id'] = str(candidate_subgraph_node_list[0])
    
    df_o = sum_feature(gpd_sub,'o_id','features')
    del df_o['o_id']
    df_o['clus_osmid'] = pd.Series(dtype='object') 
    df_o.at[0,'clus_osmid'] = candidate_subgraph_node_list
    
    df_o['clu_x'] = gpd_sub_x
    df_o['clu_y'] = gpd_sub_y
    df_o['area_clus'] = '0_0'
    df_o_d = pd.concat([df_o,df_d])
    df_o_d = df_o_d.reset_index(drop=True)    
    
    df_clus_osm = pd.DataFrame()
    for i in range(len(df_o_d)):    
        df_tmp = pd.DataFrame()
        df_tmp['source_id'] = df_o_d['clus_osmid'][i]
        df_tmp['area_clus'] = df_o_d['area_clus'][i]
        df_clus_osm = pd.concat([df_clus_osm,df_tmp])    

    df_od_data = pd.merge(df_od_data,df_clus_osm)
    df_clus_osm.columns = ['target_id','area_clus_d']
    df_od_data = pd.merge(df_od_data,df_clus_osm)
    df_od_data = df_od_data[['area_clus','area_clus_d','car_uv']]
    df_od_data = df_od_data.rename(columns={'car_uv':'flowCounts'})        
    df_o_d['key'] = 1
    # 自连接（生成笛卡尔积）
    df_cartesian = pd.merge(
        df_o_d, 
        df_o_d, 
        on='key', 
        suffixes=('', '_d')  # 原始列名 vs 其他行列名
    ).drop(columns='key')     
        
    df_cartesian = df_cartesian[df_cartesian['area_clus']!=df_cartesian['area_clus_d']]
    df_cartesian['line_dis'] = df_cartesian.apply(lambda z:math.sqrt((z.clu_x_d-z.clu_x)**2+(z.clu_y_d-z.clu_y)**2),axis=1) 
        # df_cartesian['flowCounts'] = df_cartesian.apply(lambda z:get_o_od(z.clus_osmid,z.clus_osmid_d,df_od_data),axis=1)
    df_features = df_cartesian[['clus_osmid','clus_osmid_d','area_clus','area_clus_d','line_dis','features','features_d']]    
    df_cartesian = pd.merge(df_cartesian,df_od_data,on=['area_clus','area_clus_d'],how='left')
    df_cartesian = df_cartesian.fillna(0)

    df_gb = df_cartesian.groupby(['area_clus','area_clus_d'])['flowCounts'].sum().reset_index()
    df_gb = pd.merge(df_gb,df_features,on=['area_clus','area_clus_d'])  
    return(df_gb)




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
        r_gdf_node
    df_graph_Kmeans_input = creat_input(r_gdf_node,r_df_od, zl_graph_id)
    df_graph_Kmeans_train = pd.read_csv(f'{TrainData_floder}graph_kmeans_input_data_{conv_num}.csv')
    model, X_test, y_test, eval_results = train_xgboost(df_graph_Kmeans_train.copy())

    input_df = df_graph_Kmeans_input.copy()
    input_df['input_x'] = input_df.apply(lambda z:z.features+z.features_d+[z.line_dis],axis=1)
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
    

