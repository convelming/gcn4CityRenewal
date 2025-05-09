
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

from sklearn.cluster import KMeans
import math

from src.models.DataPrepare.InputPrepareUtils import cluster_dis,sum_feature,clu_osmid,creat_input
from src.models.DataPrepare.ModelPredictUtils import fu_to_zero,calculate_rmse,calculate_std_dev,calculate_cpc,prepare_data,train_xgboost

def creat_input(gdf_node, df_od_data, candidate_subgraph_node_list):
    gpd_sub = gdf_node[gdf_node['osmid'].isin(candidate_subgraph_node_list)].reset_index(drop=True)
    gpd_sub_x = gpd_sub['x'].mean()
    gpd_sub_y = gpd_sub['y'].mean()
    gpd_rest = gdf_node[~gdf_node['osmid'].isin(candidate_subgraph_node_list)].reset_index(drop=True)
    gpd_rest['dis'] = gpd_rest.apply(lambda z: math.sqrt((z.y - gpd_sub_y) ** 2 + (z.x - gpd_sub_x) ** 2), axis=1)
    gpd_rest['dis_clus'] = gpd_rest.apply(lambda z: cluster_dis(z.dis), axis=1)
    df_all_data = pd.DataFrame()
    for c in range(1, 7):
        df_clu = gpd_rest[gpd_rest['dis_clus'] == c]
        n_c = (9 - c) * 2
        if len(df_clu) < n_c:
            n_c = len(df_clu)
        kmeans = KMeans(n_clusters=int(n_c))
        kmeans.fit(df_clu[['x', 'y']])
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        df_clu['cluster'] = labels
        df_all_clu = pd.DataFrame()
        for la in range(labels.max() + 1):
            df_clu_la = df_clu[df_clu['cluster'] == la]
            df_clu_la['clu_x'] = centroids[la][0]
            df_clu_la['clu_y'] = centroids[la][1]
            df_all_clu = pd.concat([df_all_clu, df_clu_la])
        df_all_data = pd.concat([df_all_data, df_all_clu])
    df_all_data['area_clus'] = df_all_data.apply(lambda z: str(int(z.dis_clus)) + '_' + str(int(z.cluster)), axis=1)
    df_all_clus = df_all_data[['area_clus', 'clu_x', 'clu_y']].drop_duplicates()
    df_all_clus['clus_osmid'] = df_all_clus.apply(lambda z: clu_osmid(df_all_data, z.area_clus), axis=1)
    #     return(df_all_clus)
    df_d = pd.merge(df_all_clus, sum_feature(df_all_data, 'area_clus', 'features'))
    df_d['clus_osmid'] = df_d['clus_osmid'].apply(ast.literal_eval)
    gpd_sub['o_id'] = str(candidate_subgraph_node_list[0])

    df_o = sum_feature(gpd_sub, 'o_id', 'features')
    del df_o['o_id']
    df_o['clus_osmid'] = pd.Series(dtype='object')
    df_o.at[0, 'clus_osmid'] = candidate_subgraph_node_list

    df_o['clu_x'] = gpd_sub_x
    df_o['clu_y'] = gpd_sub_y
    df_o['area_clus'] = '0_0'
    df_o_d = pd.concat([df_o, df_d])
    df_o_d = df_o_d.reset_index(drop=True)

    df_clus_osm = pd.DataFrame()
    for i in range(len(df_o_d)):
        df_tmp = pd.DataFrame()
        df_tmp['source_id'] = df_o_d['clus_osmid'][i]
        df_tmp['area_clus'] = df_o_d['area_clus'][i]
        df_clus_osm = pd.concat([df_clus_osm, df_tmp])

    df_od_data = pd.merge(df_od_data, df_clus_osm)
    df_clus_osm.columns = ['target_id', 'area_clus_d']
    df_od_data = pd.merge(df_od_data, df_clus_osm)
    df_od_data = df_od_data[['area_clus', 'area_clus_d', 'car_uv']]
    df_od_data = df_od_data.rename(columns={'car_uv': 'flowCounts'})
    df_o_d['key'] = 1
    # 自连接（生成笛卡尔积）
    df_cartesian = pd.merge(
        df_o_d,
        df_o_d,
        on='key',
        suffixes=('', '_d')  # 原始列名 vs 其他行列名
    ).drop(columns='key')

    df_cartesian = df_cartesian[df_cartesian['area_clus'] != df_cartesian['area_clus_d']]
    df_cartesian['line_dis'] = df_cartesian.apply(
        lambda z: math.sqrt((z.clu_x_d - z.clu_x) ** 2 + (z.clu_y_d - z.clu_y) ** 2), axis=1)
    # df_cartesian['flowCounts'] = df_cartesian.apply(lambda z:get_o_od(z.clus_osmid,z.clus_osmid_d,df_od_data),axis=1)
    df_features = df_cartesian[
        ['clus_osmid', 'clus_osmid_d', 'area_clus', 'area_clus_d', 'line_dis', 'features', 'features_d']]
    df_cartesian = pd.merge(df_cartesian, df_od_data, on=['area_clus', 'area_clus_d'], how='left')
    df_cartesian = df_cartesian.fillna(0)

    df_gb = df_cartesian.groupby(['area_clus', 'area_clus_d'])['flowCounts'].sum().reset_index()
    df_gb = pd.merge(df_gb, df_features, on=['area_clus', 'area_clus_d'])
    return (df_gb)


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

    #     # 绘制训练过程
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(eval_results['train']['rmse'], label='Train RMSE')
    #     plt.plot(eval_results['test']['rmse'], label='Test RMSE')
    #     plt.xlabel('Boosting Rounds')
    #     plt.ylabel('RMSE')
    #     plt.title('Training and Validation Loss')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

    # 评估模型
    #     y_pred = model.predict(dtest)
    #     mse = mean_squared_error(y_test, y_pred)
    #     mae = mean_absolute_error(y_test, y_pred)
    #     rmse = np.sqrt(mse)
    #     cpc = calculate_cpc(y_test, y_pred)

    #     print(f"\nModel Evaluation:")
    #     print(f"Test RMSE: {rmse:.4f}")
    #     print(f"Test MAE: {mae:.4f}")
    #     print(f"Test MSE: {mse:.4f}")
    #     print(f"Test CPC: {cpc:.4f}")

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
            new_data['input_x'] = new_data['input_x'].apply(ast.literal_eval)

        X_new = np.array(new_data['input_x'].tolist())
    else:
        X_new = np.array(new_data)

    dnew = xgb.DMatrix(X_new)
    predictions = model.predict(dnew)
    return predictions
