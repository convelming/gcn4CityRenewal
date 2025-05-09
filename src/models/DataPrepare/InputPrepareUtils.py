import pandas as pd
from src.models.DataPrepare.InputPrepareUtils import cluster_dis,sum_feature,get_o_od,clu_osmid
from src.utils.graphUtils import  bidirectional_search
from sklearn.cluster import KMeans
def str_to_list(df,col_name):
    if type(df[col_name].values[0]) != list:
        df[col_name] =  df[col_name].apply(ast.literal_eval)
    return(df)

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

def clu_osmid(df_data, clus_name,clus_col = 'area_clus',id_col='osmid'):
    """在聚类小区内有哪些基础单元点"""
    df = df_data[df_data[clus_col] == clus_name]
    return (str(list(df[id_col])))

def get_subgraph_list(gr,depth, central_point ):
    sub_g_avg_depth = depth  #子图深度
    subgraph_node_list = bidirectional_search(gr, central_point, sub_g_avg_depth) #创建中心子图，并记录中心子图都有哪些节点
    return (subgraph_node_list)   


def creat_input(gdf_node, df_od_data, candidate_subgraph_node_list
                gdf_node_id_col='osmid',gdf_node_x_col='x',gdf_node_y_col='y',
                df_od_data_o_id_col = 'source_id', df_od_data_d_id_col = 'target_id',df_od_data_flow = 'car_uv'): #gr为全市图网络，gdf_node为各节点的属性，df_od_data为各节点之间的od量
    gpd_sub = gdf_node[gdf_node[gdf_node_id_col].isin(candidate_subgraph_node_list)].reset_index(drop=True) #提取出中心子图的节点属性
    gpd_sub_x = gpd_sub[gdf_node_x_col].mean()#计算中心子图的中心位置
    gpd_sub_y = gpd_sub[gdf_node_y_col].mean()
    gpd_rest = gdf_node[~gdf_node[gdf_node_id_col].isin(candidate_subgraph_node_list)].reset_index(drop=True)#提取除中心子图外的节点属性
    gpd_rest['dis'] = gpd_rest.apply(lambda z: math.sqrt((z.y - gpd_sub_y) ** 2 + (z.x - gpd_sub_x) ** 2), axis=1) # 计算其他节点距中心子图中心的直线距离
    gpd_rest['dis_clus'] = gpd_rest.apply(lambda z: cluster_dis(z.dis), axis=1) #基于直线距离先分为6大类
    #在6大类中再通过kmeans聚类
    df_all_data = pd.DataFrame()
    for c in range(1, 7):
        df_clu = gpd_rest[gpd_rest['dis_clus'] == c]
        n_c = (9 - c) * 2
        if len(df_clu) < n_c:
            n_c = len(df_clu)
        kmeans = KMeans(n_clusters=int(n_c))
        kmeans.fit(df_clu[[gdf_node_x_col, gdf_node_y_col]])
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        df_clu['cluster'] = labels
        df_all_clu = pd.DataFrame()
        for la in range(labels.max() + 1):
            df_clu_la = df_clu[df_clu['cluster'] == la]
            df_clu_la['clu_x'] = centroids[la][0] #计算每一个小类的中心点
            df_clu_la['clu_y'] = centroids[la][1]
            df_all_clu = pd.concat([df_all_clu, df_clu_la])
        df_all_data = pd.concat([df_all_data, df_all_clu])

    df_all_data['area_clus'] = df_all_data.apply(lambda z: str(int(z.dis_clus)) + '_' + str(int(z.cluster)), axis=1) #距离大类+聚类小类
    df_all_clus = df_all_data[['area_clus', 'clu_x', 'clu_y']].drop_duplicates()
    df_all_clus['clus_osmid'] = df_all_clus.apply(lambda z: clu_osmid(df_all_data, z.area_clus), axis=1) #每一类中都有哪些单位节点id
    df_d = pd.merge(df_all_clus, sum_feature(df_all_data, 'area_clus', 'features'))
    df_d['clus_osmid'] = df_d['clus_osmid'].apply(ast.literal_eval)
    gpd_sub['o_id'] = str(central_point) + "_" + str(int(sub_g_avg_depth))

    df_o = sum_feature(gpd_sub, 'o_id', 'features')#计算中心区域的节点属性总和
    del df_o['o_id']
    df_o['clus_osmid'] = pd.Series(dtype='object')
    df_o.at[0, 'clus_osmid'] = candidate_subgraph_node_list#中心区域有哪些节点
    df_o['clu_x'] = gpd_sub_x
    df_o['clu_y'] = gpd_sub_y
    df_o['area_clus'] = '0_0'#中心区域的类别设为0_0
    df_o_d = pd.concat([df_o, df_d])#将中心区域与其他区域进行合并
    df_o_d = df_o_d.reset_index(drop=True)

    df_clus_osm = pd.DataFrame()
    for i in range(len(df_o_d)):
        df_tmp = pd.DataFrame()
        df_tmp[df_od_data_o_id_col] = df_o_d['clus_osmid'][i]
        df_tmp['area_clus'] = df_o_d['area_clus'][i]
        df_clus_osm = pd.concat([df_clus_osm, df_tmp])
    df_od_data = pd.merge(df_od_data, df_clus_osm)
    df_clus_osm.columns = [df_od_data_d_id_col, 'area_clus_d']
    df_od_data = pd.merge(df_od_data, df_clus_osm)
    df_od_data = df_od_data[['area_clus', 'area_clus_d', df_od_data_flow]]
    df_od_data = df_od_data.rename(columns={df_od_data_flow: 'flowCounts'}) #将OD数据从原始的网格间OD量变为类别间的OD量


    # 自连接（生成笛卡尔积）,生成N*N的OD矩阵即其features
    df_o_d['key'] = 1
    df_cartesian = pd.merge(
        df_o_d,
        df_o_d,
        on='key',
        suffixes=('', '_d')  # 原始列名 vs 其他行列名
    ).drop(columns='key')

    df_cartesian = df_cartesian[df_cartesian['area_clus'] != df_cartesian['area_clus_d']] #删除自己到自己区域的数据
    df_cartesian['line_dis'] = df_cartesian.apply(
        lambda z: math.sqrt((z.clu_x_d - z.clu_x) ** 2 + (z.clu_y_d - z.clu_y) ** 2), axis=1) #计算区域间的直线距离

    df_features = df_cartesian[['area_clus', 'area_clus_d', 'line_dis', 'features', 'features_d']]
    df_cartesian = pd.merge(df_cartesian, df_od_data, on=['area_clus', 'area_clus_d'], how='left')
    df_cartesian = df_cartesian.fillna(0)
    df_gb = df_cartesian.groupby(['area_clus', 'area_clus_d'])['flowCounts'].sum().reset_index()
    df_gb = pd.merge(df_gb, df_features, on=['area_clus', 'area_clus_d'])#统计所有从A区域到B区域的OD总量

    df_gb['input_x'] = df_gb.apply(lambda z: z.features + z.features_d + [z.line_dis], axis=1) #O区域的features + D区域的features + OD间的距离作为input_x
    return (df_gb)