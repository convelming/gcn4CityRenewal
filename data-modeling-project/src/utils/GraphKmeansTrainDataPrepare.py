import ast
import random
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import math
import pickle
import time
import multiprocessing
from multiprocessing import Pool, Manager, Lock
from functools import partial

from src.subGraphSearh.heuristicSearch import heuristic_search
from src.subGraphSearh.similarityCals import cal_graph_degree_distribution, cal_KL_divergence, cal_cluster_coe_diff, \
    cal_shortest_path_length_ratio, cal_edge_similarity, \
    cal_total_weighted_similarity, cal_graph_cosine_similarity
from src.utils.graphUtils import getSubGraphInPoly, get_graph_central_node, get_bi_dir_depth_info, \
    get_bi_avg_graph_depth, bidirectional_search

from sklearn.cluster import KMeans


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


def sum_feature(df_data, clus_col, feature_col='features'):
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


def clu_osmid(df_data, clus_name,clus_col = 'area_clus',id_col='osmid'):
    """在聚类小区内有哪些基础单元点"""
    df = df_data[df_data[clus_col] == clus_name]
    return (str(list(df[id_col])))


def creat_input(gr, gdf_node, df_od_data, depth, central_point,
                gdf_node_id_col='osmid',x_col='x',y_col='y'): #gr为全市图网络，gdf_node为各节点的属性，df_od_data为各节点之间的od量
    sub_g_avg_depth = depth  #子图深度
    candidate_subgraph_node_list = bidirectional_search(gr, central_point, sub_g_avg_depth)#创建子图，并记录子图都有哪些节点
    gpd_sub = gdf_node[gdf_node[gdf_node_id_col].isin(candidate_subgraph_node_list)].reset_index(drop=True) #
    gpd_sub_x = gpd_sub[x_col].mean()
    gpd_sub_y = gpd_sub[y_col].mean()
    gpd_rest = gdf_node[~gdf_node[gdf_node_id_col].isin(candidate_subgraph_node_list)].reset_index(drop=True)
    gpd_rest['dis'] = gpd_rest.apply(lambda z: math.sqrt((z.y - gpd_sub_y) ** 2 + (z.x - gpd_sub_x) ** 2), axis=1)
    gpd_rest['dis_clus'] = gpd_rest.apply(lambda z: cluster_dis(z.dis), axis=1)
    df_all_data = pd.DataFrame()

    for c in range(1, 7):
        df_clu = gpd_rest[gpd_rest['dis_clus'] == c]
        n_c = (9 - c) * 2
        if len(df_clu) < n_c:
            n_c = len(df_clu)

        kmeans = KMeans(n_clusters=int(n_c))
        kmeans.fit(df_clu[[x_col, y_col]])
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
    df_d = pd.merge(df_all_clus, sum_feature(df_all_data, 'area_clus', 'features'))
    df_d['clus_osmid'] = df_d['clus_osmid'].apply(ast.literal_eval)
    gpd_sub['o_id'] = str(central_point) + "_" + str(int(sub_g_avg_depth))
    df_o = sum_feature(gpd_sub, 'o_id', 'features')

    del df_o['o_id']
    df_o['clus_osmid'] = pd.Series(dtype='object')
    df_o.at[0, 'clus_osmid'] = candidate_subgraph_node_list

    df_o['clu_x'] = gpd_sub_x
    df_o['clu_y'] = gpd_sub_y
    df_o['area_clus'] = '0_0'
    df_o_d = pd.concat([df_o, df_d])
    df_o = []
    df_d = []
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
    df_clus_osm = []
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
    df_o_d = []
    df_cartesian = df_cartesian[df_cartesian['area_clus'] != df_cartesian['area_clus_d']]
    df_cartesian['line_dis'] = df_cartesian.apply(
        lambda z: math.sqrt((z.clu_x_d - z.clu_x) ** 2 + (z.clu_y_d - z.clu_y) ** 2), axis=1)
    # df_cartesian['flowCounts'] = df_cartesian.apply(lambda z:get_o_od(z.clus_osmid,z.clus_osmid_d,df_od_data),axis=1)
    df_features = df_cartesian[['area_clus', 'area_clus_d', 'line_dis', 'features', 'features_d']]
    df_cartesian = pd.merge(df_cartesian, df_od_data, on=['area_clus', 'area_clus_d'], how='left')
    df_cartesian = df_cartesian.fillna(0)

    df_gb = df_cartesian.groupby(['area_clus', 'area_clus_d'])['flowCounts'].sum().reset_index()
    df_gb = pd.merge(df_gb, df_features, on=['area_clus', 'area_clus_d'])
    df_gb['input_x'] = df_gb.apply(lambda z: z.features + z.features_d + [z.line_dis], axis=1)
    df_output = df_gb[['input_x', 'flowCounts']]
    return (df_output)


file_lock = Lock()


def read_file():
    # 每个进程独立加载数据，避免共享大数据
    r_graph = ox.load_graphml('./base_data/guangzhou_drive_feature_node&edge.graphml')
    r_gdf_node = gpd.read_file("./base_data/guangzhou_drive_feature_node&edge.gpkg", layer='nodes')
    r_gdf_node = r_gdf_node.to_crs('EPSG:4526')
    r_gdf_node['x'] = r_gdf_node.apply(lambda z: z.geometry.x, axis=1)
    r_gdf_node['y'] = r_gdf_node.apply(lambda z: z.geometry.y, axis=1)
    r_df_od = pd.read_csv('./base_data/base_od.csv')
    return r_graph, r_gdf_node, r_df_od


def do_one(index, result_queue):
    print(f'kernel_{index}_start')
    try:
        shared_graph, shared_gdf_node, shared_df_od = read_file()

        list_search_kernel = list()
        df_xy_kernel = pd.DataFrame()

        while len(list_search_kernel) < 1000:
            random_osmid = random.choice(shared_gdf_node['osmid'])
            random_depth = random.randint(4, 20)
            random_id = f"{int(random_osmid)}_{int(random_depth)}"

            if random_id not in list_search_kernel:
                df_xy = creat_input(shared_graph, shared_gdf_node,
                                    shared_df_od, random_depth, random_osmid)
                df_xy_kernel = pd.concat([df_xy_kernel, df_xy])
                list_search_kernel.append(random_id)

                if len(list_search_kernel) % 1 == 0:
                    # 将结果放入队列而不是直接写文件
                    result_queue.put((list_search_kernel.copy(), df_xy_kernel.copy()))
                    df_xy_kernel = pd.DataFrame()  # 清空临时DataFrame

        # 处理剩余未提交的结果
        if not df_xy_kernel.empty:
            result_queue.put((list_search_kernel, df_xy_kernel))

    except Exception as e:
        print(f"Process {index} failed: {str(e)}")


def result_writer(result_queue):
    """专门负责写文件的进程"""
    list_search = []
    df_all_xy = pd.DataFrame()

    try:
        # 尝试加载已有数据
        with file_lock:
            try:
                with open("./predict/list_search.pkl", "rb") as f:
                    list_search = pickle.load(f)
            except FileNotFoundError:
                pass

            try:
                df_all_xy = pd.read_csv('./predict/input_data.csv')
            except FileNotFoundError:
                pass
    except Exception as e:
        print(f"Error loading existing data: {str(e)}")

    while True:
        try:
            new_list, new_df = result_queue.get()
            if new_list == "DONE":  # 结束信号
                break

            # 合并新数据
            list_search = list(set(list_search + new_list))
            new_df['input_x'] = new_df['input_x'].apply(str)
            df_all_xy = pd.concat([df_all_xy, new_df]).drop_duplicates()

            # 写入文件
            with file_lock:
                with open("./predict/list_search.pkl", "wb") as f:
                    pickle.dump(list_search, f)
                df_all_xy.to_csv('./predict/input_data.csv', index=False)

            print(f"Writer: Current total records: {len(df_all_xy)}")

        except Exception as e:
            print(f"Error in writer process: {str(e)}")


def run_multi():
    # 使用队列收集结果
    manager = Manager()
    result_queue = manager.Queue()

    # 启动写入进程
    writer_pool = Pool(1)
    writer_pool.apply_async(result_writer, (result_queue,))

    # 启动工作进程
    worker_pool = Pool(2)
    for i in range(2):
        worker_pool.apply_async(do_one, (i, result_queue))

    worker_pool.close()
    worker_pool.join()

    # 通知写入进程结束
    result_queue.put(("DONE", None))
    writer_pool.close()
    writer_pool.join()


if __name__ == '__main__':
    run_multi()    