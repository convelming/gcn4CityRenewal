import pickle
import ast
import random
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from src.subGraphSearh.heuristicSearch import heuristic_search
from src.subGraphSearh.similarityCals import cal_graph_degree_distribution, cal_KL_divergence, cal_cluster_coe_diff, \
    cal_shortest_path_length_ratio, cal_edge_similarity, \
    cal_total_weighted_similarity, cal_graph_cosine_similarity
from src.utils.graphUtils import getSubGraphInPoly, get_graph_central_node, get_bi_dir_depth_info, \
    get_bi_avg_graph_depth, bidirectional_search
from multiprocessing import Pool, Manager
from functools import partial

gr = ox.load_graphml('./src/data/base_data/guangzhou_drive_feature_node&edge.graphml')
gdf_node = gpd.read_file('./src/data/base_data/guangzhou_drive_feature_node&edge.gpkg',layer='nodes')
#建设大马路
poly_coords = [(113.276099, 23.140708),
                    (113.275919, 23.134086),
                    (113.279790, 23.133837),
                    (113.279790, 23.139342),
                    (113.276099, 23.140708),]
#镇龙                    
# poly_coords = [(113.557801, 23.287852),
# (113.555640, 23.284914),
# (113.554069, 23.282053),
# (113.556145, 23.279295),
# (113.558951, 23.277980),
# (113.564731, 23.277516),
# (113.567201, 23.277233),
# (113.570540, 23.277310),
# (113.573907, 23.278264),
# (113.579407, 23.279269),
# (113.582381, 23.281847),
# (113.579351, 23.284733),
# (113.576713, 23.287053),
# (113.579106, 23.288529),
# (113.577815, 23.291158),
# (113.574897, 23.289663),
# (113.571810, 23.288735),
# (113.568443, 23.287446),
# (113.566703, 23.288684),
# (113.563448, 23.287292),
# (113.563055, 23.288529),
# (113.557801, 23.287852),]

G_sub = getSubGraphInPoly(gr, poly_coords)


nodes_to_process =  list(gr.nodes)
sub_g_avg_depth = get_bi_avg_graph_depth(G_sub, get_graph_central_node(G_sub))

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

def parallel_do_many(graph, graph_node_features, num_results,target_subgraph, nodes, sub_g_avg_depth, num_processes=None):
    """
    多进程并行计算多个节点的子图相似度
    
    参数:
        graph: 原始图
        target_subgraph: 目标子图
        nodes: 要处理的节点列表
        sub_g_avg_depth: 子图平均深度
        num_processes: 使用的进程数，默认为CPU核心数
    """
    if num_processes is None:
        num_processes = os.cpu_count() or 1
    
    # 创建共享数据结构
    manager = Manager()
    central_nodes = manager.list(target_subgraph.nodes)  # 共享的中心节点列表
    result_queue = manager.Queue()  # 结果队列

    # 创建部分函数，固定部分参数
    worker = partial(do_one_parallel, 
                    graph=graph,
                    target_subgraph=target_subgraph,
                    sub_g_avg_depth=sub_g_avg_depth,
                    central_nodes=central_nodes,
                    result_queue=result_queue)

    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:

        pool.map(worker, nodes)  

    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    #判断搜索的子图是否有节点在target_subgraph中
    df_features_results = pd.DataFrame()
    top_results = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (node, score) in enumerate(top_results, 1):
        judge_subgraph_node_list = bidirectional_search(gr,node,sub_g_avg_depth)
        if any(tmp_subgraph_node_id in list(G_sub.nodes) for tmp_subgraph_node_id in judge_subgraph_node_list):
            continue
        else:
            df_tmp_node_features = gdf_node[gdf_node['osmid'].isin(judge_subgraph_node_list)].copy() 
            df_tmp_node_features['typed_id'] = rank
            df_tmp_node_features = sum_feature(df_tmp_node_features, clus_col='typed_id', feature_col='features')
            df_features_results = pd.concat([df_features_results,df_tmp_node_features])
        if len(df_features_results) == num_results:
            break
    features_array = np.array(df_features_results['features'].tolist())
    # 计算每列的最大值、最小值和平均值
    max_values = features_array.max(axis=0)
    min_values = features_array.min(axis=0)
    mean_values = features_array.mean(axis=0)
    result_df = pd.DataFrame({
        'type': ['max', 'min', 'mean'],
        'features': [max_values.tolist(), min_values.tolist(), mean_values.tolist()]
    })
    return result_df

def do_one_parallel(node, graph, target_subgraph, sub_g_avg_depth, central_nodes, result_queue):
    """
    适配后的工作函数，用于多进程环境
    """
    # print(f'kernel_{node}_start')
    candidate_subgraph_node_list = bidirectional_search(graph, node, sub_g_avg_depth)

    # 使用manager.list的检查方式

    candidate_subgraph = graph.subgraph(candidate_subgraph_node_list).copy()
    
    # 更新共享数据结构
    central_nodes.append(node)
    # visited_nodes.extend(candidate_subgraph_node_list)
    
    tmp_similarity_score = cal_total_weighted_similarity(target_subgraph, candidate_subgraph)
    print(len(central_nodes))
    
    # 将结果放入队列
    result_queue.put((node, tmp_similarity_score))



results = parallel_do_many(
    graph=gr,
    num_results = 40,
    graph_node_features = gdf_node,
    target_subgraph=G_sub,
    nodes=nodes_to_process,
    sub_g_avg_depth=sub_g_avg_depth,
    num_processes=60  
)

results.to_csv('/src/subGraphSearh/Search_result/max_min_features.csv',index=False)

with open("./zl_results.pkl", "wb") as f:
    pickle.dump(results, f)

with open("./src/subGraphSearh/Search_result/zl_results.pkl", "rb") as f:
    results = pickle.load(f)




