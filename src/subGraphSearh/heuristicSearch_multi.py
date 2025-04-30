import os
import ast
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from tqdm import tqdm
from src.subGraphSearh.similarityCals import cal_total_weighted_similarity
from src.utils.graphUtils import getSubGraphInPoly, get_graph_central_node, \
    get_bi_avg_graph_depth, bidirectional_search
from multiprocessing import Pool, Manager
from functools import partial
from tqdm.contrib.concurrent import process_map

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

def parallel_do_many(graph, graph_node_features, num_results,target_subgraph, nodes, sub_g_avg_depth, num_processes=None):
    """
    多进程并行计算多个节点的子图相似度
    参数:
        graph: 原始图
        graph_node_features:原始图的节点features
        num_results：要保留的最为相似子图数量
        target_subgraph: 目标子图
        nodes: 要处理的节点列表
        sub_g_avg_depth: 子图平均深度
        num_processes: 使用的进程数，默认为CPU核心数
    """
    #设置进程数
    if num_processes is None:
        num_processes = min(os.cpu_count() ,32)
    # 创建共享数据结构
    manager = Manager()
    central_nodes = manager.list(target_subgraph.nodes)  # 共享的中心节点列表
    result_queue = manager.Queue()  # 结果队列

    # # 创建部分函数，固定部分参数
    # worker = partial(do_one_parallel,
    #                 graph=graph,
    #                 target_subgraph=target_subgraph,
    #                 sub_g_avg_depth=sub_g_avg_depth,
    #                 central_nodes=central_nodes,
    #                 result_queue=result_queue)
    # # 使用进程池并行处理
    # with Pool(processes=num_processes) as pool:
    #         pool.map(worker, nodes)

    # 使用 process_map 替代 Pool
    process_map(
        partial(
            do_one_parallel,
            graph=graph,
            target_subgraph=target_subgraph,
            sub_g_avg_depth=sub_g_avg_depth,
            central_nodes=central_nodes,
            result_queue=result_queue
        ),
        nodes,
        max_workers=num_processes,
        chunksize=700,  # 调整 chunksize 以提高性能
        desc="Processing nodes"
    )

    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    #判断搜索的子图是否有节点在target_subgraph中
    df_features_results = pd.DataFrame()
    top_results = sorted(results, key=lambda x: x[1], reverse=True)#对各个子图通过相似度进行排序
    for rank, (node, score) in enumerate(top_results, 1):
        judge_subgraph_node_list = bidirectional_search(graph,node,sub_g_avg_depth)#计算子图有哪些节点
        if any(tmp_subgraph_node_id in list(target_subgraph.nodes) for tmp_subgraph_node_id in judge_subgraph_node_list):#判断子图节点是否在目标子图中
            continue
        else:
            df_tmp_node_features = graph_node_features[graph_node_features['osmid'].isin(judge_subgraph_node_list)].copy()
            df_tmp_node_features['typed_id'] = rank
            df_tmp_node_features = sum_feature(df_tmp_node_features, clus_col='typed_id', feature_col='features')#对子图的features求和
            df_features_results = pd.concat([df_features_results,df_tmp_node_features])
        if len(df_features_results) == num_results:#当子图数量到所需数量后停止搜集
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
    计算每一个节点所生成的子图和目标子图的相似度
    """
    candidate_subgraph_node_list = bidirectional_search(graph, node, sub_g_avg_depth)#子图nodeId
    candidate_subgraph = graph.subgraph(candidate_subgraph_node_list).copy()#子图
    # 更新共享数据结构
    central_nodes.append(node)
    tmp_similarity_score = cal_total_weighted_similarity(target_subgraph, candidate_subgraph)#相似度计算
    # print(len(central_nodes))
    # 将结果放入队列
    result_queue.put((node, tmp_similarity_score))

def run_multi():
    gr = ox.load_graphml('./src/data/base_data/guangzhou_drive_feature_node&edge.graphml')
    gdf_node = gpd.read_file('./src/data/base_data/guangzhou_drive_feature_node&edge.gpkg',layer='nodes')
    #建设大马路
    poly_coords = [(113.276099, 23.140708),
                        (113.275919, 23.134086),
                        (113.279790, 23.133837),
                        (113.279790, 23.139342),
                        (113.276099, 23.140708),]
    G_sub = getSubGraphInPoly(gr, poly_coords)
    nodes_to_process =  list(gr.nodes)
    sub_g_avg_depth = get_bi_avg_graph_depth(G_sub, get_graph_central_node(G_sub))
    results = parallel_do_many(
        graph=gr,
        num_results = 40,
        graph_node_features = gdf_node,
        target_subgraph=G_sub,
        nodes=nodes_to_process,
        sub_g_avg_depth=sub_g_avg_depth,
        num_processes=40
    )
    results.to_csv('/src/subGraphSearh/Search_result/max_min_features.csv',index=False)

if __name__ == '__main__':
    run_multi()



