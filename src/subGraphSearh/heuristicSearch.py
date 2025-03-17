"""

启发式搜索算法思路
    1.初始随机选点
    在原始图中随机选取若干个节点作为起始点。
    使用已知的子图生成函数生成候选子图。
    仅保留与目标子图无重叠的子图，并计算其相似度。
    2.优先搜索策略
    根据候选子图的相似度排名，选择排名靠前的子图。
    记录这些子图的中心节点或重心节点，在下一轮迭代中优先从这些节点的邻域继续搜索。
    3.邻域扩展与更新
    在每轮迭代中，以当前最优子图的中心节点为起点，扩展其邻域继续搜索新的候选子图。
    通过控制邻域半径（如 BFS/DFS 深度）来平衡搜索范围和计算成本。
    4.终止条件
    满足指定的子图数量。
    达到最大迭代次数。
    若连续多次迭代无更优结果，提前停止。
关键策略
    启发式点选策略：
        初始随机点选可避免局部最优。
        随机点选时可引入节点度、聚类系数等特征作为权重，提高优质子图的命中率。
    邻域扩展策略：
        控制邻域深度，逐步扩大搜索范围。
        结合已知相似度函数，对每轮新扩展的子图进行筛选并替换。
"""
import heapq
import random
from heapq import heappush, heappop
import osmnx as ox

from src.subGraphSearh.similarityCals import cal_total_weighted_similarity
from src.utils.graphUtils import getSubGraphInPoly, get_graph_central_node, get_bi_dir_depth_info, \
    get_bi_avg_graph_depth, bidirectional_search, get_adj_subGraphs


def heuristic_search(graph, target_subgraph, num_results, graph_node_lon='x', graph_node_lat='y', search_step='0.005',
                     search_strategy={'random': 0.2, 'top_adj': 0.4}, max_iter=50, error = 0.00001):
    """
    随机选取初始节点并生成子图
    计算相似度
    排序后选出排名靠前的 search_strategy['top_adj']*num_results 中的每个node将会生成4个candidate subgraph（左上，左下，右上，右下）
    在graph上继续生成 num_results* search_strategy['random']
    以上search_strategy['top_adj']*num_results + num_results* search_strategy['random'] 个subgraph 与
    原来生成的num_results个sub_graphs 重新排名 获取新的list
    注意新的subgraph中的node不能与给定target_subgraph的node相同
    迭代至达到最大迭代次数
    或连续10次迭代的最优值与上一次的最优值之差小于指定误差

    :param graph:
    :param target_subgraph:
    :param num_results:
    :param expand_func:
    :param similarity_func:
    :param search_step:
    :param search_strategy:
    :param max_iter:
    :return:

    """

    # 初始化
    visited_nodes = set(target_subgraph.nodes)
    candidate_subgraphs = []
    # get target_subgraph depth
    sub_g_avg_depth = get_bi_avg_graph_depth(target_subgraph, get_graph_central_node(target_subgraph))
    # sub_g_1 = bidirectional_search(gr, 6369608427, sub_g_avg_depth)
    initial_nodes = random.sample(list(graph.nodes), min(len(graph.nodes), num_results))
    # 获取初始子图集
    sub_candi_graph_id = 0
    for node in initial_nodes:
        candidate_subgraph_node_list = bidirectional_search(graph, node, sub_g_avg_depth)
        while any(candidate_subgraph_node_id in visited_nodes for candidate_subgraph_node_id in candidate_subgraph_node_list):
            node = random.sample(list(graph.nodes), 1)[0]
            candidate_subgraph_node_list = bidirectional_search(graph, node, sub_g_avg_depth)
            # 会导致子图的数量达不到要求
        candidate_subgraph = graph.subgraph(candidate_subgraph_node_list).copy()
        ox.save_graph_geopackage(candidate_subgraph, f"/users/convel/desktop/test_hp_rnd_graph{node}.gpkg")

        tmp_similarity_score = cal_total_weighted_similarity(target_subgraph, candidate_subgraph)
        heappush(candidate_subgraphs, (tmp_similarity_score, sub_candi_graph_id, candidate_subgraph))
        sub_candi_graph_id += 1
    iTerminate = 0
    tmp_error = 9999.99
    # 迭代搜索
    for i in range(max_iter):
        tmp_best = sorted(candidate_subgraphs)[0]
        sub_candi_graph_id = 0
        # 在指定排名靠前的元素生成子图，添加到candidate_subgraphs里
        for tmp_similar_subgraph in heapq.nlargest(int(num_results*search_strategy["top_adj"]), candidate_subgraphs): # bug_fix : 返回前n个需要int
            # get tmp subgraph centroid then get coords
            tmp_central_node = get_graph_central_node(tmp_similar_subgraph[2]) # bug_fix : 加了[2]，tmp_similar_subgraph是个tuple,(权重，排名，元素)的格式
            tmp_coord = (graph.nodes.get(tmp_central_node)[graph_node_lon],
                         graph.nodes.get(tmp_central_node)[graph_node_lat]) # bug_fix : 将.get_node_attributes(graph_node_lon)更改为[graph_node_lon]，lat同样
            # calculate four coords and then get closest nodes of these four
            #
            tmp_subgraphs = get_adj_subGraphs(graph,tmp_coord, graph_node_lon, graph_node_lat,  search_step) # bug_fix : 调换tmp_coord的位置
            for tmp_subgraph in tmp_subgraphs:
                tmp_similarity_score = cal_total_weighted_similarity(target_subgraph, tmp_subgraph)
                heappush(tmp_subgraph, (tmp_similarity_score, sub_candi_graph_id, candidate_subgraph))
                sub_candi_graph_id += 1

        # 检查candidate_subgraphs, 如果数量不够，则采用random的方式补齐；若多了则删除掉排名靠后的元素
        while len(candidate_subgraphs)<num_results:
            initial_nodes = random.sample(list(graph.nodes), min(20, num_results))
            for node in initial_nodes:
                candidate_subgraph = bidirectional_search(graph, node, sub_g_avg_depth)
                if not candidate_subgraph or visited_nodes & set(candidate_subgraph.nodes):
                    continue  # 会导致子图的数量达不到要求
                similarity_score = cal_total_weighted_similarity(target_subgraph, candidate_subgraph)
                if len(candidate_subgraphs) > num_results:
                    heapq.heappop(candidate_subgraphs)
                else:
                    heappush(candidate_subgraphs, (similarity_score, sub_candi_graph_id, candidate_subgraph))
                    sub_candi_graph_id += 1
        # 计算终止条件
        tmp_error = abs(tmp_best-sorted(candidate_subgraphs)[0])
        if tmp_error < error:
            iTerminate += 1
        else:
            iTerminate = 0
        if iTerminate > 10:
            print(f"There are no dominating results for the last iterations, iteration terminate at iter: {i}.")
            return [subgraph for _, subgraph in sorted(candidate_subgraphs, key=lambda x: -x[0])[:num_results]]
    # 返回按相似度排序的结果
    return [subgraph for _, subgraph in sorted(candidate_subgraphs, key=lambda x: -x[0])[:num_results]]
