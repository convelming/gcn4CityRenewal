import random

import networkx as nx
import numpy as np
import osmnx as ox

from src.subGraphSearh.heuristicSearch import heuristic_search
from src.subGraphSearh.similarityCals import cal_graph_degree_distribution, cal_KL_divergence, cal_cluster_coe_diff, \
    cal_shortest_path_length_ratio, cal_edge_similarity, \
    cal_total_weighted_similarity, cal_graph_cosine_similarity
from src.utils.graphUtils import getSubGraphInPoly, get_graph_central_node, get_bi_dir_depth_info, \
    get_bi_avg_graph_depth, bidirectional_search
def genRndFeatures(num):
    tmp_str = ""
    for i in range(num):
        tmp_str += str(random.uniform(0, 20.0))+","
    # return tmp_str
    return [random.uniform(0, 20) for _ in range(20)]

# 示例测试
if __name__ == "__main__":
    # load the osm netowrk in wgs 84
    gr = ox.load_graphml("F:/jupyter/dg/天河北城市更新/v5/base_data/guangzhou_drive.graphml")
    poly_coords = [(113.276099,23.140708),
    (113.275919,23.134086),
    (113.279790,23.133837),
    (113.279790,23.139342),
    (113.276099,23.140708),]
    # # 获取多边形内的子图
    nx.set_node_attributes(gr, genRndFeatures(25), "features")
    sub_g = getSubGraphInPoly(gr, poly_coords)
    # # sub_g_central_node = get_graph_central_node(sub_g)
    # for node, data in gr.nodes(data=True):
    #     print(data['features'])
    # for node, data in sub_g.nodes(data=True):
    #     print(data['features'])

    ox.save_graph_geopackage(sub_g, "./sub_g_1187415550.qpkq")
    # sub_g_avg_depth = get_bi_avg_graph_depth(sub_g, sub_g_central_node)
    # # sub_nodes_ids = bidirectional_search(G, sorted_items[0][0], 20)
    # sub_g_1 = bidirectional_search(gr, 6369608427, sub_g_avg_depth)
    # ox.save_graph_geopackage(gr.subgraph(sub_g_1), "/users/convel/desktop/sub_g_1.gpkg")

    heuristic_search(gr, sub_g, 20)


    print("done!")






