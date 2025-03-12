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
    return tmp_str

# 示例测试
if __name__ == "__main__":
    # load the osm netowrk in wgs 84
    gr = ox.load_graphml("../../data/test_hp_graph.graphml")

    poly_coords = [(113.465218, 23.131286),
    (113.464738, 23.116090),
    (113.479289, 23.115122),
    (113.485907, 23.115874),
    (113.484153, 23.119977),
    (113.487303, 23.122728),
    (113.486904, 23.126671),
    (113.480013, 23.129296),
    (113.475097, 23.134175),
    (113.465218, 23.131286),]
    # # 获取多边形内的子图
    sub_g = getSubGraphInPoly(gr, poly_coords)
    # sub_g_central_node = get_graph_central_node(sub_g)
    ox.save_graph_geopackage(sub_g, "/users/convel/desktop/test_hp_tar_graph.gpkg")
    # sub_g_avg_depth = get_bi_avg_graph_depth(sub_g, sub_g_central_node)
    # # sub_nodes_ids = bidirectional_search(G, sorted_items[0][0], 20)
    # sub_g_1 = bidirectional_search(gr, 6369608427, sub_g_avg_depth)
    # ox.save_graph_geopackage(gr.subgraph(sub_g_1), "/users/convel/desktop/sub_g_1.gpkg")
    nx.set_node_attributes(gr, genRndFeatures(25), "features")
    for node, data in gr.nodes(data=True):
        print(data['features'])
    heuristic_search(gr, sub_g, 20)


    print("done!")






