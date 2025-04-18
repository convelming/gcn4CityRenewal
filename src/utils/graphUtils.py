import networkx as nx
import osmnx as ox
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point
from collections import deque
import numpy as np

def getSubGraphInPoly(G, polygon_coords):
    """
    从 NetworkX 图 G 中提取位于 polygon 内的子图。

    参数：
    - G: NetworkX 图（通常是 OSMnx 生成的）
    - polygon_coords: 多边形坐标列表 [(x1, y1), (x2, y2), ..., (x1, y1)]

    返回：
    - G_sub: 仅包含多边形内部节点和边的子图
    """
    # 构造 Shapely 多边形
    polygon = Polygon(polygon_coords)

    # 找到在多边形内部的节点
    nodes_inside = [node for node, data in G.nodes(data=True)
                    if 'x' in data and 'y' in data and polygon.contains(Point(data['x'], data['y']))]

    # 提取子图
    G_sub = G.subgraph(nodes_inside).copy()

    return G_sub


def dfs_nx(graph, start_node, n):
    """
    Perform depth-limited search on a NetworkX graph starting from `start_node` up to depth `n`.
    this is a downstream search
    Parameters:
        graph: NetworkX graph object.
        start_node: Starting node for the search.
        n: Maximum depth to search (0-based).

    Returns:
        visited_nodes: List of nodes visited during the search, ordered by visitation time.
    """
    visited = []  # To keep track of visited nodes and their depths
    stack = [(start_node, 0)]  # Stack for DFS; each element is a tuple (current_node, current_depth)

    while stack:
        node, depth = stack.pop()

        # Check if the node has already been visited
        if node not in [v[0] for v in visited]:
            visited.append((node, depth))

            # If we haven't reached the maximum depth, add neighbors to the stack
            if depth < n:
                # Push neighbors onto the stack (reverse order to maintain LIFO)
                # You can modify this part to change the search order (e.g., reverse=True for different traversal)
                for neighbor in reversed(list(nx.neighbors(graph, node))):
                    stack.append((neighbor, depth + 1))

    # Extract only the nodes from visited
    return [node for node, _ in visited]


def bidirectional_search(G, start_node, max_depth):
    if start_node not in G:
        return []

    visited = set()
    queue = deque([(start_node, 0)])
    visited.add(start_node)

    while queue:
        current_node, depth = queue.popleft()

        if depth >= max_depth:
            continue

        # Explore upstream nodes (predecessors)
        for predecessor in G.predecessors(current_node):
            if predecessor not in visited:
                visited.add(predecessor)
                queue.append((predecessor, depth + 1))

        # Explore downstream nodes (successors)
        for successor in G.successors(current_node):
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, depth + 1))

    return list(visited)
def get_graph_central_node(graph, type="closeness"):
    """
    获取子图的中心点，有以下几种方案：
    度中心性（Degree Centrality） ：衡量一个节点有多少直接连接。
    接近中心性（Closeness Centrality） ：衡量一个节点到其他所有节点的平均距离有多近。 default setups
    介数中心性（Betweenness Centrality） ：衡量一个节点在最短路径中的重要程度，即有多少对节点的最短路径经过它。
    get the graph's central node, if type is closeness return nx.closeness_centrality
    if type is degree, then returns nx.degree_centrality
    if type is betweenness, then returns nx.betweenness_centrality

    :param graph:
    :param type:
    :return:  graph node id
    """
    if type == "degree":
        centrality = nx.degree_centrality(graph)
    elif type == "betweenness":
        centrality = nx.betweenness_centrality(graph)
    elif type == "closeness":
        centrality = nx.closeness_centrality(graph)
    else:
        raise ValueError(f"Invalid type: {type}. Valid types are 'degree', 'betweenness', or 'closeness'.")

    if not centrality:
        return None  # Handle empty graph, though problem might assume non-empty

    max_node = max(centrality, key=centrality.get)
    return max_node


# 示例：获取广州的路网并裁剪
# city_name = "Guangzhou, China"
# osm_file = "/Users/convel/Documents/GIS数据/广州市/gz221123.osm"
# G = ox.graph_from_xml(filepath=osm_file)
#
# # G = ox.graph_from_xml(osm_file)
# # # G = ox.graph_from_bbox(113.42633, 113.46977, 23.14092, 23.16476, network_type="drive", simplify=True)
# # fig, ax = ox.plot_graph(G, node_size=10, edge_linewidth=0.5, show=False)
# ox.save_graphml(G, "../../data/guangzhou_all.graphml")
# G = ox.load_graphml("../../data/guangzhou_all.graphml")
# 113.42633,23.14092 : 113.46977,23.16476
# 定义一个多边形区域（注意最后一个点应和第一个点相同）

# poly_coords = [(113.4000, 23.1031), (113.5738, 23.1031), (113.5738, 23.1985), (113.5738, 23.1031), (113.4000, 23.1031)]
# # 获取多边形内的子图
# G_sub = getSubGraphInPoly(G, poly_coords)
# ox.save_graphml(G_sub, "../../data/test_hp_graph.graphml")
# ox.save_graph_geopackage(G_sub, "../../data/test_hp_graph.gpkg")
# G = ox.load_graphml("../../data/test_hp_graph.graphml")
# 可视化原图和子图
# fig, ax = ox.plot_graph(G, node_size=10, edge_linewidth=0.5, show=False)
# ox.plot_graph(G_sub, node_color="red", edge_color="red", node_size=10, edge_linewidth=1, ax=ax)
# print("done")

# closeness = nx.closeness_centrality(G)
# sorted_items = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
# top_item = (sorted_items[0][0], sorted_items[0][1])
#
# sub_nodes_ids = bidirectional_search(G, sorted_items[0][0], 20)

# for node in sub_nodes_ids:
#     print("t;"+str(G.nodes.get(node)['x'])+";"+str(G.nodes.get(node)['y']))

def get_downstream_depth_info(graph, node_id):
    if not graph or node_id not in graph:
        return []
    visited = set()
    queue = deque([node_id])
    visited.add(node_id)
    level_sizes = []
    while queue:
        level_size = len(queue)
        level_sizes.append(level_size)
        for _ in range(level_size):
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return level_sizes



def reverse_graph(G):
    """
    Reverse all the edges of a directed graph.

    Parameters:
        G (networkx.DiGraph): The original directed graph.

    Returns:
        reversed_G (networkx.DiGraph): A new graph with all edge directions reversed.
    """
    reversed_G = nx.DiGraph()
    reversed_G.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        reversed_G.add_edge(v, u, **data)

    return reversed_G
def get_bi_dir_depth_info(G, node_id):
    reversed_G = reverse_graph(G)
    return (get_downstream_depth_info(G,node_id),get_downstream_depth_info(reversed_G,node_id))
def get_bi_avg_graph_depth(G, node_id):
    max_downStream = max(0, len(get_downstream_depth_info(G, node_id))) # bug_fix:max() to len()
    reversed_G = reverse_graph(G)
    max_upStream = max(0, len(get_downstream_depth_info(reversed_G, node_id))) # bug_fix:max() to len()
    return (max_downStream+max_upStream) / 2

def get_graph_bounding_box(graph):
    """

    :param graph:
    :return:
    """

    pass
def get_adj_subGraphs(graph,center_node_coord, graph_node_lon='x', graph_node_lat='y', search_step = 0.005):
    """
    use KDTree to get the nearest node on the gpah
    :param graph: networkX graph
    :param graph_node_lon: node features with longitude recorded
    :param graph_node_lat:
    :param center_node_coord:
    :return:
    """
    # 获取节点坐标
    pos_Xs = nx.get_node_attributes(graph, graph_node_lon)
    search_step = float(search_step) # bug_fix : 限定search_step的类型为float
    # 构建 KDTree
    node_list = list(pos_Xs.keys())  # 节点编号
    coord_lon_list = list(pos_Xs.values())  # 坐标列表
    coord_lat_list = list(nx.get_node_attributes(graph, graph_node_lat).values())  # 坐标列表
    tree = KDTree(np.array(list(zip(coord_lon_list, coord_lat_list)))) # bug_fix : 增加np.array(list())

    # 查找最近节点
    _, nearest_idx_0 = tree.query((center_node_coord[0]-search_step,center_node_coord[1]-search_step))  # 返回最邻近点的索引
    _, nearest_idx_1 = tree.query((center_node_coord[0]-search_step,center_node_coord[1]+search_step))  # 返回最邻近点的索引
    _, nearest_idx_2 = tree.query((center_node_coord[0]+search_step,center_node_coord[1]+search_step))  # 返回最邻近点的索引
    _, nearest_idx_3 = tree.query((center_node_coord[0]+search_step,center_node_coord[1]-search_step))  # 返回最邻近点的索引
    return [node_list[nearest_idx_0], node_list[nearest_idx_1], node_list[nearest_idx_2], node_list[nearest_idx_3]]