import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon, Point
from collections import deque


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

        if depth == max_depth:
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

# 示例：获取广州的路网并裁剪
# city_name = "Guangzhou, China"
# osm_file = "/Users/convel/Documents/GIS数据/广州市/gz221123.osm"
# G = ox.graph_from_xml(filepath=osm_file)
#
# # G = ox.graph_from_xml(osm_file)
# # # G = ox.graph_from_bbox(113.42633, 113.46977, 23.14092, 23.16476, network_type="drive", simplify=True)
# # fig, ax = ox.plot_graph(G, node_size=10, edge_linewidth=0.5, show=False)
# ox.save_graphml(G, "../../data/guangzhou_all.graphml")
G = ox.load_graphml("../../data/guangzhou_all.graphml")
# 113.42633,23.14092 : 113.46977,23.16476
# 定义一个多边形区域（注意最后一个点应和第一个点相同）

# poly_coords = [(113.4000, 23.1031), (113.5738, 23.1031), (113.5738, 23.1985), (113.5738, 23.1031), (113.4000, 23.1031)]
# # 获取多边形内的子图
# G_sub = getSubGraphInPoly(G, poly_coords)
# ox.save_graphml(G_sub, "../../data/test_hp_graph.graphml")
# ox.save_graph_geopackage(G_sub, "../../data/test_hp_graph.gpkg")
G = ox.load_graphml("../../data/test_hp_graph.graphml")
# 可视化原图和子图
# fig, ax = ox.plot_graph(G, node_size=10, edge_linewidth=0.5, show=False)
# ox.plot_graph(G_sub, node_color="red", edge_color="red", node_size=10, edge_linewidth=1, ax=ax)
# print("done")

closeness = nx.closeness_centrality(G)
sorted_items = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
top_item = (sorted_items[0][0], sorted_items[0][1])
print(G.nodes)

sub_nodes_ids = bidirectional_search(G, sorted_items[0][0], 20)
print("id;x;y")

# for node in sub_nodes_ids:
#     print("t;"+str(G.nodes.get(node)['x'])+";"+str(G.nodes.get(node)['y']))

def compute_level_sizes(graph, start_node):
    if not graph or start_node not in graph:
        return []
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
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

depth_info = compute_level_sizes(G, sorted_items[0][0])

print("done!")
