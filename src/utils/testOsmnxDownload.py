import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random

# 获取某个城市的路网数据（默认是有向图）
# city_name = "Guangzhou, China"
# G = ox.graph_from_place(city_name, network_type="all")
#network_type="drive"：仅获取可通行的机动车道路。
# network_type="walk"：步行网络（包括步道、人行道）。
# network_type="all"：获取所有道路类型。
# 输出基本信息
# print(nx.info(G))


# 设置边界框 (南, 西, 北, 东)
G = ox.graph_from_bbox(north=23.13929, south=23.12737, east=113.47165, west=113.44993, network_type="all")
# print(nx.info(G))


# 绘制路网
fig, ax = ox.plot_graph(G, node_size=5, edge_linewidth=0.7, bgcolor="white")
plt.show()

# 访问所有节点
nodes = list(G.nodes(data=True))
print("示例节点:", nodes[:5])  # 前 5 个节点

# 访问所有边
edges = list(G.edges(data=True))
print("示例边:", edges[:5])  # 前 5 条边



# 随机选择两个节点
source, target = random.sample(G.nodes, 2)

# 计算最短路径（基于长度）
shortest_path = nx.shortest_path(G, source=source, target=target, weight="length")

print("最短路径:", shortest_path)

# 可视化最短路径
ox.plot_graph_route(G, shortest_path, route_linewidth=3, node_size=50, bgcolor="white")

# 有向转无向
G_undirected = G.to_undirected()
print(nx.info(G_undirected))


# 可以将 NetworkX 图保存到 GraphML 格式，以便以后重新加载：


ox.save_graphml(G, "guangzhou_graph.graphml")

# 重新加载
G_loaded = ox.load_graphml("guangzhou_graph.graphml")


# 如果想提取一个特定范围的子图（比如某个小区域），可以：

# 获取子图（周围 1km 半径内）
center_node = list(G.nodes)[0]  # 选择一个中心节点
G_sub = ox.truncate_graph_radius(G, source_node=center_node, max_distance=1000)

# 绘制子图
ox.plot_graph(G_sub)




# 按城市获取 OSM 数据	ox.graph_from_place("Guangzhou, China", network_type="drive")
# 按边界框获取 OSM 数据	ox.graph_from_bbox(north, south, east, west, network_type="drive")
# 可视化 OSM 路网	ox.plot_graph(G)
# 获取节点和边属性	G.nodes(data=True) / G.edges(data=True)
# 计算最短路径	nx.shortest_path(G, source, target, weight="length")
# 转换为无向图	G.to_undirected()
# 保存 OSM 图	ox.save_graphml(G, "graph.graphml")
# 加载 OSM 图	ox.load_graphml("graph.graphml")


