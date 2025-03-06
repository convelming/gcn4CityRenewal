import networkx as nx
import numpy as np

from src.subGraphSearh.similarityCals import cal_graph_degree_distribution, cal_KL_divergence, cal_cluster_coe_diff, \
    cal_shortest_path_length_ratio, cal_graph_cosin_simularity

G1 = nx.DiGraph()
G1.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 2), (2, 5)])

G2 = nx.DiGraph()
G2.add_edges_from([(30, 20), (30, 60),(20, 30), (30, 40), (40, 20), (20, 40),  (20, 50), (20, 10)])

# 计算 KL 散度
kl_value = cal_KL_divergence(G1, G2, in_weight=0.5)

graph_degrees1 = cal_graph_degree_distribution(G1, direction="in")
graph_degrees2 = cal_graph_degree_distribution(G2, direction="out")
print(graph_degrees1)
print(graph_degrees2)
print(f"KL 散度: {kl_value:.4f}")

cluster_coe_diff = cal_cluster_coe_diff(G1, G2)
print(f"cluster_coe_diff:{cluster_coe_diff:.2f}")
splr = cal_shortest_path_length_ratio(G1, G2)
print(f"splr:{splr:.2f}")



# 示例测试
if __name__ == "__main__":
    np.random.seed(42)
    X_a = np.random.rand(4, 5)  # 子图1有4个节点，每个节点5维特征
    X_b = np.random.rand(6, 5)  # 子图2有6个节点，每个节点5维特征

    sim_max_pooling = cal_graph_cosin_simularity(X_a, X_b, "max_pooling")
    sim_global_opt = cal_graph_cosin_simularity(X_a, X_b, "global_opt")

    print(f"最大池化匹配相似度: {sim_max_pooling:.4f}")
    print(f"最优传输匹配相似度: {sim_global_opt:.4f}")